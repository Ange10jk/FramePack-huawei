"""ARCache scaffold, adapted directly from DiCache's HunyuanVideo forward.

Source: Bujiazi/DiCache, fdbe20b669c9174bbed5ec994de073fd881c8010,
HunyuanVideo/run_hunyuanvideo_dicache.py (Jiazi Bu et al.).
Changes: FramePack input/output, history-only scheduling, first-segment reference,
and first-segment-guided gamma. The full/skip branches retain DiCache's structure.

Usage after wiring the F1 worker (not installed automatically):
    model.forward = MethodType(arcache_forward, model)
    initialize_arcache(model, segment_id=0, num_steps=25)
    # Call initialize_arcache again for each subsequent segment of this request.

TODO: wire worker/wrapper call sites. This scaffold
supports batch=1, real CFG=1 only; distilled guidance is still supported.
HGC monitors the first block norm1 modulated visual input, excluding only the
current noisy segment (the extra start-image anchor is included). Callers supply
the same denoising schedule for each segment.
The monitored layer is a concrete scaffold choice, not verified author layer parity.
"""
# TaylorSeer source: Shenyi-Z/TaylorSeer (Jiacheng Liu et al.),
# 704ee98c74f7f04da443daa3c0aa2cc7803d86e3,
# TaylorSeer-HunyuanVideo/hyvideo/modules/taylor_utils/__init__.py:
# derivative_approximation + taylor_formula, specialized to default max_order=1.
# Adaptation: whole-stack residual instead of per-module features; HGC scheduling.
import torch


def initialize_arcache(self, segment_id, num_steps=25, rel_l1_thresh=0.1,
                      taylor_predict=None):
    """Call once per segment. Segment 0 starts a new request and clears reference.

    taylor_predict(self, timestep) returns a residual tensor. It may read
    residual_window/full_step_window, but must not append predicted observations.
    """
    if segment_id == 0:
        self.reference_residuals = []
        self.reference_timesteps = []
        self.reference_sources = []
    self.segment_id, self.num_steps, self.cnt = segment_id, num_steps, 0
    self.rel_l1_thresh = rel_l1_thresh
    self.taylor_predict = taylor_predict if taylor_predict is not None else taylorseer_predict
    self.accumulated_rel_l1_distance = 0.0
    self.previous_history_tokens = None
    self.taylor_derivative = None
    self.residual_cache = None
    self.residual_window, self.full_step_window = [], []
    self.enable_teacache = False


def taylorseer_predict(self, timestep):
    """Official first-order Taylor formula, using step indices (not timestep).

    With only one full observation, the expansion contains just its zero order.
    Predictions never update the derivative or the full-compute anchors.
    """
    residual = self.residual_cache
    if self.taylor_derivative is not None:
        distance = self.cnt - self.full_step_window[-1]
        residual = residual + self.taylor_derivative * distance
    return residual


@torch.no_grad()
def arcache_forward(
    self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask,
    pooled_projections, guidance, latent_indices=None,
    clean_latents=None, clean_latent_indices=None,
    clean_latents_2x=None, clean_latent_2x_indices=None,
    clean_latents_4x=None, clean_latent_4x_indices=None,
    image_embeddings=None, attention_kwargs=None, return_dict=True,
):
    # FramePack input preparation replaces the original HunyuanVideo embedding.
    batch_size, _, num_frames, height, width = hidden_states.shape
    p, p_t = self.config['patch_size'], self.config['patch_size_t']
    tt, th, tw = num_frames // p_t, height // p, width // p
    original_context_length = tt * th * tw
    img, rope_freqs = self.process_input_hidden_states(
        hidden_states, latent_indices, clean_latents, clean_latent_indices,
        clean_latents_2x, clean_latent_2x_indices, clean_latents_4x, clean_latent_4x_indices)
    vec = self.gradient_checkpointing_method(self.time_text_embed, timestep, guidance, pooled_projections)
    txt = self.gradient_checkpointing_method(self.context_embedder, encoder_hidden_states, timestep, encoder_attention_mask)
    
    if self.image_projection is not None:
        extra = self.gradient_checkpointing_method(self.image_projection, image_embeddings)
        extra_mask = torch.ones((batch_size, extra.shape[1]), dtype=encoder_attention_mask.dtype,
                                device=encoder_attention_mask.device)
        txt = torch.cat([extra, txt], dim=1)
        encoder_attention_mask = torch.cat([extra_mask, encoder_attention_mask], dim=1)
    
    txt = txt[:, :int(encoder_attention_mask.sum().item())]
    attention_mask = (None, None, None, None)
    history_token_count = img.shape[1] - original_context_length
    step_t = float(timestep.flatten()[0].item())

    skip_forward = False
    self.last_gamma = None

    # --------------------- History-Guided Cache -----------------------------
    # Same extraction point as FramePack TeaCache: before the first attention.
    modulated_input = self.transformer_blocks[0].norm1(img, emb=vec)[0]
    # All clean visual tokens: 4x history, 2x history, start anchor, 1x history.
    history_tokens = modulated_input[:, :history_token_count]

    if self.cnt != 0: # estimation; only the first step forces full computation
        denominator = self.previous_history_tokens.abs().mean()
        delta_y = (history_tokens - self.previous_history_tokens).abs().mean() / denominator
        self.accumulated_rel_l1_distance += delta_y.item()
        skip_forward = self.accumulated_rel_l1_distance <= self.rel_l1_thresh
            
    self.previous_history_tokens = history_tokens.detach().clone()
    if not skip_forward:
        self.accumulated_rel_l1_distance = 0.0

    if skip_forward:
        # --------------------- Enhanced Residual Correction -----------------
        if self.segment_id == 0:
            residual = self.taylor_predict(self, timestep)
            source = "taylor_predicted"
        elif len(self.residual_window) >= 2:
            tb, ta = self.full_step_window[-2:]
            r_t, r_a, r_b = (self.reference_residuals[i] for i in (self.cnt, ta, tb))
            # Replace DiCache's probe-residual gamma with first-segment geometry.
            denominator = (r_a - r_b).abs().mean()
            gamma = ((r_t - r_b).abs().mean() / denominator).item()
            residual = self.residual_window[-2] + gamma * (self.residual_window[-1] - self.residual_window[-2])
            source = "erc_predicted"
            self.last_gamma = gamma
        else:
            residual, source = self.residual_cache, "cached"
        img = img + residual.to(img)
    else:
        ori_img = img.clone()

        # --------------------- Pass through DiT blocks -----------------------
        for block in self.transformer_blocks:
            img, txt = self.gradient_checkpointing_method(
                block, img, txt, vec, attention_mask, rope_freqs)
        for block in self.single_transformer_blocks:
            img, txt = self.gradient_checkpointing_method(
                block, img, txt, vec, attention_mask, rope_freqs)

        residual = img - ori_img  # block[0] to block[-1] residual.
        # TaylorSeer derivative_approximation with max_order=1, first_enhance=1.
        # Update only from true full observations, before replacing the old cache.
        if self.segment_id == 0 and self.residual_cache is not None:
            distance = self.cnt - self.full_step_window[-1]
            self.taylor_derivative = (residual - self.residual_cache) / distance

        self.residual_cache = residual.detach().clone()
        self.residual_window.append(self.residual_cache)
        self.full_step_window.append(self.cnt)
        self.residual_window = self.residual_window[-2:]
        self.full_step_window = self.full_step_window[-2:]
        source = "full"

    # Save the effective first-segment residual before the output head.
    if self.segment_id == 0:
        self.reference_residuals.append(residual.detach().cpu().clone())
        self.reference_timesteps.append(step_t)
        self.reference_sources.append(source)
    self.last_cache_source = source

    # ---------------------------- FramePack output --------------------------
    img = self.gradient_checkpointing_method(self.norm_out, img, vec)
    img = img[:, -original_context_length:]
    if self.high_quality_fp32_output_for_inference:
        img = img.float()
        if self.proj_out.weight.dtype != torch.float32:
            self.proj_out.to(dtype=torch.float32)
    img = self.gradient_checkpointing_method(self.proj_out, img)
    img = img.reshape(batch_size, tt, th, tw, -1, p_t, p, p)
    img = img.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(batch_size, -1, tt*p_t, th*p, tw*p)
    self.cnt += 1  # No implicit modulo reset; the worker initializes each segment.
    if return_dict:
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        return Transformer2DModelOutput(sample=img)
    return (img,)
