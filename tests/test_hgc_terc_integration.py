"""CPU integration checks without pretrained weights."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch

from diffusers_helper.models.hunyuan_video_packed_hgc_terc_morder import HunyuanVideoTransformer3DModelPacked as Model
from diffusers_helper.models.cache_functions import cache_init_terc
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan


class Block:
    def __init__(self):
        self.calls = 0

    def norm1(self, x, emb):
        return (x + emb,)

    def __call__(self, x, text, emb, mask, rope):
        self.calls += 1
        return x + emb, text


def model():
    m = SimpleNamespace(config={'patch_size': 1, 'patch_size_t': 1},
                        image_projection=None, high_quality_fp32_output_for_inference=False)
    m.transformer_blocks = [Block()]
    m.single_transformer_blocks = [Block()]
    m.gradient_checkpointing_method = lambda fn, *args: fn(*args)
    m.process_input_hidden_states = lambda x, *args: (torch.cat((torch.ones(1, 1, 1), x.reshape(1, 1, 1)), dim=1), None)
    m.time_text_embed = lambda t, *args: t.reshape(1, 1, 1)
    m.context_embedder = lambda text, *args: text
    m.norm_out = lambda x, emb: x
    m.proj_out = lambda x: x
    return m


def forward(m, section, cache=None, current=None, t=1.):
    return Model.forward(m, torch.zeros(1, 1, 1, 1, 1), torch.tensor([t]),
                         torch.zeros(1, 1, 1), torch.ones(1, 1, dtype=torch.long),
                         None, None, section_index=section, cache_dic=cache,
                         current=current, return_dict=False)[0]


class IntegrationTests(unittest.TestCase):
    def test_sections_and_request_reset(self):
        m = model()
        cache, current = cache_init_terc(5)
        Model.initialize_arcache(m, num_steps=5, rel_l1_thresh=0.3)
        for i in range(5):
            forward(m, 0, cache, current)
            self.assertEqual(current['step'], i)
        self.assertEqual(current['activated_steps'], [0, 2, 4])
        self.assertEqual(len(m.first_section_residuals), 5)
        self.assertEqual(m.transformer_blocks[0].calls, 3)
        reference = m.first_section_residuals
        Model.initialize_arcache(m, num_steps=5, rel_l1_thresh=0.3, section_idx=1)
        for _ in range(5):
            self.assertTrue(torch.isfinite(forward(m, 1)).all())
        self.assertIs(m.first_section_residuals, reference)
        self.assertEqual(len(m.current_section_residuals), 2)
        self.assertEqual(m.transformer_blocks[0].calls, 6)
        Model.initialize_arcache(m, num_steps=5, section_idx=0)
        self.assertEqual(m.first_section_residuals, [])
        self.assertEqual(m.cnt, 0)

    def test_disabled_and_taylor_step_sync(self):
        m = model()
        Model.initialize_arcache(m, enable_arcache=False)
        for _ in range(3):
            forward(m, 0)
        self.assertEqual(m.transformer_blocks[0].calls, 3)
        cache, current = cache_init_terc(5, max_order=1)
        Model.initialize_arcache(m, num_steps=5)
        m.teacache_rescale_funcs = [lambda _: 0.]
        forward(m, 0, cache, current, t=1.)
        m.teacache_rescale_funcs = [lambda _: 1.]
        forward(m, 0, cache, current, t=2.)
        m.teacache_rescale_funcs = [lambda _: 0.]
        out = forward(m, 0, cache, current, t=3.)
        self.assertEqual(out.item(), 6.)
        self.assertEqual(current['type'], 'taylor_cache')
        self.assertEqual(current['activated_steps'], [0, 1])

    def test_pipeline_forwards_cache_objects(self):
        cache, current = cache_init_terc(3)
        seen = []
        def transformer(**kwargs):
            seen.append(kwargs)
            return (torch.zeros_like(kwargs['hidden_states']),)
        def sampler(k_model, latents, sigmas, **kwargs):
            return k_model(latents, sigmas[0].expand(1), **kwargs['extra_args'])
        with patch('diffusers_helper.pipelines.k_diffusion_hunyuan.sample_unipc', sampler):
            sample_hunyuan(transformer, width=8, height=8, frames=1,
                           device='cpu', dtype=torch.float32, generator=torch.Generator(),
                           prompt_embeds=torch.zeros(1, 1, 1),
                           prompt_embeds_mask=torch.ones(1, 1), prompt_poolers=torch.zeros(1, 1),
                           section_index=0, cache_dic=cache, current=current)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]['section_index'], 0)
        self.assertIs(seen[0]['cache_dic'], cache)
        self.assertIs(seen[0]['current'], current)


if __name__ == '__main__':
    unittest.main()
