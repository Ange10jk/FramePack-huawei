"""Source-confirmed group definitions and shared FramePack-F1 parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT.parent
DATASET = BASE / "vbench_i2v_inputs"
OUTPUTS = ROOT / "outputs"
HF_CACHE = ROOT / "hf_download" / "hub"

GROUPS = {
    "vanilla100": dict(backend="huawei", mode="None", steps=25),
    "vanilla50": dict(backend="huawei", mode="None", steps=13),
    "teacache_slow": dict(backend="tea", mode="TeaCache", steps=25, threshold=0.10),
    "teacache_fast": dict(backend="tea", mode="TeaCache", steps=25, threshold=0.20),
    "taylorseer": dict(backend="taylor", mode="TaylorSeer", steps=25,
                       max_order=0, fresh_threshold=5, first_enhance=1),
    "arcache_slow": dict(backend="huawei", mode="ARCache", steps=25,
                         threshold=0.10, erc=True, order=2, ksi=0.5),
    "arcache_fast": dict(backend="huawei", mode="ARCache", steps=25,
                         threshold=0.20, erc=True, order=2, ksi=0.5),
}
BACKENDS = {"huawei": ROOT, "tea": BASE / "FramePack", "taylor": BASE / "FramePack-TaylorSeer"}
PARAMS = dict(sections=4, length_argument=5.0, latent_window=9, resolution_bucket=640,
              frames=145, raw_fps=30, eval_fps=24, crf=16, sampler="unipc",
              cfg=1.0, guidance=10.0, rescale=0.0, memory_preservation_gb=8.0,
              negative_prompt="", high_vram=True, transformer_dtype="bfloat16",
              encoders_vae_dtype="float16", transformer_output_fp32=True,
              warmup_per_process=1,
              timing="device sync before and after worker including final raw MP4 encode")
IMAGE_GROUPS = ("origin", "crop/1-1", "crop/8-5", "crop/7-4", "crop/16-9")
REPOS = {
    "hunyuanvideo-community/HunyuanVideo": (
        "e8c2aaa66fe3742a32c11a6766aecbf07c56e773",
        ("text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae"),
    ),
    "lllyasviel/flux_redux_bfl": (
        "45b801affc54ff2af4e5daf1b282e0921901db87", ("feature_extractor", "image_encoder"),
    ),
    "lllyasviel/FramePack_F1_I2V_HY_20250503": (
        "ab239828e0b384fed75580f186f078717d4020f7", (".",),
    ),
}
