import os

import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

requested = os.environ.get("FRAMEPACK_DEVICE", "auto")
if requested == "auto":
    requested = "npu:0" if torch_npu is not None and torch.npu.is_available() else "cuda:0"
gpu = torch.device(requested)
if gpu.type not in ("npu", "cuda"):
    raise ValueError("FramePack requires an NPU or CUDA device")
accelerator_api = getattr(torch, gpu.type)
accelerator_api.set_device(gpu)
