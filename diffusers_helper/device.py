"""Accelerator discovery and small cross-backend helpers.

FramePack historically assumed CUDA at import time. Keeping device discovery in
one place lets the rest of the project work with TorchNPU while preserving the
CUDA path for existing users.
"""

import os
import warnings

import torch


torch_npu = None
_npu_import_error = None

try:
    import torch_npu as _torch_npu

    torch_npu = _torch_npu
except (ImportError, OSError) as exc:
    _npu_import_error = exc


def _npu_is_available():
    return torch_npu is not None and hasattr(torch, "npu") and torch.npu.is_available()


def _select_accelerator():
    requested = os.environ.get("FRAMEPACK_DEVICE", "auto").strip().lower()
    if requested == "auto":
        if _npu_is_available():
            return torch.device(f"npu:{torch.npu.current_device()}")
        if torch.cuda.is_available():
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        warnings.warn(
            "Neither Ascend NPU nor CUDA is available. FramePack will use CPU, "
            "which is intended only for diagnostics.",
            RuntimeWarning,
        )
        return torch.device("cpu")

    requested_type = requested.split(":", 1)[0]
    if requested_type == "npu":
        if torch_npu is None:
            detail = f" ({_npu_import_error})" if _npu_import_error else ""
            raise RuntimeError(
                "FRAMEPACK_DEVICE requests an Ascend NPU, but torch_npu could not "
                f"be imported{detail}. Install a torch-npu build matching PyTorch and CANN."
            )
        if not torch.npu.is_available():
            raise RuntimeError("FRAMEPACK_DEVICE requests an Ascend NPU, but no NPU is available.")
        device = torch.device(requested)
        if device.index is None:
            device = torch.device("npu", torch.npu.current_device())
        torch.npu.set_device(device)
    elif requested_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("FRAMEPACK_DEVICE requests CUDA, but CUDA is not available.")
        device = torch.device(requested)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    elif requested_type == "cpu":
        device = torch.device(requested)
    else:
        raise ValueError(f"Unsupported FRAMEPACK_DEVICE: {requested}")

    return device


accelerator = _select_accelerator()
accelerator_type = accelerator.type
accelerator_api = getattr(torch, accelerator_type, None)


def set_device(device=None):
    """Bind the accelerator in the calling thread before allocating tensors."""
    device = accelerator if device is None else torch.device(device)
    if device.type in ("npu", "cuda"):
        getattr(torch, device.type).set_device(device)


def empty_cache(device=None):
    device = accelerator if device is None else torch.device(device)
    if device.type in ("npu", "cuda"):
        api = getattr(torch, device.type)
        # Cache operations act on the current device, which is thread-local.
        with api.device(device):
            api.empty_cache()


def synchronize(device=None):
    device = accelerator if device is None else torch.device(device)
    if device.type in ("npu", "cuda"):
        getattr(torch, device.type).synchronize(device)


def is_npu():
    return accelerator_type == "npu"
