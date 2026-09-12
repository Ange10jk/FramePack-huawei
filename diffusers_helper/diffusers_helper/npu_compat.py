"""Small PyTorch NPU compatibility shims used by FramePack."""

import torch


def install():
    """Replace unsupported 5D replicate padding with an equivalent concat."""
    original_pad = torch.nn.functional.pad
    if getattr(original_pad, "_framepack_npu_compat", False):
        return

    def compatible_pad(input, pad, mode="constant", value=None):
        if input.device.type == "npu" and mode == "replicate" and input.dim() == 5 and len(pad) == 6:
            output = input
            for dimension, before, after in ((4, pad[0], pad[1]), (3, pad[2], pad[3]), (2, pad[4], pad[5])):
                chunks = []
                if before:
                    shape = list(output.shape)
                    shape[dimension] = before
                    chunks.append(output.narrow(dimension, 0, 1).expand(*shape))
                chunks.append(output)
                if after:
                    shape = list(output.shape)
                    shape[dimension] = after
                    chunks.append(output.narrow(dimension, output.shape[dimension] - 1, 1).expand(*shape))
                output = torch.cat(chunks, dim=dimension)
            return output
        return original_pad(input, pad, mode, value)

    compatible_pad._framepack_npu_compat = True
    torch.nn.functional.pad = compatible_pad
