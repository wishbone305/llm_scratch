"""PyTorch device + precision policy. Keeps the same training code portable across CUDA/MPS/CPU."""

from __future__ import annotations

import torch


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_policy(device: torch.device) -> dict:
    """CUDA -> bf16 autocast + torch.compile; MPS/CPU -> fp32, no compile (compile is flaky on MPS)."""
    if device.type == "cuda":
        # Ampere+ (A100/L4) do bf16 in tensor cores; Turing (T4) does NOT -> use fp16 there.
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return {"amp_dtype": amp_dtype, "use_amp": True, "use_compile": True}
    return {"amp_dtype": torch.float32, "use_amp": False, "use_compile": False}
