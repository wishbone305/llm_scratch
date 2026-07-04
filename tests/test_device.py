"""PyTorch device + precision policy."""

import torch

from llmscratch.device import device_policy, pick_device


def test_pick_device_honors_preference():
    assert pick_device("cpu").type == "cpu"


def test_policy_cpu_is_fp32_no_compile():
    pol = device_policy(torch.device("cpu"))
    assert pol["use_amp"] is False
    assert pol["use_compile"] is False
    assert pol["amp_dtype"] == torch.float32


def test_policy_cuda_bf16_when_supported(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    pol = device_policy(torch.device("cuda"))
    assert pol["use_amp"] is True
    assert pol["use_compile"] is True
    assert pol["amp_dtype"] == torch.bfloat16


def test_policy_cuda_falls_back_to_fp16_on_turing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    pol = device_policy(torch.device("cuda"))
    assert pol["amp_dtype"] == torch.float16
    assert pol["use_amp"] is True
