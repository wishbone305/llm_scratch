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


def test_policy_cuda_is_bf16_with_compile():
    pol = device_policy(torch.device("cuda"))
    assert pol["use_amp"] is True
    assert pol["use_compile"] is True
    assert pol["amp_dtype"] == torch.bfloat16
