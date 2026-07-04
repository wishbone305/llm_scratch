"""Tests for autoregressive generation (both backends), written before the samplers."""

import mlx.core as mx
import torch

from llmscratch import model_mlx, model_torch
from llmscratch.config import ModelConfig
from llmscratch.sample_mlx import generate as gen_mlx
from llmscratch.sample_torch import generate as gen_torch


def cfg() -> ModelConfig:
    return ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, block_size=32, vocab_size=256)


def test_torch_generate_shape():
    m = model_torch.GPT(cfg())
    m.eval()
    out = gen_torch(m, torch.tensor([[1, 2, 3]]), max_new_tokens=5, block_size=32, top_k=10)
    assert out.shape == (1, 8)


def test_torch_generate_greedy_is_deterministic():
    m = model_torch.GPT(cfg())
    m.eval()
    idx = torch.tensor([[1, 2, 3]])
    o1 = gen_torch(m, idx, 5, 32, temperature=0.0)
    o2 = gen_torch(m, idx, 5, 32, temperature=0.0)
    assert torch.equal(o1, o2)


def test_mlx_generate_shape():
    m = model_mlx.GPT(cfg())
    out = gen_mlx(m, mx.array([[1, 2, 3]]), max_new_tokens=5, block_size=32, top_k=10)
    assert out.shape == (1, 8)


def test_mlx_generate_greedy_is_deterministic():
    m = model_mlx.GPT(cfg())
    idx = mx.array([[1, 2, 3]])
    o1 = gen_mlx(m, idx, 5, 32, temperature=0.0)
    o2 = gen_mlx(m, idx, 5, 32, temperature=0.0)
    assert bool(mx.array_equal(o1, o2))


def test_torch_generate_top_p_and_penalty():
    m = model_torch.GPT(cfg())
    m.eval()
    out = gen_torch(m, torch.tensor([[1, 2, 3, 4]]), max_new_tokens=8, block_size=32,
                    temperature=0.9, top_k=50, top_p=0.9, repetition_penalty=1.3)
    assert out.shape == (1, 12)
    assert int(out.max()) < cfg().vocab_size


def test_mlx_generate_top_p_and_penalty():
    import numpy as np
    m = model_mlx.GPT(cfg())
    out = gen_mlx(m, mx.array([[1, 2, 3, 4]]), max_new_tokens=8, block_size=32,
                  temperature=0.9, top_k=50, top_p=0.9, repetition_penalty=1.3)
    assert out.shape == (1, 12)
    assert int(np.asarray(out).max()) < cfg().vocab_size
