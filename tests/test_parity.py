"""Cross-framework parity: the MLX and PyTorch models must be numerically equivalent.

This is the safeguard that keeps the two implementations from silently drifting apart.
"""

import mlx.core as mx
import numpy as np
import torch

from llmscratch import model_mlx, model_torch
from llmscratch.config import ModelConfig, estimate_params
from llmscratch.convert import load_mlx_into_torch, load_torch_into_mlx


def cfg(**kw) -> ModelConfig:
    base = dict(d_model=64, n_layers=2, n_heads=4, d_ff=128, block_size=32, vocab_size=256)
    base.update(kw)
    return ModelConfig(**base)


def test_param_counts_match():
    c = cfg()
    m = model_mlx.GPT(c)
    t = model_torch.GPT(c)
    assert m.num_params() == t.num_params() == estimate_params(c)


def test_logits_match_mlx_into_torch():
    mx.set_default_device(mx.cpu)
    c = cfg()
    m = model_mlx.GPT(c)
    t = model_torch.GPT(c)
    load_mlx_into_torch(m, t)
    t.eval()

    idx = np.array([[5, 9, 1, 3, 7, 0, 2, 8]], dtype=np.int64)
    with torch.no_grad():
        lt = t(torch.from_numpy(idx)).numpy()
    lm = np.asarray(m(mx.array(idx)))

    assert lt.shape == lm.shape
    max_abs = float(np.max(np.abs(lt - lm)))
    assert max_abs < 1e-3, f"logits diverge: max|delta|={max_abs}"


def test_logits_match_torch_into_mlx():
    mx.set_default_device(mx.cpu)
    c = cfg()
    t = model_torch.GPT(c)
    m = model_mlx.GPT(c)
    load_torch_into_mlx(t, m)
    t.eval()

    idx = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
    with torch.no_grad():
        lt = t(torch.from_numpy(idx)).numpy()
    lm = np.asarray(m(mx.array(idx)))
    max_abs = float(np.max(np.abs(lt - lm)))
    assert max_abs < 1e-3, f"logits diverge: max|delta|={max_abs}"
