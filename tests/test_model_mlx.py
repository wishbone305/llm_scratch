"""Tests for the MLX model (written before the implementation)."""

import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from llmscratch.config import ModelConfig, estimate_params
from llmscratch.model_mlx import GPT


def tiny_cfg(**kw) -> ModelConfig:
    base = dict(d_model=64, n_layers=2, n_heads=4, d_ff=128, block_size=32, vocab_size=256)
    base.update(kw)
    return ModelConfig(**base)


def test_forward_shape():
    cfg = tiny_cfg()
    model = GPT(cfg)
    logits = model(mx.zeros((2, 16), dtype=mx.int32))
    mx.eval(logits)
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_loss_is_finite():
    cfg = tiny_cfg()
    model = GPT(cfg)
    x = mx.random.randint(0, cfg.vocab_size, (2, 16))
    y = mx.random.randint(0, cfg.vocab_size, (2, 16))
    loss = model.loss(x, y)
    mx.eval(loss)
    assert math.isfinite(float(loss))


def test_num_params_matches_analytic_estimate():
    cfg = tiny_cfg()
    model = GPT(cfg)
    assert model.num_params() == estimate_params(cfg)


def test_gradients_populate():
    cfg = tiny_cfg()
    model = GPT(cfg)
    x = mx.random.randint(0, cfg.vocab_size, (2, 16))
    y = mx.random.randint(0, cfg.vocab_size, (2, 16))

    def loss_fn(m, xx, yy):
        return m.loss(xx, yy)

    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    mx.eval(loss, grads)
    flat = tree_flatten(grads)
    assert len(flat) > 0
    assert any(float(mx.sum(mx.abs(g))) > 0 for _, g in flat)


def test_position_sensitivity_rope():
    # Swapping the order of the prefix tokens must change the last-position logits.
    # A bag-of-tokens model (no positional info) would give identical output here, so
    # this specifically exercises RoPE being wired into attention.
    cfg = tiny_cfg()
    model = GPT(cfg)
    l1 = model(mx.array([[5, 9, 1]]))[:, -1]
    l2 = model(mx.array([[9, 5, 1]]))[:, -1]
    assert float(mx.max(mx.abs(l1 - l2))) > 1e-5


def test_grad_checkpoint_equivalence():
    import numpy as np
    from mlx.utils import tree_flatten

    cfg = tiny_cfg()
    base = GPT(cfg)
    ckpt_model = GPT(cfg.with_(grad_checkpoint=True))
    ckpt_model.update(base.parameters())
    x = mx.random.randint(0, cfg.vocab_size, (2, 16))
    y = mx.random.randint(0, cfg.vocab_size, (2, 16))

    def lf(m, xx, yy):
        return m.loss(xx, yy)

    lb, gb = nn.value_and_grad(base, lf)(base, x, y)
    lc, gc = nn.value_and_grad(ckpt_model, lf)(ckpt_model, x, y)
    mx.eval(gb, gc)
    assert abs(float(lb) - float(lc)) < 1e-3  # forward must match
    gbf, gcf = dict(tree_flatten(gb)), dict(tree_flatten(gc))
    # EVERY parameter grad must match base — especially the transformer blocks, not just the
    # embedding (the earlier mx.checkpoint bug zeroed all block-param grads and this missed it).
    for k in gbf:
        assert np.allclose(np.array(gbf[k]), np.array(gcf[k]), atol=1e-4), f"grad mismatch: {k}"
    block_grad = sum(float(mx.sum(mx.abs(gcf[k]))) for k in gcf if k.startswith("blocks."))
    assert block_grad > 0  # regression guard: blocks must actually receive gradients
