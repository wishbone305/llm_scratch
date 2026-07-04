"""Tests for the PyTorch model (written before the implementation)."""

import torch

from llmscratch.config import ModelConfig, estimate_params
from llmscratch.model_torch import GPT


def tiny_cfg(**kw) -> ModelConfig:
    base = dict(d_model=64, n_layers=2, n_heads=4, d_ff=128, block_size=32, vocab_size=256)
    base.update(kw)
    return ModelConfig(**base)


def test_forward_shape():
    cfg = tiny_cfg()
    model = GPT(cfg)
    logits = model(torch.zeros(2, 16, dtype=torch.long))
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_loss_is_finite():
    cfg = tiny_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    assert torch.isfinite(model.loss(x, y))


def test_num_params_matches_analytic_estimate():
    cfg = tiny_cfg()
    model = GPT(cfg)
    assert model.num_params() == estimate_params(cfg)


def test_gradients_populate():
    cfg = tiny_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    model.loss(x, y).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_position_sensitivity_rope():
    cfg = tiny_cfg()
    model = GPT(cfg)
    model.eval()
    with torch.no_grad():
        l1 = model(torch.tensor([[5, 9, 1]]))[:, -1]
        l2 = model(torch.tensor([[9, 5, 1]]))[:, -1]
    assert (l1 - l2).abs().max().item() > 1e-5


def test_grad_checkpoint_equivalence():
    cfg = tiny_cfg()
    base = GPT(cfg)
    ckpt_model = GPT(cfg.with_(grad_checkpoint=True))
    ckpt_model.load_state_dict(base.state_dict())
    base.train()
    ckpt_model.train()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    lb, lc = base.loss(x, y), ckpt_model.loss(x, y)
    assert torch.allclose(lb, lc, atol=1e-5)  # checkpointing must not change forward values
    lb.backward()
    lc.backward()
    gb, gc = dict(base.named_parameters()), dict(ckpt_model.named_parameters())
    assert all(torch.allclose(gb[n].grad, gc[n].grad, atol=1e-4) for n in gb)
