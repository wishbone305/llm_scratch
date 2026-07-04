"""Smoke tests: the LR schedule, and that both trainers actually reduce loss on a tiny,
learnable corpus (a repeating pattern). Written before the trainers exist.
"""

import numpy as np

from llmscratch.config import ModelConfig, TrainConfig
from llmscratch.schedule import cosine_lr


def _make_tiny_data(tmp_path):
    pattern = np.arange(50, dtype=np.uint16)
    data = np.tile(pattern, 400)  # 20,000 tokens, trivially learnable
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)
    data.tofile(d / "train.bin")
    data[:2000].tofile(d / "val.bin")
    return d


def _tiny_cfgs(steps: int = 60):
    model = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, block_size=32, vocab_size=64)
    train = TrainConfig(
        batch_size=8,
        grad_accum=1,
        max_steps=steps,
        learning_rate=3e-3,
        warmup_steps=5,
        eval_interval=10_000,  # no eval during the smoke run
        eval_iters=5,
        log_interval=10_000,
    )
    return model, train


def test_cosine_lr_schedule():
    def lr(s):
        return cosine_lr(s, learning_rate=1.0, min_lr=0.1, warmup_steps=10, max_steps=100)

    assert lr(0) < lr(9)                 # warmup ramps up
    assert abs(lr(9) - 1.0) < 0.2        # ~peak at end of warmup
    assert lr(100) == 0.1                # floor at/after max_steps
    assert 0.1 < lr(50) < 1.0            # decaying in between


def test_mlx_trainer_reduces_loss(tmp_path):
    from llmscratch import train_mlx

    data = _make_tiny_data(tmp_path)
    model_cfg, train_cfg = _tiny_cfgs()
    res = train_mlx.train(
        model_cfg, train_cfg, data_dir=data, out_dir=tmp_path / "out", run_name="t", log=False
    )
    losses = res["losses"]
    assert len(losses) == train_cfg.max_steps
    assert np.mean(losses[-10:]) < np.mean(losses[:10]) * 0.8


def test_torch_trainer_reduces_loss(tmp_path):
    from llmscratch import train_torch

    data = _make_tiny_data(tmp_path)
    model_cfg, train_cfg = _tiny_cfgs()
    res = train_torch.train(
        model_cfg,
        train_cfg,
        data_dir=data,
        out_dir=tmp_path / "out",
        run_name="t",
        log=False,
        device="cpu",
    )
    losses = res["losses"]
    assert np.mean(losses[-10:]) < np.mean(losses[:10]) * 0.8
