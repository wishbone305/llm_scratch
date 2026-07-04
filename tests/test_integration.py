"""End-to-end-ish: train (with eval+checkpoint) -> load checkpoint -> generate, both backends.
Also covers grad accumulation, directory-based data prep, and empty-split handling.
"""

import mlx.core as mx
import numpy as np
import torch

from llmscratch import train_mlx, train_torch
from llmscratch.config import ModelConfig, TrainConfig
from llmscratch.data import load_split, prepare_corpus
from llmscratch.sample_mlx import generate as gen_mlx
from llmscratch.sample_mlx import load_model as load_mlx
from llmscratch.sample_torch import generate as gen_torch
from llmscratch.sample_torch import load_model as load_torch


def _tiny_data(tmp_path):
    data = np.tile(np.arange(50, dtype=np.uint16), 400)
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)
    data.tofile(d / "train.bin")
    data[:2000].tofile(d / "val.bin")
    return d


def _cfgs():
    model = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, block_size=32, vocab_size=64)
    train = TrainConfig(
        batch_size=8, grad_accum=2, max_steps=20, learning_rate=3e-3,
        warmup_steps=2, eval_interval=10, eval_iters=3, log_interval=5,
    )
    return model, train


def test_mlx_train_eval_checkpoint_sample(tmp_path):
    model_cfg, train_cfg = _cfgs()
    res = train_mlx.train(
        model_cfg, train_cfg, data_dir=_tiny_data(tmp_path), out_dir=tmp_path / "out",
        run_name="r", log=False,
    )
    assert res["best_val"] < float("inf")  # eval branch ran and checkpointed
    model, cfg = load_mlx(res["checkpoint"])
    out = gen_mlx(model, mx.array([[1, 2, 3]]), 4, cfg.block_size, temperature=0.0)
    assert out.shape == (1, 7)


def test_torch_train_eval_checkpoint_sample(tmp_path):
    model_cfg, train_cfg = _cfgs()
    res = train_torch.train(
        model_cfg, train_cfg, data_dir=_tiny_data(tmp_path), out_dir=tmp_path / "out",
        run_name="r", log=False, device="cpu",
    )
    assert res["best_val"] < float("inf")
    model, cfg = load_torch(res["checkpoint"], "cpu")
    out = gen_torch(model, torch.tensor([[1, 2, 3]]), 4, cfg.block_size, temperature=0.0)
    assert out.shape == (1, 7)


def test_prepare_corpus_directory(tmp_path):
    src = tmp_path / "texts"
    src.mkdir()
    for i in range(5):
        (src / f"doc{i}.txt").write_text("hello world " * 100, encoding="utf-8")
    stats = prepare_corpus(src, tmp_path / "data", val_frac=0.4, seed=1)
    assert stats["train"] + stats["val"] > 0


def test_load_split_empty_file(tmp_path):
    (tmp_path / "val.bin").write_bytes(b"")
    assert len(load_split(tmp_path, "val")) == 0


def test_torch_resume_from_last(tmp_path):
    from llmscratch import train_torch

    data = _tiny_data(tmp_path)
    model_cfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, block_size=32, vocab_size=64)
    out = tmp_path / "out"
    base = dict(batch_size=8, grad_accum=1, learning_rate=3e-3, warmup_steps=2,
                eval_interval=5, eval_iters=3, log_interval=100)

    train_torch.train(model_cfg, TrainConfig(max_steps=10, **base), data_dir=data,
                      out_dir=out, run_name="r", log=False, device="cpu")
    assert (out / "r.last.pt").exists()  # resumable checkpoint written

    res = train_torch.train(model_cfg, TrainConfig(max_steps=20, **base), data_dir=data,
                            out_dir=out, run_name="r", log=False, device="cpu", resume="auto")
    # last eval saved at step 9 -> resume at step 10 -> runs steps 10..19 = 10 steps
    assert len(res["losses"]) == 10
