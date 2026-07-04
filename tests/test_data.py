"""Tests for the data pipeline (written before the implementation)."""

import numpy as np
import pytest

from llmscratch.data import (
    BinDataset,
    as_mlx,
    as_torch,
    load_split,
    prepare_corpus,
    sample_batch,
)


def test_sample_batch_shapes_and_dtype():
    data = np.arange(1000, dtype=np.uint16)
    rng = np.random.default_rng(0)
    x, y = sample_batch(data, batch_size=4, block_size=8, rng=rng)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert x.dtype == np.int64 and y.dtype == np.int64


def test_targets_are_next_token():
    # data is a ramp, so the next token is always value + 1
    data = np.arange(1000, dtype=np.uint16)
    rng = np.random.default_rng(0)
    x, y = sample_batch(data, batch_size=16, block_size=8, rng=rng)
    assert np.array_equal(y, x + 1)


def test_sample_batch_too_small_raises():
    data = np.arange(8, dtype=np.uint16)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_batch(data, batch_size=2, block_size=16, rng=rng)


def test_load_split_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_split(tmp_path, "train")


def test_as_torch_and_as_mlx_wrappers():
    data = np.arange(100, dtype=np.uint16)
    rng = np.random.default_rng(0)
    batch = sample_batch(data, 2, 8, rng=rng)

    import torch

    xt, yt = as_torch(batch, device="cpu")
    assert tuple(xt.shape) == (2, 8) and xt.dtype == torch.int64

    import mlx.core as mx  # noqa: F401

    xm, ym = as_mlx(batch)
    assert xm.shape == (2, 8)


def test_prepare_corpus_roundtrip(tmp_path):
    src = tmp_path / "corpus.txt"
    src.write_text("Hello world. " * 200, encoding="utf-8")
    out = tmp_path / "data"
    stats = prepare_corpus(src, out, val_frac=0.1, seed=0)

    assert (out / "train.bin").exists()
    assert (out / "val.bin").exists()
    assert stats["train"] > 0 and stats["val"] > 0

    train = load_split(out, "train")
    assert train.dtype == np.uint16
    assert int(train.max()) < 50257  # all ids within GPT-2 vocab


def test_bindataset_next_token_alignment(tmp_path):
    src = tmp_path / "c.txt"
    src.write_text("the quick brown fox " * 500, encoding="utf-8")
    out = tmp_path / "data"
    prepare_corpus(src, out, val_frac=0.1, seed=0)

    ds = BinDataset(out, block_size=16, seed=0)
    x, y = ds.batch("train", batch_size=4)
    assert x.shape == (4, 16)
    assert np.array_equal(y[:, :-1], x[:, 1:])  # y is x shifted by one
