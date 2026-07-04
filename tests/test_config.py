"""Config validation, loading, and parameter estimation."""

import pytest

from llmscratch.config import ModelConfig, TrainConfig, estimate_params, load_config


def test_load_config_returns_validated_objects():
    name, m, t = load_config("debug_mac")
    assert name == "debug_mac"
    assert m.d_model == 384 and m.n_layers == 6 and m.head_dim == 64
    assert t.max_steps > 0


def test_load_config_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist_xyz")


def test_model_config_rejects_bad_shapes():
    with pytest.raises(ValueError):
        ModelConfig(d_model=10, n_layers=1, n_heads=3, d_ff=8, block_size=4)  # 10 % 3
    with pytest.raises(ValueError):
        ModelConfig(d_model=8, n_layers=0, n_heads=2, d_ff=8, block_size=4)   # n_layers <= 0
    with pytest.raises(ValueError):
        ModelConfig(d_model=8, n_layers=1, n_heads=4, d_ff=8, block_size=4, n_kv_heads=3)  # 4 % 3


def test_train_config_validation():
    with pytest.raises(ValueError):
        TrainConfig(batch_size=0)
    with pytest.raises(ValueError):
        TrainConfig(max_steps=10, warmup_steps=20)


def test_lr_floor_and_kv_heads_and_immutability():
    assert TrainConfig(learning_rate=1e-3).lr_floor == pytest.approx(1e-4)
    assert TrainConfig(learning_rate=1e-3, min_lr=5e-5).lr_floor == 5e-5

    mha = ModelConfig(d_model=8, n_layers=1, n_heads=4, d_ff=8, block_size=4)
    gqa = ModelConfig(d_model=8, n_layers=1, n_heads=4, d_ff=8, block_size=4, n_kv_heads=2)
    assert mha.kv_heads == 4 and gqa.kv_heads == 2

    updated = mha.with_(n_layers=3)
    assert updated.n_layers == 3 and mha.n_layers == 1  # original unchanged


def test_estimate_untied_adds_head():
    base = dict(d_model=64, n_layers=2, n_heads=4, d_ff=128, block_size=32, vocab_size=256)
    tied = ModelConfig(**base)
    untied = ModelConfig(**base, tie_embeddings=False)
    assert estimate_params(untied) == estimate_params(tied) + 256 * 64
