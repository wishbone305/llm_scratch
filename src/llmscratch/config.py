"""Shared, framework-agnostic configuration — the single source of truth for model shape.

Both the MLX and PyTorch model implementations build from the *same* ModelConfig, which is
what makes the cross-framework parity test meaningful.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, replace
from pathlib import Path

DEFAULT_VOCAB_SIZE = 50257  # tiktoken "gpt2"

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Look in the cwd first (so a user can keep local configs), then the repo's config/ dir.
_CONFIG_DIRS = (Path.cwd() / "config", _REPO_ROOT / "config")


@dataclass(frozen=True)
class ModelConfig:
    """Immutable description of the transformer. Validated at construction (the boundary)."""

    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    block_size: int
    vocab_size: int = DEFAULT_VOCAB_SIZE
    n_kv_heads: int | None = None  # None -> full multi-head attention (kv_heads == n_heads)
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    grad_checkpoint: bool = False  # recompute activations in backward to cut memory

    def __post_init__(self) -> None:
        for name in ("d_model", "n_layers", "n_heads", "d_ff", "block_size", "vocab_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_heads % self.kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.kv_heads})"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_heads(self) -> int:
        return self.n_kv_heads if self.n_kv_heads is not None else self.n_heads

    def with_(self, **changes) -> "ModelConfig":
        """Return a new config with overrides (immutable update)."""
        return replace(self, **changes)


@dataclass(frozen=True)
class TrainConfig:
    """Immutable training hyperparameters with sensible defaults."""

    batch_size: int = 16
    grad_accum: int = 1
    max_steps: int = 2000
    learning_rate: float = 3e-4
    min_lr: float | None = None  # None -> learning_rate / 10
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 10
    seed: int = 1337
    out_dir: str = "out"
    data_dir: str = "data"

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.grad_accum <= 0:
            raise ValueError("batch_size and grad_accum must be positive")
        if self.warmup_steps > self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) cannot exceed max_steps ({self.max_steps})"
            )

    @property
    def lr_floor(self) -> float:
        return self.min_lr if self.min_lr is not None else self.learning_rate * 0.1

    @property
    def tokens_per_step(self) -> int:
        # filled in by the trainer once block_size is known; convenience only
        return self.batch_size * self.grad_accum


def _find_config_file(name: str) -> Path:
    for directory in _CONFIG_DIRS:
        candidate = directory / f"{name}.py"
        if candidate.exists():
            return candidate
    searched = ", ".join(str(d) for d in _CONFIG_DIRS)
    raise FileNotFoundError(f"config '{name}' not found (looked in: {searched})")


def load_config(name: str) -> tuple[str, ModelConfig, TrainConfig]:
    """Load config/<name>.py and build validated ModelConfig + TrainConfig.

    The config file must define a ``MODEL`` dict and may define a ``TRAIN`` dict.
    """
    path = _find_config_file(name)
    spec = importlib.util.spec_from_file_location(f"llmscratch._cfg_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load config module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_kwargs = getattr(module, "MODEL", None)
    train_kwargs = getattr(module, "TRAIN", {})
    if not isinstance(model_kwargs, dict):
        raise ValueError(f"config '{name}' must define a MODEL dict")
    if not isinstance(train_kwargs, dict):
        raise ValueError(f"config '{name}' TRAIN must be a dict if present")

    return name, ModelConfig(**model_kwargs), TrainConfig(**train_kwargs)


def estimate_params(cfg: ModelConfig) -> int:
    """Analytic parameter count (tied embeddings). Used as a cross-check in tests."""
    emb = cfg.vocab_size * cfg.d_model
    # attention: q (d*d) + k,v (d*kv_dim each) + o (d*d)
    kv_dim = cfg.kv_heads * cfg.head_dim
    attn = cfg.d_model * cfg.d_model * 2 + cfg.d_model * kv_dim * 2
    # SwiGLU MLP: gate + up (d*d_ff each) + down (d_ff*d)
    mlp = 3 * cfg.d_model * cfg.d_ff
    # RMSNorm weights: 2 per block + 1 final
    norms = (2 * cfg.n_layers + 1) * cfg.d_model
    per_layer = attn + mlp
    total = emb + cfg.n_layers * per_layer + norms
    if not cfg.tie_embeddings:
        total += cfg.vocab_size * cfg.d_model
    return total
