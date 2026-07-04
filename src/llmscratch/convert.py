"""Weight bridge between the MLX and PyTorch models.

Both implementations use identical submodule names and the (out, in) Linear weight layout, so
the parameter trees map 1:1 — conversion is just an array-type change (via numpy). RoPE's
``inv_freq`` is a non-persistent buffer in torch and absent in MLX, so it never participates.

Enables: train locally in MLX -> continue/finetune in cloud PyTorch (and vice-versa), pull a
cloud checkpoint back to sample on MPS, and the cross-framework parity test.
"""

from __future__ import annotations

import numpy as np


def mlx_to_numpy(model_mlx) -> dict[str, np.ndarray]:
    from mlx.utils import tree_flatten

    return {k: np.array(v) for k, v in tree_flatten(model_mlx.parameters())}


def torch_to_numpy(model_torch) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in model_torch.state_dict().items()}


def load_mlx_into_torch(model_mlx, model_torch) -> None:
    import torch

    weights = mlx_to_numpy(model_mlx)
    state = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in weights.items()}
    model_torch.load_state_dict(state, strict=True)


def load_torch_into_mlx(model_torch, model_mlx) -> None:
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    weights = torch_to_numpy(model_torch)
    tree = tree_unflatten([(k, mx.array(v)) for k, v in weights.items()])
    model_mlx.update(tree)
