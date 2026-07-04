"""Cosine learning-rate schedule with linear warmup (framework-neutral)."""

from __future__ import annotations

import math


def cosine_lr(
    step: int, *, learning_rate: float, min_lr: float, warmup_steps: int, max_steps: int
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (learning_rate - min_lr)
