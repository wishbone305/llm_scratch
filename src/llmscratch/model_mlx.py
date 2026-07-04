"""Llama-style decoder-only transformer in MLX (the fast local-training track).

RMSNorm + RoPE + SwiGLU, pre-norm blocks, weight-tied LM head. Same architecture as
``model_torch.py`` — the parity test guarantees they stay numerically equivalent.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx.nn.utils import checkpoint as _grad_checkpoint

from .config import ModelConfig


def _causal_mask(seq_len: int, dtype) -> mx.array:
    mask = mx.full((seq_len, seq_len), float("-inf"), dtype=dtype)
    return mx.triu(mask, k=1)  # 0 on/below diagonal, -inf strictly above


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.kv_heads = cfg.kv_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim**-0.5
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.rope = nn.RoPE(cfg.head_dim, traditional=False, base=cfg.rope_theta)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        B, T, _ = x.shape
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k = self.rope(q), self.rope(k)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layers)]
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        if not cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._init_params()

    def _init_params(self) -> None:
        """GPT-style init: N(0, 0.02), residual projections scaled by 1/sqrt(2*n_layers)."""
        std = 0.02
        res_std = std / math.sqrt(2 * self.cfg.n_layers)

        def normal(shape, s):
            return mx.random.normal(shape) * s

        self.tok_emb.weight = normal(self.tok_emb.weight.shape, std)
        for blk in self.blocks:
            for lin in (blk.attn.wq, blk.attn.wk, blk.attn.wv):
                lin.weight = normal(lin.weight.shape, std)
            blk.attn.wo.weight = normal(blk.attn.wo.weight.shape, res_std)
            blk.mlp.gate.weight = normal(blk.mlp.gate.weight.shape, std)
            blk.mlp.up.weight = normal(blk.mlp.up.weight.shape, std)
            blk.mlp.down.weight = normal(blk.mlp.down.weight.shape, res_std)
        if not self.cfg.tie_embeddings:
            self.lm_head.weight = normal(self.lm_head.weight.shape, std)

    def __call__(self, idx: mx.array) -> mx.array:
        idx = idx.astype(mx.int32)
        _, T = idx.shape
        h = self.tok_emb(idx)
        mask = _causal_mask(T, h.dtype)
        for block in self.blocks:
            h = _grad_checkpoint(block)(h, mask) if self.cfg.grad_checkpoint else block(h, mask)
        h = self.norm(h)
        if self.cfg.tie_embeddings:
            return h @ self.tok_emb.weight.T
        return self.lm_head(h)

    def loss(self, idx: mx.array, targets: mx.array) -> mx.array:
        logits = self(idx)
        return nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1).astype(mx.int32),
            reduction="mean",
        )

    def num_params(self) -> int:
        return sum(p.size for _, p in tree_flatten(self.parameters()))

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "GPT":
        return cls(cfg)
