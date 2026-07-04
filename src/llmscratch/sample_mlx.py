"""Generate text from a trained MLX checkpoint (.safetensors + .json sidecar)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from .config import ModelConfig
from .model_mlx import GPT
from .tokenizer import Tokenizer


def generate(model, idx, max_new_tokens, block_size, temperature=1.0, top_k=None,
             top_p=None, repetition_penalty=1.0):
    """Autoregressively extend ``idx`` (B, T). temperature<=0 => greedy.

    Supports top-k, top-p (nucleus), and a repetition penalty (>1 discourages repeats) to curb
    the degenerate loops small models fall into.
    """
    for _ in range(max_new_tokens):
        cond = idx[:, -block_size:]
        logits = model(cond)[:, -1, :]

        if repetition_penalty and repetition_penalty != 1.0:
            ar = mx.arange(logits.shape[-1])
            seen = mx.any(cond[:, :, None] == ar[None, None, :], axis=1)  # (B, V)
            penalized = mx.where(logits > 0, logits / repetition_penalty, logits * repetition_penalty)
            logits = mx.where(seen, penalized, logits)

        if temperature <= 0.0:
            nxt = mx.argmax(logits, axis=-1, keepdims=True)
        else:
            logits = logits / temperature
            order = mx.argsort(-logits, axis=-1)                       # descending
            s_logits = mx.take_along_axis(logits, order, axis=-1)
            B, V = s_logits.shape
            keep = mx.ones((B, V), dtype=mx.bool_)
            if top_k is not None:
                keep = keep & (mx.arange(V)[None, :] < min(top_k, V))
            if top_p is not None and top_p < 1.0:
                probs = mx.softmax(s_logits, axis=-1)
                prev = mx.cumsum(probs, axis=-1) - probs               # cumulative before this token
                keep = keep & (prev < top_p)
            s_logits = mx.where(keep, s_logits, mx.array(-1e9, dtype=s_logits.dtype))
            choice = mx.random.categorical(s_logits, axis=-1)          # index in sorted order
            nxt = mx.take_along_axis(order, choice[:, None], axis=-1)  # map back to token id

        idx = mx.concatenate([idx, nxt], axis=1)
        mx.eval(idx)
    return idx


def load_model(weights_path):
    weights_path = Path(weights_path)
    cfg_path = weights_path.parent / (weights_path.stem + ".json")
    cfg = ModelConfig(**json.loads(cfg_path.read_text())).with_(grad_checkpoint=False)
    model = GPT(cfg)
    model.load_weights(str(weights_path))
    return model, cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample from an MLX checkpoint.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.3)
    args = ap.parse_args()

    model, cfg = load_model(args.ckpt)
    tok = Tokenizer()
    ids = tok.encode(args.prompt) or [tok.eot_token]
    idx = mx.array([ids])
    out = generate(model, idx, args.max_new_tokens, cfg.block_size, args.temperature,
                   args.top_k, args.top_p, args.repetition_penalty)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
