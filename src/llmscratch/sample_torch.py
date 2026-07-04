"""Generate text from a trained PyTorch checkpoint (runs on CUDA/MPS/CPU)."""

from __future__ import annotations

import argparse

import torch

from .config import ModelConfig
from .device import pick_device
from .model_torch import GPT
from .tokenizer import Tokenizer


@torch.no_grad()
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
            for b in range(idx.shape[0]):
                seen = torch.unique(cond[b])
                l = logits[b, seen]
                logits[b, seen] = torch.where(l > 0, l / repetition_penalty, l * repetition_penalty)

        if temperature <= 0.0:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            if top_p is not None and top_p < 1.0:
                s_logits, s_idx = torch.sort(logits, descending=True, dim=-1)
                cum = torch.softmax(s_logits, dim=-1).cumsum(dim=-1)
                remove = cum > top_p
                remove[:, 1:] = remove[:, :-1].clone()  # keep the token that crosses the threshold
                remove[:, 0] = False
                remove_orig = torch.zeros_like(remove).scatter(1, s_idx, remove)
                logits = logits.masked_fill(remove_orig, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, nxt], dim=1)
    return idx


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["model_cfg"]).with_(grad_checkpoint=False)
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample from a PyTorch checkpoint.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    model, cfg = load_model(args.ckpt, device)
    tok = Tokenizer()
    ids = tok.encode(args.prompt) or [tok.eot_token]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(model, idx, args.max_new_tokens, cfg.block_size, args.temperature,
                   args.top_k, args.top_p, args.repetition_penalty)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
