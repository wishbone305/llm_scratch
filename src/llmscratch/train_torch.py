"""PyTorch training loop (the cloud track; also runs on MPS/CPU).

AdamW (weight-decay param groups) + cosine warmup, gradient accumulation, grad clipping,
periodic eval, best-checkpointing, tokens/sec logging, optional TensorBoard. Device/precision
come from device.py; TF32 matmul precision is enabled on CUDA.
"""

from __future__ import annotations

import argparse
import contextlib
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .config import TrainConfig, load_config
from .data import BinDataset, as_torch
from .device import device_policy, pick_device
from .model_torch import GPT
from .schedule import cosine_lr


def _param_groups(model, weight_decay: float) -> list:
    """AdamW groups: decay only >=2D weights (matmuls/embeddings); never 1D params (norms)."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


@torch.no_grad()
def _evaluate(model, ds, cfg, device, amp_ctx) -> dict:
    model.eval()
    out = {}
    for split in ("train", "val"):
        data = ds.split(split)
        if len(data) <= ds.block_size + 1:
            continue
        total = 0.0
        for _ in range(cfg.eval_iters):
            x, y = as_torch(ds.batch(split, cfg.batch_size), device=device)
            with amp_ctx():
                total += model.loss(x, y).item()
        out[split] = total / cfg.eval_iters
    model.train()
    return out


def _save_checkpoint(raw_model, model_cfg, path: Path) -> None:
    torch.save({"model": raw_model.state_dict(), "model_cfg": asdict(model_cfg)}, path)


def _save_full(raw_model, optimizer, step, best_val, model_cfg, train_cfg, path: Path) -> None:
    """Full training state for seamless resume (survives spot preemption / crashes)."""
    torch.save({
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val": best_val,
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(train_cfg),
    }, path)


def train(model_cfg, train_cfg, *, data_dir="data", out_dir="out", run_name="model",
          device=None, log=True, resume=None) -> dict:
    torch.manual_seed(train_cfg.seed)
    dev = pick_device(device)
    pol = device_policy(dev)
    if dev.type == "cuda":  # cheap, quality-neutral matmul-precision hygiene for Ampere+
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    raw_model = GPT(model_cfg).to(dev)
    model = torch.compile(raw_model) if pol["use_compile"] else raw_model

    optimizer = torch.optim.AdamW(
        _param_groups(raw_model, train_cfg.weight_decay),
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
        fused=(dev.type == "cuda"),
    )
    ds = BinDataset(data_dir, model_cfg.block_size, seed=train_cfg.seed)
    scaler = torch.amp.GradScaler(enabled=(dev.type == "cuda" and pol["amp_dtype"] == torch.float16))

    def amp_ctx():
        if pol["use_amp"]:
            return torch.autocast(device_type=dev.type, dtype=pol["amp_dtype"])
        return contextlib.nullcontext()

    writer = None
    if log:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(Path("runs") / f"{run_name}_torch")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt = out_path / f"{run_name}.pt"
    last = out_path / f"{run_name}.last.pt"
    losses: list[float] = []
    start_step, best_val = 0, float("inf")
    if resume:
        rp = last if resume == "auto" else Path(resume)
        if rp.exists():
            # Load on CPU, then move optimizer state to the device incrementally. Loading the
            # whole checkpoint straight to the GPU spikes VRAM (fresh model + full checkpoint at
            # once) and OOMs on tight cards; this keeps the peak at steady-state size.
            st = torch.load(rp, map_location="cpu", weights_only=False)
            raw_model.load_state_dict(st["model"])
            optimizer.load_state_dict(st["optimizer"])
            for _s in optimizer.state.values():
                for _k, _v in _s.items():
                    if isinstance(_v, torch.Tensor):
                        _s[_k] = _v.to(dev)
            start_step = int(st.get("step", -1)) + 1
            best_val = float(st.get("best_val", float("inf")))
            del st
            print(f"resumed from {rp} at step {start_step}", flush=True)
    tokens_per_step = train_cfg.batch_size * model_cfg.block_size * train_cfg.grad_accum
    t_last = time.perf_counter()

    model.train()
    for step in range(start_step, train_cfg.max_steps):
        lr = cosine_lr(step, learning_rate=train_cfg.learning_rate, min_lr=train_cfg.lr_floor,
                       warmup_steps=train_cfg.warmup_steps, max_steps=train_cfg.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = torch.zeros((), device=dev)
        for _ in range(train_cfg.grad_accum):
            x, y = as_torch(ds.batch("train", train_cfg.batch_size), device=dev)
            with amp_ctx():
                loss = model.loss(x, y) / train_cfg.grad_accum
            scaler.scale(loss).backward()
            step_loss += loss.detach()  # accumulate on-device; single .item() sync after the loop
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        step_loss = step_loss.item()
        losses.append(step_loss)

        if step % train_cfg.log_interval == 0:
            now = time.perf_counter()
            dt = now - t_last
            tps = tokens_per_step * train_cfg.log_interval / dt if (step > start_step and dt > 0) else 0.0
            print(f"step {step}/{train_cfg.max_steps}  loss {step_loss:.4f}  lr {lr:.2e}  "
                  f"{tps:,.0f} tok/s", flush=True)
            if writer is not None:
                writer.add_scalar("train/loss", step_loss, step)
                writer.add_scalar("train/lr", lr, step)
                if tps:
                    writer.add_scalar("train/tokens_per_sec", tps, step)
            t_last = now

        if train_cfg.eval_interval > 0 and (step + 1) % train_cfg.eval_interval == 0:
            metrics = _evaluate(model, ds, train_cfg, dev, amp_ctx)
            if writer is not None:
                for k, v in metrics.items():
                    writer.add_scalar(f"eval/{k}_loss", v, step)
            val = metrics.get("val", metrics.get("train", step_loss))
            if val < best_val:
                best_val = val
                _save_checkpoint(raw_model, model_cfg, ckpt)
            _save_full(raw_model, optimizer, step, best_val, model_cfg, train_cfg, last)
            print(f"step {step+1}/{train_cfg.max_steps}  loss {step_loss:.4f}  "
                  f"val {metrics.get('val', float('nan')):.4f}  lr {lr:.2e}")

    _save_checkpoint(raw_model, model_cfg, ckpt)
    if writer is not None:
        writer.close()
    return {"losses": losses, "best_val": best_val, "device": str(dev), "checkpoint": str(ckpt)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the PyTorch model (cloud track).")
    ap.add_argument("--config", required=True, help="config name, e.g. gpt_200m")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--max-steps", type=int, default=None, help="override config max_steps")
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    help="resume from out/<name>.last.pt (or a given path)")
    args = ap.parse_args()

    name, model_cfg, train_cfg = load_config(args.config)
    if args.max_steps is not None:
        train_cfg = TrainConfig(
            **{**asdict(train_cfg), "max_steps": args.max_steps,
               "warmup_steps": min(train_cfg.warmup_steps, args.max_steps)}
        )
    res = train(model_cfg, train_cfg, data_dir=args.data_dir, out_dir=args.out_dir,
                run_name=name, device=args.device, log=True, resume=args.resume)
    print(f"done. device={res['device']}  final_loss={res['losses'][-1]:.4f}  "
          f"best_val={res['best_val']:.4f}  -> {res['checkpoint']}")


if __name__ == "__main__":
    main()
