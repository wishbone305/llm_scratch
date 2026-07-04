"""MLX training loop (the fast local track on Apple Silicon).

AdamW + cosine warmup, memory-flat gradient accumulation, grad clipping, periodic eval,
best-checkpointing, tokens/sec logging, optional TensorBoard. The forward+backward is wrapped
in mx.compile so MLX fuses kernels instead of dispatching every op eagerly.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

from .config import TrainConfig, load_config
from .data import BinDataset, as_mlx
from .model_mlx import GPT
from .schedule import cosine_lr


def _evaluate(model, ds, cfg) -> dict:
    out = {}
    for split in ("train", "val"):
        data = ds.split(split)
        if len(data) <= ds.block_size + 1:
            continue
        total = 0.0
        for _ in range(cfg.eval_iters):
            x, y = as_mlx(ds.batch(split, cfg.batch_size))
            total += float(model.loss(x, y))
        out[split] = total / cfg.eval_iters
    return out


def _save_checkpoint(model, model_cfg, out_path: Path, run_name: str) -> Path:
    weights = out_path / f"{run_name}_mlx.safetensors"
    model.save_weights(str(weights))
    (out_path / f"{run_name}_mlx.json").write_text(json.dumps(asdict(model_cfg)))
    return weights


def _save_full(model, model_cfg, step, best_val, out_path: Path, run_name: str) -> None:
    """Weights + step + best_val for resume. (Adam moments reinitialize on MLX resume.)"""
    model.save_weights(str(out_path / f"{run_name}_mlx.last.safetensors"))
    (out_path / f"{run_name}_mlx.meta.json").write_text(
        json.dumps({"step": step, "best_val": best_val, "model_cfg": asdict(model_cfg)}))


def train(model_cfg, train_cfg, *, data_dir="data", out_dir="out", run_name="model",
          log=True, resume=None) -> dict:
    mx.random.seed(train_cfg.seed)
    model = GPT(model_cfg)
    optimizer = optim.AdamW(
        learning_rate=train_cfg.learning_rate,
        betas=[train_cfg.beta1, train_cfg.beta2],
        weight_decay=train_cfg.weight_decay,
    )
    ds = BinDataset(data_dir, model_cfg.block_size, seed=train_cfg.seed)

    def loss_fn(m, x, y):
        return m.loss(x, y)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    writer = None
    if log:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(Path("runs") / f"{run_name}_mlx")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    losses: list[float] = []
    ckpt = out_path / f"{run_name}_mlx.safetensors"
    start_step, best_val = 0, float("inf")
    if resume:
        meta_p = out_path / f"{run_name}_mlx.meta.json"
        last_w = out_path / f"{run_name}_mlx.last.safetensors"
        if meta_p.exists() and last_w.exists():
            meta = json.loads(meta_p.read_text())
            model.load_weights(str(last_w))
            start_step = int(meta.get("step", -1)) + 1
            best_val = float(meta.get("best_val", float("inf")))
            print(f"resumed (weights+schedule) from step {start_step}; Adam moments reset", flush=True)
    tokens_per_step = train_cfg.batch_size * model_cfg.block_size * train_cfg.grad_accum
    t_last = time.perf_counter()

    for step in range(start_step, train_cfg.max_steps):
        lr = cosine_lr(step, learning_rate=train_cfg.learning_rate, min_lr=train_cfg.lr_floor,
                       warmup_steps=train_cfg.warmup_steps, max_steps=train_cfg.max_steps)
        optimizer.learning_rate = lr

        acc_grads = None
        step_loss = 0.0
        for _ in range(train_cfg.grad_accum):
            x, y = as_mlx(ds.batch("train", train_cfg.batch_size))
            loss, grads = loss_and_grad(model, x, y)
            grads = tree_map(lambda g: g / train_cfg.grad_accum, grads)
            acc_grads = grads if acc_grads is None else tree_map(lambda a, b: a + b, acc_grads, grads)
            mx.eval(acc_grads)  # materialize per micro-batch: keeps grad-accum memory-flat (MLX is lazy)
            step_loss += float(loss) / train_cfg.grad_accum
        acc_grads, _ = optim.clip_grad_norm(acc_grads, train_cfg.grad_clip)
        optimizer.update(model, acc_grads)
        mx.eval(model.state, optimizer.state)
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
            metrics = _evaluate(model, ds, train_cfg)
            if writer is not None:
                for k, v in metrics.items():
                    writer.add_scalar(f"eval/{k}_loss", v, step)
            val = metrics.get("val", metrics.get("train", step_loss))
            if val < best_val:
                best_val = val
                ckpt = _save_checkpoint(model, model_cfg, out_path, run_name)
            _save_full(model, model_cfg, step, best_val, out_path, run_name)
            print(f"step {step+1}/{train_cfg.max_steps}  loss {step_loss:.4f}  "
                  f"val {metrics.get('val', float('nan')):.4f}  lr {lr:.2e}")

    ckpt = _save_checkpoint(model, model_cfg, out_path, run_name)
    if writer is not None:
        writer.close()
    return {"losses": losses, "best_val": best_val,
            "device": str(mx.default_device()), "checkpoint": str(ckpt)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the MLX model (local track).")
    ap.add_argument("--config", required=True, help="config name, e.g. small_50m")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--max-steps", type=int, default=None, help="override config max_steps")
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    help="resume weights + schedule from out/<name>_mlx.last.safetensors")
    args = ap.parse_args()

    name, model_cfg, train_cfg = load_config(args.config)
    if args.max_steps is not None:
        train_cfg = TrainConfig(
            **{**asdict(train_cfg), "max_steps": args.max_steps,
               "warmup_steps": min(train_cfg.warmup_steps, args.max_steps)}
        )
    res = train(model_cfg, train_cfg, data_dir=args.data_dir, out_dir=args.out_dir,
                run_name=name, log=True, resume=args.resume)
    print(f"done. device={res['device']}  final_loss={res['losses'][-1]:.4f}  "
          f"best_val={res['best_val']:.4f}  -> {res['checkpoint']}")


if __name__ == "__main__":
    main()
