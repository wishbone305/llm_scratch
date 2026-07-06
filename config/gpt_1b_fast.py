"""~1B model, SPEED variant — same shape as gpt_1b but gradient checkpointing OFF.

Identical architecture to config/gpt_1b.py (1.014B params). The only change is
grad_checkpoint=False: the backward no longer recomputes each block's forward, so steps run
~20-30% faster — at the cost of ~2x activation memory. To stay safe on 48 GB, the micro-batch
drops to 12 (un-checkpointed activations cost more per sample), with grad_accum raised to 40 so
tokens/step stays 491,520 — identical to gpt_1b. That makes this a clean speed A/B: same effective
batch, same LR, same token budget, just faster wall-clock.

Estimated peak ~35 GB on 48 GB. Run gpt_1b first to confirm the pipeline; switch to this once you
trust the fit. If steady-state VRAM shows lots of free headroom, bump batch_size (drop grad_accum
to keep tokens/step ~0.5M) for more throughput; if it OOMs on step 1, use gpt_1b instead.
"""

MODEL = dict(
    d_model=2048,
    n_layers=18,
    n_heads=16,          # head_dim 128
    d_ff=5504,           # ~ (8/3) * 2048, rounded to a multiple of 128 (43 * 128)
    block_size=1024,
    vocab_size=50257,
    grad_checkpoint=False,   # <-- the difference vs gpt_1b: recompute off, ~20-30% faster
)

# tokens/step = 12 * 1024 * 40 = 491,520 (~0.5M) — same as gpt_1b. ~14.7B tokens over 30k steps.
TRAIN = dict(
    batch_size=12,
    grad_accum=40,
    max_steps=30000,
    learning_rate=3e-4,
    warmup_steps=300,
    eval_interval=500,
    eval_iters=200,
    log_interval=10,
)
