"""~205M model tuned for a 16 GB CUDA GPU (RTX 5070 Ti / 4080 / etc).

Same architecture as gpt_200m, but with gradient checkpointing + a smaller micro-batch (16 vs 32,
with more grad-accum to keep the same ~0.5M tokens/step) so the full 200M at block 1024 fits in
16 GB VRAM. On a 40-80 GB A100, use `gpt_200m` instead (faster without checkpointing).
"""

MODEL = dict(
    d_model=1024,
    n_layers=12,
    n_heads=16,
    d_ff=2816,
    block_size=1024,
    vocab_size=50257,
    grad_checkpoint=True,   # fit the full model + block 1024 in 16 GB
)

# tokens/step = 16 * 1024 * 32 = 524,288 (~0.5M). ~4.1B tokens over 8000 steps (Chinchilla-optimal for 205M).
TRAIN = dict(
    batch_size=16,
    grad_accum=32,
    max_steps=8000,
    learning_rate=6e-4,
    warmup_steps=200,
    eval_interval=500,
    eval_iters=200,
    log_interval=10,
)
