"""~80M model on FineWeb-Edu — a safe local step-up between 50M and the too-big 125M.

Sized so peak memory stays near the proven 50M run: micro-batch 6 (grad_accum keeps the
effective batch healthy). Still mildly undertrained on ~1B tokens, but a clear step above 50M.
"""

MODEL = dict(
    d_model=640,
    n_layers=10,
    n_heads=10,      # head_dim 64
    d_ff=1664,       # ~ (8/3) * 640
    block_size=256,
    vocab_size=50257,
)

# tokens/step = batch_size * block_size * grad_accum = 6 * 256 * 6 = 9,216
TRAIN = dict(
    batch_size=6,
    grad_accum=6,
    max_steps=60000,
    learning_rate=5e-4,
    warmup_steps=600,
    eval_interval=1000,
    eval_iters=100,
    log_interval=20,
)
