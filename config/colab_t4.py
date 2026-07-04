"""~125M model tuned for a free Colab T4 (16 GB, Turing).

Key differences from the M1 Pro configs: a BIG micro-batch (T4 CUDA cores need saturating, unlike
Apple unified memory) + gradient checkpointing to fit it. Uses fp16 automatically on the T4.
For an A100/L4 (Colab Pro), use `gpt_200m` instead.
"""

MODEL = dict(
    d_model=768,
    n_layers=12,
    n_heads=12,
    d_ff=2048,
    block_size=256,
    vocab_size=50257,
    grad_checkpoint=True,   # lets the big batch fit in 16 GB
)

# tokens/step = 32 * 256 * 2 = 16,384 (batch 32 keeps the T4 busy)
TRAIN = dict(
    batch_size=32,
    grad_accum=2,
    max_steps=20000,
    learning_rate=5e-4,
    warmup_steps=200,
    eval_interval=1000,
    eval_iters=100,
    log_interval=20,
)
