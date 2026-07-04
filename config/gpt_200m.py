"""The real cloud target, trained with PyTorch on a CUDA GPU (~205M params, ~10-12h on 1xA100)."""

MODEL = dict(
    d_model=1024,
    n_layers=12,
    n_heads=16,
    d_ff=2816,      # ~ (8/3) * 1024, rounded to a multiple of 128
    block_size=1024,
    vocab_size=50257,
)

# tokens/step = 32 * 1024 * 16 = 524,288 (~0.5M, GPT-2-like).
# Chinchilla-optimal for ~205M is ~4.1B tokens -> ~7800 steps. Set max_steps from YOUR corpus.
TRAIN = dict(
    batch_size=32,
    grad_accum=16,
    max_steps=8000,
    learning_rate=6e-4,
    warmup_steps=200,
    eval_interval=500,
    eval_iters=200,
    log_interval=10,
)
