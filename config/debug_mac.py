"""Tiny config to smoke-test the whole pipeline on an M1 Pro in minutes (~30M params)."""

MODEL = dict(
    d_model=384,
    n_layers=6,
    n_heads=6,
    d_ff=1024,      # ~ (8/3) * 384
    block_size=256,
    vocab_size=50257,
)

# tokens/step = batch_size * block_size * grad_accum = 16 * 256 * 1 = 4096
TRAIN = dict(
    batch_size=16,
    grad_accum=1,
    max_steps=2000,
    learning_rate=3e-4,
    warmup_steps=100,
    eval_interval=250,
    eval_iters=50,
    log_interval=10,
)
