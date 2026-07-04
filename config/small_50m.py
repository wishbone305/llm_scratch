"""The real local model, trained with MLX on the M1 Pro (~51M params).

Tuned for 16 GB unified memory: a small micro-batch + shorter context keep PEAK memory low so
the GPU isn't stalled swapping (the cause of the earlier ~46 s/step). TinyStories are short,
so 256-token context loses almost nothing. grad_accum keeps a healthy effective batch.
"""

MODEL = dict(
    d_model=512,
    n_layers=8,
    n_heads=8,
    d_ff=1408,
    block_size=256,   # was 512 -> halves activation + 4x less attention memory/compute
    vocab_size=50257,
)

# tokens/step = batch_size * block_size * grad_accum = 8 * 256 * 4 = 8,192
TRAIN = dict(
    batch_size=8,     # was 24 -> 3x lower peak memory (the key change)
    grad_accum=4,
    max_steps=40000,
    learning_rate=6e-4,
    warmup_steps=400,
    eval_interval=1000,
    eval_iters=100,
    log_interval=20,
)
