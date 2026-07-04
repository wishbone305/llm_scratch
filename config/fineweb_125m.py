"""~125M model on FineWeb-Edu — bigger local run, memory-tuned for 16GB.

Peak memory is set by the micro-batch, so batch_size is small (4) with grad_accum=8 to keep a
healthy effective batch and the same tokens/step. ~1B tokens is Chinchilla-optimal for ~50M, so
this is undertrained on the current data (still better than 50M). Stop at the val plateau.
"""

MODEL = dict(
    d_model=768,
    n_layers=12,
    n_heads=12,      # head_dim 64
    d_ff=2048,       # (8/3) * 768
    block_size=256,
    vocab_size=50257,
    grad_checkpoint=True,
)

# tokens/step = batch_size * block_size * grad_accum = 4 * 256 * 8 = 8,192 (half the peak memory of batch=8)
TRAIN = dict(
    batch_size=4,
    grad_accum=8,
    max_steps=60000,
    learning_rate=5e-4,
    warmup_steps=600,
    eval_interval=1000,
    eval_iters=100,
    log_interval=20,
)
