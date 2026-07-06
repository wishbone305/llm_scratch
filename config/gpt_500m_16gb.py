"""~500M model for a 16 GB CUDA GPU (RTX 5070 Ti / 4080).

The practical VRAM ceiling on 16 GB: at ~500M, model + AdamW states are ~8 GB, leaving room for
activations under gradient checkpointing. Uses a small micro-batch (6) + heavy grad-accum to keep
a healthy ~0.25M tokens/step. Wants ~10B tokens (build a corpus with `build_mix.py`).
If it OOMs on step 1, drop batch_size to 4 (and raise grad_accum to ~60).
"""

MODEL = dict(
    d_model=1280,
    n_layers=22,
    n_heads=16,          # head_dim 80
    d_ff=3456,           # ~ (8/3) * 1280, rounded to a multiple of 128
    block_size=1024,
    vocab_size=50257,
    grad_checkpoint=True,
)

# tokens/step = 6 * 1024 * 40 = 245,760 (~0.25M). ~9.8B tokens over 40k steps (Chinchilla for ~500M).
TRAIN = dict(
    batch_size=6,
    grad_accum=40,
    max_steps=40000,
    learning_rate=4e-4,
    warmup_steps=400,
    eval_interval=500,
    eval_iters=200,
    log_interval=10,
)
