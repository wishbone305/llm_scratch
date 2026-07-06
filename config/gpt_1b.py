"""~1B model for a single 48 GB CUDA GPU (e.g. a 48 GB RTX 4090).

At ~1.01B params, model + AdamW states (fp32 master + grads + two moments) are ~16 GB. With
gradient checkpointing, a micro-batch of 24 at block 1024 sits comfortably around ~28 GB peak,
leaving generous headroom on 48 GB. Wants ~15B tokens (build a corpus with `build_mix.py
--target-tokens 15e9`), which fits an 80 GB disk alongside checkpoints and the venv.

This is the "real" small-LLM tier: coherence and factual grounding improve noticeably over 200M.
If you ever OOM on step 1, drop batch_size (raise grad_accum to keep tokens/step constant).
To trade memory for ~20-30% speed once you trust the fit, set grad_checkpoint=False.
"""

MODEL = dict(
    d_model=2048,
    n_layers=18,
    n_heads=16,          # head_dim 128
    d_ff=5504,           # ~ (8/3) * 2048, rounded to a multiple of 128 (43 * 128)
    block_size=1024,
    vocab_size=50257,
    grad_checkpoint=True,
)

# tokens/step = 24 * 1024 * 20 = 491,520 (~0.5M). ~14.7B tokens over 30k steps (~15 tok/param).
TRAIN = dict(
    batch_size=24,
    grad_accum=20,
    max_steps=30000,
    learning_rate=3e-4,
    warmup_steps=300,
    eval_interval=500,
    eval_iters=200,
    log_interval=10,
)
