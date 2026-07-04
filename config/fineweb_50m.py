"""50M model on FineWeb-Edu (general web text), ~996M train tokens.

Same architecture + memory-safe profile as small_50m (batch 8 / block 256), but more steps
since general data is harder than TinyStories. Expect higher loss than the storyteller and a
later plateau — stop when val flattens.
"""

MODEL = dict(
    d_model=512,
    n_layers=8,
    n_heads=8,
    d_ff=1408,
    block_size=256,
    vocab_size=50257,
)

# tokens/step = 8 * 256 * 4 = 8,192
TRAIN = dict(
    batch_size=8,
    grad_accum=4,
    max_steps=60000,
    learning_rate=6e-4,
    warmup_steps=600,
    eval_interval=1000,
    eval_iters=100,
    log_interval=20,
)
