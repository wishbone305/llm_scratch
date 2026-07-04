# llmscratch

A **~200M-parameter LLM trained from scratch** — decoder-only, **Llama-style**
(RMSNorm + RoPE + SwiGLU), weight-tied head, GPT-2 BPE tokenizer (`tiktoken`).

It's **dual-track** so it fits the hardware you actually have:

| Track | Framework | Where | Use |
|-------|-----------|-------|-----|
| **Local** | **MLX** | Apple Silicon (Metal) | fast prototyping + a real *small* model (~30–50M) |
| **Cloud** | **PyTorch** | CUDA GPU (A100/4090) | the full **~205M** run, ~10–12 h, ~$15–25 |

Both tracks share one `config`, one tokenizer, and one on-disk data format. A **parity test**
loads identical weights into both model implementations and asserts matching logits — so they
can never silently drift apart.

## Why two frameworks?

MLX is meaningfully faster than PyTorch/MPS on Apple Silicon (better unified-memory use), but
it can't run on NVIDIA/CUDA. PyTorch runs on both. So: train fast locally with MLX, and run the
big model in the cloud with PyTorch using the *same* architecture and data.

### Hardware reality (this repo was set up on an M1 Pro / 16 GB)

A 200M model **fits** locally (model + AdamW ≈ 3.2 GB fp32) but needs ~4B tokens to train well —
that's **weeks** of nonstop Metal compute. So the plan is: real small model locally, real 200M
in the cloud. The code scales to either.

## Setup

```bash
uv sync          # installs torch + mlx (+ tiktoken, tensorboard, pytest). mlx auto-skips off-Mac.
```

## Quickstart — local (MLX)

```bash
# 1. Get data. RECOMMENDED for the 50M model: TinyStories (coherent output at this size):
uv run python scripts/download_tinystories.py
#    ...or a quick tiny demo (tinyshakespeare; falls back to bundled text):
uv run python scripts/download_sample.py
#    ...or bring your own (a .txt file, or a directory of .txt files):
uv run python scripts/prepare_data.py path/to/corpus.txt

# 2. Train (start tiny to confirm everything works, then go bigger):
uv run python -m llmscratch.train_mlx --config debug_mac --max-steps 200
uv run python -m llmscratch.train_mlx --config small_50m          # the real local model

# 3. Generate:
uv run python -m llmscratch.sample_mlx --ckpt out/small_50m_mlx.safetensors \
    --prompt "Once upon a time" --max-new-tokens 200

# 4. Watch training curves:
uv run tensorboard --logdir runs
```

## The real ~205M run — cloud (PyTorch)

```bash
# On a rented CUDA box (Lambda / RunPod / vast.ai):
git clone <your-repo> && cd llm_scratch
uv sync                                              # mlx is skipped automatically on Linux
uv run python scripts/prepare_data.py /data/corpus  # or rsync prebuilt data/*.bin
uv run python -m llmscratch.train_torch --config gpt_200m   # auto: CUDA + bf16 + torch.compile

# Pull out/gpt_200m.pt back home and sample on your Mac (PyTorch runs on MPS too):
uv run python -m llmscratch.sample_torch --ckpt out/gpt_200m.pt --prompt "..." --device mps
```

Set `--max-steps` from your corpus size. Chinchilla-optimal for 205M ≈ **4.1B tokens**; at the
default batch (~0.5M tokens/step) that's ~7,800 steps.

## Configs (`config/*.py`)

| name | params | d_model × layers × heads | ctx | for |
|------|--------|--------------------------|-----|-----|
| `debug_mac` | ~30M | 384 × 6 × 6 | 256 | smoke-test the pipeline in minutes |
| `small_50m` | ~51M | 512 × 8 × 8 | 512 | real local model (MLX), ~0.5–1.5 days |
| `gpt_200m`  | ~205M | 1024 × 12 × 16 | 1024 | real cloud model (PyTorch) |

Edit these or add your own (`config/<name>.py` defining `MODEL` and `TRAIN` dicts).

## Your data

`prepare_data.py` accepts **a single `.txt` file** (split contiguously) or **a directory of
`.txt` files** (streamed one at a time — the memory-friendly path for large corpora). It writes
`data/train.bin` + `data/val.bin` as uint16 GPT-2 token ids.

## Project layout

```
src/llmscratch/
  config.py        # ModelConfig/TrainConfig dataclasses + loader  (single source of truth)
  tokenizer.py     # GPT-2 BPE wrapper
  data.py          # uint16 memmap sampler + prepare_corpus()
  model_mlx.py     # MLX model        model_torch.py   # PyTorch model
  train_mlx.py     # MLX trainer      train_torch.py   # PyTorch trainer
  sample_mlx.py    # MLX sampler      sample_torch.py  # PyTorch sampler
  schedule.py      # cosine LR + warmup     device.py  # torch device/precision policy
  convert.py       # MLX <-> PyTorch weight bridge (powers parity + cross-track transfer)
config/  scripts/  tests/  data/  out/
```

## Testing

```bash
uv run pytest                       # 40 tests
uv run pytest --cov=llmscratch      # coverage (~87%)
```

The key one is `tests/test_parity.py`: same weights in both models → matching logits, the
guard that keeps the MLX and PyTorch tracks equivalent.

## Notes

- **MPS**: PyTorch runs in fp32 with `torch.compile` off (it's flaky on Metal); CUDA gets
  bf16 + compile automatically. If you hit an unsupported op on MPS, set
  `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- **Checkpoints**: MLX → `out/<name>_mlx.safetensors` (+ `.json` config); PyTorch → `out/<name>.pt`.
- **Architecture knobs** (in any config): `n_kv_heads` for GQA, `rope_theta`, `block_size`,
  `tie_embeddings`.

## Run on Google Colab

Prefer a GPU over the local Mac? Open **`colab_train.ipynb`** in Google Colab
([colab.research.google.com](https://colab.research.google.com) → File → Open notebook → GitHub →
`wishbone305/llm_scratch`). It clones the repo, installs deps, mounts Drive for data +
checkpoints, and trains the **PyTorch/CUDA track** with `--resume` so Colab's disconnects don't
lose progress. Free T4 → `fineweb_125m`; A100/L4 → the full `gpt_200m`.
