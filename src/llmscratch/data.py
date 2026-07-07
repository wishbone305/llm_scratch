"""Data pipeline: tokenize raw text to uint16 .bin shards, and sample training batches.

Framework-neutral by design — sampling returns numpy arrays; ``as_torch`` / ``as_mlx`` wrap
them with lazy imports so this module pulls in neither torch nor mlx at import time. This is
what lets the same data path feed both training tracks.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .tokenizer import Tokenizer

# A batch is (x, y), each an int64 array of shape (batch_size, block_size); y is x shifted by 1.
Batch = tuple[np.ndarray, np.ndarray]

_REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- reading
def _resolve_split_path(data_dir: str | Path, split: str) -> Path:
    """Find <data_dir>/<split>.bin. For a relative data_dir, try the cwd first then the repo
    root, so `python -m llmscratch.train_*` works from anywhere in the project, not just root."""
    name = f"{split}.bin"
    base = Path(data_dir)
    if base.is_absolute():
        return base / name
    for candidate in (Path.cwd() / base / name, _REPO_ROOT / base / name):
        if candidate.exists():
            return candidate
    return Path.cwd() / base / name


def load_split(data_dir: str | Path, split: str) -> np.ndarray:
    """Memory-map a tokenized split (uint16). Returns an empty array for a 0-byte file."""
    path = _resolve_split_path(data_dir, split)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/prepare_data.py first")
    if path.stat().st_size == 0:
        return np.zeros(0, dtype=np.uint16)
    return np.memmap(path, dtype=np.uint16, mode="r")


def sample_batch(
    data: np.ndarray, batch_size: int, block_size: int, *, rng: np.random.Generator
) -> Batch:
    """Sample ``batch_size`` random windows of length ``block_size`` (+1 for the target shift)."""
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError(
            f"dataset ({len(data)} tokens) too small for block_size={block_size}"
        )
    starts = rng.integers(0, max_start + 1, size=batch_size)
    # Vectorized gather: a (batch, block+1) index matrix fancy-indexes the memmap in one shot
    # (returns a fresh ndarray of just the sampled windows), replacing the per-row Python loop.
    idx = starts[:, None] + np.arange(block_size + 1, dtype=np.int64)[None, :]
    windows = np.asarray(data[idx], dtype=np.int64)
    x = np.ascontiguousarray(windows[:, :-1])
    y = np.ascontiguousarray(windows[:, 1:])
    return x, y


def as_torch(batch: Batch, device: str = "cpu"):
    """Convert a numpy batch to torch tensors on ``device`` (lazy import)."""
    import torch

    x, y = batch
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def as_mlx(batch: Batch):
    """Convert a numpy batch to MLX arrays (lazy import)."""
    import mlx.core as mx

    x, y = batch
    return mx.array(x), mx.array(y)


class BinDataset:
    """Convenience over the .bin shards: caches memmaps and samples with a seeded RNG."""

    def __init__(self, data_dir: str | Path, block_size: int, seed: int = 1337) -> None:
        self.data_dir = Path(data_dir)
        self.block_size = block_size
        self._rng = np.random.default_rng(seed)
        self._cache: dict[str, np.ndarray] = {}

    def split(self, name: str) -> np.ndarray:
        if name not in self._cache:
            self._cache[name] = load_split(self.data_dir, name)
        return self._cache[name]

    def batch(self, split: str, batch_size: int) -> Batch:
        return sample_batch(self.split(split), batch_size, self.block_size, rng=self._rng)


# --------------------------------------------------------------------------- writing
def _gather_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.rglob("*.txt"))
        if not files:
            raise FileNotFoundError(f"no .txt files found under {input_path}")
        return files
    raise FileNotFoundError(input_path)


def prepare_corpus(
    input_path: str | Path,
    out_dir: str | Path,
    val_frac: float = 0.1,
    seed: int = 1337,
    encoding: str = "gpt2",
) -> dict[str, int]:
    """Tokenize raw text into ``train.bin`` + ``val.bin`` (uint16). Returns token counts.

    A single input file is split contiguously (tail held out for val). A directory of ``.txt``
    files streams one file at a time and assigns whole files to val with probability ``val_frac``
    — the memory-friendly path for large corpora.
    """
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(encoding)
    rng = np.random.default_rng(seed)
    files = _gather_files(input_path)
    counts = {"train": 0, "val": 0}

    with open(out_dir / "train.bin", "wb") as tf, open(out_dir / "val.bin", "wb") as vf:
        if len(files) == 1:
            text = files[0].read_text(encoding="utf-8", errors="ignore")
            ids = np.asarray(tok.encode(text, add_eot=True), dtype=np.uint16)
            n_val = int(len(ids) * val_frac)
            if n_val > 0:
                ids[:-n_val].tofile(tf)
                ids[-n_val:].tofile(vf)
                counts["train"], counts["val"] = len(ids) - n_val, n_val
            else:
                ids.tofile(tf)
                counts["train"] = len(ids)
        else:
            for fp in files:
                target = "val" if rng.random() < val_frac else "train"
                text = fp.read_text(encoding="utf-8", errors="ignore")
                ids = np.asarray(tok.encode(text, add_eot=True), dtype=np.uint16)
                ids.tofile(vf if target == "val" else tf)
                counts[target] += len(ids)

    return counts


def iter_documents(path: str | Path, separator: str = "<|endoftext|>", chunk_size: int = 1 << 20):
    """Stream non-empty documents from a text file, split on ``separator`` (memory-bounded).

    Used for corpora like TinyStories where documents are delimited by a literal ``<|endoftext|>``.
    """
    buffer = ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            parts = buffer.split(separator)
            buffer = parts.pop()  # trailing fragment (may hold a partial separator) carries over
            for part in parts:
                doc = part.strip()
                if doc:
                    yield doc
    tail = buffer.strip()
    if tail:
        yield tail


def write_documents(documents, out_path: str | Path, tokenizer: Tokenizer | None = None) -> int:
    """Tokenize an iterable of documents (a *real* EOT token appended after each) into a uint16
    .bin file. Streams, so it stays memory-bounded over large corpora. Returns the token count.
    """
    tok = tokenizer or Tokenizer()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, "wb") as f:
        for doc in documents:
            text = doc.strip()
            if not text:
                continue
            ids = np.asarray(tok.encode(text, add_eot=True), dtype=np.uint16)
            ids.tofile(f)
            count += len(ids)
    return count


def write_mixed(sources, out_dir: str | Path, target_tokens: int, val_frac: float = 0.005,
                seed: int = 1337, tokenizer: Tokenizer | None = None,
                log_every_tokens: int = 0) -> dict[str, int]:
    """Interleave multiple text sources into train.bin + val.bin, up to ``target_tokens``.

    ``sources``: list of ``(name, texts, weight)`` where ``texts`` is an iterable of strings.
    Each source is capped at ``weight / sum(weights) * target_tokens`` and drawn by remaining
    budget, so the blend is interleaved doc-by-doc (not concatenated). A real EOT is appended
    per document. Returns ``{name: token_count}``. A source that runs dry is simply dropped.

    ``log_every_tokens``: if > 0, print a progress line (tokens, %, rate, ETA) roughly every that
    many tokens — so a multi-hour build isn't a silent black box. 0 keeps it silent (the default).
    """
    import random

    tok = tokenizer or Tokenizer()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [n for n, _, _ in sources]
    total_w = sum(max(w, 0.0) for _, _, w in sources) or 1.0
    budgets = [max(w, 0.0) / total_w * target_tokens for _, _, w in sources]
    iters = [iter(t) for _, t, _ in sources]
    counts = [0] * len(sources)
    active = {i for i in range(len(sources)) if budgets[i] > 0}
    rng = random.Random(seed)
    total = 0
    start = time.time()
    next_log = log_every_tokens  # token threshold at which to print the next progress line

    with open(out_dir / "train.bin", "wb") as tf, open(out_dir / "val.bin", "wb") as vf:
        while active and total < target_tokens:
            choices = list(active)
            weights = [max(budgets[i] - counts[i], 0.0) for i in choices]
            if sum(weights) <= 0:
                break
            i = rng.choices(choices, weights=weights)[0]
            try:
                text = next(iters[i])
            except StopIteration:
                active.discard(i)
                continue
            if not text or not text.strip():
                continue
            ids = np.asarray(tok.encode(text.strip(), add_eot=True), dtype=np.uint16)
            ids.tofile(vf if rng.random() < val_frac else tf)
            counts[i] += len(ids)
            total += len(ids)
            if counts[i] >= budgets[i]:
                active.discard(i)
            if log_every_tokens and total >= next_log:
                elapsed = max(time.time() - start, 1e-9)
                rate = total / elapsed
                eta_min = (target_tokens - total) / rate / 60 if rate else 0.0
                print(f"[build] {total / 1e9:.2f}B / {target_tokens / 1e9:.1f}B tokens "
                      f"({100 * total / target_tokens:.1f}%) | {rate / 1e6:.2f}M tok/s | "
                      f"ETA {eta_min:.0f}m", flush=True)
                next_log = total + log_every_tokens

    return dict(zip(names, counts))
