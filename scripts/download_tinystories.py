"""Download TinyStories (V2, GPT-4) and tokenize it into data/train.bin + data/val.bin.

TinyStories is the best dataset for a ~50M model — small models produce coherent output on it.
Source files delimit stories with a literal ``<|endoftext|>``; each story is tokenized with a
real EOT token appended.

Usage:
    uv run python scripts/download_tinystories.py              # full train + val (~2GB download)
    uv run python scripts/download_tinystories.py --valid-only # quick check (small)
    uv run python scripts/download_tinystories.py --limit 5000 # cap stories per split (testing)

Note: this overwrites data/train.bin and data/val.bin.
"""

import argparse
import urllib.request
from pathlib import Path

from llmscratch.data import iter_documents, write_documents

BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
FILES = {"train": "TinyStoriesV2-GPT4-train.txt", "val": "TinyStoriesV2-GPT4-valid.txt"}
BIN = {"train": "train.bin", "val": "val.bin"}


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  {dest.name} already present ({dest.stat().st_size / 1e6:.0f} MB), skipping download")
        return
    print(f"  downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  saved {dest.stat().st_size / 1e6:.0f} MB")


def _limited(it, n: int):
    for i, x in enumerate(it):
        if i >= n:
            break
        yield x


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--valid-only", action="store_true", help="only fetch/tokenize the val split")
    ap.add_argument("--limit", type=int, default=0, help="max stories per split (0 = all)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = ["val"] if args.valid_only else ["train", "val"]

    for split in splits:
        raw = out / FILES[split]
        _download(f"{BASE}/{FILES[split]}", raw)
        docs = iter_documents(raw)
        if args.limit:
            docs = _limited(docs, args.limit)
        tokens = write_documents(docs, out / BIN[split])
        print(f"  {split}: {tokens:,} tokens -> {out / BIN[split]} ({tokens * 2 / 1e6:.0f} MB)")

    print("done. Train with:  uv run python -m llmscratch.train_mlx --config small_50m")


if __name__ == "__main__":
    main()
