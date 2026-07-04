"""CLI: tokenize raw text into data/train.bin + data/val.bin (GPT-2 BPE, uint16).

Usage:
    uv run python scripts/prepare_data.py path/to/corpus.txt
    uv run python scripts/prepare_data.py path/to/dir_of_txt_files --val-frac 0.05
"""

import argparse
from pathlib import Path

from llmscratch.data import prepare_corpus


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="a .txt file or a directory containing .txt files")
    ap.add_argument("--out-dir", default="data", help="output directory (default: data)")
    ap.add_argument("--val-frac", type=float, default=0.1, help="validation fraction (default: 0.1)")
    ap.add_argument("--seed", type=int, default=1337, help="RNG seed for the split (default: 1337)")
    args = ap.parse_args()

    counts = prepare_corpus(args.input, args.out_dir, val_frac=args.val_frac, seed=args.seed)
    total = counts["train"] + counts["val"]
    out = Path(args.out_dir).resolve()
    print(f"Wrote {counts['train']:,} train + {counts['val']:,} val = {total:,} tokens to {out}")
    print(f"  train.bin: {counts['train'] * 2 / 1e6:.1f} MB    val.bin: {counts['val'] * 2 / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
