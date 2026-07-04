"""Tokenize a directory of .txt shards into data/train.bin + data/val.bin.

For corpora exported as text shards where documents are delimited by a literal <|endoftext|>
(e.g. FineWeb-Edu). Each document is split out and tokenized with a real EOT token appended;
documents are assigned to train/val at the document level with probability --val-frac.
Streams file-by-file, so memory stays bounded over multi-GB corpora.
"""

import argparse
from pathlib import Path

import numpy as np

from llmscratch.data import iter_documents
from llmscratch.tokenizer import Tokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_dir", nargs="?", default="data/txt")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-frac", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    files = sorted(in_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"no .txt files in {in_dir}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer()
    rng = np.random.default_rng(args.seed)
    counts = {"train": 0, "val": 0}
    docs = 0

    with open(out / "train.bin", "wb") as tf, open(out / "val.bin", "wb") as vf:
        for fp in files:
            print(f"tokenizing {fp.name} ...", flush=True)
            for doc in iter_documents(fp):
                ids = np.asarray(tok.encode(doc, add_eot=True), dtype=np.uint16)
                target = "val" if rng.random() < args.val_frac else "train"
                ids.tofile(vf if target == "val" else tf)
                counts[target] += len(ids)
                docs += 1
            print(f"  cumulative: {counts['train']:,} train / {counts['val']:,} val tokens", flush=True)

    print(f"DONE: {docs:,} docs -> {counts['train']:,} train + {counts['val']:,} val tokens "
          f"({(counts['train']+counts['val'])*2/1e9:.2f} GB of .bin)")


if __name__ == "__main__":
    main()
