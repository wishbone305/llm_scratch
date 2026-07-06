"""Stream a blend of HuggingFace datasets into data/train.bin + data/val.bin.

Grows the corpus past a single FineWeb-Edu shard by interleaving several curated sources. Each
source streams, is capped at weight * --target-tokens, and is mixed in doc-by-doc.

Requires the `datasets` library:  pip install datasets   (or: uv pip install datasets)

Usage:
    uv run python scripts/build_mix.py --target-tokens 4e9
"""

import argparse
from pathlib import Path

from llmscratch.data import write_mixed

# (hf_id, subset_or_None, text_field, weight) — edit freely. Verified curated blend for a small model.
BLEND = [
    ("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", "text", 0.40),  # on-distribution bulk
    ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2",     "text", 0.18),  # synthetic quality lift
    ("mlfoundations/dclm-baseline-1.0-parquet", None,    "text", 0.15),  # general-web diversity
    ("wikimedia/wikipedia",         "20231101.en",       "text", 0.10),  # factual grounding
    ("HuggingFaceTB/finemath",      "finemath-4plus",    "text", 0.07),  # reasoning signal
    ("DKYoon/SlimPajama-6B",        None,                "text", 0.06),  # multi-source diversity
    ("common-pile/project_gutenberg", None,              "text", 0.04),  # public-domain books (long-form)
]


def _texts(hf_id, subset, field):
    """Yield the text field from a streaming HF dataset; a failed load yields nothing."""
    try:
        from datasets import load_dataset
        args = [hf_id] + ([subset] if subset else [])
        ds = load_dataset(*args, split="train", streaming=True)
    except Exception as exc:  # gated / missing / config error -> skip this source
        print(f"! {hf_id}: load failed ({exc}) — skipping", flush=True)
        return
    for ex in ds:
        t = ex.get(field)
        if t:
            yield t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-tokens", type=float, default=4e9, help="total tokens to write (default 4B)")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-frac", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    try:
        import datasets  # noqa: F401
    except ImportError:
        raise SystemExit("This needs the 'datasets' library:  pip install datasets")

    sources = []
    for hf_id, subset, field, weight in BLEND:
        name = f"{hf_id}" + (f":{subset}" if subset else "")
        sources.append((name, _texts(hf_id, subset, field), weight))
        print(f"+ {name}  (weight {weight})", flush=True)

    print(f"building ~{int(args.target_tokens):,} tokens (interleaved) ...", flush=True)
    counts = write_mixed(sources, args.out_dir, int(args.target_tokens),
                         val_frac=args.val_frac, seed=args.seed)
    total = sum(counts.values())
    print(f"\nDONE: {total:,} tokens -> {Path(args.out_dir).resolve()}")
    for name, c in counts.items():
        pct = 100 * c / max(total, 1)
        print(f"  {c:>14,}  {pct:5.1f}%  {name}")


if __name__ == "__main__":
    main()
