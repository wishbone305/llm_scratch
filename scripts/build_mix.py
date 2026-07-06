"""Stream a blend of HuggingFace datasets into data/train.bin + data/val.bin.

Grows the corpus past a single FineWeb-Edu shard by interleaving several curated sources. Each
source streams, is capped at weight * --target-tokens, and is mixed in doc-by-doc.

Requires the `datasets` library:  pip install datasets   (or: uv pip install datasets)

Usage:
    uv run python scripts/build_mix.py --target-tokens 4e9
"""

import argparse
import os
import time
from pathlib import Path

from llmscratch.data import write_mixed

# A transient network error (server disconnected, timeout, 5xx) is retried this many times
# *without making progress* before a source is dropped. Progress resets the budget, so a source
# can survive many disconnects over its lifetime as long as it keeps streaming between them.
MAX_RETRIES = 6
BACKOFF_CAP_SECONDS = 60

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
    """Yield the text field from a streaming HF dataset, resilient to transient network errors.

    A failed initial load or a mid-stream "server disconnected" is retried with exponential
    backoff: the stream is re-opened and fast-forwarded past what we already yielded (``.skip``),
    so we resume rather than restart. Any progress between failures resets the retry budget. After
    ``MAX_RETRIES`` failures *with no progress* the source ends gracefully — the build continues
    with the other sources instead of crashing. Honours ``HF_TOKEN`` automatically via ``datasets``.
    """
    from datasets import load_dataset

    ds_args = [hf_id] + ([subset] if subset else [])
    seen = 0        # docs already yielded (the resume point after a reconnect)
    attempt = 0     # consecutive failures with no progress
    while True:
        progress_before = seen
        try:
            ds = load_dataset(*ds_args, split="train", streaming=True)
            if seen:
                ds = ds.skip(seen)  # resume past what we already consumed
            for ex in ds:
                seen += 1
                t = ex.get(field)
                if t:
                    yield t
            return  # stream exhausted cleanly
        except Exception as exc:  # noqa: BLE001 — network resilience: retry, then drop the source
            if seen > progress_before:
                attempt = 0  # made progress this round; the disconnect was transient
            attempt += 1
            if attempt > MAX_RETRIES:
                print(f"! {hf_id}: giving up at doc {seen} after {MAX_RETRIES} stalled retries "
                      f"({type(exc).__name__}: {exc}) — dropping remainder", flush=True)
                return
            wait = min(2 ** attempt, BACKOFF_CAP_SECONDS)
            print(f"! {hf_id}: {type(exc).__name__} at doc {seen} — "
                  f"retry {attempt}/{MAX_RETRIES} in {wait}s", flush=True)
            time.sleep(wait)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-tokens", type=float, default=4e9, help="total tokens to write (default 4B)")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-frac", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    # Faster, more resilient HF downloads. Must be set BEFORE huggingface_hub/datasets import.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")  # default 10s -> fewer spurious timeouts
    try:
        import hf_transfer  # noqa: F401  (Rust-accelerated downloads, if installed)
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    except ImportError:
        pass

    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("HF token detected — authenticated streaming (higher rate limits, fewer disconnects).",
              flush=True)
    else:
        print("No HF_TOKEN set — anonymous access is throttled and disconnects more.\n"
              "  For faster, more reliable downloads:  export HF_TOKEN=hf_...   "
              "(get one at https://huggingface.co/settings/tokens)", flush=True)

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
