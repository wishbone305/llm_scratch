"""Stream a blend of HuggingFace datasets into data/train.bin + data/val.bin.

Grows the corpus past a single FineWeb-Edu shard by interleaving several curated sources. Each
source streams, is capped at weight * --target-tokens, and is mixed in doc-by-doc.

Requires the `datasets` library:  pip install datasets   (or: uv pip install datasets)

Usage:
    uv run python scripts/build_mix.py --target-tokens 4e9

On a box that forces a broken HF mirror via a startup hook (a shell `export HF_ENDPOINT` gets
clobbered at interpreter start), override it from inside the process:
    python scripts/build_mix.py --target-tokens 15e9 --hf-endpoint https://huggingface.co
"""

import argparse
import os
import sys
import time
from pathlib import Path

from llmscratch.data import write_mixed

# A transient network error (server disconnected, timeout, 5xx) is retried this many times
# *without making progress* before a source is dropped. Progress resets the budget, so a source
# can survive many disconnects over its lifetime as long as it keeps streaming between them.
MAX_RETRIES = 6
BACKOFF_CAP_SECONDS = 60
CHECKPOINT_EVERY = 1000  # snapshot stream position every N docs for cheap resume after a disconnect


def _force_hf_endpoint(endpoint: str) -> None:
    """Point huggingface_hub/datasets at ``endpoint``, beating any box startup hook that forces a
    broken mirror. Some rental images inject an ``HF_ENDPOINT`` override via a sitecustomize/.pth
    that runs at interpreter start — *after* your shell ``export`` — so the only reliable fix is to
    set it from inside the process before ``datasets`` imports. Sets the env var and also patches
    the ``ENDPOINT`` constant directly, covering both env-injected and patched-constant mirrors.
    """
    os.environ["HF_ENDPOINT"] = endpoint
    for name in ("huggingface_hub", "huggingface_hub.constants"):
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "ENDPOINT", None):
            mod.ENDPOINT = endpoint
    try:  # force-import + patch for the common case where hf isn't imported yet
        import huggingface_hub.constants as _c
        _c.ENDPOINT = endpoint
    except Exception:  # noqa: BLE001 — best effort; the env var alone still applies at import
        pass

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

    On a mid-stream "server disconnected" the stream is re-opened and resumed. Resume prefers
    ``IterableDataset.state_dict()`` / ``load_state_dict()`` (datasets >= 2.19): a cheap seek back
    to the saved shard+row that does NOT re-download earlier shards. It falls back to ``.skip(n)``
    (which re-streams from the start) only when state_dict is unavailable. Progress resets the
    retry budget, so a source survives many disconnects; after ``MAX_RETRIES`` *stalled* attempts
    it ends gracefully and the build continues with the other sources. ``HF_TOKEN`` is honoured
    automatically via ``datasets``.
    """
    from datasets import load_dataset

    ds_args = [hf_id] + ([subset] if subset else [])
    state = None          # last saved IterableDataset position (cheap-resume checkpoint)
    checkpoint_seen = 0   # doc count at that saved position
    seen = 0              # docs consumed so far
    attempt = 0           # consecutive stalled (no-progress) retries
    while True:
        advanced = False
        try:
            ds = load_dataset(*ds_args, split="train", streaming=True)
            can_checkpoint = hasattr(ds, "state_dict") and hasattr(ds, "load_state_dict")
            if state is not None and can_checkpoint:
                ds.load_state_dict(state)   # cheap seek back to where we were (no full re-read)
                seen = checkpoint_seen
            elif seen and hasattr(ds, "skip"):
                ds = ds.skip(seen)          # fallback: re-stream past what we've consumed
            for ex in ds:
                advanced = True
                seen += 1
                t = ex.get(field)
                if t:
                    yield t
                if can_checkpoint and seen % CHECKPOINT_EVERY == 0:
                    state, checkpoint_seen = ds.state_dict(), seen
            return  # stream exhausted cleanly
        except Exception as exc:  # noqa: BLE001 — network resilience: retry, then drop the source
            if advanced:
                attempt = 0  # made progress this round; the disconnect was transient
            attempt += 1
            if attempt > MAX_RETRIES:
                print(f"! {hf_id}: giving up at doc {seen} after {MAX_RETRIES} stalled retries "
                      f"({type(exc).__name__}: {exc}) — dropping remainder", flush=True)
                return
            wait = min(2 ** attempt, BACKOFF_CAP_SECONDS)
            print(f"! {hf_id}: {type(exc).__name__} at doc {seen} — "
                  f"resume retry {attempt}/{MAX_RETRIES} in {wait}s", flush=True)
            time.sleep(wait)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-tokens", type=float, default=4e9, help="total tokens to write (default 4B)")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-frac", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--hf-endpoint", default=None,
                    help="force the HuggingFace endpoint (e.g. https://huggingface.co), overriding a "
                         "box mirror injected via a sitecustomize/.pth startup hook")
    args = ap.parse_args()

    # Override a forced mirror from *inside* the process (a shell `export HF_ENDPOINT` loses to a
    # startup hook). Must run before any datasets/huggingface_hub import below.
    if args.hf_endpoint:
        _force_hf_endpoint(args.hf_endpoint)
        print(f"HF endpoint forced to {args.hf_endpoint}", flush=True)

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
