"""Fetch a tiny public-domain corpus and tokenize it, so the pipeline can be smoke-tested
end-to-end without the user's real dataset. Falls back to bundled text if offline.
"""

import urllib.request
from pathlib import Path

from llmscratch.data import prepare_corpus

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

_FALLBACK = (
    "It is a truth universally acknowledged, that a single man in possession of a good "
    "fortune, must be in want of a wife. However little known the feelings or views of "
    "such a man may be on his first entering a neighbourhood, this truth is so well fixed "
    "in the minds of the surrounding families, that he is considered the rightful property "
    "of some one or other of their daughters. The quick brown fox jumps over the lazy dog. "
)


def main() -> None:
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "sample.txt"
    try:
        urllib.request.urlretrieve(URL, raw)
        print(f"Downloaded tinyshakespeare to {raw}")
    except Exception as exc:  # offline / blocked: use bundled text
        print(f"Download failed ({exc}); writing bundled fallback text instead.")
        raw.write_text(_FALLBACK * 4000, encoding="utf-8")  # ~2MB of text

    stats = prepare_corpus(raw, out_dir, val_frac=0.1)
    total = stats["train"] + stats["val"]
    print(f"Tokenized {total:,} tokens ({stats['train']:,} train / {stats['val']:,} val) into {out_dir}/")


if __name__ == "__main__":
    main()
