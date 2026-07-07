"""Streaming document iteration + per-document EOT tokenization (TinyStories-style)."""

import numpy as np

from llmscratch.data import iter_documents, load_split, write_documents, write_mixed
from llmscratch.tokenizer import Tokenizer


def test_iter_documents_streams_split(tmp_path):
    p = tmp_path / "stories.txt"
    p.write_text("alpha<|endoftext|>beta<|endoftext|>gamma", encoding="utf-8")
    assert list(iter_documents(p)) == ["alpha", "beta", "gamma"]


def test_iter_documents_skips_empty(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("one<|endoftext|>   <|endoftext|>two<|endoftext|>", encoding="utf-8")
    assert list(iter_documents(p)) == ["one", "two"]


def test_iter_documents_handles_separator_across_chunks(tmp_path):
    # tiny chunk size forces the separator to span multiple reads
    p = tmp_path / "s.txt"
    p.write_text("aaaa<|endoftext|>bbbb", encoding="utf-8")
    assert list(iter_documents(p, chunk_size=3)) == ["aaaa", "bbbb"]


def test_write_documents_appends_eot_per_doc(tmp_path):
    eot = Tokenizer().eot_token
    n = write_documents(["hello world", "goodbye now"], tmp_path / "train.bin")
    arr = np.asarray(load_split(tmp_path, "train"))
    assert int((arr == eot).sum()) == 2   # exactly one EOT per document
    assert arr[-1] == eot
    assert len(arr) == n


def test_write_mixed_interleaves_by_weight(tmp_path):
    srcA = ("A", ("alpha beta gamma delta " for _ in range(5000)), 0.7)
    srcB = ("B", ("epsilon zeta " for _ in range(5000)), 0.3)
    counts = write_mixed([srcA, srcB], tmp_path / "data", target_tokens=3000, val_frac=0.0, seed=0)
    arr = load_split(tmp_path / "data", "train")
    assert len(arr) >= 2500          # roughly hit the target (per-doc granularity)
    assert counts["A"] > counts["B"]  # 0.7 weight beats 0.3
    assert int(arr.max()) < 50257


def test_write_mixed_logs_progress_when_enabled(tmp_path, capsys):
    srcA = ("A", ("alpha beta gamma delta " for _ in range(5000)), 1.0)
    write_mixed([srcA], tmp_path / "data", target_tokens=3000, val_frac=0.0, seed=0,
                log_every_tokens=500)
    out = capsys.readouterr().out
    assert "[build]" in out                     # progress lines printed
    assert "tok/s" in out and "ETA" in out       # include rate + ETA


def test_write_mixed_silent_by_default(tmp_path, capsys):
    srcA = ("A", ("alpha beta gamma delta " for _ in range(5000)), 1.0)
    write_mixed([srcA], tmp_path / "data", target_tokens=3000, val_frac=0.0, seed=0)
    assert "[build]" not in capsys.readouterr().out  # silent unless explicitly enabled
