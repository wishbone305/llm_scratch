"""Tests for scripts/build_mix.py `_texts` — network resilience of the streaming loader.

build_mix.py is a CLI script (not a package module), so it's loaded from its path via importlib.
`datasets` is faked in sys.modules, so these tests need no network and no real datasets install.
"""

import importlib.util
import sys
import types
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_mix.py"


def _load_build_mix(monkeypatch, load_dataset):
    """Import build_mix.py fresh with a fake `datasets.load_dataset`, and no-op sleeps."""
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    spec = importlib.util.spec_from_file_location("build_mix_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)  # don't actually back off in tests
    return mod


class _FakeStream:
    """A streaming IterableDataset stand-in over an in-memory doc list.

    `fail_at` raises ConnectionError once when the iterator reaches that index; `.skip(n)` returns
    a fresh stream resuming at offset n (mirroring datasets.IterableDataset.skip), never re-failing.
    """

    def __init__(self, docs, fail_at=None, offset=0):
        self._docs = docs
        self._fail_at = fail_at
        self._offset = offset

    def skip(self, n):
        return _FakeStream(self._docs, fail_at=None, offset=n)

    def __iter__(self):
        i = self._offset
        while i < len(self._docs):
            if self._fail_at is not None and i == self._fail_at:
                self._fail_at = None  # fail only once per stream
                raise ConnectionError("server disconnected")
            yield {"text": self._docs[i]}
            i += 1


def test_clean_stream_yields_all(monkeypatch):
    docs = [f"doc{i}" for i in range(5)]
    mod = _load_build_mix(monkeypatch, lambda *a, **k: _FakeStream(docs))
    assert list(mod._texts("x", None, "text")) == docs


def test_midstream_disconnect_resumes_without_duplicates(monkeypatch):
    docs = [f"doc{i}" for i in range(10)]
    calls = {"n": 0}

    def load_dataset(*_a, **_k):
        calls["n"] += 1
        # Only the first stream fails (at index 3); the reconnect resumes via .skip and completes.
        return _FakeStream(docs, fail_at=3 if calls["n"] == 1 else None)

    mod = _load_build_mix(monkeypatch, load_dataset)
    assert list(mod._texts("x", None, "text")) == docs  # every doc once, in order
    assert calls["n"] == 2  # exactly one reconnect


def test_permanent_failure_drops_source_gracefully(monkeypatch):
    def load_dataset(*_a, **_k):
        raise ConnectionError("server disconnected")

    mod = _load_build_mix(monkeypatch, load_dataset)
    # Never raises out of the generator; yields nothing after exhausting retries.
    assert list(mod._texts("x", None, "text")) == []


def test_empty_and_missing_text_fields_are_skipped(monkeypatch):
    docs = ["keep1", "", "keep2", None]
    mod = _load_build_mix(monkeypatch, lambda *a, **k: _FakeStream(docs))
    assert list(mod._texts("x", None, "text")) == ["keep1", "keep2"]
