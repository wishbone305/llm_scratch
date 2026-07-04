"""GPT-2 BPE tokenizer wrapper (tiktoken). Shared by both training tracks and data prep."""

from __future__ import annotations

from collections.abc import Iterable

import tiktoken


class Tokenizer:
    """Thin wrapper around tiktoken's GPT-2 encoding (vocab = 50257, eot = 50256)."""

    def __init__(self, encoding_name: str = "gpt2") -> None:
        self._enc = tiktoken.get_encoding(encoding_name)
        self.name = encoding_name

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    @property
    def eot_token(self) -> int:
        return self._enc.eot_token

    def encode(self, text: str, *, add_eot: bool = False) -> list[int]:
        """Encode raw text. Special tokens in the text are treated as ordinary text."""
        ids = self._enc.encode_ordinary(text)
        if add_eot:
            ids.append(self.eot_token)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return self._enc.decode(list(ids))
