"""Embedding backends behind one tiny interface.

`fastembed` runs an ONNX model locally (no torch, no API key, no per-token cost).
`hash` is a deterministic bag-of-words embedder used for offline tests and CI.
"""

from __future__ import annotations

import re
import zlib
from typing import Protocol

import numpy as np

# bge models are trained with this prefix on the *query* side only.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.clip(norms, 1e-12, None)).astype(np.float32)


class FastEmbedEmbedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding  # heavy import, keep lazy

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        self.dim = len(next(iter(self._model.embed(["probe"]))))
        self._query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _l2_normalize(np.array(list(self._model.embed(texts, batch_size=32))))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2_normalize(np.array(list(self._model.embed([self._query_prefix + text]))))[0]


_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the a an and or of to in on for is are was were be by with at as it this that from".split()
)


class HashEmbedder:
    """Feature-hashed bag of words. Lexical only, but deterministic and dependency-free."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in _WORD.findall(text.lower()):
            if w in _STOP:
                continue
            h = zlib.crc32(w.encode())
            v[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _l2_normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2_normalize(self._vec(text)[None, :])[0]


def build_embedder(kind: str, model_name: str) -> Embedder:
    if kind == "hash":
        return HashEmbedder()
    return FastEmbedEmbedder(model_name)
