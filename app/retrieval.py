"""Hybrid retrieval: dense (embedding cosine) + sparse (BM25) fused with Reciprocal Rank Fusion.

Dense catches paraphrases ("time off" ~ "vacation"); BM25 catches exact tokens that embeddings
blur (policy codes, names, numbers). RRF merges *ranks*, so we never have to normalise two
incomparable score scales. The absolute dense cosine of the best hit is kept separately as the
"is there anything relevant at all?" confidence signal used by the guardrail.
"""

from __future__ import annotations

from dataclasses import dataclass

from .embeddings import Embedder
from .store import Chunk, Store

RRF_K = 60


@dataclass
class Hit:
    chunk: Chunk
    dense: float  # cosine similarity with the query
    rrf: float


@dataclass
class Retrieval:
    hits: list[Hit]
    top_dense: float  # best cosine over the whole corpus (0 if empty)


class Retriever:
    def __init__(self, store: Store, embedder: Embedder, candidates: int = 20, top_k: int = 6):
        self.store, self.embedder = store, embedder
        self.candidates, self.top_k = candidates, top_k

    def retrieve(self, query: str) -> Retrieval:
        dense = self.store.dense_scores(self.embedder.embed_query(query))
        if not dense:
            return Retrieval([], 0.0)
        dense_rank = sorted(dense, key=dense.get, reverse=True)[: self.candidates]
        sparse_rank = self.store.bm25_search(query, self.candidates)

        fused: dict[int, float] = {}
        for ranking in (dense_rank, sparse_rank):
            for pos, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + pos + 1)

        top_ids = sorted(fused, key=fused.get, reverse=True)[: self.top_k]
        chunks = self.store.get_chunks(top_ids)
        hits = [Hit(chunks[i], dense[i], fused[i]) for i in top_ids if i in chunks]
        return Retrieval(hits, max(dense.values()))
