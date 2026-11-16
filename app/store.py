"""SQLite-backed store: documents, chunks, vectors, a BM25 (FTS5) index and a query log.

Why not a vector DB? At this scale (thousands of chunks) a brute-force matrix product
over an in-memory float32 matrix takes < 1 ms, and SQLite gives metadata, keyword search
and persistence in one zero-ops file. The `Store` interface is the seam where
pgvector / Qdrant / OpenSearch would slot in (see README).
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE,
  n_chunks INTEGER NOT NULL, n_pages INTEGER, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  idx INTEGER NOT NULL, page INTEGER, heading TEXT, text TEXT NOT NULL, embedding BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, heading, tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS query_log(
  id INTEGER PRIMARY KEY, ts REAL NOT NULL, trace_id TEXT, payload TEXT NOT NULL);
"""


@dataclass
class Chunk:
    id: int
    doc_id: int
    doc_name: str
    idx: int
    page: int | None
    heading: str | None
    text: str


class Store:
    def __init__(self, path: str, dim: int):
        self.dim = dim
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        self._ids: np.ndarray | None = None  # cached vector index
        self._matrix: np.ndarray | None = None

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def find_by_hash(self, sha256: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM documents WHERE sha256=?", (sha256,)).fetchone()
        return dict(row) if row else None

    def add_document(self, name: str, sha256: str, drafts: list, vectors: np.ndarray) -> int:
        pages = {d.page for d in drafts if d.page is not None}
        with self._lock, self._db:  # one transaction: all chunks or none
            cur = self._db.execute(
                "INSERT INTO documents(name, sha256, n_chunks, n_pages, created_at) VALUES(?,?,?,?,?)",
                (name, sha256, len(drafts), len(pages) or None, time.time()),
            )
            doc_id = cur.lastrowid
            for i, (d, v) in enumerate(zip(drafts, vectors)):
                c = self._db.execute(
                    "INSERT INTO chunks(doc_id, idx, page, heading, text, embedding) VALUES(?,?,?,?,?,?)",
                    (doc_id, i, d.page, d.heading, d.text, v.astype(np.float32).tobytes()),
                )
                self._db.execute(
                    "INSERT INTO chunks_fts(rowid, text, heading) VALUES(?,?,?)",
                    (c.lastrowid, d.text, d.heading or ""),
                )
            self._invalidate()
        return doc_id

    def list_documents(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, doc_id: int) -> bool:
        with self._lock, self._db:
            ids = [r[0] for r in self._db.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,))]
            self._db.executemany("DELETE FROM chunks_fts WHERE rowid=?", [(i,) for i in ids])
            n = self._db.execute("DELETE FROM documents WHERE id=?", (doc_id,)).rowcount
            self._invalidate()
        return n > 0

    def _invalidate(self) -> None:
        self._ids = self._matrix = None

    def _load_index(self) -> None:
        if self._matrix is not None:
            return
        rows = self._db.execute("SELECT id, embedding FROM chunks ORDER BY id").fetchall()
        self._ids = np.array([r["id"] for r in rows], dtype=np.int64)
        self._matrix = (
            np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            if rows
            else np.zeros((0, self.dim), dtype=np.float32)
        )

    def dense_scores(self, qvec: np.ndarray) -> dict[int, float]:
        """Cosine similarity of the query against every chunk (vectors are unit-norm)."""
        with self._lock:
            self._load_index()
            if len(self._ids) == 0:
                return {}
            scores = self._matrix @ qvec
            return dict(zip(self._ids.tolist(), scores.tolist()))

    def bm25_search(self, query: str, k: int) -> list[int]:
        """Chunk ids ranked by BM25. Query terms are OR-ed and quoted (no FTS syntax injection)."""
        terms = re.findall(r"\w+", query.lower())
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        with self._lock:
            rows = self._db.execute(
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts, 1.0, 0.5) LIMIT ?",
                (match, k),
            ).fetchall()
        return [r[0] for r in rows]

    def get_chunks(self, ids: list[int]) -> dict[int, Chunk]:
        if not ids:
            return {}
        q = ",".join("?" * len(ids))
        with self._lock:
            rows = self._db.execute(
                f"SELECT c.*, d.name AS doc_name FROM chunks c JOIN documents d ON d.id=c.doc_id WHERE c.id IN ({q})",
                ids,
            ).fetchall()
        return {
            r["id"]: Chunk(r["id"], r["doc_id"], r["doc_name"], r["idx"], r["page"], r["heading"], r["text"])
            for r in rows
        }

    def log_query(self, trace: dict) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO query_log(ts, trace_id, payload) VALUES(?,?,?)",
                (time.time(), trace.get("trace_id"), json.dumps(trace)),
            )

    def stats(self, window: int = 500) -> dict:
        with self._lock:
            rows = self._db.execute(
                "SELECT payload FROM query_log ORDER BY id DESC LIMIT ?", (window,)
            ).fetchall()
            n_docs = self._db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            n_chunks = self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        traces = [json.loads(r[0]) for r in rows]
        lat = sorted(t["timings_ms"]["total"] for t in traces if "timings_ms" in t)

        def pct(p: float) -> float | None:
            return lat[min(len(lat) - 1, int(p * len(lat)))] if lat else None

        n = len(traces)
        return {
            "documents": n_docs,
            "chunks": n_chunks,
            "queries": n,
            "refusal_rate": round(sum(not t.get("grounded", True) for t in traces) / n, 3) if n else None,
            "flagged_rate": round(sum(bool(t.get("flags")) for t in traces) / n, 3) if n else None,
            "latency_ms": {"p50": pct(0.5), "p95": pct(0.95)},
            "tokens": {
                "input": sum(t.get("usage", {}).get("input_tokens", 0) for t in traces),
                "output": sum(t.get("usage", {}).get("output_tokens", 0) for t in traces),
            },
        }
