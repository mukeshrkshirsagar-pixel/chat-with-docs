"""Upload -> sections -> chunks -> embeddings -> store."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from .chunking import chunk_sections
from .config import Settings
from .embeddings import Embedder
from .loaders import load_document
from .store import Store

log = logging.getLogger("ingest")


@dataclass
class IngestResult:
    name: str
    status: str  # indexed | duplicate
    doc_id: int
    chunks: int


def embed_text(draft, doc_name: str) -> str:
    """Prefix a cheap 'document > heading' header: disambiguates chunks that read the same out of context."""
    header = doc_name if not draft.heading else f"{doc_name} > {draft.heading}"
    return f"{header}\n{draft.text}"


def ingest(store: Store, embedder: Embedder, cfg: Settings, filename: str, data: bytes) -> IngestResult:
    sha = hashlib.sha256(data).hexdigest()
    if (existing := store.find_by_hash(sha)) is not None:
        return IngestResult(existing["name"], "duplicate", existing["id"], existing["n_chunks"])

    sections = load_document(filename, data)
    drafts = chunk_sections(sections, cfg.chunk_size, cfg.chunk_overlap)
    vectors = embedder.embed_documents([embed_text(d, filename) for d in drafts])
    doc_id = store.add_document(filename, sha, drafts, vectors)
    log.info("indexed", extra={"doc": filename, "chunks": len(drafts), "bytes": len(data)})
    return IngestResult(filename, "indexed", doc_id, len(drafts))
