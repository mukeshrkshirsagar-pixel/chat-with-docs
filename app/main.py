"""FastAPI app: REST for documents, SSE for chat, static single-page UI."""

from __future__ import annotations

import base64
import json
import logging
import secrets
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .embeddings import Embedder, build_embedder
from .ingestion import ingest, ingest_samples
from .llm import LLM, build_llm
from .loaders import IngestError
from .observability import Timer, new_id, request_id, setup_logging
from .rag import RagPipeline
from .retrieval import Retriever
from .store import Store

log = logging.getLogger("api")
STATIC = Path(__file__).parent / "static"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[ChatMessage] = []


def _has_password(request: Request, password: str) -> bool:
    """HTTP basic auth where only the password matters (any user name is accepted)."""
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "basic":
        return False
    try:
        supplied = base64.b64decode(token).decode().partition(":")[2]
    except ValueError:
        return False
    return secrets.compare_digest(supplied.encode(), password.encode())


def create_app(
    cfg: Settings | None = None, embedder: Embedder | None = None, llm: LLM | None = None
) -> FastAPI:
    """App factory: tests inject a fake embedder/LLM and a temp data dir."""
    cfg = cfg or get_settings()
    setup_logging()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    embedder = embedder or build_embedder(cfg.embedder, cfg.embedding_model)
    llm = llm or build_llm(cfg)
    store = Store(str(cfg.db_path), embedder.dim)
    pipeline = RagPipeline(cfg, store, Retriever(store, embedder, cfg.candidates, cfg.top_k), llm)

    if cfg.load_samples_on_start:
        ingest_samples(store, embedder, cfg)

    app = FastAPI(title="Chat With Your Docs")

    if cfg.access_password:

        @app.middleware("http")
        async def require_password(request: Request, call_next):
            if request.url.path == "/healthz" or _has_password(request, cfg.access_password):
                return await call_next(request)
            return Response(
                "Authentication required", status_code=401, headers={"WWW-Authenticate": 'Basic realm="docs"'}
            )

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        rid = request.headers.get("x-request-id") or new_id()
        request_id.set(rid)
        timer = Timer()
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        if request.url.path.startswith("/api"):
            log.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "ms": timer.ms(),
                },
            )
        return response

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "llm": type(llm).__name__, "embedder": type(embedder).__name__}

    @app.get("/api/documents")
    def list_documents() -> list[dict]:
        return store.list_documents()

    @app.post("/api/documents")
    def upload(files: list[UploadFile] = File(...)) -> list[dict]:
        results = []
        limit = cfg.max_upload_mb * 1024 * 1024
        for f in files:
            name = Path(f.filename or "upload").name  # strip any client-supplied path
            try:
                data = f.file.read(limit + 1)
                if len(data) > limit:
                    raise IngestError(f"File exceeds the {cfg.max_upload_mb} MB limit.")
                r = ingest(store, embedder, cfg, name, data)
                results.append({"name": r.name, "status": r.status, "id": r.doc_id, "chunks": r.chunks})
            except IngestError as e:
                results.append({"name": name, "status": "error", "detail": str(e)})
            except Exception:
                log.exception("ingest failed", extra={"doc": name})
                results.append(
                    {"name": name, "status": "error", "detail": "Unexpected error while indexing."}
                )
        return results

    @app.post("/api/samples")
    def load_samples() -> list[dict]:
        """One-click demo data: index the bundled sample documents."""
        return [
            {"name": r.name, "status": r.status, "chunks": r.chunks}
            for r in ingest_samples(store, embedder, cfg)
        ]

    @app.delete("/api/documents/{doc_id}")
    def delete_document(doc_id: int) -> dict:
        if not store.delete_document(doc_id):
            raise HTTPException(404, "Document not found")
        return {"deleted": doc_id}

    @app.post("/api/chat")
    def chat(req: ChatRequest) -> StreamingResponse:
        question = req.question.strip()
        if not question or len(question) > cfg.max_question_chars:
            raise HTTPException(422, f"Question must be 1-{cfg.max_question_chars} characters.")

        def sse() -> Iterator[str]:
            try:
                for ev in pipeline.answer_stream(question, [m.model_dump() for m in req.history]):
                    yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'])}\n\n"
            except Exception:
                log.exception("chat failed")
                yield f"event: error\ndata: {json.dumps('Something went wrong. Please try again.')}\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/stats")
    def stats() -> dict:
        return store.stats()

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def app_factory() -> FastAPI:  # `uvicorn app.main:app_factory --factory`
    return create_app()
