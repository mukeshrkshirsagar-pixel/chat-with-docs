"""The RAG pipeline: rewrite -> retrieve -> guard -> generate -> verify -> trace.

`answer_stream` is a generator of UI events (`sources`, `token`, `done`, `error`) so the HTTP layer
is a thin SSE wrapper and the whole flow is testable without a server.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator

from .config import Settings
from .llm import LLM, LLMError
from .observability import Timer, new_id
from .prompts import NO_ANSWER_TEXT, PROMPT_VERSION, build_user_message, format_sources
from .retrieval import Retriever
from .store import Store

log = logging.getLogger("rag")
_CITATION = re.compile(r"[\[【](\d+)[\]】]")  # some models emit 【1】 instead of [1]


def clean_history(history: list[dict], cfg: Settings) -> list[dict]:
    """Keep the last N turns, truncate long assistant answers, enforce a user-first transcript."""
    msgs = [
        {"role": m["role"], "content": str(m["content"])[: cfg.history_answer_chars]}
        for m in history
        if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()
    ][-cfg.history_turns * 2 :]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return msgs


class RagPipeline:
    def __init__(self, cfg: Settings, store: Store, retriever: Retriever, llm: LLM):
        self.cfg, self.store, self.retriever, self.llm = cfg, store, retriever, llm

    def answer_stream(self, question: str, history: list[dict] | None = None) -> Iterator[dict]:
        cfg = self.cfg
        total = Timer()
        trace: dict = {"trace_id": new_id(), "prompt_version": PROMPT_VERSION, "question": question}
        timings: dict[str, int] = {}
        flags: list[str] = []
        history = clean_history(history or [], cfg)

        # 1. Condense a follow-up into a standalone query (skipped on the first turn: no LLM call).
        t = Timer()
        query = self.llm.rewrite(history, question) if history else question
        timings["rewrite"] = t.ms()
        trace["search_query"] = query

        # 2. Retrieve.
        t = Timer()
        retrieval = self.retriever.retrieve(query)
        timings["retrieve"] = t.ms()
        trace["top_dense"] = round(retrieval.top_dense, 3)
        trace["retrieved"] = [
            {"doc": h.chunk.doc_name, "page": h.chunk.page, "dense": round(h.dense, 3)}
            for h in retrieval.hits
        ]

        # 3. Guardrail: nothing relevant -> refuse *without* calling the LLM (cheaper, and can't hallucinate).
        if not retrieval.hits or retrieval.top_dense < cfg.min_similarity:
            yield {"event": "sources", "data": []}
            yield {"event": "token", "data": NO_ANSWER_TEXT}
            timings["total"] = total.ms()
            trace.update(grounded=False, reason="low_retrieval_confidence", timings_ms=timings, flags=flags)
            yield from self._finish(trace)
            return

        sources_block, n_used = format_sources(retrieval.hits, cfg.max_context_chars)
        used_hits = retrieval.hits[:n_used]
        yield {
            "event": "sources",
            "data": [
                {
                    "n": i,
                    "doc": h.chunk.doc_name,
                    "page": h.chunk.page,
                    "heading": h.chunk.heading,
                    "score": round(h.dense, 3),
                    "text": h.chunk.text,
                }
                for i, h in enumerate(used_hits, start=1)
            ],
        }

        # 4. Generate (streamed).
        answer_parts: list[str] = []
        end: dict = {}
        t = Timer()
        first_token_ms: int | None = None
        try:
            for kind, value in self.llm.stream_answer(history, build_user_message(sources_block, question)):
                if kind == "text":
                    if first_token_ms is None:
                        first_token_ms = t.ms()
                    answer_parts.append(value)
                    yield {"event": "token", "data": value}
                else:
                    end = value
        except LLMError as e:
            log.error("generation failed", extra={"trace_id": trace["trace_id"], "error": str(e)})
            timings["total"] = total.ms()
            trace.update(grounded=False, reason="llm_error", error=str(e), timings_ms=timings, flags=flags)
            self.store.log_query(trace)
            yield {"event": "error", "data": str(e)}
            return
        timings["generate"] = t.ms()
        timings["ttft"] = first_token_ms or 0

        # 5. Verify the output: citations must point at sources we actually supplied.
        answer = "".join(answer_parts)
        cited = sorted({int(n) for n in _CITATION.findall(answer)})
        invalid = [n for n in cited if not 1 <= n <= n_used]
        if invalid:
            flags.append("invalid_citation")
        if not cited:
            flags.append("no_citations")  # legitimate for "not in the documents" answers; tracked as a signal
        if end.get("stop_reason") == "refusal":
            flags.append("model_refusal")
        elif end.get("stop_reason") == "max_tokens":
            flags.append("truncated")

        timings["total"] = total.ms()
        trace.update(
            grounded=True,
            cited=cited,
            invalid_citations=invalid,
            flags=flags,
            usage=end.get("usage", {}),
            stop_reason=end.get("stop_reason"),
            timings_ms=timings,
            answer_chars=len(answer),
        )
        yield from self._finish(trace)

    def _finish(self, trace: dict) -> Iterator[dict]:
        self.store.log_query(trace)
        log.info("query", extra={k: v for k, v in trace.items() if k not in ("question", "retrieved")})
        # Don't ship the raw log payload to the browser; send the parts the UI shows.
        yield {
            "event": "done",
            "data": {
                k: trace.get(k)
                for k in (
                    "trace_id",
                    "grounded",
                    "reason",
                    "flags",
                    "cited",
                    "invalid_citations",
                    "search_query",
                    "top_dense",
                    "timings_ms",
                    "usage",
                )
            },
        }
