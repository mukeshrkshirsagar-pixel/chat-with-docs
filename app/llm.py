"""LLM access behind a small interface so the pipeline is testable and the provider swappable.

Stream protocol: `stream_answer` yields ("text", str) deltas and finishes with
("end", {"stop_reason": ..., "usage": {...}}).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Protocol

import anthropic

from .config import Settings
from .prompts import REWRITE_PROMPT, SYSTEM_PROMPT

log = logging.getLogger("llm")


class LLMError(Exception):
    """A user-presentable LLM failure (rate limit, auth, outage...)."""


class LLM(Protocol):
    def rewrite(self, history: list[dict], question: str) -> str: ...

    def stream_answer(self, history: list[dict], user_content: str) -> Iterator[tuple[str, object]]: ...


class AnthropicLLM:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        # SDK retries 408/409/429/5xx with backoff (max_retries=2 by default); we just set a ceiling.
        self.client = anthropic.Anthropic(api_key=cfg.anthropic_api_key, timeout=60.0, max_retries=2)

    @staticmethod
    def _friendly(e: Exception) -> LLMError:
        if isinstance(e, anthropic.AuthenticationError):
            return LLMError("The Anthropic API key was rejected. Check ANTHROPIC_API_KEY.")
        if isinstance(e, anthropic.RateLimitError):
            return LLMError("Rate limited by the LLM provider. Please retry in a moment.")
        if isinstance(e, anthropic.APIConnectionError):
            return LLMError("Could not reach the LLM provider.")
        if isinstance(e, anthropic.APIStatusError):
            return LLMError(f"The LLM provider returned an error ({e.status_code}).")
        return LLMError("Unexpected LLM failure.")

    def rewrite(self, history: list[dict], question: str) -> str:
        transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
        try:
            resp = self.client.messages.create(
                model=self.cfg.rewrite_model,
                max_tokens=1024,  # thinking tokens share this budget; the query itself is tiny
                system=REWRITE_PROMPT,
                output_config={"effort": "low"},
                messages=[
                    {
                        "role": "user",
                        "content": f"<history>\n{transcript}\n</history>\n\nLatest message: {question}",
                    }
                ],
            )
        except anthropic.APIError as e:
            # Rewrite is an optimisation: degrade to the raw question instead of failing the turn.
            log.warning("rewrite failed, using raw question", extra={"error": type(e).__name__})
            return question
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return text or question

    def stream_answer(self, history: list[dict], user_content: str) -> Iterator[tuple[str, object]]:
        messages = [*history, {"role": "user", "content": user_content}]
        try:
            with self.client.messages.stream(
                model=self.cfg.generation_model,
                max_tokens=self.cfg.max_answer_tokens,
                system=SYSTEM_PROMPT,
                output_config={"effort": self.cfg.generation_effort},
                messages=messages,
            ) as stream:
                for delta in stream.text_stream:
                    yield "text", delta
                final = stream.get_final_message()
        except anthropic.APIError as e:
            raise self._friendly(e) from e
        yield (
            "end",
            {
                "stop_reason": final.stop_reason,
                "usage": {
                    "input_tokens": final.usage.input_tokens,
                    "output_tokens": final.usage.output_tokens,
                },
            },
        )


class ExtractiveLLM:
    """No-API-key fallback: returns the retrieved passages verbatim instead of a synthesised answer.

    Keeps the app demoable and the whole retrieval stack testable without credentials.
    """

    def rewrite(self, history: list[dict], question: str) -> str:
        return question

    def stream_answer(self, history: list[dict], user_content: str) -> Iterator[tuple[str, object]]:
        import re

        blocks = re.findall(r'<source id="(\d+)"[^>]*>\n(.*?)\n</source>', user_content, flags=re.S)
        yield "text", "*Retrieval-only mode (no `ANTHROPIC_API_KEY` set). Most relevant passages:*\n\n"
        for sid, body in blocks[:3]:
            yield "text", f"> {body[:400].strip()}… [{sid}]\n\n"
        yield "end", {"stop_reason": "end_turn", "usage": {"input_tokens": 0, "output_tokens": 0}}


class XAILLM:
    """Grok via xAI's OpenAI-compatible chat-completions API (any OpenAI-compatible endpoint works)."""

    # chat-completions finish_reason -> the stop_reason vocabulary the pipeline uses
    _STOP = {"stop": "end_turn", "length": "max_tokens", "content_filter": "refusal"}

    def __init__(self, cfg: Settings, client=None):
        import openai

        self.cfg = cfg
        self._openai = openai
        self.client = client or openai.OpenAI(
            api_key=cfg.xai_api_key, base_url=cfg.xai_base_url, timeout=60.0, max_retries=2
        )

    def _friendly(self, e: Exception) -> LLMError:
        o = self._openai
        if isinstance(e, o.AuthenticationError):
            return LLMError("The xAI API key was rejected. Check XAI_API_KEY.")
        if isinstance(e, o.RateLimitError):
            return LLMError("Rate limited by the LLM provider. Please retry in a moment.")
        if isinstance(e, o.APIConnectionError):
            return LLMError("Could not reach the LLM provider.")
        if isinstance(e, o.NotFoundError):
            return LLMError(f"Model not found: check XAI_GENERATION_MODEL ({self.cfg.xai_generation_model}).")
        if isinstance(e, o.APIStatusError):
            return LLMError(f"The LLM provider returned an error ({e.status_code}).")
        return LLMError("Unexpected LLM failure.")

    def rewrite(self, history: list[dict], question: str) -> str:
        transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
        try:
            resp = self.client.chat.completions.create(
                model=self.cfg.xai_rewrite_model,
                messages=[
                    {"role": "system", "content": REWRITE_PROMPT},
                    {
                        "role": "user",
                        "content": f"<history>\n{transcript}\n</history>\n\nLatest message: {question}",
                    },
                ],
            )
        except self._openai.OpenAIError as e:
            log.warning("rewrite failed, using raw question", extra={"error": type(e).__name__})
            return question
        return (resp.choices[0].message.content or "").strip() or question

    def stream_answer(self, history: list[dict], user_content: str) -> Iterator[tuple[str, object]]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": user_content},
        ]
        finish, usage = None, None
        try:
            stream = self.client.chat.completions.create(
                model=self.cfg.xai_generation_model,
                messages=messages,
                max_completion_tokens=self.cfg.max_answer_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                if chunk.usage:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.delta and choice.delta.content:
                    yield "text", choice.delta.content
                if choice.finish_reason:
                    finish = choice.finish_reason
        except self._openai.OpenAIError as e:
            raise self._friendly(e) from e
        yield (
            "end",
            {
                "stop_reason": self._STOP.get(finish, finish),
                "usage": {
                    "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                },
            },
        )


def build_llm(cfg: Settings) -> LLM:
    provider = cfg.resolved_provider
    if provider == "anthropic":
        return AnthropicLLM(cfg)
    if provider == "xai":
        return XAILLM(cfg)
    return ExtractiveLLM()
