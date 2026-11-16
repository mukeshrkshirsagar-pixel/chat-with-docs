from types import SimpleNamespace as NS

import httpx
import openai
import pytest

from app.config import Settings
from app.llm import XAILLM, AnthropicLLM, ExtractiveLLM, LLMError, build_llm


def chunk(text=None, finish=None, usage=None, choices=True):
    delta = NS(content=text)
    return NS(choices=[NS(delta=delta, finish_reason=finish)] if choices else [], usage=usage)


class FakeCompletions:
    def __init__(self, chunks=None, error=None, text="standalone query"):
        self.chunks, self.error, self.text, self.kwargs = chunks or [], error, text, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        if kwargs.get("stream"):
            return iter(self.chunks)
        return NS(choices=[NS(message=NS(content=self.text))])


def make(completions, **cfg):
    client = NS(chat=NS(completions=completions))
    return XAILLM(
        Settings(_env_file=None, xai_api_key="k", **cfg), client=client
    )  # ignore the developer's .env


def test_provider_resolution():
    assert isinstance(build_llm(Settings(_env_file=None)), ExtractiveLLM)
    assert isinstance(build_llm(Settings(_env_file=None, xai_api_key="x")), XAILLM)
    assert isinstance(build_llm(Settings(_env_file=None, anthropic_api_key="a")), AnthropicLLM)
    both = Settings(_env_file=None, anthropic_api_key="a", xai_api_key="x")
    assert both.resolved_provider == "anthropic"  # auto prefers Anthropic...
    assert (
        Settings(_env_file=None, anthropic_api_key="a", xai_api_key="x", llm_provider="xai").resolved_provider
        == "xai"
    )
    assert (
        Settings(_env_file=None, llm_provider="xai", anthropic_api_key="a").resolved_provider is None
    )  # ...but explicit wins


def test_xai_streams_text_and_maps_usage_and_stop_reason():
    comp = FakeCompletions(
        [
            chunk("Three "),
            chunk("days [1]."),
            chunk(finish="stop"),
            chunk(choices=False, usage=NS(prompt_tokens=120, completion_tokens=7)),
        ]
    )
    events = list(make(comp).stream_answer([{"role": "user", "content": "hi"}], "SOURCES + Q"))
    assert "".join(v for k, v in events if k == "text") == "Three days [1]."
    assert events[-1] == (
        "end",
        {"stop_reason": "end_turn", "usage": {"input_tokens": 120, "output_tokens": 7}},
    )
    sent = comp.kwargs
    assert sent["messages"][0]["role"] == "system" and sent["messages"][-1]["content"] == "SOURCES + Q"
    assert sent["stream"] is True and sent["model"] == "grok-4"


def test_xai_length_and_content_filter_map_to_pipeline_flags():
    for finish, expected in (("length", "max_tokens"), ("content_filter", "refusal")):
        events = list(make(FakeCompletions([chunk("x"), chunk(finish=finish)])).stream_answer([], "q"))
        assert events[-1][1]["stop_reason"] == expected


def test_xai_errors_become_friendly_llm_errors():
    req = httpx.Request("POST", "https://api.x.ai/v1/chat/completions")
    err = openai.AuthenticationError("no", response=httpx.Response(401, request=req), body=None)
    with pytest.raises(LLMError, match="XAI_API_KEY"):
        list(make(FakeCompletions(error=err)).stream_answer([], "q"))


def test_xai_rewrite_returns_query_and_degrades_gracefully():
    llm = make(FakeCompletions(text="  remote work abroad  "))
    assert llm.rewrite([{"role": "user", "content": "a"}], "and abroad?") == "remote work abroad"
    req = httpx.Request("POST", "https://api.x.ai/v1/chat/completions")
    failing = make(FakeCompletions(error=openai.APIConnectionError(request=req)))
    assert failing.rewrite([], "and abroad?") == "and abroad?"
