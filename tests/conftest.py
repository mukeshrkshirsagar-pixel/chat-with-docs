import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.embeddings import HashEmbedder
from app.main import create_app


class ScriptedLLM:
    """Deterministic stand-in for the LLM: records what it was sent, replies with a canned answer."""

    def __init__(self, reply: str = "Remote work is allowed three days a week [1]."):
        self.reply = reply
        self.calls: list[dict] = []
        self.rewrites: list[str] = []

    def rewrite(self, history, question):
        self.rewrites.append(question)
        return "remote work days per week"

    def stream_answer(self, history, user_content):
        self.calls.append({"history": history, "user_content": user_content})
        for word in self.reply.split(" "):
            yield "text", word + " "
        yield "end", {"stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 5}}


@pytest.fixture
def cfg(tmp_path):
    # HashEmbedder is lexical, so its cosines are lower than a neural model's: relax the gate.
    return Settings(data_dir=tmp_path, embedder="hash", min_similarity=0.15, anthropic_api_key=None)


@pytest.fixture
def llm():
    return ScriptedLLM()


@pytest.fixture
def client(cfg, llm):
    return TestClient(create_app(cfg, embedder=HashEmbedder(), llm=llm))
