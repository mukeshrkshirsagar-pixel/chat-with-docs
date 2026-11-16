"""All tunables in one place, overridable via env vars / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM ---
    llm_provider: str = "auto"  # auto | anthropic | xai   (auto: whichever API key is set)
    anthropic_api_key: str | None = None
    generation_model: str = "claude-opus-5-5"
    generation_effort: str = "low"  # grounded QA doesn't need deep reasoning
    rewrite_model: str = "claude-haiku-5-5"  # cheap + fast; only used on follow-ups
    xai_api_key: str | None = None
    xai_base_url: str = "https://api.x.ai/v1"
    xai_generation_model: str = "grok-4"  # verify against your account's model list
    xai_rewrite_model: str = "grok-3-mini"
    max_answer_tokens: int = 2048

    @property
    def resolved_provider(self) -> str | None:
        """Which LLM backend to use, or None for retrieval-only mode."""
        if self.llm_provider == "anthropic":
            return "anthropic" if self.anthropic_api_key else None
        if self.llm_provider == "xai":
            return "xai" if self.xai_api_key else None
        if self.anthropic_api_key:
            return "anthropic"
        return "xai" if self.xai_api_key else None

    # --- Embeddings ---
    embedder: str = "fastembed"  # "fastembed" (real) | "hash" (offline, tests)
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- Chunking ---
    chunk_size: int = 900  # characters (~200 tokens)
    chunk_overlap: int = 150

    # --- Retrieval ---
    candidates: int = 20  # per retriever (dense / BM25) before fusion
    top_k: int = 6  # chunks sent to the LLM
    min_similarity: float = 0.50  # cosine gate; calibrated with `python -m eval.run_eval --sweep`
    max_context_chars: int = 12000

    # --- Conversation ---
    history_turns: int = 6
    history_answer_chars: int = 1200
    max_question_chars: int = 2000

    # --- Limits / storage ---
    max_upload_mb: int = 20
    data_dir: Path = Path("data")
    sample_dir: Path = Path("sample_docs")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "chat_with_docs.sqlite3"


@lru_cache
def get_settings() -> Settings:
    return Settings()
