"""
app/utils/config.py

Centralised application configuration using Pydantic Settings.

All environment variables are read from here. No other module should
read os.environ directly — import get_settings() instead.

Phase 3 additions:
- CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL.

Phase 5 additions:
- CONFIDENCE_THRESHOLD: minimum average relevance score for retrieved
  chunks to be considered GOOD_CONTEXT. Below this, the adaptive
  pipeline triggers query rewriting.
- ADAPTIVE_MAX_REWRITES: safety cap on how many rewrite+re-retrieval
  cycles the adaptive pipeline will attempt per query.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All fields have sensible defaults so the app starts without a .env
    file in development, but production deployments must supply real values.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # OpenAI / OpenRouter
    # ------------------------------------------------------------------
    OPENAI_API_KEY: str = ""
    OPENAI_API_BASE: str = "https://openrouter.ai/api/v1"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_MODELS: str = ""
    """
    Comma-separated list of models for fallback.
    The first entry is the primary model; subsequent entries are
    fallbacks tried in order on failure.
    If left empty, LLM_MODEL is used as a single model (no fallback).
    """
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # ------------------------------------------------------------------
    # ChromaDB  (Phase 3)
    # ------------------------------------------------------------------
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "adaptive_rag"

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    EMBED_BATCH_SIZE: int = 50
    RAW_DOCUMENTS_DIR: str = "./data/raw_documents"
    PROCESSED_DOCUMENTS_DIR: str = "./data/processed_documents"

    # ------------------------------------------------------------------
    # Adaptive Retrieval  (Phase 5)
    # ------------------------------------------------------------------
    CONFIDENCE_THRESHOLD: float = 0.45
    """
    Minimum average relevance score (0.0–1.0) across top-K retrieved chunks
    to be judged GOOD_CONTEXT. Scores below this trigger query rewriting.

    Tuning guidance:
      - 0.35–0.45 is a practical starting range for cosine similarity
        with OpenAI text-embedding-3-small.
      - Lower values mean the pipeline rewrites more aggressively.
      - Higher values mean it rewrites on nearly every query.
    """

    ADAPTIVE_MAX_REWRITES: int = 1
    """
    Maximum number of rewrite+re-retrieval cycles per query.
    Kept at 1 for Phase 5 — the spec describes a single rewrite loop.
    Raise in future phases if multi-hop rewriting is introduced.
    """

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    APP_TITLE: str = "Adaptive RAG API"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = False

    @property
    def is_production(self) -> bool:
        return self.APP_ENV.lower() == "production"

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    EVALUATION_RESULTS_DIR: str = "./evaluation_results"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    lru_cache ensures a single Settings object is created per process,
    avoiding repeated .env file reads on every import.
    """
    return Settings()