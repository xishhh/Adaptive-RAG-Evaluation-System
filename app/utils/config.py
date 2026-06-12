"""
app/utils/config.py

Central application configuration.

All settings are loaded from environment variables (or a .env file via
python-dotenv).  Every module that needs a setting imports `get_settings()`
and reads from the returned object — never reads os.environ directly.

Using @lru_cache ensures the .env file is parsed exactly once per process.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings resolved from environment variables.

    Pydantic-Settings automatically reads from:
      1. Actual environment variables
      2. A .env file in the working directory (if present)

    Field names map directly to environment variable names (case-insensitive).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── OpenAI ─────────────────────────────────────────────────────────────────
    openai_api_key: str
    openai_model: str = "meta-llama/llama-3.2-3b-instruct:free"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── ChromaDB ───────────────────────────────────────────────────────────────
    chroma_persist_directory: str = "./chroma_db"
    chroma_collection_name: str = "adaptive_rag"

    # ── Retrieval ──────────────────────────────────────────────────────────────
    retrieval_top_k: int = 5
    retrieval_confidence_threshold: float = 0.5

    # ── Chunking ───────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 50

    # ── Application ────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    app_env: str = "development"

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    """
    Return the cached application settings instance.

    Use this function everywhere rather than instantiating Settings directly.
    The @lru_cache decorator guarantees the environment is parsed only once.
    """
    return Settings()