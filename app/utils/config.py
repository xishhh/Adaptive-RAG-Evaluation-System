"""
app/utils/config.py

Centralised application configuration using Pydantic Settings.

All environment variables are read from here. No other module should
read os.environ directly — import get_settings() instead.

Changes in Phase 3:
- Added CHROMA_PERSIST_DIR: filesystem path for ChromaDB persistence.
- Added CHROMA_COLLECTION_NAME: logical name of the ChromaDB collection.
- Added EMBEDDING_MODEL: configurable OpenAI embedding model name.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All fields have sensible defaults so the app starts without a .env file
    in development, but production deployments must supply real values.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------
    OPENAI_API_KEY: str = ""
    OPENAI_API_BASE: str = "https://openrouter.ai/api/v1"  # Add this line
    LLM_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # <--- Restore this line


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
    RAW_DOCUMENTS_DIR: str = "./data/raw_documents"
    PROCESSED_DOCUMENTS_DIR: str = "./data/processed_documents"

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