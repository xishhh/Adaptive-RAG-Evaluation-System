"""
Environment-based configuration for the Adaptive RAG system.

All configurable values are read from environment variables with
sensible defaults. Import the `settings` singleton; never instantiate
Settings directly in application code.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration object.

    Values are loaded from environment variables (case-insensitive).
    An .env file in the project root is also supported via pydantic-settings.
    """

    

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        
        
    )

    app_env: str = "development"
    log_level: str = "INFO"

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"
    

    # ------------------------------------------------------------------ #
    # OpenAI                                                               #
    # ------------------------------------------------------------------ #
    openai_api_key: str = ""

    # ------------------------------------------------------------------ #
    # Chunking                                                             #
    # ------------------------------------------------------------------ #
    chunk_size: int = 1000
    """Target character count per chunk."""

    chunk_overlap: int = 200
    """Character overlap between consecutive chunks to preserve context."""

    # ------------------------------------------------------------------ #
    # ChromaDB  (stubbed — used from Phase 3 onward)                      #
    # ------------------------------------------------------------------ #
    chroma_persist_directory: str = "./data/chroma_db"
    chroma_collection_name: str = "adaptive_rag"

    # ------------------------------------------------------------------ #
    # Paths                                                                #
    # ------------------------------------------------------------------ #
    raw_documents_dir: str = "./data/raw_documents"
    processed_documents_dir: str = "./data/processed_documents"
    evaluation_results_dir: str = "./evaluation_results"

    # ------------------------------------------------------------------ #
    # Retrieval  (stubbed — used from Phase 4 onward)                     #
    # ------------------------------------------------------------------ #
    retrieval_top_k: int = 5
    retrieval_confidence_threshold: float = 0.75


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.

    Using lru_cache means the .env file is read once per process,
    not on every import.
    """
    return Settings()


# Module-level singleton — import this everywhere.
settings: Settings = get_settings()