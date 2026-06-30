from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    OPENAI_API_KEY: str = ""
    OPENAI_API_BASE: str = "https://openrouter.ai/api/v1"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_MODELS: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "adaptive_rag"

    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    EMBED_BATCH_SIZE: int = 50
    RAW_DOCUMENTS_DIR: str = "./data/raw_documents"

    CONFIDENCE_THRESHOLD: float = 0.45
    ADAPTIVE_MAX_REWRITES: int = 1

    APP_VERSION: str = "0.1.0"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    GENERATE_EVAL_DATASET: bool = False
    EVALUATION_RESULTS_DIR: str = "./evaluation_results"

    @property
    def is_production(self) -> bool:
        return self.APP_ENV.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
