from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import evaluate, health, query, session, upload
from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    logger.info(
        f"Starting Adaptive RAG API | env={settings.APP_ENV} | "
        f"log_level={settings.LOG_LEVEL}"
    )
    yield
    logger.info("Adaptive RAG API shutting down — clearing session data ...")

    from app.api.dependencies import get_chroma_manager, get_ingestion_tracker
    from app.api.session import reset_all_data

    reset_all_data(
        chroma_manager=get_chroma_manager(),
        tracker=get_ingestion_tracker(),
    )


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Adaptive RAG API",
        description=(
            "A production-oriented Adaptive Retrieval-Augmented Generation system "
            "that ingests documents, retrieves context adaptively, generates grounded "
            "answers with citations, and evaluates answer quality automatically."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(upload.router)
    app.include_router(query.router)
    app.include_router(evaluate.router)
    app.include_router(session.router)

    logger.info("All routers registered")
    return app


app = create_app()
