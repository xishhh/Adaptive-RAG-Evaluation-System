"""
app/main.py

FastAPI application entry point.

This module creates the FastAPI application instance, registers all API
routers, and configures application-level middleware and lifecycle events.

Run locally:
    uvicorn app.main:app --reload

Run in Docker:
    CMD set in Dockerfile.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import evaluate, health, query, session, upload
from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.

    Code before ``yield`` runs on startup.
    Code after ``yield`` runs on shutdown — all user data is cleared
    so every session starts and ends with a clean slate.
    """
    settings = get_settings()
    logger.info(
        f"Starting Adaptive RAG API | env={settings.APP_ENV} | "
        f"log_level={settings.LOG_LEVEL}"
    )
    yield
    logger.info("Adaptive RAG API shutting down — clearing session data …")

    from app.api.dependencies import (
        get_chroma_manager,
        get_ingestion_tracker,
    )
    from app.api.session import reset_all_data

    reset_all_data(
        chroma_manager=get_chroma_manager(),
        tracker=get_ingestion_tracker(),
    )


# ── Application Factory ────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Construct and configure the FastAPI application.

    Using a factory function (rather than a bare module-level instance)
    makes the app easier to test — tests can call create_app() to get a
    fresh instance with test settings.
    """
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

    # ── Middleware ──────────────────────────────────────────────────────────────
    # CORS — permissive in development; tighten allow_origins in production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ────────────────────────────────────────────────────────────────
    # All routers are registered at the root prefix.
    # Individual endpoints define their own paths (e.g. /health, /upload).
    app.include_router(health.router)
    app.include_router(upload.router)
    app.include_router(query.router)
    app.include_router(evaluate.router)
    app.include_router(session.router)

    logger.info("All routers registered")
    return app


# ── Application Instance ───────────────────────────────────────────────────────
# This is the object uvicorn targets: `uvicorn app.main:app`
app = create_app()
