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

from app.api import evaluate, health, query, upload
from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
from app.api.query import router as query_router


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.

    Code before `yield` runs on startup.
    Code after `yield` runs on shutdown.

    In later phases, this will initialise the ChromaDB client and warm up
    the embedding model. For Phase 1, it only logs the startup event.
    """
    settings = get_settings()
    logger.info(
        f"Starting Adaptive RAG API | env={settings.APP_ENV} | "
        f"log_level={settings.LOG_LEVEL}"
    )
    yield
    logger.info("Adaptive RAG API shutting down")


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

    logger.info("All routers registered")
    return app


# ── Application Instance ───────────────────────────────────────────────────────
# This is the object uvicorn targets: `uvicorn app.main:app`
app = create_app()
app.include_router(query_router)