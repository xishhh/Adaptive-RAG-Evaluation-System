"""
app/api/health.py

GET /health endpoint.

Phase 3 change:
- Now reports ChromaDB collection statistics so operators can verify
  the vector store is live and contains the expected number of chunks.
"""

import logging

from fastapi import APIRouter

from app.models.responses import CollectionStatsResponse, HealthResponse
from app.utils.config import get_settings
from app.vectorstore.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)

router = APIRouter()
settings = get_settings()

_chroma_manager = ChromaManager()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns service status and ChromaDB collection statistics.",
)
def health_check() -> HealthResponse:
    """
    Verify the service is running and report vector store state.
    """
    try:
        stats = _chroma_manager.collection_stats()
        vector_store = CollectionStatsResponse(**stats)
    except Exception:
        logger.exception("ChromaDB health check failed.")
        vector_store = None

    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        vector_store=vector_store,
    )