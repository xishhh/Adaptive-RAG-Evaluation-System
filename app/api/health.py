import logging

from fastapi import APIRouter, Depends

from app.api.dependencies import get_chroma_manager
from app.models.responses import CollectionStatsResponse, HealthResponse
from app.utils.config import get_settings
from app.vectorstore.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)

router = APIRouter()
settings = get_settings()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns service status and ChromaDB collection statistics.",
)
def health_check(
    chroma_manager: ChromaManager = Depends(get_chroma_manager),
) -> HealthResponse:
    try:
        stats = chroma_manager.collection_stats()
        vector_store = CollectionStatsResponse(**stats)
    except Exception:
        logger.exception("ChromaDB health check failed.")
        vector_store = None

    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        vector_store=vector_store,
    )
