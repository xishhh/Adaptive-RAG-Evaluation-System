"""
app/api/health.py

Health check endpoint.

GET /health is the only endpoint fully implemented in Phase 1.
It is the Phase 1 completion criterion: if this responds 200, the app boots.

In later phases, additional checks (ChromaDB connectivity, OpenAI key
validity) will be added to this endpoint.
"""

from datetime import datetime, timezone

from fastapi import APIRouter

from app.models.responses import HealthResponse
from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Application version — increment this as phases complete
APP_VERSION = "0.1.0"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns the current health status of the application.",
    tags=["Health"],
)
async def health_check() -> HealthResponse:
    """
    Lightweight health check used by Docker HEALTHCHECK, load balancers,
    and uptime monitors.

    Returns HTTP 200 with a JSON body when the application is healthy.
    No external dependencies (database, OpenAI) are checked here in Phase 1.
    """
    settings = get_settings()
    logger.debug("Health check requested")

    return HealthResponse(
        status="ok",
        version=APP_VERSION,
        environment=settings.app_env,
        timestamp=datetime.now(timezone.utc),
    )