"""
app/api/evaluate.py

Evaluation endpoints.

POST /evaluate runs the RAGAS evaluation harness against a dataset.
GET /metrics returns historical evaluation metrics.

Status: STUB — returns HTTP 501 until Phase 6 (Evaluation Harness) is complete.
"""

from fastapi import APIRouter, HTTPException, status

from app.models.requests import EvaluateRequest
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/evaluate",
    summary="Run Evaluation",
    description="Run the RAGAS evaluation harness against a QA dataset.",
    tags=["Evaluation"],
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
)
async def run_evaluation(request: EvaluateRequest) -> dict:
    """
    Evaluate system quality using RAGAS metrics.

    Implemented in Phase 6.
    """
    logger.warning("POST /evaluate called but not yet implemented (Phase 6)")
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Evaluation is not yet implemented. Coming in Phase 6.",
    )


@router.get(
    "/metrics",
    summary="Get Evaluation Metrics",
    description="Retrieve historical evaluation metrics and the latest run summary.",
    tags=["Evaluation"],
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
)
async def get_metrics() -> dict:
    """
    Return historical evaluation metrics.

    Implemented in Phase 6.
    """
    logger.warning("GET /metrics called but not yet implemented (Phase 6)")
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Metrics retrieval is not yet implemented. Coming in Phase 6.",
    )