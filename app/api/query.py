"""
app/api/query.py

Question answering endpoint.

POST /query accepts a natural language question, runs it through the
adaptive retrieval pipeline, and returns a grounded answer with citations.

Status: STUB — returns HTTP 501 until Phase 4 (Basic RAG) is complete.
"""

from fastapi import APIRouter, HTTPException, status

from app.models.requests import QueryRequest
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/query",
    summary="Query Knowledge Base",
    description="Submit a natural language question and receive a grounded answer with citations.",
    tags=["Query"],
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
)
async def query_knowledge_base(request: QueryRequest) -> dict:
    """
    Answer a question using the RAG pipeline.

    Implemented in Phase 4.
    """
    logger.warning("POST /query called but not yet implemented (Phase 4)")
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Query answering is not yet implemented. Coming in Phase 4.",
    )