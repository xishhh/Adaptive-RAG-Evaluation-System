"""
app/api/upload.py

Document upload endpoint.

POST /upload accepts a multipart file upload and triggers the ingestion
pipeline (loading → chunking → embedding → vector store).

Status: STUB — returns HTTP 501 until Phase 2 (Document Ingestion) is complete.
"""

from fastapi import APIRouter, HTTPException, UploadFile, status

from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/upload",
    summary="Upload Document",
    description="Upload a document (PDF, DOCX, TXT, XLSX) for ingestion into the knowledge base.",
    tags=["Ingestion"],
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
)
async def upload_document(file: UploadFile) -> dict:
    """
    Ingest a document into the knowledge base.

    Implemented in Phase 2.
    """
    logger.warning("POST /upload called but not yet implemented (Phase 2)")
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Document ingestion is not yet implemented. Coming in Phase 2.",
    )