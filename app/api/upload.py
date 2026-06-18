"""
app/api/upload.py

POST /upload and GET /upload/status/{job_id} endpoints.

Upload accepts a document file and kicks off asynchronous ingestion in the
background. The endpoint returns immediately; all processing (loading,
chunking, embedding, ChromaDB storage, optional eval dataset generation)
happens via FastAPI BackgroundTasks.

The status endpoint lets callers poll for job completion without adding
an external database — state is kept in an in-memory IngestionTracker.
"""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status

from app.api.dependencies import get_ingestion_service, get_ingestion_tracker
from app.ingestion.ingestion_service import DocumentIngestionService
from app.models.responses import IngestionStatusResponse, UploadResponse
from app.services.ingestion_tracker import IngestionTracker
from app.utils.config import get_settings
from app.utils.helpers import get_file_extension

logger = logging.getLogger(__name__)

router = APIRouter()

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".xlsx"}


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
    description=(
        "Accepts a PDF, DOCX, TXT, or XLSX file. "
        "Returns immediately and processes the document in the background: "
        "extracts text, splits into chunks, generates embeddings, "
        "and stores everything in the ChromaDB vector database. "
        "Re-uploading the same filename replaces the existing vectors. "
        "Use GET /upload/status/{job_id} to poll for completion."
    ),
)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    ingestion_service: DocumentIngestionService = Depends(get_ingestion_service),
    tracker: IngestionTracker = Depends(get_ingestion_tracker),
) -> UploadResponse:
    """
    Accept a document for ingestion.

    The endpoint returns immediately with a job_id. The full ingestion
    pipeline runs as a FastAPI BackgroundTask to avoid blocking the
    client while embeddings are generated and ChromaDB is written.
    """
    # --- 1. Validate file type ---
    original_filename = file.filename or "unknown"
    ext = get_file_extension(original_filename)

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
            ),
        )

    logger.info("Received upload: '%s'", original_filename)

    # --- 2. Create job & persist file ---
    # The file is saved with a UUID prefix to prevent filename collisions
    # when multiple uploads arrive with the same name.  The original
    # filename is preserved in metadata / the tracker for display.
    job_id = tracker.create_job(original_filename)

    settings = get_settings()
    raw_dir = Path(settings.RAW_DOCUMENTS_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)

    save_path = raw_dir / f"{job_id}{ext}"

    try:
        content = await file.read()
        save_path.write_bytes(content)
    except OSError as exc:
        tracker.mark_failed(job_id, str(exc))
        logger.exception("Failed to save '%s'.", original_filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file.",
        ) from exc

    logger.info(
        "Saved '%s' to '%s' | job_id=%s",
        original_filename,
        save_path,
        job_id,
    )

    # --- 3. Schedule background ingestion ---
    background_tasks.add_task(
        ingestion_service.process_document,
        file_path=save_path,
        original_filename=original_filename,
        job_id=job_id,
    )

    return UploadResponse(
        job_id=job_id,
        filename=original_filename,
        status="processing",
    )


@router.get(
    "/upload/status/{job_id}",
    response_model=IngestionStatusResponse,
    summary="Poll ingestion job status",
    description=(
        "Returns the current status of a background ingestion job. "
        "Possible values: 'processing', 'completed', 'failed'."
    ),
)
async def get_ingestion_status(
    job_id: str,
    tracker: IngestionTracker = Depends(get_ingestion_tracker),
) -> IngestionStatusResponse:
    """Return the current status of an ingestion job."""
    job = tracker.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ingestion job '{job_id}' not found.",
        )

    return IngestionStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        error=job.get("error"),
    )