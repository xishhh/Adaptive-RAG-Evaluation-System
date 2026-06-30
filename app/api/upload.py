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
    description="Accepts PDF, DOCX, TXT, or XLSX. Returns immediately and processes in the background.",
)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    ingestion_service: DocumentIngestionService = Depends(get_ingestion_service),
    tracker: IngestionTracker = Depends(get_ingestion_tracker),
) -> UploadResponse:
    original_filename = file.filename or "unknown"
    ext = get_file_extension(original_filename)

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    logger.info("Received upload: '%s'", original_filename)

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
    description="Returns current status of a background ingestion job.",
)
async def get_ingestion_status(
    job_id: str,
    tracker: IngestionTracker = Depends(get_ingestion_tracker),
) -> IngestionStatusResponse:
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
