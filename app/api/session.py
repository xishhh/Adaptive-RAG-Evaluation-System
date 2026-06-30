from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel

from app.api.dependencies import get_chroma_manager, get_ingestion_tracker
from app.services.ingestion_tracker import IngestionTracker
from app.utils.config import get_settings
from app.vectorstore.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)

router = APIRouter()


class SessionResetResponse(BaseModel):
    detail: str


def reset_all_data(
    chroma_manager: ChromaManager,
    tracker: IngestionTracker,
) -> None:
    settings = get_settings()

    chroma_manager.reset_collection()

    raw_dir = Path(settings.RAW_DOCUMENTS_DIR)
    if raw_dir.exists():
        _rmtree(raw_dir)
        logger.info("Deleted raw documents directory: %s", raw_dir)

    eval_dir = Path("data/evaluation_dataset")
    if eval_dir.exists():
        _rmtree(eval_dir)
        logger.info("Deleted evaluation dataset directory: %s", eval_dir)

    tracker.clear()
    logger.info("Session reset complete — all user data cleared.")


def _rmtree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


@router.delete(
    "/session",
    status_code=status.HTTP_200_OK,
    summary="Reset all user data for a fresh session",
    description="Deletes all stored data: ChromaDB vectors, uploaded documents, evaluation datasets, and ingestion job history.",
)
async def delete_session(
    chroma_manager: ChromaManager = Depends(get_chroma_manager),
    tracker: IngestionTracker = Depends(get_ingestion_tracker),
) -> SessionResetResponse:
    reset_all_data(chroma_manager, tracker)
    return SessionResetResponse(detail="Session reset complete — all data cleared.")
