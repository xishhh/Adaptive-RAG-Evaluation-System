"""
app/api/session.py

DELETE /session endpoint — clears all user data: ChromaDB vectors,
uploaded files, evaluation datasets, and ingestion jobs.

Useful for:
  - Resetting between demo sessions
  - Cleaning up after a batch of testing
  - Triggered automatically on server shutdown via lifespan.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel

from app.api.dependencies import (
    get_chroma_manager,
    get_ingestion_tracker,
)
from app.services.ingestion_tracker import IngestionTracker
from app.utils.config import get_settings
from app.vectorstore.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)

router = APIRouter()


class SessionResetResponse(BaseModel):
    detail: str

# ---------------------------------------------------------------------------
# Public helpers (used by lifespan shutdown as well)
# ---------------------------------------------------------------------------

def reset_all_data(
    chroma_manager: ChromaManager,
    tracker: IngestionTracker,
) -> None:
    """Reset ChromaDB, delete uploaded files & eval datasets, clear tracker."""
    settings = get_settings()

    # 1. Reset ChromaDB collection
    chroma_manager.reset_collection()

    # 2. Delete uploaded raw documents
    raw_dir = Path(settings.RAW_DOCUMENTS_DIR)
    if raw_dir.exists():
        _rmtree(raw_dir)
        logger.info("Deleted raw documents directory: %s", raw_dir)

    # 3. Delete evaluation datasets
    eval_dir = Path("data/evaluation_dataset")
    if eval_dir.exists():
        _rmtree(eval_dir)
        logger.info("Deleted evaluation dataset directory: %s", eval_dir)

    # 4. Clear in-memory ingestion tracker
    tracker.clear()

    logger.info("Session reset complete — all user data cleared.")


def _rmtree(path: Path) -> None:
    """Remove a directory tree, recreating the empty directory afterwards."""
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.delete(
    "/session",
    status_code=status.HTTP_200_OK,
    summary="Reset all user data for a fresh session",
    description=(
        "Deletes all stored data: ChromaDB vector collection, uploaded "
        "raw documents, evaluation datasets, and ingestion job history. "
        "The server remains running and ready for a new session."
    ),
)
async def delete_session(
    chroma_manager: ChromaManager = Depends(get_chroma_manager),
    tracker: IngestionTracker = Depends(get_ingestion_tracker),
) -> SessionResetResponse:
    """Clear all user data and return a confirmation."""
    reset_all_data(chroma_manager, tracker)
    return SessionResetResponse(detail="Session reset complete — all data cleared.")
