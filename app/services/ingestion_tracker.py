"""
app/services/ingestion_tracker.py

Lightweight in-memory tracker for document ingestion status.

Provides a simple dict-backed store so the API can report whether a
background ingestion job is still processing, completed, or failed.

No external dependencies — pure Python.  Entries are lost on process
restart, which is acceptable for the MVP phase.
"""

from __future__ import annotations

import datetime
import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class IngestionTracker:
    """
    Tracks background ingestion jobs in memory.

    Each job is identified by a UUID and holds:
      - job_id       (str)
      - filename     (str)
      - status       ("processing" | "completed" | "failed")
      - error        (str | None)
      - created_at   (ISO-8601 str)

    Thread-safety note: FastAPI BackgroundTasks run in the same event loop
    on the same thread as the server, so plain dict access is safe for
    this MVP.  A future version with Celery workers would need a lock.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}

    def create_job(self, filename: str) -> str:
        """Register a new ingestion job and return its job_id."""
        job_id = uuid.uuid4().hex
        self._jobs[job_id] = {
            "job_id": job_id,
            "filename": filename,
            "status": "processing",
            "error": None,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        logger.info("Ingestion job created | job_id=%s | file='%s'", job_id, filename)
        return job_id

    def mark_completed(self, job_id: str) -> None:
        """Mark a job as completed."""
        job = self._jobs.get(job_id)
        if job is not None:
            job["status"] = "completed"
            logger.info("Ingestion job completed | job_id=%s", job_id)

    def mark_failed(self, job_id: str, error: str) -> None:
        """Mark a job as failed with an error message."""
        job = self._jobs.get(job_id)
        if job is not None:
            job["status"] = "failed"
            job["error"] = error
            logger.info("Ingestion job failed | job_id=%s | error=%s", job_id, error)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Return the job dict, or None if the job_id is unknown."""
        return self._jobs.get(job_id)

    def clear(self) -> None:
        """Remove all tracked jobs. Used during session reset."""
        self._jobs.clear()
        logger.info("Ingestion tracker cleared.")
