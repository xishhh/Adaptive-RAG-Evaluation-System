"""
app/ingestion/ingestion_service.py

Background document ingestion service.

Encapsulates the full ingestion pipeline:
    load -> chunk -> batch embed -> batch ChromaDB insert -> optional eval generation

Chunks are processed in configurable batches (EMBED_BATCH_SIZE) so that large
documents don't create a single oversized embedding API call.  Each batch is
embedded and stored independently, providing partial-failure recovery: if one
batch fails, already-stored batches remain in ChromaDB.
"""

import logging
from pathlib import Path

from app.evaluators.eval_dataset_generator import EvalDatasetGenerator
from app.ingestion.chunker import DocumentChunker
from app.ingestion.loaders import DocumentLoader
from app.services.ingestion_tracker import IngestionTracker
from app.utils.config import get_settings
from app.vectorstore.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)


class DocumentIngestionService:
    """
    Orchestrates the synchronous document ingestion pipeline.

    Each call to process_document() runs:
      1. DocumentLoader.load()                          — extract text
      2. DocumentChunker.chunk()                        — split into overlapping chunks
      3. ChromaManager.delete_by_document_name()        — remove stale vectors for re-uploads
      4. _store_in_batches()                            — batch embed + batch ChromaDB insert
      5. EvalDatasetGenerator.generate_from_chunks()    — optional Q&A dataset

    Errors are always logged but never raised, making this safe to
    call from a FastAPI BackgroundTask without crashing the worker.
    """

    def __init__(
        self,
        chroma_manager: ChromaManager,
        eval_generator: EvalDatasetGenerator,
        tracker: IngestionTracker,
    ) -> None:
        self._chroma_manager = chroma_manager
        self._eval_generator = eval_generator
        self._tracker = tracker
        self._batch_size = get_settings().EMBED_BATCH_SIZE

    def process_document(
        self,
        file_path: Path,
        original_filename: str,
        job_id: str,
    ) -> None:
        """
        Run the full ingestion pipeline for a single document.

        Args:
            file_path:        Absolute path to the saved file on disk.
            original_filename: User-facing filename (e.g. "contract_a.pdf").
            job_id:           Ingestion job identifier for status tracking.
        """
        logger.info(
            "Background ingestion started | job_id=%s | file='%s' | path='%s'",
            job_id,
            original_filename,
            file_path,
        )

        chunks = None
        try:
            # ------------------------------------------------------------------
            # Step 1 & 2 — Load -> Chunk
            # ------------------------------------------------------------------
            loader = DocumentLoader()
            raw_doc = loader.load(file_path, original_filename=original_filename)

            chunker = DocumentChunker()
            chunks = chunker.chunk(raw_doc)

            if not chunks:
                logger.error(
                    "No chunks produced for '%s' — aborting ingestion.",
                    original_filename,
                )
                self._tracker.mark_failed(job_id, "No text content could be extracted.")
                return

            logger.info(
                "Loaded and chunked '%s' — %d chunks.",
                original_filename,
                len(chunks),
            )

            # ------------------------------------------------------------------
            # Step 3 — Delete existing vectors for this document (idempotent re-upload)
            # ------------------------------------------------------------------
            try:
                deleted_count = self._chroma_manager.delete_by_document_name(
                    original_filename
                )
                if deleted_count > 0:
                    logger.info(
                        "Replaced %d existing chunks for '%s' (re-upload).",
                        deleted_count,
                        original_filename,
                    )
            except Exception as exc:
                logger.warning(
                    "Could not delete existing chunks for '%s': %s",
                    original_filename,
                    exc,
                )

            # ------------------------------------------------------------------
            # Step 4 — Batch embed + batch ChromaDB insert
            # ------------------------------------------------------------------
            total_stored = self._store_in_batches(chunks, original_filename)

            if total_stored == 0:
                logger.error(
                    "No chunks were stored for '%s' — all batches failed.",
                    original_filename,
                )
                self._tracker.mark_failed(
                    job_id, "All embedding batches failed — see logs."
                )
                return

            # ------------------------------------------------------------------
            # Step 5 — Generate evaluation dataset (non-fatal)
            # ------------------------------------------------------------------
            try:
                samples_written = self._eval_generator.generate_from_chunks(
                    chunks=chunks,
                    document_name=original_filename,
                )
                if samples_written > 0:
                    logger.info(
                        "Eval dataset generated | file='%s' | samples=%d",
                        original_filename,
                        samples_written,
                    )
            except Exception as exc:
                logger.error(
                    "Eval dataset generation failed for '%s' (ingestion OK): %s",
                    original_filename,
                    exc,
                )

            self._tracker.mark_completed(job_id)
            logger.info(
                "Background ingestion complete | job_id=%s | file='%s' | total_stored=%d",
                job_id,
                original_filename,
                total_stored,
            )

        except Exception as exc:
            self._tracker.mark_failed(job_id, str(exc))
            logger.exception(
                "Background ingestion failed | job_id=%s | file='%s': %s",
                job_id,
                original_filename,
                exc,
            )
        finally:
            self._cleanup(file_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _store_in_batches(
        self,
        chunks: list,
        original_filename: str,
    ) -> int:
        """
        Embed and store chunks in sub-batches of EMBED_BATCH_SIZE.

        Each batch is a separate call to ChromaManager.add_chunks(), which
        internally does a single embed_documents() call for the batch followed
        by a single ChromaDB insert.  This keeps individual API calls small
        and provides partial-failure recovery.

        Args:
            chunks:             All Chunk objects for the document.
            original_filename:  User-facing filename (for logging).

        Returns:
            Total number of chunks successfully stored across all batches.
        """
        settings = get_settings()
        batch_size = settings.EMBED_BATCH_SIZE

        total_stored = 0
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        logger.info(
            "Storing %d chunks in batches of %d (%d batch(es)) | file='%s'",
            len(chunks),
            batch_size,
            total_batches,
            original_filename,
        )

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(chunks))
            batch = chunks[start:end]

            # Filter out empty chunks — they waste API calls and add noise.
            valid = [c for c in batch if c.chunk_text.strip()]
            skipped_in_batch = len(batch) - len(valid)
            if skipped_in_batch:
                logger.warning(
                    "Skipped %d empty chunk(s) in batch %d/%d | file='%s'",
                    skipped_in_batch,
                    batch_idx + 1,
                    total_batches,
                    original_filename,
                )

            if not valid:
                logger.warning(
                    "Batch %d/%d is empty after filtering — skipping | file='%s'",
                    batch_idx + 1,
                    total_batches,
                    original_filename,
                )
                continue

            try:
                batch_dumps = [c.model_dump() for c in valid]
                stored = self._chroma_manager.add_chunks(batch_dumps)
                total_stored += stored
                logger.info(
                    "Batch %d/%d stored | file='%s' | batch_chunks=%d | cumulative=%d",
                    batch_idx + 1,
                    total_batches,
                    original_filename,
                    stored,
                    total_stored,
                )
            except Exception as exc:
                logger.exception(
                    "Batch %d/%d failed for '%s' (previous batches preserved): %s",
                    batch_idx + 1,
                    total_batches,
                    original_filename,
                    exc,
                )
                # Continue with remaining batches — partial storage is better
                # than losing everything.  The caller logs total_stored so
                # operators can see how much was actually persisted.

        return total_stored

    @staticmethod
    def _cleanup(file_path: Path) -> None:
        """Remove the persisted file after processing."""
        try:
            file_path.unlink(missing_ok=True)
            logger.debug("Cleaned up '%s'.", file_path)
        except Exception as exc:
            logger.warning("Failed to clean up '%s': %s", file_path, exc)
