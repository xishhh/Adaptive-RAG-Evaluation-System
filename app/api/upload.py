"""
app/api/upload.py

POST /upload endpoint.

Accepts a document file, runs it through the ingestion pipeline
(load → chunk), then stores the resulting chunks in ChromaDB via ChromaManager.

Phase 3 change:
- ChromaManager.add_chunks() replaces the Phase 2 stub that returned chunks
  without storing them anywhere.
"""

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, status

from app.ingestion.chunker import DocumentChunker
from app.ingestion.loaders import DocumentLoader
from app.models.responses import UploadResponse
from app.vectorstore.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Single shared ChromaManager instance per process.
# In Phase 7 (API Completion) this will move to a dependency injection pattern.
_chroma_manager = ChromaManager()

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".xlsx"}


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
    description=(
        "Accepts a PDF, DOCX, TXT, or XLSX file. "
        "Extracts text, splits into chunks, generates embeddings, "
        "and stores everything in the ChromaDB vector database."
    ),
)
async def upload_document(file: UploadFile) -> UploadResponse:
    """
    Ingest a document into the vector knowledge base.

    Steps:
    1. Validate file type.
    2. Save to a temp file (loaders need a filesystem path).
    3. Load and extract text via DocumentLoader.
    4. Chunk text via DocumentChunker.
    5. Store chunks + embeddings in ChromaDB.
    6. Return ingestion summary.
    """
    # --- 1. Validate ---
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
            ),
        )

    logger.info("Received upload: '%s'", file.filename)

    # --- 2. Write to temp file ---
    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)
    except OSError as exc:
        logger.exception("Failed to write temp file for '%s'.", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file.",
        ) from exc

    # --- 3 & 4. Load → Chunk ---
    try:
        loader = DocumentLoader()
        raw_doc = loader.load(tmp_path)

        chunker = DocumentChunker()
        chunks = chunker.chunk(raw_doc)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Ingestion pipeline failed for '%s'.", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document processing failed.",
        ) from exc
    finally:
        # Clean up temp file regardless of outcome.
        tmp_path.unlink(missing_ok=True)

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text content could be extracted from the document.",
        )

    # --- 5. Store in ChromaDB ---
    try:
        stored_count = _chroma_manager.add_chunks(
    [chunk.model_dump() for chunk in chunks]
)
    except Exception as exc:
        logger.exception("ChromaDB storage failed for '%s'.", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store document in vector database.",
        ) from exc

    logger.info(
        "Ingestion complete | file='%s' | chunks=%d",
        file.filename,
        stored_count,
    )

    return UploadResponse(
        filename=file.filename or "unknown",
        chunks_stored=stored_count,
        message=f"Successfully ingested '{file.filename}' into the knowledge base.",
    )