"""
Document chunker for the Adaptive RAG ingestion pipeline.

Responsibilities:
  - Accept a RawDocument (produced by loaders.py).
  - Split its full_text into overlapping chunks using
    LangChain's RecursiveCharacterTextSplitter.
  - Annotate each chunk with complete metadata:
      - document_name
      - chunk_id  (UUID, assigned here)
      - chunk_index
      - page_number (where determinable)
      - all metadata inherited from the RawDocument

Why RecursiveCharacterTextSplitter?
  It splits on semantic boundaries (paragraphs → sentences → words)
  before falling back to character splits, which preserves coherence
  better than naive fixed-size splitting. This is the production
  standard for text-based RAG pipelines.

This module does NOT generate embeddings or write to ChromaDB.
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.documents import Chunk, RawDocument
from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentChunker:
    """
    Splits a RawDocument into a list of Chunk objects.

    Args:
        chunk_size: Target character count per chunk. Defaults to the
                    value in settings (env: CHUNK_SIZE).
        chunk_overlap: Character overlap between chunks. Defaults to the
                       value in settings (env: CHUNK_OVERLAP).

    Usage:
        chunker = DocumentChunker()
        chunks = chunker.chunk(raw_document)
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        settings = get_settings()
        self._chunk_size = chunk_size or settings.CHUNK_SIZE
        self._chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            # Hierarchy of separators tried in order.
            # The splitter falls back to the next separator only when
            # the current one produces chunks that are still too large.
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

        logger.debug(
            "DocumentChunker initialised — chunk_size=%d, chunk_overlap=%d",
            self._chunk_size,
            self._chunk_overlap,
        )

    def chunk(self, raw_doc: RawDocument) -> list[Chunk]:
        """
        Split a RawDocument into a list of annotated Chunks.

        Args:
            raw_doc: A loaded and text-extracted document.

        Returns:
            List of Chunk objects, each with complete metadata.

        Raises:
            ValueError: If raw_doc.full_text is empty.
        """
        if not raw_doc.full_text.strip():
            raise ValueError(
                f"Cannot chunk '{raw_doc.document_name}': full_text is empty."
            )

        logger.info(
            "Chunking '%s' (%d chars, type=%s) …",
            raw_doc.document_name,
            len(raw_doc.full_text),
            raw_doc.file_type,
        )

        # LangChain splitter returns plain strings.
        text_chunks: list[str] = self._splitter.split_text(raw_doc.full_text)

        if not text_chunks:
            raise ValueError(
                f"Chunking produced no output for '{raw_doc.document_name}'."
            )

        chunks: list[Chunk] = []
        for index, chunk_text in enumerate(text_chunks):
            page_number = self._resolve_page_number(
                chunk_text=chunk_text,
                chunk_index=index,
                raw_doc=raw_doc,
            )

            # Inherit all RawDocument metadata and add chunk-level fields.
            chunk_metadata: dict[str, Any] = {
                **raw_doc.metadata,
                "file_type": raw_doc.file_type,
            }

            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                document_name=raw_doc.document_name,
                chunk_text=chunk_text.strip(),
                page_number=page_number,
                chunk_index=index,
                metadata=chunk_metadata,
            )
            chunks.append(chunk)

        logger.info(
            "Produced %d chunks from '%s'.",
            len(chunks),
            raw_doc.document_name,
        )
        return chunks

    def chunk_batch(self, raw_docs: list[RawDocument]) -> list[Chunk]:
        """
        Chunk multiple documents and return a flat list of all Chunks.

        Args:
            raw_docs: List of RawDocument objects.

        Returns:
            Flat list of Chunk objects across all documents.
        """
        all_chunks: list[Chunk] = []
        for raw_doc in raw_docs:
            try:
                all_chunks.extend(self.chunk(raw_doc))
            except ValueError as exc:
                # Log and continue — one bad document should not abort the batch.
                logger.error(
                    "Skipping '%s' during batch chunking: %s",
                    raw_doc.document_name,
                    exc,
                )
        return all_chunks

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _resolve_page_number(
        self,
        chunk_text: str,
        chunk_index: int,
        raw_doc: RawDocument,
    ) -> int | None:
        """
        Attempt to determine which page a chunk originated from.

        For PDFs: uses the page_char_offsets list stored in metadata
        by the PDF loader to map the chunk's approximate position in
        full_text back to a page number.

        For all other formats: returns None because page boundaries
        are not reliably determinable without rendering the document.

        Args:
            chunk_text: The text content of the chunk.
            chunk_index: Zero-based position of this chunk.
            raw_doc: The source RawDocument.

        Returns:
            1-based page number, or None.
        """
        if raw_doc.file_type != "pdf":
            return None

        page_char_offsets: list[int] | None = raw_doc.metadata.get(
            "page_char_offsets"
        )

        if not page_char_offsets:
            return None

        # Find the approximate character position of this chunk in full_text.
        char_position = raw_doc.full_text.find(chunk_text[:50])  # use first 50 chars

        if char_position == -1:
            # Fallback: estimate position proportionally.
            total_chars = len(raw_doc.full_text)
            total_chunks = max(
                len(raw_doc.full_text) // max(self._chunk_size, 1), 1
            )
            char_position = int(
                (chunk_index / total_chunks) * total_chars
            )

        # Find the last page whose start offset is ≤ our char_position.
        page_number = 1
        for page_idx, offset in enumerate(page_char_offsets):
            if offset <= char_position:
                page_number = page_idx + 1  # convert to 1-based
            else:
                break

        return page_number