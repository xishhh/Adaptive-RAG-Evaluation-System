from __future__ import annotations

import uuid
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.documents import Chunk, RawDocument
from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentChunker:
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
        if not raw_doc.full_text.strip():
            raise ValueError(
                f"Cannot chunk '{raw_doc.document_name}': full_text is empty."
            )

        logger.info(
            "Chunking '%s' (%d chars, type=%s) ...",
            raw_doc.document_name,
            len(raw_doc.full_text),
            raw_doc.file_type,
        )

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

        logger.info("Produced %d chunks from '%s'.", len(chunks), raw_doc.document_name)
        return chunks

    def _resolve_page_number(
        self,
        chunk_text: str,
        chunk_index: int,
        raw_doc: RawDocument,
    ) -> int | None:
        if raw_doc.file_type != "pdf":
            return None

        page_char_offsets: list[int] | None = raw_doc.metadata.get("page_char_offsets")
        if not page_char_offsets:
            return None

        char_position = raw_doc.full_text.find(chunk_text[:50])
        if char_position == -1:
            total_chars = len(raw_doc.full_text)
            total_chunks = max(len(raw_doc.full_text) // max(self._chunk_size, 1), 1)
            char_position = int((chunk_index / total_chunks) * total_chars)

        page_number = 1
        for page_idx, offset in enumerate(page_char_offsets):
            if offset <= char_position:
                page_number = page_idx + 1
            else:
                break

        return page_number
