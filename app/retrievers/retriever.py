from __future__ import annotations

import logging
from typing import Optional

from app.models.responses import ChunkResult
from app.vectorstore.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        chroma_manager: ChromaManager,
        default_top_k: int = 5,
    ) -> None:
        self._chroma = chroma_manager
        self._default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[ChunkResult]:
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        k = top_k if top_k is not None else self._default_top_k

        logger.info("Retrieving top-%d chunks for query: '%s'", k, query[:80])

        raw_results = self._chroma.similarity_search(
            query=query,
            top_k=k,
        )

        if not raw_results:
            logger.warning("No chunks retrieved for query: '%s'", query[:80])
            return []

        chunk_results = [self._to_chunk_result(r) for r in raw_results]

        logger.info("Retrieved %d chunks | top score=%.4f", len(chunk_results), chunk_results[0].relevance_score if chunk_results else 0.0)
        return chunk_results

    @staticmethod
    def _to_chunk_result(raw: dict) -> ChunkResult:
        return ChunkResult(
            chunk_id=raw["chunk_id"],
            chunk_text=raw["chunk_text"],
            document_name=raw["document_name"],
            page_number=raw.get("page_number", 0),
            chunk_index=raw.get("chunk_index", 0),
            relevance_score=raw["relevance_score"],
        )
