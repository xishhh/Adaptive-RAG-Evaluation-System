"""
app/retrievers/retriever.py

Basic vector similarity retriever for Phase 4.

Responsibilities:
  - Accept a query string and top_k parameter.
  - Delegate the similarity search to ChromaManager.
  - Map raw result dicts to ChunkResult Pydantic objects.
  - Return a ranked list of ChunkResult objects.

Phase 5 will wrap this retriever inside an adaptive layer that adds
query classification, confidence evaluation, and query rewriting.
This module must remain unaware of those concerns.

Design decision:
  Retriever takes a ChromaManager instance via constructor injection
  rather than instantiating it internally. This makes the retriever
  testable in isolation (pass a mock) and avoids hidden side effects.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.models.responses import ChunkResult
from app.vectorstore.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves the top-K most relevant chunks for a given query.

    Args:
        chroma_manager: An initialised ChromaManager instance.
        default_top_k:  Default number of chunks to retrieve if not
                        specified per call. Defaults to 5.

    Usage:
        retriever = Retriever(chroma_manager=ChromaManager())
        results = retriever.retrieve("What is the termination clause?")
    """

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
        filter_metadata: Optional[dict] = None,
    ) -> list[ChunkResult]:
        """
        Run a similarity search and return ranked ChunkResult objects.

        Args:
            query:           Natural-language query string.
            top_k:           Number of chunks to return. Falls back to
                             default_top_k if not provided.
            filter_metadata: Optional ChromaDB metadata filter.
                             Example: {"document_name": "contract_a.pdf"}

        Returns:
            List of ChunkResult objects sorted by relevance_score descending.
            Returns an empty list if the vector store is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        k = top_k if top_k is not None else self._default_top_k

        logger.info("Retrieving top-%d chunks for query: '%s'", k, query[:80])

        raw_results = self._chroma.similarity_search(
            query=query,
            top_k=k,
            filter_metadata=filter_metadata,
        )

        if not raw_results:
            logger.warning("No chunks retrieved for query: '%s'", query[:80])
            return []

        chunk_results = [self._to_chunk_result(r) for r in raw_results]

        logger.info(
            "Retrieved %d chunks | top score=%.4f",
            len(chunk_results),
            chunk_results[0].relevance_score if chunk_results else 0.0,
        )
        return chunk_results

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_chunk_result(raw: dict) -> ChunkResult:
        """
        Map a raw similarity_search result dict to a ChunkResult.

        ChromaManager guarantees these keys in every result dict:
            chunk_id, chunk_text, document_name,
            page_number, chunk_index, relevance_score.
        """
        return ChunkResult(
            chunk_id=raw["chunk_id"],
            chunk_text=raw["chunk_text"],
            document_name=raw["document_name"],
            page_number=raw.get("page_number", 0),
            chunk_index=raw.get("chunk_index", 0),
            relevance_score=raw["relevance_score"],
        )