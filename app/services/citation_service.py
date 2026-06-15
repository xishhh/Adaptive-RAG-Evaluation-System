"""
app/services/citation_service.py

Builds structured citations from retrieved chunks.

Responsibilities:
  - Accept the LLM-generated answer and the list of retrieved ChunkResults.
  - Filter chunks to those actually relevant to the answer.
  - Return an ordered, deduplicated list of ChunkResult citations.

Why a separate service?
  Citation logic is independently testable and will evolve in Phase 5
  (confidence thresholding, deduplication across documents). Keeping it
  separate from rag_service.py preserves single-responsibility.

Current strategy (Phase 4):
  Return all retrieved chunks as citations, sorted by relevance score.
  Phase 5 will introduce confidence-based filtering once the confidence
  evaluator is in place.
"""

from __future__ import annotations

import logging

from app.models.responses import ChunkResult

logger = logging.getLogger(__name__)


class CitationService:
    """
    Produces ordered citation lists from retrieval results.

    Usage:
        service = CitationService()
        citations = service.build_citations(chunks=retrieved_chunks)
    """

    def build_citations(
        self,
        chunks: list[ChunkResult],
        min_relevance_score: float = 0.0,
    ) -> list[ChunkResult]:
        """
        Filter and sort chunks to produce a citation list.

        Args:
            chunks:               Retrieved chunks from the Retriever.
            min_relevance_score:  Minimum score threshold. Chunks below
                                  this threshold are excluded. Defaults to
                                  0.0 (include all) for Phase 4. Phase 5
                                  will raise this threshold adaptively.

        Returns:
            Deduplicated list of ChunkResult objects sorted by
            relevance_score descending.
        """
        if not chunks:
            logger.debug("build_citations called with empty chunk list.")
            return []

        # Filter by minimum relevance score.
        filtered = [c for c in chunks if c.relevance_score >= min_relevance_score]

        # Deduplicate by chunk_id — the same chunk can appear multiple times
        # if the retriever is called more than once (Phase 5 re-retrieval).
        seen: set[str] = set()
        deduplicated: list[ChunkResult] = []
        for chunk in filtered:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                deduplicated.append(chunk)

        # Sort by relevance score descending.
        deduplicated.sort(key=lambda c: c.relevance_score, reverse=True)

        logger.debug(
            "Citations built: %d chunks in → %d citations out (threshold=%.2f).",
            len(chunks),
            len(deduplicated),
            min_relevance_score,
        )
        return deduplicated

    def format_context_block(self, chunks: list[ChunkResult]) -> str:
        """
        Render retrieved chunks into a plain-text context block for the LLM prompt.

        Each chunk is prefixed with its source so the LLM can reference
        it in the answer. The format is designed to be unambiguous and
        easy for the model to cite.

        Args:
            chunks: Ordered list of ChunkResult objects.

        Returns:
            Multi-line string ready to be injected into a prompt template.

        Example output:
            [Source 1 | contract_a.pdf | page 3]
            The termination clause states that either party may …

            [Source 2 | contract_b.pdf | page 7]
            Penalty for late delivery is calculated as …
        """
        if not chunks:
            return "No relevant context found."

        blocks: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            page_info = f" | page {chunk.page_number}" if chunk.page_number else ""
            header = f"[Source {i} | {chunk.document_name}{page_info}]"
            blocks.append(f"{header}\n{chunk.chunk_text}")

        return "\n\n".join(blocks)