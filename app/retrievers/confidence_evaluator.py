"""
app/retrievers/confidence_evaluator.py

Evaluates whether retrieved chunks provide sufficient context to answer
a user query with confidence.

Architecture reference: Component 5 (Retrieval Quality Evaluator) in
SYSTEM_ARCHITECTURE.md.

Possible outcomes:
    GOOD_CONTEXT    — Average relevance score ≥ threshold. Proceed to
                      answer generation.
    LOW_CONFIDENCE  — Average relevance score < threshold, or no chunks
                      returned. Trigger query rewriting and re-retrieval.

Design decisions:
  1. Score-based evaluation only — no extra LLM call.
     ChromaDB's cosine similarity scores (0.0–1.0) are a reliable proxy
     for retrieval quality. Adding an LLM call here would double latency
     on every query and introduce a circular dependency (using an LLM
     to evaluate context for an LLM).
  2. The threshold is configurable via Settings.CONFIDENCE_THRESHOLD so
     it can be tuned without code changes.
  3. Zero chunks is always LOW_CONFIDENCE regardless of threshold.
     A RAG system with no retrieved context cannot produce a grounded answer.
  4. The evaluator is stateless — no constructor state beyond the threshold.
     This makes it trivially unit-testable.
"""

from __future__ import annotations

import logging
from typing import Literal

from app.models.responses import ChunkResult

logger = logging.getLogger(__name__)

ConfidenceOutcome = Literal["GOOD_CONTEXT", "LOW_CONFIDENCE"]


class ConfidenceEvaluator:
    """
    Evaluates retrieval quality and returns a confidence verdict.

    Args:
        threshold: Minimum average relevance score (0.0–1.0) required
                   for a GOOD_CONTEXT verdict. Defaults to 0.45.
                   In production, pass settings.CONFIDENCE_THRESHOLD.

    Usage:
        evaluator = ConfidenceEvaluator(threshold=settings.CONFIDENCE_THRESHOLD)
        outcome = evaluator.evaluate(chunks)
        # → "GOOD_CONTEXT" or "LOW_CONFIDENCE"
    """

    def __init__(self, threshold: float = 0.45) -> None:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"Confidence threshold must be between 0.0 and 1.0, got {threshold}."
            )
        self._threshold = threshold

    def evaluate(self, chunks: list[ChunkResult]) -> ConfidenceOutcome:
        """
        Evaluate a list of retrieved chunks and return the confidence outcome.

        Args:
            chunks: Ordered list of ChunkResult objects from the Retriever.
                    May be empty.

        Returns:
            "GOOD_CONTEXT" if average relevance_score ≥ threshold.
            "LOW_CONFIDENCE" if no chunks retrieved or score < threshold.
        """
        if not chunks:
            logger.warning(
                "ConfidenceEvaluator: no chunks retrieved — LOW_CONFIDENCE."
            )
            return "LOW_CONFIDENCE"

        avg_score = sum(c.relevance_score for c in chunks) / len(chunks)
        top_score = chunks[0].relevance_score  # chunks are sorted by score descending

        logger.info(
            "Confidence evaluation | chunks=%d | avg_score=%.4f | "
            "top_score=%.4f | threshold=%.4f",
            len(chunks),
            avg_score,
            top_score,
            self._threshold,
        )

        if avg_score >= self._threshold:
            logger.info("Confidence outcome: GOOD_CONTEXT")
            return "GOOD_CONTEXT"

        logger.info(
            "Confidence outcome: LOW_CONFIDENCE (avg %.4f < threshold %.4f)",
            avg_score,
            self._threshold,
        )
        return "LOW_CONFIDENCE"

    @property
    def threshold(self) -> float:
        """The configured confidence threshold."""
        return self._threshold