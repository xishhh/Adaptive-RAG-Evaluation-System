from __future__ import annotations

import logging
from typing import Literal

from app.models.responses import ChunkResult

logger = logging.getLogger(__name__)

ConfidenceOutcome = Literal["GOOD_CONTEXT", "LOW_CONFIDENCE"]


class ConfidenceEvaluator:
    def __init__(self, threshold: float = 0.45) -> None:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {threshold}.")
        self._threshold = threshold

    def evaluate(self, chunks: list[ChunkResult]) -> ConfidenceOutcome:
        if not chunks:
            logger.warning("ConfidenceEvaluator: no chunks retrieved — LOW_CONFIDENCE.")
            return "LOW_CONFIDENCE"

        avg_score = sum(c.relevance_score for c in chunks) / len(chunks)
        top_score = chunks[0].relevance_score

        logger.info(
            "Confidence evaluation | chunks=%d | avg_score=%.4f | top_score=%.4f | threshold=%.4f",
            len(chunks), avg_score, top_score, self._threshold,
        )

        if avg_score >= self._threshold:
            logger.info("Confidence outcome: GOOD_CONTEXT")
            return "GOOD_CONTEXT"

        logger.info("Confidence outcome: LOW_CONFIDENCE (avg %.4f < threshold %.4f)", avg_score, self._threshold)
        return "LOW_CONFIDENCE"

    @property
    def threshold(self) -> float:
        return self._threshold
