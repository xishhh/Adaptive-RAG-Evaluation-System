"""
app/api/dependencies.py

Central place for all FastAPI Depends() providers.
Holds singleton instances of ChromaManager, AdaptiveRAGService,
RagasEvaluator, MetricsTracker, and EvalDatasetGenerator.
"""

from functools import lru_cache

from fastapi import Depends

from app.evaluators.eval_dataset_generator import EvalDatasetGenerator
from app.evaluators.metrics_tracker import MetricsTracker
from app.evaluators.ragas_evaluator import RagasEvaluator
from app.services.adaptive_rag_service import AdaptiveRAGService
from app.vectorstore.chroma_manager import ChromaManager


@lru_cache
def get_chroma_manager() -> ChromaManager:
    """Provides a singleton instance of ChromaManager."""
    return ChromaManager()


@lru_cache
def get_eval_generator() -> EvalDatasetGenerator:
    """Provides a singleton instance of EvalDatasetGenerator."""
    return EvalDatasetGenerator()


@lru_cache
def get_ragas_evaluator() -> RagasEvaluator:
    """Provides a singleton instance of RagasEvaluator."""
    return RagasEvaluator()


@lru_cache
def get_metrics_tracker() -> MetricsTracker:
    """Provides a singleton instance of MetricsTracker."""
    return MetricsTracker()


# We use a global variable to cache AdaptiveRAGService because
# it depends on ChromaManager, and @lru_cache doesn't cleanly support
# arguments injected via Depends() without hashing issues.
_adaptive_rag_service_instance: AdaptiveRAGService | None = None


def get_adaptive_rag_service(
    chroma_manager: ChromaManager = Depends(get_chroma_manager),
) -> AdaptiveRAGService:
    """Provides a singleton instance of AdaptiveRAGService."""
    global _adaptive_rag_service_instance
    if _adaptive_rag_service_instance is None:
        _adaptive_rag_service_instance = AdaptiveRAGService(chroma_manager=chroma_manager)
    return _adaptive_rag_service_instance
