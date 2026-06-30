from functools import lru_cache

from fastapi import Depends

from app.evaluators.eval_dataset_generator import EvalDatasetGenerator
from app.evaluators.metrics_tracker import MetricsTracker
from app.evaluators.ragas_evaluator import RagasEvaluator
from app.ingestion.ingestion_service import DocumentIngestionService
from app.services.adaptive_rag_service import AdaptiveRAGService
from app.services.ingestion_tracker import IngestionTracker
from app.services.rag_service import RAGService
from app.vectorstore.chroma_manager import ChromaManager


@lru_cache
def get_chroma_manager() -> ChromaManager:
    return ChromaManager()


@lru_cache
def get_rag_service(
    chroma_manager: ChromaManager = Depends(get_chroma_manager),
) -> RAGService:
    return RAGService(chroma_manager=chroma_manager)


@lru_cache
def get_eval_generator(
    chroma_manager: ChromaManager = Depends(get_chroma_manager),
    rag_service: RAGService = Depends(get_rag_service),
) -> EvalDatasetGenerator:
    return EvalDatasetGenerator(
        chroma_manager=chroma_manager,
        rag_service=rag_service,
    )


@lru_cache
def get_ragas_evaluator() -> RagasEvaluator:
    return RagasEvaluator()


@lru_cache
def get_metrics_tracker() -> MetricsTracker:
    return MetricsTracker()


_adaptive_rag_service_instance: AdaptiveRAGService | None = None


def get_adaptive_rag_service(
    chroma_manager: ChromaManager = Depends(get_chroma_manager),
) -> AdaptiveRAGService:
    global _adaptive_rag_service_instance
    if _adaptive_rag_service_instance is None:
        _adaptive_rag_service_instance = AdaptiveRAGService(chroma_manager=chroma_manager)
    return _adaptive_rag_service_instance


@lru_cache
def get_ingestion_tracker() -> IngestionTracker:
    return IngestionTracker()


@lru_cache
def get_ingestion_service(
    chroma_manager: ChromaManager = Depends(get_chroma_manager),
    eval_generator: EvalDatasetGenerator = Depends(get_eval_generator),
    tracker: IngestionTracker = Depends(get_ingestion_tracker),
) -> DocumentIngestionService:
    return DocumentIngestionService(
        chroma_manager=chroma_manager,
        eval_generator=eval_generator,
        tracker=tracker,
    )
