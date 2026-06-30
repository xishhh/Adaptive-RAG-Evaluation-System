from __future__ import annotations

import logging
from typing import Any, Generator

from app.models.responses import AdaptiveQueryResponse, ChunkResult
from app.retrievers.confidence_evaluator import ConfidenceEvaluator
from app.retrievers.query_classifier import QueryClassifier
from app.retrievers.query_rewriter import QueryRewriter
from app.retrievers.retriever import Retriever
from app.services.citation_service import build_citations, format_context_block
from app.services.rag_service import RAGService
from app.utils.config import get_settings
from app.utils.llm_factory import create_llm_with_fallback
from app.vectorstore.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)


class AdaptiveRAGService:
    def __init__(
        self,
        chroma_manager: ChromaManager,
        top_k: int = 5,
    ) -> None:
        settings = get_settings()

        llm = create_llm_with_fallback(
            temperature=0.0,
            max_tokens=1024,
        )

        self._classifier = QueryClassifier(llm=llm)
        self._evaluator = ConfidenceEvaluator(threshold=settings.CONFIDENCE_THRESHOLD)
        self._rewriter = QueryRewriter(llm=llm)
        self._retriever = Retriever(chroma_manager=chroma_manager, default_top_k=top_k)

        self._rag_service = RAGService(chroma_manager=chroma_manager, top_k=top_k)

        self._max_rewrites: int = settings.ADAPTIVE_MAX_REWRITES
        self._top_k = top_k

        logger.info(
            "AdaptiveRAGService initialised | top_k=%d | "
            "confidence_threshold=%.2f | max_rewrites=%d",
            top_k,
            settings.CONFIDENCE_THRESHOLD,
            self._max_rewrites,
        )

    def query(self, question: str, top_k: int | None = None) -> AdaptiveQueryResponse:
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        effective_top_k = top_k if top_k is not None else self._top_k

        logger.info("Adaptive RAG query: '%s'", question[:120])

        query_type = self._classifier.classify(question)

        if query_type == "DIRECT_LLM":
            logger.info("Path: DIRECT_LLM — skipping retrieval.")
            answer = self._rag_service.call_llm(
                question=question,
                context="No document context is required for this question.",
            )
            return AdaptiveQueryResponse(
                question=question,
                answer=answer,
                sources=[],
                query_type="DIRECT_LLM",
                rewritten_query=None,
                retrieval_strategy="direct_llm",
            )

        logger.info("Path: KNOWLEDGE_QUERY — running retrieval.")

        chunks: list[ChunkResult] = self._retriever.retrieve(
            query=question,
            top_k=effective_top_k,
        )

        rewritten_query: str | None = None
        retrieval_strategy = "single_retrieval"

        confidence = self._evaluator.evaluate(chunks)

        if confidence == "LOW_CONFIDENCE":
            for attempt in range(self._max_rewrites):
                logger.info(
                    "LOW_CONFIDENCE — rewrite attempt %d/%d",
                    attempt + 1,
                    self._max_rewrites,
                )
                rewritten_query = self._rewriter.rewrite(question)
                chunks = self._retriever.retrieve(
                    query=rewritten_query,
                    top_k=effective_top_k,
                )
                retrieval_strategy = "rewritten_retrieval"

                new_confidence = self._evaluator.evaluate(chunks)
                logger.info(
                    "Post-rewrite confidence: %s (attempt %d)",
                    new_confidence,
                    attempt + 1,
                )
                if new_confidence == "GOOD_CONTEXT":
                    break

        context_block = format_context_block(chunks)
        answer = self._rag_service.call_llm(question=question, context=context_block)
        citations = build_citations(chunks=chunks)

        logger.info(
            "Adaptive RAG complete | strategy=%s | chunks_used=%d | answer_length=%d",
            retrieval_strategy,
            len(citations),
            len(answer),
        )

        return AdaptiveQueryResponse(
            question=question,
            answer=answer,
            sources=citations,
            query_type="KNOWLEDGE_QUERY",
            rewritten_query=rewritten_query,
            retrieval_strategy=retrieval_strategy,
        )

    def stream_query(
        self,
        question: str,
        top_k: int | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        if not question or not question.strip():
            yield {"event": "error", "data": {"message": "Question must not be empty."}}
            return

        effective_top_k = top_k if top_k is not None else self._top_k

        logger.info("Adaptive RAG stream query: '%s'", question[:120])

        try:
            query_type = self._classifier.classify(question)
        except Exception as exc:
            logger.exception("Query classification failed: %s", exc)
            yield {"event": "error", "data": {"message": f"Classification failed: {exc}"}}
            return

        if query_type == "DIRECT_LLM":
            logger.info("Stream path: DIRECT_LLM — skipping retrieval.")
            yield {
                "event": "metadata",
                "data": {
                    "query_type": "DIRECT_LLM",
                    "sources": [],
                    "rewritten_query": None,
                    "retrieval_strategy": "direct_llm",
                },
            }
            context = "No document context is required for this question."
            yield from _yield_tokens(self._rag_service, question, context)
            return

        logger.info("Stream path: KNOWLEDGE_QUERY — running retrieval.")

        try:
            chunks: list[ChunkResult] = self._retriever.retrieve(
                query=question,
                top_k=effective_top_k,
            )
        except Exception as exc:
            logger.exception("Retrieval failed: %s", exc)
            yield {"event": "error", "data": {"message": f"Retrieval failed: {exc}"}}
            return

        rewritten_query: str | None = None
        retrieval_strategy = "single_retrieval"

        try:
            confidence = self._evaluator.evaluate(chunks)
            if confidence == "LOW_CONFIDENCE":
                for attempt in range(self._max_rewrites):
                    logger.info(
                        "LOW_CONFIDENCE — rewrite attempt %d/%d",
                        attempt + 1,
                        self._max_rewrites,
                    )
                    rewritten_query = self._rewriter.rewrite(question)
                    chunks = self._retriever.retrieve(
                        query=rewritten_query,
                        top_k=effective_top_k,
                    )
                    retrieval_strategy = "rewritten_retrieval"
                    new_confidence = self._evaluator.evaluate(chunks)
                    if new_confidence == "GOOD_CONTEXT":
                        break
        except Exception as exc:
            logger.exception("Confidence evaluation / rewrite failed: %s", exc)
            yield {
                "event": "error",
                "data": {"message": f"Pipeline error: {exc}"},
            }
            return

        context_block = format_context_block(chunks)
        citations = build_citations(chunks=chunks)

        yield {
            "event": "metadata",
            "data": {
                "query_type": "KNOWLEDGE_QUERY",
                "sources": [c.model_dump() for c in citations],
                "rewritten_query": rewritten_query,
                "retrieval_strategy": retrieval_strategy,
            },
        }

        yield from _yield_tokens(self._rag_service, question, context_block)


def _yield_tokens(
    rag_service: RAGService,
    question: str,
    context: str,
) -> Generator[dict[str, Any], None, None]:
    try:
        for token in rag_service.stream_llm(question, context):
            yield {"event": "token", "data": token}
    except Exception as exc:
        logger.exception("LLM streaming failed: %s", exc)
        yield {"event": "error", "data": {"message": f"LLM error: {exc}"}}
        return

    yield {"event": "done", "data": {"completed": True}}
