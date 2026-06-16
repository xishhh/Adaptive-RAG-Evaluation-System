"""
app/services/adaptive_rag_service.py

Orchestrates the full adaptive RAG pipeline for Phase 5.

Architecture reference: Components 3–8 in SYSTEM_ARCHITECTURE.md.

Pipeline flow:
    User question
        ↓
    QueryClassifier
        ├── DIRECT_LLM    → LLM answers directly (no retrieval)
        │                   retrieval_strategy = "direct_llm"
        │
        └── KNOWLEDGE_QUERY
                ↓
            Retriever (first pass)
                ↓
            ConfidenceEvaluator
                ├── GOOD_CONTEXT  → proceed to RAGService
                │                   retrieval_strategy = "single_retrieval"
                │
                └── LOW_CONFIDENCE
                        ↓
                    QueryRewriter
                        ↓
                    Retriever (second pass with rewritten query)
                        ↓
                    RAGService (with re-retrieved chunks)
                        retrieval_strategy = "rewritten_retrieval"

Relationship to RAGService:
    AdaptiveRAGService does NOT replace RAGService.
    It delegates answer generation to RAGService._call_llm() and
    citation building to CitationService. This avoids code duplication
    and respects the single-responsibility principle: RAGService owns
    LLM invocation; AdaptiveRAGService owns retrieval strategy.

Design decisions:
  1. All four adaptive components (classifier, retriever, evaluator,
     rewriter) are instantiated once in __init__ and reused across calls.
     This avoids re-creating LLM clients on every request.
  2. DIRECT_LLM answers use the LLM with an empty context block and
     return an AdaptiveQueryResponse with sources=[]. This keeps the
     response shape consistent for all callers.
  3. ADAPTIVE_MAX_REWRITES is respected but currently capped at 1 as
     specified. The loop structure is in place for future extension.
  4. All intermediate decisions are logged at INFO level to make the
     adaptive behaviour observable in production logs.
"""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI

from app.models.responses import AdaptiveQueryResponse, ChunkResult
from app.retrievers.confidence_evaluator import ConfidenceEvaluator
from app.retrievers.query_classifier import QueryClassifier
from app.retrievers.query_rewriter import QueryRewriter
from app.retrievers.retriever import Retriever
from app.services.citation_service import CitationService
from app.services.rag_service import RAGService
from app.utils.config import get_settings
from app.vectorstore.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)


class AdaptiveRAGService:
    """
    End-to-end adaptive RAG pipeline with query classification,
    confidence evaluation, and query rewriting.

    Args:
        chroma_manager: An initialised ChromaManager instance.
        top_k:          Number of chunks to retrieve per pass. Defaults to 5.

    Usage:
        service = AdaptiveRAGService(chroma_manager=ChromaManager())
        response = service.query("What is the termination clause?")
    """

    def __init__(
        self,
        chroma_manager: ChromaManager,
        top_k: int = 5,
    ) -> None:
        settings = get_settings()

        # Shared LLM instance — one client, used by classifier, rewriter,
        # and (via RAGService) the answer generator.
        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_API_BASE,
            temperature=0.0,
            max_tokens=1024,
        )

        # Adaptive components (Phase 5)
        self._classifier = QueryClassifier(llm=llm)
        self._evaluator = ConfidenceEvaluator(threshold=settings.CONFIDENCE_THRESHOLD)
        self._rewriter = QueryRewriter(llm=llm)
        self._retriever = Retriever(chroma_manager=chroma_manager, default_top_k=top_k)

        # Answer generation (Phase 4, reused here)
        self._rag_service = RAGService(chroma_manager=chroma_manager, top_k=top_k)
        self._citation_service = CitationService()

        self._max_rewrites: int = settings.ADAPTIVE_MAX_REWRITES
        self._top_k = top_k

        logger.info(
            "AdaptiveRAGService initialised | model=%s | top_k=%d | "
            "confidence_threshold=%.2f | max_rewrites=%d",
            settings.LLM_MODEL,
            top_k,
            settings.CONFIDENCE_THRESHOLD,
            self._max_rewrites,
        )

    def query(self, question: str, top_k: int | None = None) -> AdaptiveQueryResponse:
        """
        Run the adaptive RAG pipeline for a user question.

        Args:
            question: Natural-language question from the user.
            top_k:    Override the retriever's default top_k for this call.

        Returns:
            AdaptiveQueryResponse with answer, sources, and adaptive metadata.

        Raises:
            ValueError: If the question is empty.
        """
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        effective_top_k = top_k if top_k is not None else self._top_k

        logger.info("Adaptive RAG query: '%s'", question[:120])

        # ------------------------------------------------------------------ #
        # Step 1 — Query Classification                                        #
        # ------------------------------------------------------------------ #
        query_type = self._classifier.classify(question)

        # ------------------------------------------------------------------ #
        # Step 2a — DIRECT_LLM path                                            #
        # ------------------------------------------------------------------ #
        if query_type == "DIRECT_LLM":
            logger.info("Path: DIRECT_LLM — skipping retrieval.")
            answer = self._rag_service._call_llm(
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

        # ------------------------------------------------------------------ #
        # Step 2b — KNOWLEDGE_QUERY path                                       #
        # ------------------------------------------------------------------ #
        logger.info("Path: KNOWLEDGE_QUERY — running retrieval.")

        chunks: list[ChunkResult] = self._retriever.retrieve(
            query=question,
            top_k=effective_top_k,
        )

        # ------------------------------------------------------------------ #
        # Step 3 — Confidence Evaluation + optional Rewrite loop               #
        # ------------------------------------------------------------------ #
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

                # Re-evaluate after rewrite; break early if now sufficient.
                new_confidence = self._evaluator.evaluate(chunks)
                logger.info(
                    "Post-rewrite confidence: %s (attempt %d)",
                    new_confidence,
                    attempt + 1,
                )
                if new_confidence == "GOOD_CONTEXT":
                    break
                # If still LOW_CONFIDENCE and more attempts remain, loop again.

        # ------------------------------------------------------------------ #
        # Step 4 — Answer Generation                                           #
        # ------------------------------------------------------------------ #
        context_block = self._citation_service.format_context_block(chunks)
        answer = self._rag_service._call_llm(question=question, context=context_block)
        citations = self._citation_service.build_citations(chunks=chunks)

        logger.info(
            "Adaptive RAG complete | strategy=%s | chunks_used=%d | "
            "answer_length=%d | rewritten_query=%s",
            retrieval_strategy,
            len(citations),
            len(answer),
            f"'{rewritten_query[:60]}'" if rewritten_query else "None",
        )

        return AdaptiveQueryResponse(
            question=question,
            answer=answer,
            sources=citations,
            query_type="KNOWLEDGE_QUERY",
            rewritten_query=rewritten_query,
            retrieval_strategy=retrieval_strategy,
        )