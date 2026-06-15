"""
app/services/rag_service.py

Orchestrates the basic RAG pipeline for Phase 4.

Flow:
    User question
        → Retriever  (top-K chunks from ChromaDB)
        → CitationService.format_context_block  (assemble prompt context)
        → LLM via LangChain  (generate grounded answer)
        → CitationService.build_citations  (structured citation list)
        → QueryResponse

Phase 5 will introduce AdaptiveRAGService, which wraps this service
with query classification, confidence evaluation, and query rewriting.
This service must remain unaware of those concerns.

LLM integration:
  Uses LangChain's ChatOpenAI configured to point at OpenRouter
  (OPENAI_API_BASE in settings). The prompt instructs the model to
  answer only from the provided context, avoiding hallucination.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.models.responses import ChunkResult, QueryResponse
from app.retrievers.retriever import Retriever
from app.services.citation_service import CitationService
from app.utils.config import get_settings
from app.vectorstore.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)

# System prompt that instructs the LLM to stay grounded in the context.
# Keeping it here as a module-level constant makes it easy to version
# and test independently of the service logic.
_SYSTEM_PROMPT = """You are a precise document question-answering assistant.

Answer the user's question using ONLY the context provided below.
Do not use any knowledge outside of the provided context.

Rules:
- If the context contains the answer, provide it clearly and concisely.
- Always reference the source when possible (e.g. "According to contract_a.pdf…").
- If the context does NOT contain enough information to answer, respond with:
  "I could not find sufficient information in the provided documents to answer this question."
- Do not speculate, infer beyond the context, or fabricate information.
"""


class RAGService:
    """
    End-to-end RAG pipeline: retrieve context → generate answer → cite sources.

    Args:
        chroma_manager:   An initialised ChromaManager instance.
        top_k:            Number of chunks to retrieve per query. Defaults to 5.

    Usage:
        service = RAGService(chroma_manager=ChromaManager())
        response = service.query("What is the termination clause?")
    """

    def __init__(
        self,
        chroma_manager: ChromaManager,
        top_k: int = 5,
    ) -> None:
        settings = get_settings()

        self._retriever = Retriever(chroma_manager=chroma_manager, default_top_k=top_k)
        self._citation_service = CitationService()

        self._llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_API_BASE,
            temperature=0.0,   # deterministic answers for a QA system
            max_tokens=1024,
        )

        logger.info(
            "RAGService initialised | model=%s | top_k=%d",
            settings.LLM_MODEL,
            top_k,
        )

    def query(self, question: str, top_k: int | None = None) -> QueryResponse:
        """
        Run the full RAG pipeline for a user question.

        Args:
            question: Natural-language question from the user.
            top_k:    Override the default top_k for this call only.

        Returns:
            QueryResponse with the generated answer and source citations.

        Raises:
            ValueError: If the question is empty.
        """
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        logger.info("RAG query: '%s'", question[:120])

        # Step 1: Retrieve relevant chunks.
        chunks: list[ChunkResult] = self._retriever.retrieve(
            query=question,
            top_k=top_k,
        )

        # Step 2: Build a context block for the prompt.
        context_block = self._citation_service.format_context_block(chunks)

        # Step 3: Call the LLM.
        answer = self._call_llm(question=question, context=context_block)

        # Step 4: Build structured citations from retrieved chunks.
        citations = self._citation_service.build_citations(chunks=chunks)

        logger.info(
            "RAG complete | answer_length=%d chars | citations=%d",
            len(answer),
            len(citations),
        )

        return QueryResponse(
            question=question,
            answer=answer,
            sources=citations,
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _call_llm(self, question: str, context: str) -> str:
        """
        Send the question and context to the LLM and return the answer text.

        The prompt is structured as:
          - SystemMessage: grounding rules
          - HumanMessage:  context block + question

        Separating context from the system prompt is intentional:
        it makes the context block easy to swap or extend in Phase 5
        without touching the system rules.

        Args:
            question: The user's question.
            context:  Formatted context block from CitationService.

        Returns:
            The LLM's answer as a plain string.
        """
        human_content = (
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        logger.debug("Calling LLM | model messages assembled.")

        response = self._llm.invoke(messages)

        # LangChain's ChatOpenAI returns an AIMessage; .content is the string.
        return response.content.strip()