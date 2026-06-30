from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.models.responses import ChunkResult, QueryResponse
from app.retrievers.retriever import Retriever
from app.services.citation_service import build_citations, format_context_block
from app.utils.config import get_settings
from app.utils.llm_factory import create_llm_with_fallback
from app.vectorstore.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)

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
    def __init__(
        self,
        chroma_manager: ChromaManager,
        top_k: int = 5,
    ) -> None:
        settings = get_settings()

        self._retriever = Retriever(chroma_manager=chroma_manager, default_top_k=top_k)

        self._llm = create_llm_with_fallback(
            temperature=0.0,
            max_tokens=1024,
        )

        logger.info(
            "RAGService initialised | model=%s | top_k=%d",
            settings.LLM_MODEL,
            top_k,
        )

    def query(self, question: str, top_k: int | None = None) -> QueryResponse:
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        logger.info("RAG query: '%s'", question[:120])

        chunks: list[ChunkResult] = self._retriever.retrieve(
            query=question,
            top_k=top_k,
        )

        context_block = format_context_block(chunks)
        answer = self.call_llm(question=question, context=context_block)
        citations = build_citations(chunks=chunks)

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

    def call_llm(self, question: str, context: str) -> str:
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
        return response.content.strip()

    def stream_llm(self, question: str, context: str):
        human_content = (
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        logger.debug("Streaming LLM | model messages assembled.")

        for chunk in self._llm.stream(messages):
            token = chunk.content
            if token:
                yield token
