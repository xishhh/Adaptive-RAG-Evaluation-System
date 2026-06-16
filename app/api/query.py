"""
app/api/query.py

FastAPI router for POST /query.

Phase 4: Used RAGService → QueryResponse.
Phase 5: Upgraded to AdaptiveRAGService → AdaptiveQueryResponse.

The response shape is a strict superset of Phase 4's QueryResponse
(three additional fields: query_type, rewritten_query, retrieval_strategy),
so existing clients that ignore unknown fields remain unaffected.

Responsibilities:
  - Accept a QueryRequest (question + optional top_k).
  - Wire up ChromaManager and AdaptiveRAGService.
  - Delegate all logic to AdaptiveRAGService.query().
  - Return an AdaptiveQueryResponse.
  - Translate known domain errors to appropriate HTTP responses.

Note:
  ChromaManager and AdaptiveRAGService are instantiated per-request.
  Phase 7 (API Completion) will lift these into FastAPI dependency
  injection for proper lifecycle management.
"""

from __future__ import annotations

import logging

import openai
from fastapi import APIRouter, HTTPException, status

from app.models.requests import QueryRequest
from app.models.responses import AdaptiveQueryResponse
from app.services.adaptive_rag_service import AdaptiveRAGService
from app.vectorstore.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/query",
    response_model=AdaptiveQueryResponse,
    summary="Ask a question against the knowledge base (adaptive retrieval).",
    description=(
        "Submit a natural-language question. The adaptive pipeline classifies "
        "the query, retrieves relevant document chunks from ChromaDB, evaluates "
        "retrieval confidence, and rewrites the query if confidence is low before "
        "generating a grounded answer with source citations. "
        "The response includes retrieval_strategy and query_type fields that "
        "expose what the adaptive pipeline decided."
    ),
)
async def query_endpoint(request: QueryRequest) -> AdaptiveQueryResponse:
    """
    POST /query

    Request body:
        question (str): The user's question. Required.
        top_k    (int): Number of chunks to retrieve. Optional, defaults to 5.

    Returns:
        AdaptiveQueryResponse with answer, citations, and adaptive metadata.

    Error responses:
        400 — Empty or invalid question.
        422 — Request body validation failure (handled by FastAPI automatically).
        429 — LLM provider rate limit hit.
        502 — LLM API authentication or connection error.
        500 — Unexpected internal error.
    """
    logger.info("POST /query | question='%s'", request.question[:120])

    try:
        chroma = ChromaManager()
        service = AdaptiveRAGService(chroma_manager=chroma, top_k=request.top_k)
        return service.query(question=request.question)

    except ValueError as exc:
        logger.warning("Invalid query request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except openai.RateLimitError as exc:
        logger.warning("LLM rate limit hit: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                "The LLM provider is temporarily rate-limited. "
                "Please wait a moment and try again. "
                f"(Provider detail: {exc})"
            ),
        )
    except openai.APIError as exc:
        logger.error("OpenAI/OpenRouter API error during query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM API error: {exc}",
        )
    except Exception as exc:
        logger.exception("Unexpected error during query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {exc}",
        )