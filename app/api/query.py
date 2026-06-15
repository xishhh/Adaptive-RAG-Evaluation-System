"""
app/api/query.py

FastAPI router for POST /query.

Responsibilities:
  - Accept a QueryRequest (question + optional top_k).
  - Wire up ChromaManager and RAGService.
  - Delegate all logic to RAGService.query().
  - Return a QueryResponse.
  - Translate known domain errors to appropriate HTTP responses.

Design:
  ChromaManager and RAGService are instantiated per-request here for
  simplicity in Phase 4. Phase 7 (API Completion) will move these into
  FastAPI dependency injection for proper lifecycle management.
"""

from __future__ import annotations

import logging

import openai
from fastapi import APIRouter, HTTPException, status

from app.models.requests import QueryRequest
from app.models.responses import QueryResponse
from app.services.rag_service import RAGService
from app.vectorstore.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question against the knowledge base.",
    description=(
        "Submit a natural-language question. The system retrieves relevant "
        "document chunks from ChromaDB, generates a grounded answer using the "
        "configured LLM, and returns the answer with source citations."
    ),
)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    POST /query

    Request body:
        question (str):  The user's question. Required.
        top_k    (int):  Number of chunks to retrieve. Optional, defaults to 5.

    Returns:
        QueryResponse with answer text and source citations.

    Error responses:
        400 — Empty or invalid question.
        422 — Request body validation failure (handled by FastAPI).
        500 — Unexpected error during retrieval or LLM call.
    """
    logger.info("POST /query | question='%s'", request.question[:120])

    try:
        chroma = ChromaManager()
        service = RAGService(chroma_manager=chroma, top_k=request.top_k)
        return service.query(question=request.question)

    except ValueError as exc:
        logger.warning("Invalid query request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except openai.RateLimitError as exc:
        # The free-tier OpenRouter model is rate-limited upstream.
        # Surface the provider's message so the caller knows to retry.
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
        # Covers AuthenticationError, APIConnectionError, etc.
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