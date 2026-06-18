"""
app/api/query.py

FastAPI routers for POST /query and POST /query/stream.

Phase 4: Used RAGService → QueryResponse.
Phase 5: Upgraded to AdaptiveRAGService → AdaptiveQueryResponse.
Phase 8: Added /query/stream for SSE token streaming.

The response shape of /query is a strict superset of Phase 4's QueryResponse
(three additional fields: query_type, rewritten_query, retrieval_strategy),
so existing clients that ignore unknown fields remain unaffected.

Responsibilities:
  - Accept a QueryRequest (question + optional top_k).
  - /query:       Delegates to AdaptiveRAGService.query() → AdaptiveQueryResponse.
  - /query/stream: Delegates to AdaptiveRAGService.stream_query() → SSE stream.
  - Translate known domain errors to appropriate HTTP responses.
"""

from __future__ import annotations

import json
import logging

import openai
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_adaptive_rag_service
from app.models.requests import QueryRequest
from app.models.responses import AdaptiveQueryResponse
from app.services.adaptive_rag_service import AdaptiveRAGService

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
async def query_endpoint(
    request: QueryRequest,
    service: AdaptiveRAGService = Depends(get_adaptive_rag_service),
) -> AdaptiveQueryResponse:
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
        return service.query(question=request.question, top_k=request.top_k)

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


@router.post(
    "/query/stream",
    summary="Ask a question and stream the answer via SSE",
    description=(
        "Same adaptive RAG pipeline as POST /query, but the answer is "
        "streamed token-by-token using Server-Sent Events. "
        "Events:\n"
        "  - metadata: query_type, sources, rewritten_query, retrieval_strategy\n"
        "  - token:    a single chunk of generated answer text\n"
        "  - error:    a pipeline error (terminal)\n"
        "  - done:     signals completion"
    ),
)
async def query_stream_endpoint(
    request: QueryRequest,
    service: AdaptiveRAGService = Depends(get_adaptive_rag_service),
) -> StreamingResponse:
    """
    POST /query/stream

    Same request body as POST /query. Returns a Server-Sent Events stream
    instead of a JSON body. The client should listen for:
      - ``event: metadata``  (received once, before tokens)
      - ``event: token``     (received 0..N times)
      - ``event: error``     (terminal, if the pipeline fails)
      - ``event: done``      (terminal, on success)

    Example client (JavaScript):
        const es = new EventSource("/query/stream");
        es.addEventListener("token", (e) => console.log(e.data));
        es.addEventListener("done", (e) => es.close());
    """
    logger.info("POST /query/stream | question='%s'", request.question[:120])

    def event_stream():
        """
        Synchronous generator that yields SSE-formatted lines.

        Yields:
            Lines in SSE format: event + data fields, double-newline separated.
        """
        for event in service.stream_query(
            question=request.question,
            top_k=request.top_k,
        ):
            data = json.dumps(event["data"], ensure_ascii=False, default=str)
            yield f"event: {event['event']}\ndata: {data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )