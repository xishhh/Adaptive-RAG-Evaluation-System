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
    description="Submit a question. The adaptive pipeline classifies the query, retrieves "
    "relevant chunks, evaluates confidence (rewriting if low), and returns a grounded answer.",
)
async def query_endpoint(
    request: QueryRequest,
    service: AdaptiveRAGService = Depends(get_adaptive_rag_service),
) -> AdaptiveQueryResponse:
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
            detail=f"The LLM provider is temporarily rate-limited. ({exc})",
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
    description="Same as POST /query but streams answer tokens via Server-Sent Events.",
)
async def query_stream_endpoint(
    request: QueryRequest,
    service: AdaptiveRAGService = Depends(get_adaptive_rag_service),
) -> StreamingResponse:
    logger.info("POST /query/stream | question='%s'", request.question[:120])

    def event_stream():
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
