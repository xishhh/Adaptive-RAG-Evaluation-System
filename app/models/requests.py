"""
Pydantic request models for the Adaptive RAG API.

These models define and validate the shape of incoming HTTP request
payloads. They are intentionally kept separate from internal data
models (see responses.py and the ingestion package).
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Payload for POST /query."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural language question to answer.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve.",
    )


class EvaluateRequest(BaseModel):
    """Payload for POST /evaluate."""

    dataset_path: str = Field(
        ...,
        description="Path to the JSONL evaluation dataset file.",
    )
    run_label: str = Field(
        default="default",
        description="Human-readable label for this evaluation run.",
    )