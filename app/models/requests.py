"""
Pydantic request models for the Adaptive RAG API.

These models define and validate the shape of incoming HTTP request
payloads. They are intentionally kept separate from internal data
models (see responses.py and the ingestion package).
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    question: str = Field(
        ...,
        min_length=1,
        description="The natural-language question to answer.",
        examples=["What is the termination clause in contract A?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve from the vector store.",
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