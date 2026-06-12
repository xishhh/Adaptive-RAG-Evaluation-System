"""
app/models/responses.py

Pydantic response models for all API endpoints.

These models define and validate the shape of outgoing responses.
Using explicit response models ensures the API contract is clear and
prevents accidental leakage of internal fields.
"""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


# ── Health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = Field(description="Application status. 'ok' when healthy.")
    version: str = Field(description="Application version string.")
    environment: str = Field(description="Runtime environment (development/production).")
    timestamp: datetime = Field(description="UTC timestamp of the health check.")


# ── Upload ─────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Response for POST /upload."""

    filename: str = Field(description="Original filename of the uploaded document.")
    chunks_created: int = Field(description="Number of chunks produced from the document.")
    document_id: str = Field(description="Unique identifier assigned to the ingested document.")
    message: str = Field(description="Human-readable status message.")


# ── Query ──────────────────────────────────────────────────────────────────────

class CitationModel(BaseModel):
    """A single source citation attached to a query answer."""

    document_name: str = Field(description="Source document filename.")
    chunk_id: str = Field(description="Unique chunk identifier within the document.")
    page_number: int | None = Field(
        default=None,
        description="Page number in the source document, if available.",
    )
    chunk_text: str = Field(description="The raw chunk text used as context.")


class QueryResponse(BaseModel):
    """Response for POST /query."""

    question: str = Field(description="The original user question.")
    answer: str = Field(description="The generated, context-grounded answer.")
    citations: list[CitationModel] = Field(
        default_factory=list,
        description="Source chunks used to ground the answer.",
    )
    retrieval_strategy: str = Field(
        description="Strategy used: 'direct_llm', 'retrieval', or 'adaptive_retrieval'."
    )
    query_rewritten: bool = Field(
        default=False,
        description="Whether the query was rewritten before re-retrieval.",
    )
    rewritten_query: str | None = Field(
        default=None,
        description="The rewritten query, if query rewriting was triggered.",
    )


# ── Evaluate ───────────────────────────────────────────────────────────────────

class EvaluationMetrics(BaseModel):
    """Aggregate metrics from a single evaluation run."""

    recall_at_k: float | None = Field(default=None, description="Recall@K score.")
    precision_at_k: float | None = Field(default=None, description="Precision@K score.")
    faithfulness: float | None = Field(default=None, description="RAGAS faithfulness score.")
    answer_relevance: float | None = Field(default=None, description="RAGAS answer relevance score.")


class EvaluationResponse(BaseModel):
    """Response for POST /evaluate."""

    run_id: str = Field(description="Unique identifier for this evaluation run.")
    run_label: str | None = Field(default=None, description="Optional human-readable run label.")
    dataset_path: str = Field(description="Dataset used for this evaluation.")
    metrics: EvaluationMetrics = Field(description="Aggregate evaluation metrics.")
    sample_count: int = Field(description="Number of QA pairs evaluated.")
    report_path: str = Field(description="Path to the full JSON evaluation report.")
    timestamp: datetime = Field(description="UTC timestamp when evaluation completed.")


# ── Metrics ────────────────────────────────────────────────────────────────────

class MetricsSummaryResponse(BaseModel):
    """Response for GET /metrics."""

    total_runs: int = Field(description="Total number of evaluation runs recorded.")
    latest_run: EvaluationResponse | None = Field(
        default=None,
        description="Most recent evaluation run, if any.",
    )
    historical_runs: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Summary of all historical evaluation runs.",
    )