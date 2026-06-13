"""
Pydantic models for API responses AND internal data structures.

Internal models (RawDocument, Chunk) live here because they are
consumed by multiple layers (ingestion, vectorstore, retrieval) and
do not belong exclusively to any single package.

API response models (UploadResponse, QueryResponse, etc.) define
the shape of HTTP responses returned by the FastAPI routes.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field
from datetime import datetime


# ======================================================================= #
# Internal data models                                                      #
# ======================================================================= #


class RawDocument(BaseModel):
    """
    Represents a document after loading and text extraction,
    before chunking.

    Produced by: app/ingestion/loaders.py
    Consumed by: app/ingestion/chunker.py
    """

    document_name: str = Field(
        ...,
        description="Original filename, e.g. 'contract_2024.pdf'.",
    )
    file_type: str = Field(
        ...,
        description="Lowercase file extension without dot, e.g. 'pdf'.",
    )
    full_text: str = Field(
        ...,
        description="Complete extracted text content of the document.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arbitrary key-value pairs extracted alongside the text. "
            "For PDFs this may include page count; for XLSX, sheet names."
        ),
    )


class Chunk(BaseModel):
    """
    Represents a single text chunk ready for embedding and storage.

    Produced by: app/ingestion/chunker.py
    Consumed by: app/ingestion/embeddings.py (Phase 3)
                 app/vectorstore/chroma_manager.py (Phase 3)

    The schema matches exactly what SYSTEM_ARCHITECTURE.md
    Component 2 specifies for stored metadata.
    """

    chunk_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this chunk.",
    )
    document_name: str = Field(
        ...,
        description="Source document filename.",
    )
    chunk_text: str = Field(
        ...,
        description="Text content of this chunk.",
    )
    page_number: int | None = Field(
        default=None,
        description=(
            "Page number this chunk originated from, if determinable. "
            "None for formats without page structure (e.g. TXT, XLSX)."
        ),
    )
    chunk_index: int = Field(
        ...,
        ge=0,
        description="Zero-based position of this chunk within its source document.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata propagated from the source RawDocument.",
    )


# ======================================================================= #
# API response models                                                       #
# ======================================================================= #


class UploadResponse(BaseModel):
    """Response for POST /upload."""

    message: str
    document_name: str
    chunks_created: int
    file_type: str


class Citation(BaseModel):
    """A single source citation included in a query answer."""

    document_name: str
    chunk_id: str
    page_number: int | None = None
    excerpt: str = Field(description="Short excerpt from the source chunk.")


class QueryResponse(BaseModel):
    """Response for POST /query."""

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieval_strategy: str = Field(
        description=(
            "Which retrieval path was used: 'direct_llm', "
            "'standard_retrieval', or 'adaptive_retrieval'."
        )
    )


class EvaluationMetrics(BaseModel):
    """Aggregate metrics from one evaluation run."""

    recall_at_k: float | None = None
    precision_at_k: float | None = None
    faithfulness: float | None = None
    answer_relevance: float | None = None


class EvaluateResponse(BaseModel):
    """Response for POST /evaluate."""

    run_label: str
    dataset_path: str
    metrics: EvaluationMetrics
    report_path: str = Field(description="Path to the saved JSON report.")


class MetricsResponse(BaseModel):
    """Response for GET /metrics."""

    runs: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of historical evaluation run summaries.",
    )


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = "ok"
    version: str = "1.0.0"
    environment: str
    timestamp: datetime