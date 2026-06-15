"""
app/models/responses.py

Pydantic response models for all API endpoints.

Phase 3 additions:
- UploadResponse: now includes chunks_stored count (previously was a stub).
- CollectionStatsResponse: new model for the /health endpoint to report
  ChromaDB collection state.
- ChunkResult: represents a single retrieved chunk from ChromaDB.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Upload
# =============================================================================

class UploadResponse(BaseModel):
    """Response returned after a successful document ingestion."""

    filename: str = Field(..., description="Original filename of the uploaded document.")
    chunks_stored: int = Field(..., description="Number of chunks stored in ChromaDB.", ge=0)
    message: str = Field(..., description="Human-readable status message.")

    model_config = {"json_schema_extra": {
        "example": {
            "filename": "contract_a.pdf",
            "chunks_stored": 42,
            "message": "Successfully ingested 'contract_a.pdf' into the knowledge base.",
        }
    }}


# =============================================================================
# Query  (placeholder — Phase 4 will complete this)
# =============================================================================

class ChunkResult(BaseModel):
    """A single retrieved chunk from ChromaDB."""

    chunk_id: str = Field(..., description="Unique identifier for this chunk.")
    chunk_text: str = Field(..., description="Raw text of the chunk.")
    document_name: str = Field(..., description="Source document filename.")
    page_number: int = Field(0, description="Page number within the source document.")
    chunk_index: int = Field(0, description="Position of this chunk within the document.")
    relevance_score: float = Field(
        ..., description="Cosine similarity score (0–1, higher = more relevant).", ge=0.0, le=1.0
    )


class QueryResponse(BaseModel):
    """Response returned after a question-answering request."""

    question: str = Field(..., description="The original question asked by the user.")
    answer: str = Field(..., description="The generated answer.")
    sources: list[ChunkResult] = Field(
        default_factory=list, description="Chunks used to generate the answer."
    )
    model_config = {"json_schema_extra": {
        "example": {
            "question": "What is the termination clause?",
            "answer": "The contract may be terminated with 30 days' written notice …",
            "sources": [],
        }
    }}


# =============================================================================
# Evaluation  (placeholder — Phase 6 will complete this)
# =============================================================================

class EvaluationResponse(BaseModel):
    """Response returned after running the evaluation harness."""

    run_id: str = Field(..., description="Unique identifier for this evaluation run.")
    metrics: dict[str, Any] = Field(
        default_factory=dict, description="Evaluation metric results."
    )
    message: str = Field(..., description="Human-readable status message.")


# =============================================================================
# Health
# =============================================================================

class CollectionStatsResponse(BaseModel):
    """ChromaDB collection statistics returned by /health."""

    collection_name: str
    total_chunks: int
    persist_dir: str
    embedding_model: str


class HealthResponse(BaseModel):
    """Response returned by GET /health."""

    status: str = Field(..., description="'ok' when the service is healthy.")
    version: str = Field(..., description="API version string.")
    vector_store: Optional[CollectionStatsResponse] = Field(
        None, description="ChromaDB collection statistics."
    )

    model_config = {"json_schema_extra": {
        "example": {
            "status": "ok",
            "version": "0.1.0",
            "vector_store": {
                "collection_name": "adaptive_rag",
                "total_chunks": 142,
                "persist_dir": "./data/chroma_db",
                "embedding_model": "text-embedding-3-small",
            },
        }
    }}