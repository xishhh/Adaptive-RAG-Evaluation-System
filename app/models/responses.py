from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    job_id: str = Field(..., description="Unique identifier for this ingestion job.")
    filename: str = Field(..., description="Original filename of the uploaded document.")
    status: str = Field(..., description="Processing status.", examples=["processing"])

    model_config = {"json_schema_extra": {
        "example": {"job_id": "a1b2c3d4e5f6...", "filename": "contract_a.pdf", "status": "processing"}
    }}


class IngestionStatusResponse(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the ingestion job.")
    status: str = Field(..., description="Current processing status.", examples=["processing", "completed", "failed"])
    error: Optional[str] = Field(None, description="Error message if the job failed.")


class ChunkResult(BaseModel):
    chunk_id: str = Field(..., description="Unique identifier for this chunk.")
    chunk_text: str = Field(..., description="Raw text of the chunk.")
    document_name: str = Field(..., description="Source document filename.")
    page_number: int = Field(0, description="Page number within the source document.")
    chunk_index: int = Field(0, description="Position of this chunk within the document.")
    relevance_score: float = Field(
        ..., description="Cosine similarity score (0–1, higher = more relevant).", ge=0.0, le=1.0
    )


class QueryResponse(BaseModel):
    question: str = Field(..., description="The original question asked by the user.")
    answer: str = Field(..., description="The generated answer.")
    sources: list[ChunkResult] = Field(default_factory=list, description="Chunks used to generate the answer.")

    model_config = {"json_schema_extra": {
        "example": {
            "question": "What is the termination clause?",
            "answer": "The contract may be terminated with 30 days' written notice ...",
            "sources": [],
        }
    }}


class AdaptiveQueryResponse(QueryResponse):
    query_type: Literal["DIRECT_LLM", "KNOWLEDGE_QUERY"] = Field(
        ..., description="Classification of the incoming query."
    )
    rewritten_query: Optional[str] = Field(
        None, description="The expanded query from QueryRewriter if confidence was low."
    )
    retrieval_strategy: str = Field(
        ..., description="Short label describing the adaptive path taken."
    )

    model_config = {"json_schema_extra": {
        "example": {
            "question": "What penalties exist for delayed delivery?",
            "answer": "According to contract_a.pdf, a penalty of 0.5% per day applies ...",
            "sources": [],
            "query_type": "KNOWLEDGE_QUERY",
            "rewritten_query": "Penalty clauses, liquidated damages, delayed delivery penalties",
            "retrieval_strategy": "rewritten_retrieval",
        }
    }}


class EvaluationResponse(BaseModel):
    run_id: str = Field(..., description="Unique identifier for this evaluation run.")
    run_label: str = Field(..., description="Human-readable label supplied in the request.")
    dataset_path: str = Field(..., description="Path of the JSONL dataset that was evaluated.")
    sample_count: int = Field(..., description="Number of samples evaluated.", ge=0)
    created_at: str = Field(..., description="ISO-8601 UTC timestamp of when the run completed.")
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="RAGAS metric scores. Values: float 0.0–1.0 or null.",
    )
    message: str = Field(..., description="Human-readable status message.")

    model_config = {"json_schema_extra": {
        "example": {
            "run_id": "20240615T143022_contract_review",
            "run_label": "contract_review",
            "dataset_path": "data/evaluation_dataset/sample_eval.jsonl",
            "sample_count": 10,
            "created_at": "2024-06-15T14:30:22+00:00",
            "metrics": {"context_precision": 0.85, "context_recall": 0.78, "faithfulness": 0.91, "answer_relevancy": 0.87},
            "message": "Evaluation run 'contract_review' completed successfully.",
        }
    }}


class EvaluationRunRecord(BaseModel):
    run_id: str
    run_label: str
    dataset_path: str
    sample_count: int
    created_at: str
    metrics: dict[str, Any]


class MetricsSummaryResponse(BaseModel):
    total_runs: int = Field(..., description="Total number of evaluation runs recorded.")
    aggregate_metrics: dict[str, Any] = Field(
        default_factory=dict, description="Mean score per metric across all historical runs."
    )
    runs: list[EvaluationRunRecord] = Field(
        default_factory=list, description="Most recent evaluation runs, newest first (up to 50)."
    )

    model_config = {"json_schema_extra": {
        "example": {
            "total_runs": 3,
            "aggregate_metrics": {"context_precision": 0.82, "context_recall": 0.75, "faithfulness": 0.89, "answer_relevancy": 0.84},
            "runs": [],
        }
    }}


class CollectionStatsResponse(BaseModel):
    collection_name: str
    total_chunks: int
    persist_dir: str
    embedding_model: str


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' when the service is healthy.")
    version: str = Field(..., description="API version string.")
    vector_store: Optional[CollectionStatsResponse] = Field(None, description="ChromaDB collection statistics.")

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
