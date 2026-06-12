"""
app/models/requests.py

Pydantic request models for all API endpoints.

These models define and validate the shape of incoming request bodies.
They are imported by route handlers in app/api/.

File upload requests are handled with FastAPI's UploadFile directly
(multipart form data), so no Pydantic model is needed for /upload.
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Request body for POST /query.

    Attributes:
        question:   The user's natural language question.
        top_k:      Number of chunks to retrieve. Overrides the default
                    from settings when provided.
        use_adaptive: Whether to use the adaptive retrieval pipeline
                    (classification, confidence scoring, rewriting).
                    Defaults to True.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's natural language question.",
        examples=["What are the termination clauses in Contract A?"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of chunks to retrieve. Defaults to RETRIEVAL_TOP_K from settings.",
    )
    use_adaptive: bool = Field(
        default=True,
        description="Enable adaptive retrieval (query classification, rewriting, re-retrieval).",
    )


class EvaluationRequest(BaseModel):
    """
    Request body for POST /evaluate.

    Attributes:
        dataset_path: Path (relative to data/evaluation_dataset/) of the
                      evaluation dataset file to run against.
        run_label:    Optional human-readable label for this evaluation run.
    """

    dataset_path: str = Field(
        ...,
        description="Path to the evaluation dataset file, relative to data/evaluation_dataset/.",
        examples=["qa_pairs.json"],
    )
    run_label: str | None = Field(
        default=None,
        max_length=100,
        description="Optional label for this evaluation run.",
    )