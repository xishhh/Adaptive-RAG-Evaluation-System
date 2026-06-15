"""
Internal document models used by ingestion and vector storage.
"""

from typing import Any

from pydantic import BaseModel, Field


class RawDocument(BaseModel):
    document_name: str
    file_type: str
    full_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    document_name: str
    chunk_text: str
    page_number: int | None = None
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)