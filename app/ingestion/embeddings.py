"""
app/ingestion/embeddings.py

Provides the EmbeddingService used by the ingestion pipeline to convert
document chunks into vector representations before storing them in ChromaDB.

Design decisions:
- This module is responsible ONLY for generating embeddings.
- Storage is delegated to ChromaManager — this preserves separation of concerns.
- Batch processing is used to minimise API round-trips.
- The OpenAI embedding model is configurable via environment variables.
"""

import logging
from typing import Any

from langchain_openai import OpenAIEmbeddings

from app.utils.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


class EmbeddingService:
    """
    Wraps LangChain's OpenAIEmbeddings for document chunk embedding.

    Responsibilities:
    - Embed a list of text strings in a single batched API call.
    - Embed a single query string for retrieval.
    - Attach embeddings back onto chunk dicts for downstream use.

    This service does NOT interact with ChromaDB.
    ChromaManager calls embed_documents internally when add_chunks() is called,
    so direct use of EmbeddingService is only needed when you want the raw
    vectors before storage (e.g. for debugging or pre-processing pipelines).
    """

    def __init__(self) -> None:
        self._model = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_API_BASE,  # Override API Base
        )

        logger.info(
            "EmbeddingService initialised | model=%s", settings.EMBEDDING_MODEL
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of text strings.

        Args:
            texts: Plain text strings to embed. Typically chunk_text values.

        Returns:
            List of float vectors, one per input string.
            Order is preserved — output[i] corresponds to texts[i].

        Notes:
        - LangChain handles OpenAI's batch size limits internally.
        - Empty strings in the input will produce a valid (but meaningless)
          embedding — callers should filter these out before calling.
        """
        if not texts:
            logger.warning("embed_documents called with empty list.")
            return []

        logger.info("Embedding %d document chunks …", len(texts))
        vectors = self._model.embed_documents(texts)
        logger.info("Embedding complete | dimensions=%d", len(vectors[0]) if vectors else 0)
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string for similarity search.

        OpenAI uses a slightly different instruction for query embeddings
        vs document embeddings internally — LangChain handles this correctly
        by calling embed_query() rather than embed_documents() for retrieval.

        Args:
            query: The user's natural-language query.

        Returns:
            A single float vector representing the query.
        """
        if not query or not query.strip():
            raise ValueError("Query string must not be empty.")

        logger.debug("Embedding query: '%s'", query[:80])
        return self._model.embed_query(query)

    def embed_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Attach embedding vectors to a list of chunk dicts in-place.

        This is a convenience method for pipelines that need the embedding
        vectors attached to chunk metadata before storage.

        Args:
            chunks: List of chunk dicts containing at minimum a `chunk_text` key.

        Returns:
            The same list with an `embedding` key added to each dict.

        Example input chunk:
            {
                "chunk_id": "doc1_chunk_0",
                "chunk_text": "This is the first paragraph …",
                "document_name": "contract.pdf",
                "page_number": 1,
                "chunk_index": 0,
            }

        Example output chunk (same dict, mutated):
            {
                "chunk_id": "doc1_chunk_0",
                "chunk_text": "This is the first paragraph …",
                "document_name": "contract.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "embedding": [0.012, -0.034, …],  # 1536 floats for text-embedding-3-small
            }
        """
        texts = [chunk["chunk_text"] for chunk in chunks]
        vectors = self.embed_documents(texts)

        for chunk, vector in zip(chunks, vectors):
            chunk["embedding"] = vector

        return chunks