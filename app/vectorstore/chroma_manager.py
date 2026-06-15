"""
app/vectorstore/chroma_manager.py

Manages all ChromaDB interactions: collection lifecycle, embedding storage,
similarity search, and metadata filtering.

This module is the single point of contact between the application and ChromaDB.
No other module should import chromadb directly.
"""

import logging
from typing import Any, Optional

import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings

from app.utils.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


class ChromaManager:
    """
    Manages a persistent ChromaDB collection.

    Responsibilities:
    - Initialize and persist a ChromaDB collection.
    - Add document chunks with embeddings and metadata.
    - Perform similarity searches and return structured results.
    - Support metadata filtering for targeted retrieval.

    Design decisions:
    - Uses LangChain's OpenAIEmbeddings so the embedding model is consistent
      across ingestion and retrieval — preventing embedding space mismatches.
    - Persistence is always enabled; ephemeral collections are not production-safe.
    - The collection is created on first use and reused on subsequent starts.
    """

    def __init__(self) -> None:
        self._embedding_fn = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_base=settings.OPENAI_API_BASE,  # Override API Base
        )

        self._client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._get_or_create_collection()
        logger.info(
            "ChromaManager initialised | collection=%s | persist_dir=%s",
            settings.CHROMA_COLLECTION_NAME,
            settings.CHROMA_PERSIST_DIR,
        )

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    def _get_or_create_collection(self) -> chromadb.Collection:
        """
        Return the existing collection or create it if absent.

        ChromaDB's get_or_create_collection is idempotent — safe to call
        on every startup without destroying existing data.
        """
        return self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine distance for semantic search
        )

    def reset_collection(self) -> None:
        """
        Drop and recreate the collection.

        Use only during testing or when a full re-ingestion is required.
        NOT exposed via the API — must be called explicitly in code.
        """
        logger.warning(
            "Resetting collection '%s'. All data will be lost.",
            settings.CHROMA_COLLECTION_NAME,
        )
        try:
            self._client.delete_collection(settings.CHROMA_COLLECTION_NAME)
        except Exception:
                logger.warning(
                    "Collection '%s' did not exist or could not be deleted cleanly.",
                    settings.CHROMA_COLLECTION_NAME,
                )

        self._collection = self._get_or_create_collection()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """
        Embed and store a list of document chunks.

        Each chunk dict must contain:
            chunk_id      (str)  — globally unique identifier
            chunk_text    (str)  — raw text to embed
            document_name (str)  — source filename
            page_number   (int)  — page number, or 0 if unavailable
            chunk_index   (int)  — position within the document

        Returns:
            Number of chunks successfully added.

        Notes:
        - Chunks are embedded in a single batch call for efficiency.
        - ChromaDB uses `ids` as the deduplication key. Re-adding the same
          chunk_id will raise a DuplicateIDError; callers should guard against
          re-ingesting the same document without resetting first.
        """
        if not chunks:
            logger.warning("add_chunks called with empty list — skipping.")
            return 0

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            ids.append(chunk["chunk_id"])
            documents.append(chunk["chunk_text"])

            # Retrieve the full metadata dict from the Chunk object.
            chunk_metadata = chunk.get("metadata", {})

            # Combine the nested metadata with the explicit chunk fields,
            # then sanitize so only ChromaDB-safe scalar values remain.
            combined_metadata = {
                **chunk_metadata,
                "document_name": chunk["document_name"],
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk.get("page_number") or 0,
                "chunk_index": chunk.get("chunk_index", 0),
                "chunk_text": chunk["chunk_text"],
            }
            metadatas.append(self._sanitize_metadata(combined_metadata))

            

        logger.info("Generating embeddings for %d chunks …", len(chunks))
        embeddings = self._embedding_fn.embed_documents(documents)

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info("Stored %d chunks in ChromaDB.", len(chunks))
        return len(chunks)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Perform a cosine-similarity search against the collection.

        Args:
            query:           Natural-language query string.
            top_k:           Maximum number of results to return.
            filter_metadata: Optional ChromaDB `where` filter dict.
                             Example: {"document_name": "contract_a.pdf"}

        Returns:
            List of result dicts, each containing:
                chunk_id      (str)
                chunk_text    (str)
                document_name (str)
                page_number   (int)
                chunk_index   (int)
                distance      (float)  — lower = more similar (cosine distance)
                relevance_score (float) — 1 - distance, higher = more similar

        Raises:
            ValueError: If the collection is empty and a search is attempted.
        """
        count = self._collection.count()
        if count == 0:
            logger.warning("Similarity search on empty collection.")
            return []

        query_embedding = self._embedding_fn.embed_query(query)

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, count),  # guard against top_k > collection size
            "include": ["metadatas", "documents", "distances"],
        }
        if filter_metadata:
            query_kwargs["where"] = filter_metadata

        results = self._collection.query(**query_kwargs)

        return self._parse_query_results(results)

    def get_chunk_by_id(self, chunk_id: str) -> Optional[dict[str, Any]]:
        """
        Fetch a single chunk by its exact ID.

        Returns None if the chunk does not exist.
        """
        result = self._collection.get(
            ids=[chunk_id],
            include=["metadatas", "documents"],
        )
        if not result["ids"] or not result["ids"][0]:
            return None

        return {
            "chunk_id": result["ids"][0],
            "chunk_text": result["documents"][0],
            **result["metadatas"][0],
        }

    def collection_stats(self) -> dict[str, Any]:
        """
        Return basic statistics about the current collection.

        Useful for the GET /health endpoint and operational monitoring.
        """
        count = self._collection.count()
        return {
            "collection_name": settings.CHROMA_COLLECTION_NAME,
            "total_chunks": count,
            "persist_dir": settings.CHROMA_PERSIST_DIR,
            "embedding_model": settings.EMBEDDING_MODEL,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Ensure all metadata values are ChromaDB-safe scalars.

        ChromaDB only accepts str, int, float, and bool as metadata values.
        Lists (e.g. page_char_offsets, sheet_names), dicts, None, and any
        other type will cause a ValueError at insertion time.

        Strategy:
        - None      → 0 (for numeric fields) or "" (for string fields)
        - list/dict → JSON-serialised string
        - Everything else → str(value)
        """
        import json

        safe: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                safe[key] = value
            elif value is None:
                # Keep numeric sentinel keys as int 0, everything else as "".
                safe[key] = 0 if key in ("page_number", "chunk_index") else ""
            elif isinstance(value, (list, dict)):
                safe[key] = json.dumps(value)
            else:
                safe[key] = str(value)
        return safe

    @staticmethod
    def _parse_query_results(raw: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Flatten ChromaDB's nested query response into a clean list of dicts.

        ChromaDB returns results as parallel arrays wrapped in an extra list
        (one per query). Since we always issue single-query requests, we
        unwrap the outer list before processing.
        """
        parsed: list[dict[str, Any]] = []

        # ChromaDB wraps results in a list — one element per query embedding.
        ids = raw.get("ids", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for chunk_id, metadata, document, distance in zip(
            ids, metadatas, documents, distances
        ):
            relevance_score = round(max(0.0, min(1.0, 1.0 - distance)), 6)
            parsed.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_text": document,
                    "document_name": metadata.get("document_name", "unknown"),
                    "page_number": metadata.get("page_number", 0),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "distance": round(distance, 6),
                    "relevance_score": relevance_score,
                }
            )

        # Sort by relevance descending (highest similarity first).
        parsed.sort(key=lambda x: x["relevance_score"], reverse=True)
        return parsed