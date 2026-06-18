"""
app/vectorstore/chroma_manager.py

Manages all ChromaDB interactions: collection lifecycle, embedding storage,
similarity search, and metadata filtering.

This module is the single point of contact between the application and ChromaDB.
No other module should import chromadb directly.

Fix #5 — Re-upload creates duplicate vector entries:
  Added delete_by_document_name(document_name) which removes all existing
  chunks for a given document before a re-ingest. Called from upload.py
  before add_chunks() so re-uploads replace rather than stack.

Fix #11 — chunk_text duplicated in Chroma metadata:
  Removed "chunk_text" from combined_metadata inside add_chunks(). The text
  is already stored in the `documents` field of the ChromaDB record. Storing
  it again in metadata doubles the on-disk size per chunk and can cause
  Chroma metadata-size rejections on large chunks.
"""

import json
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
    - Delete all chunks for a given document (for idempotent re-upload).
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
            openai_api_base=settings.OPENAI_API_BASE,
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

    def delete_by_document_name(self, document_name: str) -> int:
        """
        Delete all chunks belonging to the given document.

        Fix #5: Called from upload.py before add_chunks() so that
        re-uploading the same file replaces its vectors rather than
        creating a second set, which would cause the same passages to
        be retrieved multiple times and degrade answer quality.

        Args:
            document_name: The user-facing filename (e.g. "contract_a.pdf").
                           Must match the value stored in the `document_name`
                           metadata field during ingestion.

        Returns:
            Number of chunks deleted (0 if the document was not found).
        """
        if not document_name:
            return 0

        # Fetch all IDs for this document_name via a metadata filter.
        # We use get() instead of query() because we don't need embeddings —
        # we only need the IDs to delete.
        result = self._collection.get(
            where={"document_name": document_name},
            include=[],  # IDs only — no need to fetch text or embeddings
        )

        ids_to_delete: list[str] = result.get("ids", [])
        if not ids_to_delete:
            logger.debug(
                "delete_by_document_name: no chunks found for '%s'.", document_name
            )
            return 0

        self._collection.delete(ids=ids_to_delete)
        logger.info(
            "Deleted %d existing chunk(s) for document '%s'.",
            len(ids_to_delete),
            document_name,
        )
        return len(ids_to_delete)

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
        - Fix #11: chunk_text is NOT stored in metadata. It is already stored
          in the `documents` field of the ChromaDB record. Storing it twice
          doubled on-disk size and risked hitting Chroma's metadata size limits.
        - Callers should call delete_by_document_name() before this method
          if the document may already exist (Fix #5).
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

            # Retrieve the nested metadata dict from the Chunk object.
            chunk_metadata = chunk.get("metadata", {})

            # Combine nested metadata with explicit chunk fields.
            # Fix #11: "chunk_text" is intentionally excluded from metadata.
            # It lives in the `documents` field; duplicating it here wastes
            # storage and can exceed ChromaDB's per-record metadata size limit.
            combined_metadata = {
                **chunk_metadata,
                "document_name": chunk["document_name"],
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk.get("page_number") or 0,
                "chunk_index": chunk.get("chunk_index", 0),
                # chunk_text deliberately omitted — stored in `documents` field
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
                chunk_id        (str)
                chunk_text      (str)
                document_name   (str)
                page_number     (int)
                chunk_index     (int)
                distance        (float) — lower = more similar (cosine distance)
                relevance_score (float) — 1 - distance, higher = more similar
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