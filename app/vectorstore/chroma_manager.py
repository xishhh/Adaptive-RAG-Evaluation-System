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
        logger.info("ChromaManager initialised | collection=%s | persist_dir=%s", settings.CHROMA_COLLECTION_NAME, settings.CHROMA_PERSIST_DIR)

    def _get_or_create_collection(self) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def reset_collection(self) -> None:
        logger.warning("Resetting collection '%s'. All data will be lost.", settings.CHROMA_COLLECTION_NAME)
        try:
            self._client.delete_collection(settings.CHROMA_COLLECTION_NAME)
        except Exception:
            logger.warning("Collection '%s' did not exist or could not be deleted cleanly.", settings.CHROMA_COLLECTION_NAME)

        self._collection = self._get_or_create_collection()

    def delete_by_document_name(self, document_name: str) -> int:
        if not document_name:
            return 0

        result = self._collection.get(
            where={"document_name": document_name},
            include=[],
        )

        ids_to_delete: list[str] = result.get("ids", [])
        if not ids_to_delete:
            logger.debug("delete_by_document_name: no chunks found for '%s'.", document_name)
            return 0

        self._collection.delete(ids=ids_to_delete)
        logger.info("Deleted %d existing chunk(s) for document '%s'.", len(ids_to_delete), document_name)
        return len(ids_to_delete)

    def add_chunks(self, chunks: list[dict[str, Any]]) -> int:
        if not chunks:
            logger.warning("add_chunks called with empty list — skipping.")
            return 0

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            ids.append(chunk["chunk_id"])
            documents.append(chunk["chunk_text"])

            chunk_metadata = chunk.get("metadata", {})

            combined_metadata = {
                **chunk_metadata,
                "document_name": chunk["document_name"],
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk.get("page_number") or 0,
                "chunk_index": chunk.get("chunk_index", 0),
            }
            metadatas.append(self._sanitize_metadata(combined_metadata))

        logger.info("Generating embeddings for %d chunks ...", len(chunks))
        embeddings = self._embedding_fn.embed_documents(documents)

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info("Stored %d chunks in ChromaDB.", len(chunks))
        return len(chunks)

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        count = self._collection.count()
        if count == 0:
            logger.warning("Similarity search on empty collection.")
            return []

        query_embedding = self._embedding_fn.embed_query(query)

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, count),
            "include": ["metadatas", "documents", "distances"],
        }
        if filter_metadata:
            query_kwargs["where"] = filter_metadata

        results = self._collection.query(**query_kwargs)

        return self._parse_query_results(results)

    def collection_stats(self) -> dict[str, Any]:
        count = self._collection.count()
        return {
            "collection_name": settings.CHROMA_COLLECTION_NAME,
            "total_chunks": count,
            "persist_dir": settings.CHROMA_PERSIST_DIR,
            "embedding_model": settings.EMBEDDING_MODEL,
        }

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                safe[key] = value
            elif value is None:
                safe[key] = 0 if key in ("page_number", "chunk_index") else ""
            elif isinstance(value, (list, dict)):
                safe[key] = json.dumps(value)
            else:
                safe[key] = str(value)
        return safe

    @staticmethod
    def _parse_query_results(raw: dict[str, Any]) -> list[dict[str, Any]]:
        parsed: list[dict[str, Any]] = []

        ids = raw.get("ids", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for chunk_id, metadata, document, distance in zip(ids, metadatas, documents, distances):
            relevance_score = round(max(0.0, min(1.0, 1.0 - distance)), 6)
            parsed.append({
                "chunk_id": chunk_id,
                "chunk_text": document,
                "document_name": metadata.get("document_name", "unknown"),
                "page_number": metadata.get("page_number", 0),
                "chunk_index": metadata.get("chunk_index", 0),
                "distance": round(distance, 6),
                "relevance_score": relevance_score,
            })

        parsed.sort(key=lambda x: x["relevance_score"], reverse=True)
        return parsed
