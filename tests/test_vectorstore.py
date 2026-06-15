"""
tests/test_vectorstore.py

Integration tests for Phase 3: ChromaDB storage and similarity search.

These tests use an in-memory / temp-dir ChromaDB instance so they never
touch the production persistence directory.

Run with:
    pytest tests/test_vectorstore.py -v
"""

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.vectorstore.chroma_manager import ChromaManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_chroma_dir(tmp_path: Path) -> Path:
    """Provide an isolated temp directory for each test's ChromaDB instance."""
    return tmp_path / "chroma_test"


@pytest.fixture
def mock_settings(temp_chroma_dir: Path):
    """
    Patch get_settings() so ChromaManager uses a temp directory
    and a unique collection name — guarantees test isolation.
    """
    collection_name = f"test_{uuid.uuid4().hex[:8]}"
    with patch("app.vectorstore.chroma_manager.settings") as mock_s:
        mock_s.CHROMA_PERSIST_DIR = str(temp_chroma_dir)
        mock_s.CHROMA_COLLECTION_NAME = collection_name
        mock_s.EMBEDDING_MODEL = "text-embedding-3-small"
        mock_s.OPENAI_API_KEY = "sk-test-key"
        yield mock_s


@pytest.fixture
def mock_embeddings():
    """
    Patch OpenAIEmbeddings so tests don't make real API calls.

    Returns deterministic 4-dimensional vectors for predictability.
    """
    with patch("app.vectorstore.chroma_manager.OpenAIEmbeddings") as mock_cls:
        instance = MagicMock()
        # embed_documents returns one vector per input string.
        instance.embed_documents.side_effect = lambda texts: [
            [float(i), 0.1, 0.2, 0.3] for i, _ in enumerate(texts)
        ]
        # embed_query returns a single vector.
        instance.embed_query.return_value = [0.0, 0.1, 0.2, 0.3]
        mock_cls.return_value = instance
        yield instance


def make_chunk(
    document_name: str = "test_doc.pdf",
    chunk_index: int = 0,
    page_number: int = 1,
    text: str = "Sample chunk text for testing purposes.",
) -> dict:
    """Build a valid chunk dict for testing."""
    return {
        "chunk_id": f"{document_name}_chunk_{chunk_index}",
        "chunk_text": text,
        "document_name": document_name,
        "page_number": page_number,
        "chunk_index": chunk_index,
    }


# ---------------------------------------------------------------------------
# Tests: add_chunks
# ---------------------------------------------------------------------------

class TestAddChunks:
    def test_add_single_chunk(self, mock_settings, mock_embeddings):
        manager = ChromaManager()
        chunk = make_chunk(text="The contract terminates after 30 days notice.")
        count = manager.add_chunks([chunk])
        assert count == 1

    def test_add_multiple_chunks(self, mock_settings, mock_embeddings):
        manager = ChromaManager()
        chunks = [make_chunk(chunk_index=i, text=f"Chunk text {i}") for i in range(5)]
        count = manager.add_chunks(chunks)
        assert count == 5

    def test_add_empty_list_returns_zero(self, mock_settings, mock_embeddings):
        manager = ChromaManager()
        count = manager.add_chunks([])
        assert count == 0

    def test_embeddings_called_once_per_batch(self, mock_settings, mock_embeddings):
        manager = ChromaManager()
        chunks = [make_chunk(chunk_index=i) for i in range(3)]
        manager.add_chunks(chunks)
        # embed_documents should be called exactly once with all texts together.
        mock_embeddings.embed_documents.assert_called_once()
        call_args = mock_embeddings.embed_documents.call_args[0][0]
        assert len(call_args) == 3


# ---------------------------------------------------------------------------
# Tests: similarity_search
# ---------------------------------------------------------------------------

class TestSimilaritySearch:
    def test_search_empty_collection_returns_empty_list(
        self, mock_settings, mock_embeddings
    ):
        manager = ChromaManager()
        results = manager.similarity_search("termination clause")
        assert results == []

    def test_search_returns_results_after_add(self, mock_settings, mock_embeddings):
        manager = ChromaManager()
        chunks = [
            make_chunk(chunk_index=0, text="Termination requires 30 days notice."),
            make_chunk(chunk_index=1, text="Payment is due within 14 days."),
        ]
        manager.add_chunks(chunks)
        results = manager.similarity_search("termination clause", top_k=2)
        assert len(results) >= 1

    def test_search_result_schema(self, mock_settings, mock_embeddings):
        manager = ChromaManager()
        chunks = [make_chunk(chunk_index=0, text="Force majeure clause details.")]
        manager.add_chunks(chunks)
        results = manager.similarity_search("force majeure")
        assert len(results) == 1
        result = results[0]
        # Verify all expected keys are present.
        assert "chunk_id" in result
        assert "chunk_text" in result
        assert "document_name" in result
        assert "page_number" in result
        assert "chunk_index" in result
        assert "distance" in result
        assert "relevance_score" in result

    def test_search_top_k_respected(self, mock_settings, mock_embeddings):
        manager = ChromaManager()
        chunks = [make_chunk(chunk_index=i, text=f"Clause {i} details.") for i in range(10)]
        manager.add_chunks(chunks)
        results = manager.similarity_search("clause", top_k=3)
        assert len(results) <= 3

    def test_relevance_score_between_0_and_1(self, mock_settings, mock_embeddings):
        manager = ChromaManager()
        chunks = [make_chunk(text="Indemnification obligations of both parties.")]
        manager.add_chunks(chunks)
        results = manager.similarity_search("indemnification")
        for r in results:
            assert 0.0 <= r["relevance_score"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: get_chunk_by_id
# ---------------------------------------------------------------------------

class TestGetChunkById:
    def test_get_existing_chunk(self, mock_settings, mock_embeddings):
        manager = ChromaManager()
        chunk = make_chunk(chunk_index=0, text="Liability cap is $1,000,000.")
        manager.add_chunks([chunk])
        result = manager.get_chunk_by_id(chunk["chunk_id"])
        assert result is not None
        assert result["chunk_id"] == chunk["chunk_id"]

    def test_get_nonexistent_chunk_returns_none(self, mock_settings, mock_embeddings):
        manager = ChromaManager()
        result = manager.get_chunk_by_id("nonexistent_id_xyz")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: collection_stats
# ---------------------------------------------------------------------------

class TestCollectionStats:
    def test_stats_empty_collection(self, mock_settings, mock_embeddings):
        manager = ChromaManager()
        stats = manager.collection_stats()
        assert stats["total_chunks"] == 0
        assert "collection_name" in stats
        assert "persist_dir" in stats
        assert "embedding_model" in stats

    def test_stats_count_after_add(self, mock_settings, mock_embeddings):
        manager = ChromaManager()
        chunks = [make_chunk(chunk_index=i) for i in range(7)]
        manager.add_chunks(chunks)
        stats = manager.collection_stats()
        assert stats["total_chunks"] == 7


# ---------------------------------------------------------------------------
# Tests: reset_collection
# ---------------------------------------------------------------------------

class TestResetCollection:
    def test_reset_clears_all_chunks(self, mock_settings, mock_embeddings):
        manager = ChromaManager()
        chunks = [make_chunk(chunk_index=i) for i in range(5)]
        manager.add_chunks(chunks)
        assert manager.collection_stats()["total_chunks"] == 5

        manager.reset_collection()
        assert manager.collection_stats()["total_chunks"] == 0