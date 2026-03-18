"""Tests for the health check endpoint.

Mocks are applied at the QdrantClient / FastEmbedEmbeddings / ChatGroq level so
the real QdrantStore.initialize() code runs — startup bugs are caught here.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _make_app_client() -> TestClient:
    """Build a TestClient where the real lifespan runs but external I/O is mocked."""
    mock_qdrant = MagicMock()
    mock_qdrant.get_collections.return_value.collections = []
    mock_qdrant.create_collection.return_value = None

    mock_vector_store = MagicMock()
    mock_groq = MagicMock()

    with (
        patch("src.storage.qdrant_store.QdrantClient", return_value=mock_qdrant),
        patch("src.storage.qdrant_store.FastEmbedEmbeddings"),
        patch("src.storage.qdrant_store.QdrantVectorStore", return_value=mock_vector_store),
        patch("src.generation.chain.ChatGroq", return_value=mock_groq),
    ):
        from src.main import app

        return TestClient(app)


@pytest.fixture
def client() -> TestClient:
    """TestClient with real startup code running (external I/O mocked)."""
    with _make_app_client() as c:
        yield c


def test_health_returns_ok(client: TestClient) -> None:
    """Health endpoint returns 200 with expected payload."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "rag-knowledge-assistant"


def test_health_does_not_require_auth(client: TestClient) -> None:
    """Health endpoint is accessible without Bearer token."""
    response = client.get("/health")
    assert response.status_code == 200


def test_startup_creates_qdrant_collection() -> None:
    """Startup calls create_collection when no collection exists."""
    mock_qdrant = MagicMock()
    mock_qdrant.get_collections.return_value.collections = []

    with (
        patch("src.storage.qdrant_store.QdrantClient", return_value=mock_qdrant),
        patch("src.storage.qdrant_store.FastEmbedEmbeddings"),
        patch("src.storage.qdrant_store.QdrantVectorStore"),
        patch("src.generation.chain.ChatGroq"),
    ):
        from src.main import app

        with TestClient(app):
            mock_qdrant.create_collection.assert_called_once()


def test_startup_recreates_collection_on_dimension_mismatch() -> None:
    """Startup deletes and recreates collection when vector dimensions changed."""
    mock_qdrant = MagicMock()

    existing_collection = MagicMock()
    existing_collection.name = "knowledge_base"
    mock_qdrant.get_collections.return_value.collections = [existing_collection]

    # Existing collection has 1536 dims (old OpenAI), config expects 384 (FastEmbed)
    mock_qdrant.get_collection.return_value.config.params.vectors.size = 1536

    with (
        patch("src.storage.qdrant_store.QdrantClient", return_value=mock_qdrant),
        patch("src.storage.qdrant_store.FastEmbedEmbeddings"),
        patch("src.storage.qdrant_store.QdrantVectorStore"),
        patch("src.generation.chain.ChatGroq"),
    ):
        from src.main import app

        with TestClient(app):
            mock_qdrant.delete_collection.assert_called_once_with("knowledge_base")
            mock_qdrant.create_collection.assert_called_once()
