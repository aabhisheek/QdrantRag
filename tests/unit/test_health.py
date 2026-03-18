"""Tests for the health check endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    """Create a test client with mocked lifespan dependencies."""
    with patch("src.main.QdrantStore") as mock_store_cls:
        mock_store = AsyncMock()
        mock_store.initialize = AsyncMock()
        mock_store.close = AsyncMock()
        mock_store_cls.return_value = mock_store

        from src.main import app

        with TestClient(app) as c:
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
