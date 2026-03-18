"""Tests for the ingestion pipeline logic."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.common.config import Settings

if TYPE_CHECKING:
    from pathlib import Path
from src.common.errors import ValidationError
from src.ingestion.pipeline import IngestionPipeline


@pytest.fixture
def settings() -> Settings:
    """Create test settings."""
    return Settings(
        openai_api_key="test-key",
        api_key="test-api-key",
        chunk_size=100,
        chunk_overlap=20,
    )


@pytest.fixture
def mock_store() -> MagicMock:
    """Create a mock QdrantStore."""
    store = MagicMock()
    store.add_documents = AsyncMock(return_value=["id-1", "id-2"])
    return store


@pytest.fixture
def pipeline(settings: Settings, mock_store: MagicMock) -> IngestionPipeline:
    """Create an IngestionPipeline with mocked store."""
    return IngestionPipeline(settings, mock_store)


@pytest.mark.asyncio
async def test_ingest_rejects_unsupported_file_type(
    pipeline: IngestionPipeline,
    tmp_path: Path,
) -> None:
    """Unsupported file types raise ValidationError."""
    with pytest.raises(ValidationError, match="Unsupported file type"):
        await pipeline.ingest_file(tmp_path / "test.xlsx", "test.xlsx")


@pytest.mark.asyncio
async def test_ingest_text_file(
    pipeline: IngestionPipeline,
    mock_store: MagicMock,
    tmp_path: Path,
) -> None:
    """Text file ingestion produces chunks and stores them."""
    test_file = tmp_path / "sample.txt"
    test_file.write_text("This is a test document with some content for chunking. " * 20)

    with patch("src.ingestion.pipeline.TextLoader") as mock_loader_cls:
        mock_loader = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "This is a test document with some content for chunking. " * 20
        mock_doc.metadata = {}
        mock_loader.load.return_value = [mock_doc]
        mock_loader_cls.return_value = mock_loader

        result = await pipeline.ingest_file(test_file, "sample.txt")

    assert result["filename"] == "sample.txt"
    assert result["chunk_count"] > 0
    assert "document_id" in result
    mock_store.add_documents.assert_called_once()


@pytest.mark.asyncio
async def test_ingest_assigns_metadata(
    pipeline: IngestionPipeline,
    mock_store: MagicMock,
    tmp_path: Path,
) -> None:
    """Ingested chunks have correct metadata fields."""
    test_file = tmp_path / "doc.txt"
    test_file.write_text("Some content here.")

    with patch("src.ingestion.pipeline.TextLoader") as mock_loader_cls:
        mock_loader = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "Some content here."
        mock_doc.metadata = {}
        mock_loader.load.return_value = [mock_doc]
        mock_loader_cls.return_value = mock_loader

        await pipeline.ingest_file(test_file, "doc.txt")

    stored_docs = mock_store.add_documents.call_args[0][0]
    for doc in stored_docs:
        assert "document_id" in doc.metadata
        assert doc.metadata["filename"] == "doc.txt"
        assert doc.metadata["file_type"] == ".txt"
        assert "chunk_index" in doc.metadata
