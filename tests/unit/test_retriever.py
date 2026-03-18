"""Tests for the retriever service."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.common.config import Settings
from src.retrieval.retriever import RetrieverService


def _make_settings() -> Settings:
    return Settings(
        openai_api_key="test-key",
        api_key="test-api-key",
        top_k=3,
        score_threshold=0.5,
    )


def test_format_context_empty() -> None:
    """Empty document list returns 'no context' message."""
    service = RetrieverService(_make_settings(), MagicMock())
    result = service.format_context([])
    assert result == "No relevant context found."


def test_format_context_with_documents() -> None:
    """Documents are formatted with source citations."""
    docs = [
        Document(
            page_content="RAG stands for Retrieval-Augmented Generation.",
            metadata={"filename": "guide.pdf", "chunk_index": 0},
        ),
        Document(
            page_content="Qdrant is a vector database.",
            metadata={"filename": "qdrant.txt", "chunk_index": 2},
        ),
    ]
    service = RetrieverService(_make_settings(), MagicMock())
    result = service.format_context(docs)

    assert "[Source 1: guide.pdf, chunk 0]" in result
    assert "[Source 2: qdrant.txt, chunk 2]" in result
    assert "RAG stands for" in result


def test_extract_sources_deduplicates() -> None:
    """Sources are deduplicated by document_id."""
    meta_1 = {"document_id": "doc-1", "filename": "a.pdf", "chunk_index": 0}
    meta_1b = {"document_id": "doc-1", "filename": "a.pdf", "chunk_index": 1}
    meta_2 = {"document_id": "doc-2", "filename": "b.txt", "chunk_index": 0}
    docs = [
        Document(page_content="a", metadata=meta_1),
        Document(page_content="b", metadata=meta_1b),
        Document(page_content="c", metadata=meta_2),
    ]
    service = RetrieverService(_make_settings(), MagicMock())
    sources = service.extract_sources(docs)

    assert len(sources) == 2
    assert sources[0]["document_id"] == "doc-1"
    assert sources[1]["document_id"] == "doc-2"
