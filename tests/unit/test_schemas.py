"""Tests for Pydantic request/response schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    DocumentInfo,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceInfo,
)


def test_health_response_defaults() -> None:
    """HealthResponse has correct default values."""
    resp = HealthResponse()
    assert resp.status == "ok"
    assert resp.service == "rag-knowledge-assistant"


def test_query_request_valid() -> None:
    """QueryRequest accepts valid input."""
    req = QueryRequest(question="What is RAG?")
    assert req.question == "What is RAG?"
    assert req.top_k is None


def test_query_request_empty_question_rejected() -> None:
    """QueryRequest rejects empty question."""
    with pytest.raises(ValidationError):
        QueryRequest(question="")


def test_query_request_top_k_bounds() -> None:
    """QueryRequest validates top_k range."""
    req = QueryRequest(question="test", top_k=5)
    assert req.top_k == 5

    with pytest.raises(ValidationError):
        QueryRequest(question="test", top_k=0)

    with pytest.raises(ValidationError):
        QueryRequest(question="test", top_k=21)


def test_ingest_response() -> None:
    """IngestResponse serializes correctly."""
    resp = IngestResponse(document_id="abc-123", filename="test.pdf", chunk_count=10)
    assert resp.document_id == "abc-123"
    assert resp.chunk_count == 10


def test_query_response_with_sources() -> None:
    """QueryResponse includes sources correctly."""
    resp = QueryResponse(
        answer="The answer is 42.",
        sources=[SourceInfo(document_id="doc-1", filename="guide.pdf", chunk_index=3)],
        context_used=1,
    )
    assert len(resp.sources) == 1
    assert resp.sources[0].filename == "guide.pdf"


def test_document_info() -> None:
    """DocumentInfo round-trips correctly."""
    info = DocumentInfo(
        document_id="doc-1",
        filename="notes.txt",
        file_type=".txt",
        chunk_count=5,
    )
    assert info.file_type == ".txt"
