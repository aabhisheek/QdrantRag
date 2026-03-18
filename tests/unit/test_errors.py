"""Tests for typed application errors."""

from __future__ import annotations

from src.common.errors import (
    AppError,
    DocumentNotFoundError,
    IngestionError,
    RetrievalError,
    ValidationError,
)


def test_app_error_defaults() -> None:
    """AppError has default status 500."""
    err = AppError("something broke")
    assert err.message == "something broke"
    assert err.status_code == 500


def test_validation_error_status() -> None:
    """ValidationError has status 422."""
    err = ValidationError("bad input")
    assert err.status_code == 422


def test_ingestion_error_status() -> None:
    """IngestionError has status 500."""
    err = IngestionError("parse failed")
    assert err.status_code == 500


def test_document_not_found_error() -> None:
    """DocumentNotFoundError includes document ID and has status 404."""
    err = DocumentNotFoundError("doc-123")
    assert "doc-123" in err.message
    assert err.status_code == 404


def test_retrieval_error_status() -> None:
    """RetrievalError has status 500."""
    err = RetrievalError("search failed")
    assert err.status_code == 500
