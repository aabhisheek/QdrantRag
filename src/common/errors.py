"""Typed application errors for the RAG Knowledge Assistant."""

from __future__ import annotations


class AppError(Exception):
    """Base error for the application."""

    def __init__(self, message: str, *, status_code: int = 500) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class ValidationError(AppError):
    """Raised when input validation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=422)


class IngestionError(AppError):
    """Raised when document ingestion fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=500)


class DocumentNotFoundError(AppError):
    """Raised when a requested document does not exist."""

    def __init__(self, document_id: str) -> None:
        super().__init__(f"Document not found: {document_id}", status_code=404)


class RetrievalError(AppError):
    """Raised when vector search or retrieval fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=500)
