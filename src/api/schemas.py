"""Pydantic v2 request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(strict=True)

    status: str = "ok"
    service: str = "rag-knowledge-assistant"


class IngestResponse(BaseModel):
    """Response after successful document ingestion."""

    model_config = ConfigDict(strict=True)

    document_id: str
    filename: str
    chunk_count: int


class QueryRequest(BaseModel):
    """Request body for the query endpoint."""

    model_config = ConfigDict(strict=True)

    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceInfo(BaseModel):
    """Source document reference in query response."""

    model_config = ConfigDict(strict=True)

    document_id: str
    filename: str
    chunk_index: int | None = None


class QueryResponse(BaseModel):
    """Response from the query endpoint."""

    model_config = ConfigDict(strict=True)

    answer: str
    sources: list[SourceInfo]
    context_used: int


class DocumentInfo(BaseModel):
    """Information about an ingested document."""

    model_config = ConfigDict(strict=True)

    document_id: str
    filename: str
    file_type: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    """Response listing all ingested documents."""

    model_config = ConfigDict(strict=True)

    documents: list[DocumentInfo]
    total: int


class DeleteResponse(BaseModel):
    """Response after document deletion."""

    model_config = ConfigDict(strict=True)

    message: str
    document_id: str


class ErrorResponse(BaseModel):
    """Standard error response."""

    model_config = ConfigDict(strict=True)

    detail: str
