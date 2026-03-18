"""FastAPI route definitions for the RAG Knowledge Assistant."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Request, UploadFile

from src.api.schemas import (
    DeleteResponse,
    DocumentListResponse,
    ErrorResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceInfo,
)
from src.common.auth import verify_token

logger = structlog.get_logger()

health_router = APIRouter(tags=["health"])
ingest_router = APIRouter(tags=["ingestion"], dependencies=[Depends(verify_token)])
query_router = APIRouter(tags=["query"], dependencies=[Depends(verify_token)])
documents_router = APIRouter(tags=["documents"], dependencies=[Depends(verify_token)])


@health_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health_check() -> HealthResponse:
    """Return service health status."""
    return HealthResponse()


@ingest_router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Ingest a document",
)
async def ingest_document(
    request: Request,
    file: UploadFile,
    _token: Annotated[str, Depends(verify_token)],
) -> IngestResponse:
    """Upload and ingest a PDF or text document into the knowledge base.

    Args:
        request: The FastAPI request (carries app state).
        file: The uploaded file.
        _token: Validated auth token.

    Returns:
        Ingestion result with document ID and chunk count.
    """
    pipeline = request.app.state.ingestion_pipeline

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = await pipeline.ingest_file(tmp_path, file.filename)
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(**result)


@query_router.post(
    "/query",
    response_model=QueryResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Query the knowledge base",
)
async def query_knowledge_base(
    request: Request,
    body: QueryRequest,
    _token: Annotated[str, Depends(verify_token)],
) -> QueryResponse:
    """Ask a question and get a context-aware answer from the knowledge base.

    Args:
        request: The FastAPI request (carries app state).
        body: The query request with the question.
        _token: Validated auth token.

    Returns:
        Answer with sources and context count.
    """
    retriever = request.app.state.retriever
    chain = request.app.state.generation_chain

    documents = await retriever.retrieve(body.question)
    context = retriever.format_context(documents)
    sources = retriever.extract_sources(documents)

    answer = await chain.generate(context, body.question)

    return QueryResponse(
        answer=answer,
        sources=[SourceInfo(**s) for s in sources],
        context_used=len(documents),
    )


@documents_router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all documents",
)
async def list_documents(
    request: Request,
    _token: Annotated[str, Depends(verify_token)],
) -> DocumentListResponse:
    """List all ingested documents in the knowledge base.

    Args:
        request: The FastAPI request (carries app state).
        _token: Validated auth token.

    Returns:
        List of documents with metadata.
    """
    store = request.app.state.qdrant_store
    docs = await store.list_documents()
    return DocumentListResponse(
        documents=docs,
        total=len(docs),
    )


@documents_router.delete(
    "/documents/{document_id}",
    response_model=DeleteResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Delete a document",
)
async def delete_document(
    request: Request,
    document_id: str,
    _token: Annotated[str, Depends(verify_token)],
) -> DeleteResponse:
    """Delete a document and all its chunks from the knowledge base.

    Args:
        request: The FastAPI request (carries app state).
        document_id: The UUID of the document to delete.
        _token: Validated auth token.

    Returns:
        Confirmation of deletion.
    """
    store = request.app.state.qdrant_store
    await store.delete_by_document_id(document_id)
    return DeleteResponse(
        message="Document deleted successfully",
        document_id=document_id,
    )
