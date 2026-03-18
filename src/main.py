"""FastAPI application entry point for the RAG Knowledge Assistant."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.routes import documents_router, health_router, ingest_router, query_router
from src.common.config import get_settings
from src.common.errors import AppError
from src.common.logging import setup_logging
from src.generation.chain import GenerationChain
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.retriever import RetrieverService
from src.storage.qdrant_store import QdrantStore

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown lifecycle.

    Initializes Qdrant store, ingestion pipeline, retriever, and generation chain
    at startup. Cleans up resources at shutdown.
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    logger.info("app_starting", provider=settings.llm_provider, model=settings.llm_model)

    store = QdrantStore(settings)
    await store.initialize()

    app.state.qdrant_store = store
    app.state.ingestion_pipeline = IngestionPipeline(settings, store)
    app.state.retriever = RetrieverService(settings, store)
    app.state.generation_chain = GenerationChain(settings)

    logger.info("app_started")
    yield

    await store.close()
    logger.info("app_shutdown")


app = FastAPI(
    title="RAG Knowledge Assistant",
    description="AI Knowledge Assistant using LangChain + Qdrant for document-grounded Q&A",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Handle application-specific errors with proper status codes."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message},
    )


app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(documents_router)
