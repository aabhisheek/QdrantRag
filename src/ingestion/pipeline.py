"""Document ingestion pipeline: load, chunk, embed, and store documents."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.common.errors import IngestionError, ValidationError

if TYPE_CHECKING:
    from langchain_core.documents import Document

    from src.common.config import Settings
    from src.storage.qdrant_store import QdrantStore

logger = structlog.get_logger()

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class IngestionPipeline:
    """Handles document loading, chunking, and storage into Qdrant.

    Args:
        settings: Application settings for chunk size/overlap configuration.
        store: The Qdrant vector store for persisting embedded chunks.
    """

    def __init__(self, settings: Settings, store: QdrantStore) -> None:
        self._settings = settings
        self._store = store
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    async def ingest_file(self, file_path: Path, filename: str) -> dict:
        """Ingest a single file: load, chunk, embed, and store.

        Args:
            file_path: Path to the temporary uploaded file.
            filename: Original filename for metadata.

        Returns:
            Dict with document_id, filename, and chunk_count.

        Raises:
            ValidationError: If the file type is not supported.
            IngestionError: If loading or processing fails.
        """
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValidationError(
                f"Unsupported file type: {suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        document_id = str(uuid.uuid4())
        logger.info("ingestion_started", document_id=document_id, filename=filename)

        try:
            raw_docs = await self._load_document(file_path, suffix)
        except Exception as e:
            raise IngestionError(f"Failed to load document: {e}") from e

        chunks = self._splitter.split_documents(raw_docs)

        for i, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "file_type": suffix,
                    "chunk_index": i,
                }
            )

        await self._store.add_documents(chunks)

        logger.info(
            "ingestion_completed",
            document_id=document_id,
            filename=filename,
            chunk_count=len(chunks),
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "chunk_count": len(chunks),
        }

    async def _load_document(self, file_path: Path, suffix: str) -> list[Document]:
        """Load a document using the appropriate LangChain loader.

        Args:
            file_path: Path to the file.
            suffix: File extension (e.g., '.pdf', '.txt').

        Returns:
            List of loaded Document objects.
        """
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path), encoding="utf-8")

        return await asyncio.to_thread(loader.load)
