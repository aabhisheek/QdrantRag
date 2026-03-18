"""Qdrant vector store wrapper for collection management and document operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, VectorParams

if TYPE_CHECKING:
    from langchain_core.documents import Document

    from src.common.config import Settings

logger = structlog.get_logger()


class QdrantStore:
    """Manages the Qdrant vector store lifecycle and operations.

    Args:
        settings: Application settings for Qdrant connection and collection config.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            check_compatibility=False,
        )
        self._embeddings = FastEmbedEmbeddings(
            model_name=settings.embedding_model,
        )
        self._vector_store: QdrantVectorStore | None = None

    async def initialize(self) -> None:
        """Create the collection if it doesn't exist and initialize the vector store.

        If the collection exists with a different vector dimension (e.g. embedding model
        was changed), it is deleted and recreated automatically.
        """
        collection_name = self._settings.qdrant_collection
        required_dim = self._settings.embedding_dimensions

        collections = self._client.get_collections().collections
        existing_names = {c.name for c in collections}

        if collection_name in existing_names:
            existing_dim = self._get_collection_dimensions(collection_name)
            if existing_dim is not None and existing_dim != required_dim:
                logger.warning(
                    "collection_dimension_mismatch",
                    collection=collection_name,
                    existing_dim=existing_dim,
                    required_dim=required_dim,
                    action="recreating",
                )
                self._client.delete_collection(collection_name)
                existing_names.discard(collection_name)

        if collection_name not in existing_names:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=required_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("collection_created", collection=collection_name)

        self._vector_store = QdrantVectorStore(
            client=self._client,
            collection_name=collection_name,
            embedding=self._embeddings,
        )
        logger.info("qdrant_store_initialized", collection=collection_name)

    def _get_collection_dimensions(self, collection_name: str) -> int | None:
        """Return the vector size of an existing collection, or None if undetectable.

        Args:
            collection_name: The Qdrant collection name.

        Returns:
            Vector dimension size, or None.
        """
        info = self._client.get_collection(collection_name)
        vectors_config = info.config.params.vectors
        # Unnamed vector: has a direct .size attribute
        size = getattr(vectors_config, "size", None)
        if isinstance(size, int):
            return size
        # Named vectors: dict-like root model
        root = getattr(vectors_config, "root", vectors_config)
        if isinstance(root, dict):
            first = next(iter(root.values()), None)
            if first is not None:
                named_size = getattr(first, "size", None)
                if isinstance(named_size, int):
                    return named_size
        return None

    @property
    def vector_store(self) -> QdrantVectorStore:
        """Return the initialized vector store.

        Returns:
            The QdrantVectorStore instance.

        Raises:
            RuntimeError: If the store has not been initialized.
        """
        if self._vector_store is None:
            raise RuntimeError("QdrantStore not initialized. Call initialize() first.")
        return self._vector_store

    @property
    def client(self) -> QdrantClient:
        """Return the raw Qdrant client."""
        return self._client

    @property
    def embeddings(self) -> FastEmbedEmbeddings:
        """Return the embeddings model."""
        return self._embeddings

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store in batch.

        Args:
            documents: List of LangChain Document objects to embed and store.

        Returns:
            List of document IDs assigned by Qdrant.
        """
        ids = self.vector_store.add_documents(documents)
        logger.info("documents_added", count=len(documents))
        return ids

    async def delete_by_document_id(self, document_id: str) -> None:
        """Delete all chunks belonging to a specific source document.

        Args:
            document_id: The source document identifier.
        """
        results = self._client.scroll(
            collection_name=self._settings.qdrant_collection,
            scroll_filter={
                "must": [{"key": "metadata.document_id", "match": {"value": document_id}}]
            },
            limit=10000,
        )
        point_ids = [point.id for point in results[0]]
        if point_ids:
            self._client.delete(
                collection_name=self._settings.qdrant_collection,
                points_selector=PointIdsList(points=point_ids),
            )
        logger.info("document_deleted", document_id=document_id, chunks_removed=len(point_ids))

    async def list_documents(self) -> list[dict]:
        """List all unique source documents in the collection.

        Returns:
            List of dicts with document metadata.
        """
        results = self._client.scroll(
            collection_name=self._settings.qdrant_collection,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )
        seen: dict[str, dict] = {}
        for point in results[0]:
            metadata = point.payload.get("metadata", {})
            doc_id = metadata.get("document_id", "unknown")
            if doc_id not in seen:
                seen[doc_id] = {
                    "document_id": doc_id,
                    "filename": metadata.get("filename", "unknown"),
                    "file_type": metadata.get("file_type", "unknown"),
                    "chunk_count": 0,
                }
            seen[doc_id]["chunk_count"] += 1
        return list(seen.values())

    async def close(self) -> None:
        """Close the Qdrant client connection."""
        self._client.close()
        logger.info("qdrant_store_closed")
