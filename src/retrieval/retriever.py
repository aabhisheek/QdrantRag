"""Vector search and context assembly for RAG retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from langchain_core.documents import Document

    from src.common.config import Settings
    from src.storage.qdrant_store import QdrantStore

logger = structlog.get_logger()


class RetrieverService:
    """Performs similarity search against Qdrant and assembles context.

    Args:
        settings: Application settings for retrieval parameters.
        store: The Qdrant vector store to search against.
    """

    def __init__(self, settings: Settings, store: QdrantStore) -> None:
        self._settings = settings
        self._store = store

    async def retrieve(self, query: str) -> list[Document]:
        """Retrieve relevant document chunks for a query.

        Args:
            query: The user's question or search query.

        Returns:
            List of relevant Document objects with metadata.
        """
        results = self._store.vector_store.similarity_search_with_score(
            query=query,
            k=self._settings.top_k,
        )

        filtered = [doc for doc, score in results if score >= self._settings.score_threshold]

        logger.info(
            "retrieval_completed",
            query_length=len(query),
            total_results=len(results),
            filtered_results=len(filtered),
        )

        return filtered

    def format_context(self, documents: list[Document]) -> str:
        """Format retrieved documents into a context string for the LLM.

        Args:
            documents: List of retrieved Document objects.

        Returns:
            Formatted context string with source citations.
        """
        if not documents:
            return "No relevant context found."

        context_parts: list[str] = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("filename", "unknown")
            chunk_idx = doc.metadata.get("chunk_index", "?")
            context_parts.append(f"[Source {i}: {source}, chunk {chunk_idx}]\n{doc.page_content}")

        return "\n\n---\n\n".join(context_parts)

    def extract_sources(self, documents: list[Document]) -> list[dict]:
        """Extract source metadata from retrieved documents.

        Args:
            documents: List of retrieved Document objects.

        Returns:
            List of source metadata dicts.
        """
        sources: list[dict] = []
        seen: set[str] = set()
        for doc in documents:
            doc_id = doc.metadata.get("document_id", "unknown")
            if doc_id not in seen:
                seen.add(doc_id)
                sources.append(
                    {
                        "document_id": doc_id,
                        "filename": doc.metadata.get("filename", "unknown"),
                        "chunk_index": doc.metadata.get("chunk_index"),
                    }
                )
        return sources
