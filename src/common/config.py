"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the RAG Knowledge Assistant."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI (optional, kept for compatibility)
    openai_api_key: str = ""

    # Groq
    groq_api_key: str = ""

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "knowledge_base"

    # Embeddings (FastEmbed — runs locally, no API key needed)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384

    # LLM
    llm_model: str = "llama-3.3-70b-versatile"
    llm_provider: str = "groq"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"

    # Chunking
    chunk_size: int = 1024
    chunk_overlap: int = 128

    # Retrieval
    top_k: int = 5
    score_threshold: float = 0.2

    # Auth
    api_key: str = ""

    # Logging
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
