"""Tests for application configuration."""

from __future__ import annotations

from src.common.config import Settings


def test_settings_defaults() -> None:
    """Settings have sensible defaults."""
    settings = Settings(openai_api_key="test-key", api_key="test-api-key")
    assert settings.qdrant_host == "localhost"
    assert settings.qdrant_port == 6333
    assert settings.qdrant_collection == "knowledge_base"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimensions == 1536
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.llm_provider == "openai"
    assert settings.chunk_size == 1024
    assert settings.chunk_overlap == 128
    assert settings.top_k == 5
    assert settings.score_threshold == 0.5
    assert settings.log_level == "INFO"


def test_settings_custom_values() -> None:
    """Settings can be overridden."""
    settings = Settings(
        openai_api_key="sk-test",
        qdrant_host="qdrant-server",
        qdrant_port=6334,
        chunk_size=512,
        llm_provider="ollama",
        llm_model="llama3.1",
        api_key="custom-key",
    )
    assert settings.qdrant_host == "qdrant-server"
    assert settings.qdrant_port == 6334
    assert settings.chunk_size == 512
    assert settings.llm_provider == "ollama"
    assert settings.llm_model == "llama3.1"
