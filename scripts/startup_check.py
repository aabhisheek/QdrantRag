"""Startup smoke test — verifies the app can boot without real external services.

Run this after any src/ code change:
    python scripts/startup_check.py
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch


def run() -> int:
    """Run the startup smoke test.

    Returns:
        0 on success, 1 on failure.
    """
    mock_qdrant = MagicMock()
    mock_qdrant.get_collections.return_value.collections = []
    mock_qdrant.create_collection.return_value = None

    try:
        with (
            patch("src.storage.qdrant_store.QdrantClient", return_value=mock_qdrant),
            patch("src.storage.qdrant_store.FastEmbedEmbeddings"),
            patch("src.storage.qdrant_store.QdrantVectorStore"),
            patch("src.generation.chain.ChatGroq"),
        ):
            from fastapi.testclient import TestClient

            from src.main import app

            with TestClient(app) as client:
                resp = client.get("/health")
                if resp.status_code != 200:
                    print(f"FAIL: /health returned {resp.status_code}: {resp.text}")
                    return 1
                print("OK: app started and /health returned 200")
                return 0

    except Exception as e:  # noqa: BLE001
        print(f"FAIL: app startup raised {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(run())
