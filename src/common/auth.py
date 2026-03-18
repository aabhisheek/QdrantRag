"""Bearer token authentication dependency for FastAPI."""

from __future__ import annotations

from typing import Annotated

import bcrypt
import structlog
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.common.config import Settings, get_settings

logger = structlog.get_logger()
security = HTTPBearer()


def _verify_api_key(provided_key: str, stored_key: str) -> bool:
    """Verify an API key against the stored value.

    Args:
        provided_key: The key from the Authorization header.
        stored_key: The key from configuration.

    Returns:
        True if the key matches.
    """
    if stored_key.startswith("$2b$"):
        return bcrypt.checkpw(
            provided_key.encode("utf-8"),
            stored_key.encode("utf-8"),
        )
    return provided_key == stored_key


async def verify_token(
    credentials: Annotated[HTTPAuthorizationCredentials, Security(security)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> str:
    """Validate the Bearer token from the request.

    Args:
        credentials: The HTTP Bearer credentials.
        settings: Application settings.

    Returns:
        The validated token string.

    Raises:
        HTTPException: If the token is invalid.
    """
    if not settings.api_key:
        return credentials.credentials

    if not _verify_api_key(credentials.credentials, settings.api_key):
        logger.warning("invalid_api_key_attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return credentials.credentials
