"""FastAPI dependency injection providers."""

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

from src.config import Settings, get_settings

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def verify_api_key(
    api_key: str = Depends(api_key_header),  # noqa: B008
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> str:
    """Validate the incoming API key against the configured master key.

    Raises:
        HTTPException: If the API key is missing or invalid (403).
    """
    if api_key == settings.kok_api_key:
        return api_key
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN,
        detail="Could not validate credentials. Invalid or missing API Key.",
    )


def get_collection(request: Request):
    """Retrieve the ChromaDB collection from application state."""
    return request.app.state.collection


def get_ollama_client(request: Request):
    """Retrieve the Ollama client from application state."""
    return request.app.state.ollama_client


def get_cross_encoder(request: Request):
    """Retrieve the cross-encoder model from application state."""
    return request.app.state.cross_encoder
