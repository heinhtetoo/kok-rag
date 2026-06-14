"""Health check endpoint."""

from fastapi import APIRouter, Request

from src.models.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Check service health and dependency connectivity.

    Verifies that both ChromaDB and Ollama are reachable.
    Returns an overall status of ``healthy`` or ``degraded``.
    """
    chromadb_status = "unknown"
    ollama_status = "unknown"

    # Check ChromaDB
    try:
        collection = request.app.state.collection
        collection.count()
        chromadb_status = "healthy"
    except Exception:
        chromadb_status = "unhealthy"

    # Check Ollama
    try:
        client = request.app.state.ollama_client
        client.list()
        ollama_status = "healthy"
    except Exception:
        ollama_status = "unhealthy"

    overall = (
        "healthy" if chromadb_status == "healthy" and ollama_status == "healthy" else "degraded"
    )

    return HealthResponse(
        status=overall,
        chromadb=chromadb_status,
        ollama=ollama_status,
    )
