"""Pydantic request and response schemas for the Kök RAG API."""

from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Request body for the ``/ask`` endpoint."""

    question: str
    cuisine_filter: str | None = None
    dish_type_filter: str | None = None

    model_config = {"json_schema_extra": {"examples": [{"question": "How do I make Mohinga?"}]}}


class QueryResponse(BaseModel):
    """Response body for the ``/ask`` endpoint."""

    answer: str
    sources: list[str]
    tool_used: str = "none"  # "recipe_book" | "web_search" | "none"


class IngestRequest(BaseModel):
    """Request body for the ``/ingest`` endpoint."""

    url: str
    cuisine: str = "Unknown"
    dish_type: str = "Unknown"

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "url": "https://theburmalicious.com/recipe/mohinga",
                    "cuisine": "Burmese",
                    "dish_type": "Soup",
                }
            ]
        }
    }


class IngestResponse(BaseModel):
    """Response body for the ``/ingest`` endpoint."""

    message: str
    title: str
    chunks_added: int


class HealthResponse(BaseModel):
    """Response body for the ``/health`` endpoint."""

    status: str
    chromadb: str
    ollama: str
