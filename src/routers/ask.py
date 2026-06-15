"""Recipe Q&A endpoint — powered by the agentic tool-calling loop."""

from fastapi import APIRouter, Depends, HTTPException

from src.config import get_settings
from src.dependencies import (
    get_collection,
    get_cross_encoder,
    get_ollama_client,
    verify_api_key,
)
from src.models.schemas import QueryRequest, QueryResponse
from src.services.agent import run_agent

router = APIRouter(tags=["Recipes"])


@router.post(
    "/ask",
    response_model=QueryResponse,
    dependencies=[Depends(verify_api_key)],
)
async def ask_kok(
    request: QueryRequest,
    collection=Depends(get_collection),  # noqa: B008
    ollama_client=Depends(get_ollama_client),  # noqa: B008
    cross_encoder=Depends(get_cross_encoder),  # noqa: B008
) -> QueryResponse:
    """Ask a natural-language question about recipes or culinary topics.

    The underlying agent autonomously decides which tool to invoke:
    - ``search_recipe_book`` — for questions about saved recipes (default)
    - ``search_web`` — when the user explicitly requests web results

    The ``tool_used`` field in the response indicates which tool was called.
    """
    settings = get_settings()

    try:
        result = run_agent(
            question=request.question,
            ollama_client=ollama_client,
            model=settings.ollama_model,
            collection=collection,
            cross_encoder=cross_encoder,
            parent_store_path=settings.parent_store_path,
            web_search_max_results=settings.web_search_max_results,
            max_iterations=settings.agent_max_iterations,
        )

        return QueryResponse(
            answer=result.answer,
            sources=result.sources,
            tool_used=result.tool_used,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
