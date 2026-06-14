"""Recipe Q&A endpoint."""

from fastapi import APIRouter, Depends, HTTPException

from src.config import get_settings
from src.dependencies import (
    get_collection,
    get_cross_encoder,
    get_ollama_client,
    verify_api_key,
)
from src.models.schemas import QueryRequest, QueryResponse
from src.services.retrieval import (
    extract_filters_from_query,
    generate_answer,
    retrieve_and_rerank,
)

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
    """Ask a natural-language question about your recipes.

    The pipeline extracts metadata filters, retrieves relevant chunks,
    resolves parent documents, re-ranks with a cross-encoder, and
    generates a grounded answer via the LLM.
    """
    settings = get_settings()

    try:
        # Extract metadata filters from the question
        extracted_filters = extract_filters_from_query(
            request.question, ollama_client, settings.ollama_model
        )

        # Retrieve and re-rank parent documents
        top_parents = retrieve_and_rerank(
            question=request.question,
            collection=collection,
            cross_encoder=cross_encoder,
            parent_store_path=settings.parent_store_path,
            extracted_filters=extracted_filters,
        )

        if not top_parents:
            return QueryResponse(
                answer="I don't have that in your recipe book.",
                sources=[],
            )

        # Generate a grounded answer
        answer = generate_answer(
            question=request.question,
            context_documents=top_parents,
            ollama_client=ollama_client,
            model=settings.ollama_model,
        )

        return QueryResponse(answer=answer, sources=top_parents)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
