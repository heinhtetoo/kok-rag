"""Recipe ingestion endpoint."""

from fastapi import APIRouter, Depends, HTTPException

from src.config import get_settings
from src.dependencies import get_bm25_index, get_collection, verify_api_key
from src.models.schemas import IngestRequest, IngestResponse
from src.services.embedding import embed_chunks
from src.services.ingestion import chunk_recipe, load_parent_store
from src.services.scraper import scrape_recipe

router = APIRouter(tags=["Recipes"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(verify_api_key)],
)
async def ingest_url(
    request: IngestRequest,
    collection=Depends(get_collection),  # noqa: B008
    bm25_index=Depends(get_bm25_index),  # noqa: B008
) -> IngestResponse:
    """Ingest a recipe from a supported URL.

    Scrapes the recipe page, splits it into parent-child chunks, and
    embeds the child chunks into the vector database.
    """
    settings = get_settings()

    try:
        # Scrape the recipe
        recipe = scrape_recipe(request.url, request.cuisine, request.dish_type)
        if not recipe:
            raise HTTPException(
                status_code=400,
                detail="Failed to scrape the provided URL. Please ensure it's a supported recipe URL.",
            )

        # Split into chunks
        chunks, parent_id = chunk_recipe(recipe, settings.parent_store_path)
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Failed to ingest the recipe. No chunks were created.",
            )

        # Embed into vector database
        chunks_added = embed_chunks(
            chunks,
            parent_id,
            collection,
            recipe.to_metadata(parent_id),
        )
        if not chunks_added:
            raise HTTPException(
                status_code=400,
                detail="Failed to embed the recipe chunks.",
            )

        # Rebuild BM25 index with new document
        bm25_index.build(load_parent_store(settings.parent_store_path))

        return IngestResponse(
            message="Successfully ingested with section-aware chunking!",
            title=recipe.title,
            chunks_added=chunks_added,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
