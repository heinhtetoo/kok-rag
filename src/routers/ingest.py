"""Recipe ingestion endpoint."""

from fastapi import APIRouter, Depends, HTTPException

from src.config import get_settings
from src.dependencies import get_collection, verify_api_key
from src.models.schemas import IngestRequest, IngestResponse
from src.services.embedding import embed_chunks
from src.services.ingestion import chunk_recipe
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
) -> IngestResponse:
    """Ingest a recipe from a supported URL.

    Scrapes the recipe page, splits it into parent-child chunks, and
    embeds the child chunks into the vector database.
    """
    settings = get_settings()

    try:
        # Scrape the recipe
        recipe_file = scrape_recipe(request.url, settings.recipe_dir)
        if not recipe_file:
            raise HTTPException(
                status_code=400,
                detail="Failed to scrape the provided URL. Please ensure it's a supported recipe URL.",
            )

        # Split into chunks
        chunks, parent_id = chunk_recipe(
            recipe_file, settings.recipe_dir, settings.parent_store_path
        )
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
            request.url,
            request.cuisine,
            request.dish_type,
        )
        if not chunks_added:
            raise HTTPException(
                status_code=400,
                detail="Failed to embed the recipe chunks.",
            )

        return IngestResponse(
            message="Successfully ingested with Parent-Child chunking!",
            title=recipe_file,
            chunks_added=chunks_added,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
