"""Migration script to re-chunk and re-embed all recipes."""

import contextlib
import os
import sys

# Add the project root to sys.path so we can import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chromadb
from chromadb.utils import embedding_functions

from src.config import get_settings
from src.models.recipe import Recipe
from src.services.embedding import embed_chunks
from src.services.ingestion import chunk_recipe_by_section, load_parent_store


def parse_raw_text_to_recipe(text: str, parent_id: str) -> Recipe:
    """Roughly reconstruct a Recipe dataclass from raw text."""
    lines = text.split("\n")
    title = parent_id
    source_url = "Unknown"

    for line in lines:
        if line.startswith("Title: "):
            title = line[len("Title: ") :].strip()
        elif line.startswith("Source: "):
            source_url = line[len("Source: ") :].strip()

    return Recipe(
        title=title,
        source_url=source_url,
        ingredients=[],  # Extracted if needed, but not strictly required for re-chunking
        instructions=[],
        cuisine="Unknown",
        dish_type="Unknown",
        raw_text=text,
    )


def reindex():
    settings = get_settings()

    print(f"Loading parent store from {settings.parent_store_path}...")
    parent_store = load_parent_store(settings.parent_store_path)
    if not parent_store:
        print("No recipes found in parent store. Exiting.")
        return

    print(f"Found {len(parent_store)} recipes to re-index.")

    # Initialize ChromaDB client
    db_client = chromadb.PersistentClient(path=settings.vector_db_dir)

    # Re-create collection
    print(f"Deleting collection '{settings.collection_name}' (if exists)...")
    with contextlib.suppress(Exception):
        db_client.delete_collection(name=settings.collection_name)

    print(
        f"Creating collection '{settings.collection_name}' with model '{settings.embedding_model}'..."
    )
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )
    collection = db_client.create_collection(name=settings.collection_name, embedding_function=ef)

    # Process each recipe
    total_chunks = 0
    for parent_id, raw_text in parent_store.items():
        recipe = parse_raw_text_to_recipe(raw_text, parent_id)

        chunks = chunk_recipe_by_section(recipe)
        if not chunks:
            print(f"Warning: No chunks generated for '{parent_id}'")
            continue

        chunks_added = embed_chunks(
            chunks=chunks,
            parent_id=parent_id,
            collection=collection,
            metadata_template=recipe.to_metadata(parent_id),
        )
        total_chunks += chunks_added
        print(f"Re-indexed '{parent_id}': {chunks_added} chunks")

    print(f"Re-indexing complete! Total chunks embedded: {total_chunks}")


if __name__ == "__main__":
    reindex()
