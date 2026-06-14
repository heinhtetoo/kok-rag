"""Vector embedding and ChromaDB storage service."""

import chromadb

from src.core.logging import get_logger

logger = get_logger(__name__)


def embed_chunks(
    chunks: list[str],
    parent_id: str,
    collection: chromadb.Collection,
    url: str,
    cuisine: str,
    dish_type: str,
) -> int:
    """Embed text chunks and upsert them into the vector database.

    Each chunk is stored with metadata linking it back to its parent document,
    source URL, cuisine, and dish type. Uses ``upsert`` for idempotent writes.

    Args:
        chunks: List of text chunks to embed.
        parent_id: The parent document identifier.
        collection: ChromaDB collection instance.
        url: Source URL of the recipe.
        cuisine: Cuisine category metadata.
        dish_type: Dish type metadata.

    Returns:
        Number of chunks successfully stored.
    """
    ids = [f"{parent_id}_child_{i}" for i in range(len(chunks))]

    metadatas = [
        {
            "parent_id": parent_id,
            "source": url,
            "cuisine": cuisine,
            "dish_type": dish_type,
        }
        for _ in chunks
    ]

    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)

    logger.info(
        "Stored %d chunks in vector database for parent '%s'",
        len(chunks),
        parent_id,
    )
    return len(chunks)
