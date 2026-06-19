"""Vector embedding and ChromaDB storage service."""

import chromadb

from src.core.logging import get_logger

logger = get_logger(__name__)


def embed_chunks(
    chunks: list[str],
    parent_id: str,
    collection: chromadb.Collection,
    metadata_template: dict,
) -> int:
    """Embed text chunks and upsert them into the vector database.

    Each chunk is stored with metadata linking it back to its parent document,
    source URL, cuisine, dish type, and more. Uses ``upsert`` for idempotent writes.

    Args:
        chunks: List of text chunks to embed.
        parent_id: The parent document identifier.
        collection: ChromaDB collection instance.
        metadata_template: Dictionary with baseline metadata for all chunks.

    Returns:
        Number of chunks successfully stored.
    """
    # Suggestion 7: Delete existing chunks before re-ingesting to prevent stale orphans
    collection.delete(where={"parent_id": parent_id})

    ids = [f"{parent_id}_child_{i}" for i in range(len(chunks))]

    metadatas = []
    for chunk in chunks:
        meta = metadata_template.copy()

        # Simple heuristic to add 'section' metadata for targeted filtering
        if "INGREDIENTS:" in chunk and "INSTRUCTIONS:" not in chunk:
            meta["section"] = "ingredients"
        elif "INSTRUCTIONS:" in chunk:
            meta["section"] = "instructions"

        metadatas.append(meta)

    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)

    logger.info(
        "Stored %d chunks in vector database for parent '%s'",
        len(chunks),
        parent_id,
    )
    return len(chunks)
