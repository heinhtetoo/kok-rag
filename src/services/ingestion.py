"""Recipe ingestion and chunking service."""

import json
import os
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.logging import get_logger
from src.models.recipe import Recipe

logger = get_logger(__name__)


def chunk_recipe_by_section(recipe: Recipe) -> list[str]:
    """Split a recipe into semantic chunks by section."""
    title_line = f"Title: {recipe.title}"
    text = recipe.raw_text

    # Split on boundaries
    sections = re.split(r"\n(?=INGREDIENTS:|INSTRUCTIONS:)", text)

    chunks = []
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", "."],
    )

    for section in sections:
        sec_text = section.strip()
        if not sec_text:
            continue

        # Ignore the title/source block alone if it was split
        if (
            sec_text.startswith("Title:")
            and "INGREDIENTS:" not in sec_text
            and "INSTRUCTIONS:" not in sec_text
        ):
            continue

        chunk_content = f"{title_line}\n\n{sec_text}"
        if len(chunk_content) > 500:
            sub_chunks = fallback_splitter.split_text(chunk_content)
            chunks.extend(sub_chunks)
        else:
            chunks.append(chunk_content)

    return chunks


def chunk_recipe(
    recipe: Recipe,
    parent_store_path: str,
) -> tuple[list[str], str]:
    """Split a recipe object into chunks using the section-aware strategy.

    The full recipe text is saved as the parent document in a JSON sidecar
    store, and section-aware child chunks are returned for vector embedding.

    Args:
        recipe: Recipe object to chunk.
        parent_store_path: Path to the parent document JSON store.

    Returns:
        A tuple of ``(child_chunks, parent_id)``.
    """
    # Create a safe parent ID from the title
    parent_id = re.sub(r"[^a-z0-9]", "-", recipe.title.lower())
    if not parent_id:
        import uuid

        parent_id = uuid.uuid4().hex

    save_parent(parent_store_path, parent_id, recipe.raw_text)

    chunks = chunk_recipe_by_section(recipe)
    logger.info("Split '%s' into %d child chunks", recipe.title, len(chunks))

    return chunks, parent_id


def load_parent_store(store_path: str) -> dict[str, str]:
    """Load the parent document store from disk.

    Args:
        store_path: Path to the JSON parent store file.

    Returns:
        Dictionary mapping parent IDs to full document text.
    """
    if os.path.exists(store_path):
        with open(store_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_parent(parent_store_path: str, parent_id: str, text: str) -> None:
    """Save a parent document to the JSON store.

    Args:
        parent_store_path: Path to the JSON parent store file.
        parent_id: Unique identifier for the parent document.
        text: Full text content of the parent document.
    """
    store = load_parent_store(parent_store_path)
    store[parent_id] = text

    os.makedirs(os.path.dirname(parent_store_path), exist_ok=True)
    with open(parent_store_path, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)
