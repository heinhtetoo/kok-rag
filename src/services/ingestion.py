"""Recipe ingestion and chunking service."""

import json
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.logging import get_logger

logger = get_logger(__name__)


def chunk_recipe(
    filename: str,
    recipe_dir: str,
    parent_store_path: str,
) -> tuple[list[str], str]:
    """Split a recipe file into chunks using the parent-child strategy.

    The full recipe text is saved as the parent document in a JSON sidecar
    store, and small child chunks are returned for vector embedding.

    Args:
        filename: Name of the recipe text file.
        recipe_dir: Directory containing recipe files.
        parent_store_path: Path to the parent document JSON store.

    Returns:
        A tuple of ``(child_chunks, parent_id)``.

    Raises:
        FileNotFoundError: If the recipe file does not exist.
    """
    parent_id = filename.removesuffix(".txt")

    file_path = os.path.join(recipe_dir, filename)
    if not os.path.isfile(file_path) or not filename.endswith(".txt"):
        raise FileNotFoundError(f"Recipe file not found: {file_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20,
        separators=["\n\n", "\n", ". ", " ", "."],
    )

    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    save_parent(parent_store_path, parent_id, text)

    chunks = splitter.split_text(text)
    logger.info("Split '%s' into %d child chunks", filename, len(chunks))

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
