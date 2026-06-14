"""Retrieval, re-ranking, and answer generation service."""

import json

import chromadb
from ollama import Client as OllamaClient
from sentence_transformers import CrossEncoder

from src.core.logging import get_logger
from src.services.ingestion import load_parent_store

logger = get_logger(__name__)


def extract_filters_from_query(
    query: str,
    ollama_client: OllamaClient,
    model: str,
) -> dict[str, str | None]:
    """Use the LLM to extract metadata filters from a natural-language query.

    The LLM acts as a lightweight routing agent, parsing the user's query to
    extract structured ``cuisine`` and ``dish_type`` values for filtered
    vector retrieval.

    Args:
        query: The user's input query.
        ollama_client: Ollama client instance.
        model: Name of the LLM model to use.

    Returns:
        Dictionary with ``cuisine`` and ``dish_type`` keys (values may be ``None``).
    """
    system_prompt = """
    You are a strict data extraction routing agent. Analyze the user's query and extract the 'cuisine' and 'dish_type' if they are mentioned.

    Rules:
    - If a cuisine is mentioned (e.g., Burmese, Italian), set "cuisine" to that value.
    - If a single-word dish type is mentioned (e.g., Soup, Salad, Curry, Noodle), set "dish_type" to that value.
    - If either is missing, set the value to null.
    - You must ONLY output a valid JSON object. Do not add any conversational text.

    Example Output:
    {"cuisine": "Burmese", "dish_type": "Soup"}
    """

    try:
        response = ollama_client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            options={"temperature": 0.0},
        )

        raw_output = response["message"]["content"].strip()

        # Strip markdown code fences if present
        if raw_output.startswith("```json"):
            raw_output = raw_output[len("```json") :].strip()
            if raw_output.endswith("```"):
                raw_output = raw_output[: -len("```")].strip()

        return json.loads(raw_output)

    except Exception as e:
        logger.error("Filter extraction failed, defaulting to no filters: %s", e)
        return {"cuisine": None, "dish_type": None}


def retrieve_and_rerank(
    question: str,
    collection: chromadb.Collection,
    cross_encoder: CrossEncoder,
    parent_store_path: str,
    extracted_filters: dict[str, str | None],
) -> list[str]:
    """Retrieve candidate documents, resolve parents, and re-rank.

    Implements the core retrieval pipeline:
    1. Builds metadata filters from extracted query attributes.
    2. Performs cosine-similarity search over child-chunk embeddings.
    3. Maps child chunks to full parent documents.
    4. Scores parents against the query with a cross-encoder.
    5. Returns only positively-scored top results.

    Args:
        question: The user's question.
        collection: ChromaDB collection to search.
        cross_encoder: Cross-encoder model for re-ranking.
        parent_store_path: Path to the parent document store.
        extracted_filters: Metadata filters extracted from the query.

    Returns:
        List of top-ranked parent document texts (may be empty).
    """
    # Build metadata filter
    where_clause: dict = {}
    conditions: list[dict] = []

    if extracted_filters.get("cuisine"):
        conditions.append({"cuisine": extracted_filters["cuisine"].capitalize()})
    if extracted_filters.get("dish_type"):
        conditions.append({"dish_type": extracted_filters["dish_type"].capitalize()})

    if len(conditions) == 1:
        where_clause = conditions[0]
    elif len(conditions) > 1:
        where_clause = {"$and": conditions}

    # Semantic retrieval
    results = collection.query(
        query_texts=[question],
        n_results=20,
        where=where_clause if where_clause else None,
    )

    if not results["documents"][0]:
        logger.info("No documents found for query")
        return []

    # Parent resolution
    children_metadata = results["metadatas"][0]
    unique_parent_ids = list({meta["parent_id"] for meta in children_metadata})
    logger.info(
        "Retrieved %d chunks mapping to %d parent(s): %s",
        len(results["documents"][0]),
        len(unique_parent_ids),
        unique_parent_ids,
    )

    parent_store = load_parent_store(parent_store_path)
    candidate_parents = [parent_store[pid] for pid in unique_parent_ids if pid in parent_store]

    if not candidate_parents:
        logger.warning("No parent documents found in store for IDs: %s", unique_parent_ids)
        return []

    # Cross-encoder re-ranking
    pairs = [[question, parent_text] for parent_text in candidate_parents]
    scores = cross_encoder.predict(pairs)

    scored_parents = sorted(
        zip(scores, candidate_parents, strict=False), key=lambda x: x[0], reverse=True
    )

    # Keep top 2 with positive relevance score
    top_parents = [doc for score, doc in scored_parents[:2] if score > 0]
    logger.info(
        "Re-ranking complete: %d/%d parents passed threshold",
        len(top_parents),
        len(candidate_parents),
    )

    return top_parents


def generate_answer(
    question: str,
    context_documents: list[str],
    ollama_client: OllamaClient,
    model: str,
) -> str:
    """Generate an answer using the LLM with retrieved context.

    Constructs a grounded system prompt from the context documents and
    generates a deterministic response with ``temperature=0.0``.

    Args:
        question: The user's question.
        context_documents: List of relevant parent documents.
        ollama_client: Ollama client instance.
        model: Name of the LLM model to use.

    Returns:
        The generated answer string.
    """
    context = "\n\n---\n\n".join(context_documents)

    system_prompt = f"""
    You are a precise and helpful sous-chef, named Kök. Answer the user's question using ONLY the context provided below.
    If the provided context does not contain the answer, politely say "I don't have that in your recipe book."
    Do not make up cooking times, ingredients, or instructions. Format your response cleanly.

    Context:
    {context}
    """

    response = ollama_client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        options={"temperature": 0.0, "top_p": 0.9},
    )

    return response["message"]["content"].strip()
