"""Retrieval, re-ranking, and answer generation service."""

import chromadb
from sentence_transformers import CrossEncoder

from src.config import get_settings
from src.core.logging import get_logger
from src.services.ingestion import load_parent_store
from src.services.search import BM25Index, reciprocal_rank_fusion

logger = get_logger(__name__)


def retrieve_and_rerank(
    question: str,
    collection: chromadb.Collection,
    cross_encoder: CrossEncoder,
    parent_store_path: str,
    bm25_index: BM25Index,
    extracted_filters: dict[str, str | None] = None,
) -> list[str]:
    """Retrieve candidate documents using hybrid search and re-rank.

    Implements the core retrieval pipeline:
    1. ChromaDB vector search → parent IDs (ranked list A)
    2. BM25 keyword search → parent IDs (ranked list B)
    3. RRF fusion → combined ranked list
    4. Cross-encoder re-rank on fused list
    5. Return top results

    Args:
        question: The user's question.
        collection: ChromaDB collection to search.
        cross_encoder: Cross-encoder model for re-ranking.
        parent_store_path: Path to the parent document store.
        bm25_index: The BM25Index instance.
        extracted_filters: Optional metadata filters.

    Returns:
        List of top-ranked parent document texts (may be empty).
    """
    if extracted_filters is None:
        extracted_filters = {}

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

    # Semantic retrieval (dense)
    results = collection.query(
        query_texts=[question],
        n_results=10,  # Reduced from 20 due to better chunking
        where=where_clause if where_clause else None,
    )

    vector_parent_ids = []
    if results["documents"][0]:
        children_metadata = results["metadatas"][0]
        # Keep order while removing duplicates
        seen = set()
        for meta in children_metadata:
            pid = meta["parent_id"]
            if pid not in seen:
                vector_parent_ids.append(pid)
                seen.add(pid)

    # Keyword retrieval (sparse)
    bm25_results = bm25_index.search(question, top_k=10)

    logger.info("Vector parents: %d, BM25 parents: %d", len(vector_parent_ids), len(bm25_results))

    # Fusion
    if not vector_parent_ids and not bm25_results:
        logger.info("No documents found for query")
        return []

    fused_parent_ids = reciprocal_rank_fusion(vector_parent_ids, bm25_results)

    # Parent resolution
    parent_store = load_parent_store(parent_store_path)
    candidate_parents = [parent_store[pid] for pid in fused_parent_ids if pid in parent_store]

    if not candidate_parents:
        logger.warning("No parent documents found in store for IDs: %s", fused_parent_ids)
        return []

    # Cross-encoder re-ranking
    pairs = [[question, parent_text] for parent_text in candidate_parents]
    scores = cross_encoder.predict(pairs)

    logger.debug("Re-rank scores: %s", list(zip(fused_parent_ids, scores, strict=False)))

    scored_parents = sorted(
        zip(scores, candidate_parents, strict=False), key=lambda x: x[0], reverse=True
    )

    settings = get_settings()
    threshold = settings.reranker_score_threshold

    top_parents = [doc for score, doc in scored_parents[:2] if score > threshold]
    logger.info(
        "Re-ranking complete: %d/%d parents passed threshold",
        len(top_parents),
        len(candidate_parents),
    )

    return top_parents
