"""Search utilities including BM25 and Reciprocal Rank Fusion."""

import re

from rank_bm25 import BM25Okapi

from src.core.logging import get_logger

logger = get_logger(__name__)


def tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25."""
    return [word for word in re.split(r"\W+", text.lower()) if word]


class BM25Index:
    """In-memory BM25 index for sparse keyword retrieval."""

    def __init__(self):
        self.bm25 = None
        self.parent_ids = []

    def build(self, parent_store: dict[str, str]):
        """Build the index from the parent document store."""
        if not parent_store:
            logger.info("Parent store is empty, skipping BM25 index build.")
            self.bm25 = None
            self.parent_ids = []
            return

        self.parent_ids = list(parent_store.keys())
        corpus = list(parent_store.values())
        tokenized_corpus = [tokenize(doc) for doc in corpus]

        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("Built BM25 index over %d documents.", len(self.parent_ids))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search the BM25 index and return (parent_id, score) pairs."""
        if not self.bm25:
            return []

        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        results = [
            (self.parent_ids[i], scores[i]) for i in range(len(self.parent_ids)) if scores[i] > 0
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]


def reciprocal_rank_fusion(
    vector_results: list[str], bm25_results: list[tuple[str, float]], k: int = 60
) -> list[str]:
    """Merge ranked lists using Reciprocal Rank Fusion (RRF).

    Args:
        vector_results: List of parent_ids from vector search (already ordered).
        bm25_results: List of (parent_id, score) from BM25.
        k: RRF constant.

    Returns:
        Combined list of parent_ids ordered by RRF score.
    """
    rrf_scores = {}

    # Process vector results
    for rank, parent_id in enumerate(vector_results):
        if parent_id not in rrf_scores:
            rrf_scores[parent_id] = 0.0
        rrf_scores[parent_id] += 1.0 / (k + rank + 1)

    # Process BM25 results
    # bm25_results is sorted by score
    for rank, (parent_id, _) in enumerate(bm25_results):
        if parent_id not in rrf_scores:
            rrf_scores[parent_id] = 0.0
        rrf_scores[parent_id] += 1.0 / (k + rank + 1)

    # Sort by combined score descending
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [parent_id for parent_id, _ in fused]
