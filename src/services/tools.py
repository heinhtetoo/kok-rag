"""Tool definitions for the Kök agentic loop.

Each tool is a plain Python callable whose signature, type hints, and
docstring are automatically converted by the Ollama SDK into the JSON
schema sent to the model. Dependencies are injected at construction time
via ``make_tools()`` — the LLM only sees the ``query: str`` parameter.

Side-effects (which tool ran, what sources were retrieved) are recorded
into a ``ToolState`` instance shared between the tools and the agent harness,
avoiding the need to parse results a second time after the loop.
"""

from dataclasses import dataclass, field

import chromadb
from duckduckgo_search import DDGS
from sentence_transformers import CrossEncoder

from src.core.logging import get_logger
from src.services.retrieval import retrieve_and_rerank
from src.services.search import BM25Index

logger = get_logger(__name__)


@dataclass
class ToolState:
    """Mutable side-channel written by tools during execution.

    The agent harness reads this after the loop to populate ``AgentResult``
    without having to re-parse tool output strings.
    """

    tool_used: str = "none"
    sources: list[str] = field(default_factory=list)


def make_tools(
    collection: chromadb.Collection,
    cross_encoder: CrossEncoder,
    parent_store_path: str,
    bm25_index: BM25Index,
    web_search_max_results: int,
    state: ToolState,
) -> list:
    """Build the list of tool callables with all dependencies pre-bound.

    Returns plain functions whose signatures are serialised into JSON schema
    by the Ollama SDK — the model only sees ``query: str``.

    Args:
        collection: ChromaDB collection to search.
        cross_encoder: Cross-encoder model for re-ranking recipe results.
        parent_store_path: Path to the parent document JSON store.
        web_search_max_results: Maximum DuckDuckGo results to return.
        state: Shared mutable state updated as tools execute.

    Returns:
        List of callables ready to be passed to ``ollama_client.chat(tools=...)``.
    """

    def search_recipe_book(query: str) -> str:
        """Search the personal recipe book for recipes, ingredients, and cooking instructions.

        Use this for any question about a saved recipe, dish, ingredient, or
        cooking method. Do not use this to look up general culinary knowledge
        or information the user explicitly wants from the internet.

        Args:
            query: A clear, concise search query derived from the user's question.

        Returns:
            The most relevant recipe text from the book, or a not-found message.
        """
        logger.info("Tool call — search_recipe_book(query=%r)", query)

        top_parents = retrieve_and_rerank(
            question=query,
            collection=collection,
            cross_encoder=cross_encoder,
            parent_store_path=parent_store_path,
            bm25_index=bm25_index,
            # Filters are intentionally omitted here: the LLM has already
            # distilled the query, so semantic search alone is sufficient.
            extracted_filters={"cuisine": None, "dish_type": None},
        )

        if not top_parents:
            logger.info("search_recipe_book — no results found")
            return "No relevant recipes were found in the recipe book."

        state.tool_used = "recipe_book"
        state.sources = top_parents
        logger.info("search_recipe_book — returned %d parent document(s)", len(top_parents))
        return "\n\n---\n\n".join(top_parents)

    def search_web(query: str) -> str:
        """Search the internet for general cooking knowledge, food science, or techniques.

        Use this ONLY when the user explicitly requests web or internet results,
        or asks about something that is clearly not in a personal recipe collection
        (e.g., food history, nutrition science, restaurant recommendations).

        Args:
            query: A concise search query suitable for a web search engine.

        Returns:
            Top web search results formatted as title + snippet + source URL.
        """
        logger.info("Tool call — search_web(query=%r)", query)

        try:
            # DDGS is synchronous; acceptable here as Ollama inference
            # is also synchronous and dominates latency.
            results = list(DDGS().text(query, max_results=web_search_max_results))
        except Exception as e:
            logger.error("DuckDuckGo search raised an exception: %s", e)
            return f"Web search failed: {e}"

        if not results:
            logger.info("search_web — no results returned")
            return "No web results found for that query."

        state.tool_used = "web_search"
        logger.info("search_web — returned %d result(s)", len(results))

        return "\n\n".join(f"**{r['title']}**\n{r['body']}\nSource: {r['href']}" for r in results)

    return [search_recipe_book, search_web]
