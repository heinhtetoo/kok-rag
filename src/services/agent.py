"""Agentic tool-calling loop for Kök RAG.

Implements the standard Think → Act → Observe → Think pattern:

1. Build initial message history (system prompt + user question).
2. Call the LLM with the tool list.
3. If the model requests a tool → execute it, append both the tool-call
   message and the tool-result message, then loop back to step 2.
4. If the model returns plain text → that is the final answer; return it.

A ``max_iterations`` guard prevents infinite loops.
"""

from dataclasses import dataclass, field

import chromadb
from ollama import Client as OllamaClient
from sentence_transformers import CrossEncoder

from src.core.logging import get_logger
from src.services.search import BM25Index
from src.services.tools import ToolState, make_tools

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are Kök, a knowledgeable and helpful culinary assistant.
You have access to two tools:

1. search_recipe_book — searches the user's personal recipe collection.
2. search_web — searches the internet for culinary information.

Guidelines:
- For questions about saved recipes, ingredients, or cooking methods, \
always call search_recipe_book first.
- Only call search_web when the user explicitly asks for internet/web results, \
or when the topic is clearly general knowledge not likely saved in a personal recipe book.
- After receiving tool results, synthesise a clear, well-formatted answer.
- If no tool results are useful, say so politely — never invent recipes or facts.
- You may call at most one tool per turn.\
"""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class AgentResult:
    """The final output of a single agentic run.

    Attributes:
        answer: Final natural-language answer from the LLM.
        tool_used: Which tool was invoked — ``"recipe_book"``, ``"web_search"``,
            or ``"none"`` if the model answered directly.
        sources: Parent recipe texts returned by ``search_recipe_book``
            (empty for web search or direct answers).
    """

    answer: str
    tool_used: str = "none"
    sources: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent harness
# ---------------------------------------------------------------------------


def run_agent(
    question: str,
    ollama_client: OllamaClient,
    model: str,
    collection: chromadb.Collection,
    cross_encoder: CrossEncoder,
    parent_store_path: str,
    bm25_index: BM25Index,
    web_search_max_results: int,
    max_iterations: int = 5,
) -> AgentResult:
    """Run the agentic Think → Act → Observe loop.

    The LLM decides autonomously which tool to call (or none) based on the
    user's question and the tool docstrings. The loop continues until the
    model produces a plain-text final answer or ``max_iterations`` is reached.

    Args:
        question: The user's natural-language question.
        ollama_client: Ollama client instance.
        model: Ollama model identifier (e.g. ``"qwen2.5:7b"``).
        collection: ChromaDB collection for recipe retrieval.
        cross_encoder: Cross-encoder model for re-ranking.
        parent_store_path: Path to the parent document JSON store.
        web_search_max_results: Max DuckDuckGo results per search.
        max_iterations: Maximum number of Think–Act–Observe cycles before
            giving up. Prevents runaway agentic loops.

    Returns:
        An ``AgentResult`` with the final answer, tool metadata, and sources.
    """
    # Shared state written by tools, read by this harness after the loop
    state = ToolState()

    # Build tool callables and dispatch registry
    tools = make_tools(
        collection=collection,
        cross_encoder=cross_encoder,
        parent_store_path=parent_store_path,
        bm25_index=bm25_index,
        web_search_max_results=web_search_max_results,
        state=state,
    )
    # Map tool name → callable for O(1) dispatch
    tool_registry: dict[str, callable] = {fn.__name__: fn for fn in tools}

    # Initialise conversation history — the "source of truth" for the loop
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for iteration in range(1, max_iterations + 1):
        logger.info("Agent iteration %d/%d", iteration, max_iterations)

        response = ollama_client.chat(
            model=model,
            messages=messages,
            tools=tools,
            options={"temperature": 0.0},
        )

        # ── Terminal condition: no tool calls → model has its final answer ──
        if not response.message.tool_calls:
            answer = (response.message.content or "").strip()
            logger.info(
                "Agent finished after %d iteration(s) — tool_used=%r",
                iteration,
                state.tool_used,
            )
            return AgentResult(
                answer=answer,
                tool_used=state.tool_used,
                sources=state.sources,
            )

        # ── Append the assistant's tool-call message to history ──
        # The SDK's Message object is accepted directly by ollama_client.chat()
        messages.append(response.message)

        # ── Execute each requested tool and append its result ──
        for tool_call in response.message.tool_calls:
            fn_name = tool_call.function.name
            fn_args: dict = tool_call.function.arguments or {}

            logger.info("Dispatching tool %r with args %s", fn_name, fn_args)

            func = tool_registry.get(fn_name)
            if func is None:
                # Unknown tool — tell the model so it can recover
                tool_result = f"Error: tool '{fn_name}' does not exist."
                logger.warning("Model requested unknown tool: %r", fn_name)
            else:
                try:
                    tool_result = func(**fn_args)
                except Exception as exc:
                    tool_result = f"Tool '{fn_name}' raised an error: {exc}"
                    logger.error("Tool %r raised: %s", fn_name, exc)

            # Append tool result with role="tool" so the model can observe it
            messages.append({"role": "tool", "content": tool_result})

    # ── Safety net: max_iterations exceeded ──
    logger.warning("Agent exceeded max_iterations=%d without a final answer", max_iterations)
    return AgentResult(
        answer=(
            "I wasn't able to reach a confident answer after several attempts. "
            "Please try rephrasing your question."
        ),
        tool_used=state.tool_used,
        sources=state.sources,
    )
