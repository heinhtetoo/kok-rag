"""FastAPI application factory with lifespan management."""

from contextlib import asynccontextmanager

import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from ollama import Client as OllamaClient
from sentence_transformers import CrossEncoder

from src.config import get_settings
from src.core.logging import get_logger, setup_logging
from src.routers import ask, health, ingest

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown.

    Initialises all heavyweight resources (ChromaDB, Ollama, cross-encoder)
    during startup and stores them on ``app.state`` for dependency injection.
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    logger.info("Starting Kök RAG API...")

    # Initialise ChromaDB
    db_client = chromadb.PersistentClient(path=settings.vector_db_dir)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )
    app.state.collection = db_client.get_or_create_collection(
        name=settings.collection_name, embedding_function=ef
    )
    logger.info("ChromaDB collection '%s' ready", settings.collection_name)

    # Initialise Ollama client
    app.state.ollama_client = OllamaClient(host=settings.ollama_host)
    logger.info("Ollama client configured for %s", settings.ollama_host)

    # Initialise Cross-Encoder
    logger.info("Loading cross-encoder model '%s'...", settings.reranker_model)
    app.state.cross_encoder = CrossEncoder(settings.reranker_model)
    logger.info("Cross-encoder ready")

    yield

    logger.info("Shutting down Kök RAG API...")


app = FastAPI(
    title="Kök RAG API",
    version="1.0.0",
    description="Retrieval-Augmented Generation API for culinary recipes",
    lifespan=lifespan,
)

app.include_router(ask.router)
app.include_router(ingest.router)
app.include_router(health.router)
