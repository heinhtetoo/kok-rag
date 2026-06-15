"""Centralised application configuration using Pydantic Settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings

from src.core.constants import (
    COLLECTION_NAME,
    PARENT_STORE_PATH,
    RECIPE_DIR,
    VECTOR_DB_DIR,
)


class Settings(BaseSettings):
    """Application settings loaded from environment variables and ``.env`` file."""

    # API Security
    kok_api_key: str

    # Ollama LLM
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    # Storage paths
    recipe_dir: str = RECIPE_DIR
    vector_db_dir: str = VECTOR_DB_DIR
    collection_name: str = COLLECTION_NAME
    parent_store_path: str = PARENT_STORE_PATH

    # ML Models
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Agent
    agent_max_iterations: int = 5
    web_search_max_results: int = 5

    # Logging
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of application settings."""
    return Settings()
