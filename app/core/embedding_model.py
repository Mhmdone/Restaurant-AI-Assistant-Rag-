"""Centralized loading and caching for the sentence-transformer model."""

import logging
import os
import time
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# This model is used only to create embeddings for queries and documents.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


def local_files_only_enabled() -> bool:
    """Return whether the embedding model must be loaded from local files."""
    load_dotenv(PROJECT_ROOT / ".env")
    # Runtime services should use the already-downloaded model and avoid
    # unexpected network calls during application startup.
    return os.getenv(
        "EMBEDDING_LOCAL_FILES_ONLY",
        "true",
    ).lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once per application process."""
    started_at = time.perf_counter()
    local_files_only = local_files_only_enabled()
    logger.info(
        "embedding_model_load_started model=%s local_files_only=%s",
        MODEL_NAME,
        local_files_only,
    )

    # The cache decorator ensures every caller receives the same model object.
    model = SentenceTransformer(
        MODEL_NAME,
        local_files_only=local_files_only,
    )

    duration_ms = (time.perf_counter() - started_at) * 1000
    logger.info(
        "embedding_model_load_completed duration_ms=%.2f",
        duration_ms,
    )
    return model
