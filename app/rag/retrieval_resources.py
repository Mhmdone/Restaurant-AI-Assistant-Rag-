"""Shared RAG resources loaded once and reused by all application paths."""

import logging
import time
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.embedding_model import get_embedding_model
from app.rag.retrieve import (
    load_documents,
    load_embeddings,
)


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_retrieval_resources() -> tuple[
    SentenceTransformer,
    list[dict],
    np.ndarray,
]:
    """Load shared retrieval resources once per application process."""
    started_at = time.perf_counter()
    logger.info("retrieval_resources_load_started")

    # The model loader has its own cache, so this call never creates a second
    # embedding model within the same process.
    model = get_embedding_model()
    # Documents and their precomputed vectors are static application data.
    documents = load_documents()
    embeddings = load_embeddings()

    duration_ms = (time.perf_counter() - started_at) * 1000
    logger.info(
        "retrieval_resources_load_completed duration_ms=%.2f documents=%s "
        "embedding_shape=%s embedding_model_cached=true",
        duration_ms,
        len(documents),
        embeddings.shape,
    )

    return model, documents, embeddings
