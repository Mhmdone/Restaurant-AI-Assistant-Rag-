"""Offline script that generates vectors for prepared RAG documents."""

import json
from pathlib import Path

import numpy as np

from app.core.embedding_model import get_embedding_model

# Input documents are created by rag_loader.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_PATH = PROJECT_ROOT / "data/offonburg_documents.json"
EMBEDDINGS_PATH = PROJECT_ROOT / "data/offonburg_embeddings.npy"
EMBEDDED_DOCUMENTS_PATH = PROJECT_ROOT / "data/offonburg_embedded_documents.json"


def load_documents(path: Path) -> list[dict]:
    """Load prepared RAG documents from JSON."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    """Generate and save embeddings for all documents."""
    documents = load_documents(DOCUMENTS_PATH)
    # The centralized loader keeps this preprocessing script consistent with
    # the model used later for query embeddings during retrieval.
    model = get_embedding_model()

    texts = [document["content"] for document in documents]

    # Convert each document into a multilingual numerical vector.
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    np.save(EMBEDDINGS_PATH, embeddings)

    for document, embedding_index in zip(documents, range(len(embeddings))):
        document["embedding_index"] = embedding_index

    with EMBEDDED_DOCUMENTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Created embeddings with shape: {embeddings.shape}")
    print(f"Saved embeddings to: {EMBEDDINGS_PATH}")
    print(f"Saved document metadata to: {EMBEDDED_DOCUMENTS_PATH}")


if __name__ == "__main__":
    main()
