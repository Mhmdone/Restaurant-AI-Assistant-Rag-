"""Hybrid semantic and keyword retrieval for restaurant documents."""

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

# These files are generated during the offline document preparation phase.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_PATH = PROJECT_ROOT / "data/offonburg_embeddings.npy"
DOCUMENTS_PATH = PROJECT_ROOT / "data/offonburg_embedded_documents.json"

TOP_K = 3
FUZZY_ITEM_NAME_THRESHOLD = 0.82
CATALOG_BURGER_CATEGORIES = {"ساندویچ", "اسپشیال"}

# These replacements make common Persian and Arabic spelling variants
# comparable before exact or fuzzy matching is performed.
PERSIAN_CHARACTER_TRANSLATIONS = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
        "ؤ": "و",
        "ئ": "ی",
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ـ": "",
    }
)
ARABIC_DIACRITICS_PATTERN = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670]")
NON_WORD_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize Persian and Arabic character differences."""
    # NFKC handles compatible Unicode forms before language-specific cleanup.
    normalized_text = unicodedata.normalize("NFKC", text).lower()
    normalized_text = normalized_text.translate(PERSIAN_CHARACTER_TRANSLATIONS)
    normalized_text = normalized_text.replace("\u200c", " ")
    normalized_text = ARABIC_DIACRITICS_PATTERN.sub("", normalized_text)
    normalized_text = NON_WORD_PATTERN.sub(" ", normalized_text)
    return WHITESPACE_PATTERN.sub(" ", normalized_text).strip()


def tokenize(text: str) -> set[str]:
    """Extract normalized word tokens from text."""
    # Unicode-aware tokenization keeps Persian words available for keyword search.
    normalized_text = normalize_text(text)
    return set(re.findall(r"\w+", normalized_text, flags=re.UNICODE))


def load_documents() -> list[dict]:
    """Load embedded document metadata."""
    with DOCUMENTS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_embeddings() -> np.ndarray:
    """Load document embeddings from disk."""
    # The array index must stay aligned with the document list index.
    return np.load(EMBEDDINGS_PATH)


def calculate_keyword_score(query: str, document: dict) -> float:
    """Calculate lexical similarity between the query and a document."""
    normalized_query = normalize_text(query)
    content = normalize_text(document["content"])
    item_name = normalize_text(
        document.get("metadata", {}).get("item_name") or ""
    )

    # Give the highest lexical score to an exact normalized menu item match.
    if item_name and item_name in normalized_query:
        return 1.0

    # Fuzzy matching handles small typos and missing or substituted characters.
    item_name_similarity = (
        fuzz.partial_ratio(item_name, normalized_query) / 100
        if item_name
        else 0.0
    )

    query_tokens = tokenize(query)
    content_tokens = tokenize(content)

    if not query_tokens:
        return 0.0

    # Token overlap provides a lightweight lexical signal alongside embeddings.
    overlap = query_tokens.intersection(content_tokens)
    token_score = len(overlap) / len(query_tokens)

    # Only accept fuzzy names above the threshold to avoid unrelated matches.
    if item_name_similarity >= FUZZY_ITEM_NAME_THRESHOLD:
        return max(item_name_similarity, token_score)

    return token_score


def canonicalize_menu_category(category: str | None) -> str | None:
    """Map common English or Persian category variants to stored values."""
    if not category:
        return None

    normalized_category = normalize_text(category)
    aliases = {
        "sandwich": "ساندویچ",
        "ساندویچ": "ساندویچ",
        "burger": "ساندویچ",
        "برگر": "ساندویچ",
        "همبرگر": "ساندویچ",
        "special": "اسپشیال",
        "اسپشیال": "اسپشیال",
        "salad": "سالاد و پیش غذا",
        "سالاد و پیش غذا": "سالاد و پیش غذا",
        "appetizer": "سالاد و پیش غذا",
        "sauce": "سس ها",
        "سس": "سس ها",
        "سس ها": "سس ها",
        "drink": "نوشیدنی",
        "drinks": "نوشیدنی",
        "نوشیدنی": "نوشیدنی",
        "extra": "اضافات",
        "extras": "اضافات",
        "اضافه": "اضافات",
        "اضافات": "اضافات",
    }
    return aliases.get(normalized_category, category)


def search_catalog(
    query: str,
    documents: list[dict],
    menu_category: str | None = None,
) -> list[dict]:
    """Return the complete catalog subset needed by list or comparison queries."""
    normalized_query = normalize_text(query)
    canonical_category = canonicalize_menu_category(menu_category)
    normalized_category = normalize_text(canonical_category or "")

    candidates = list(documents)
    if normalized_category:
        candidates = [
            document
            for document in candidates
            if normalize_text(
                document.get("metadata", {}).get("category") or ""
            )
            == normalized_category
        ]

    burger_terms = ("برگر", "همبرگر", "burger")
    is_burger_query = any(term in normalized_query for term in burger_terms)
    explicitly_requests_extras = any(
        term in normalized_query
        for term in ("اضافه", "نان", "گوشت")
    )

    if is_burger_query:
        burger_candidates = []
        for document in candidates:
            metadata = document.get("metadata", {})
            category = normalize_text(metadata.get("category") or "")
            content = normalize_text(document.get("content", ""))
            contains_burger = any(term in content for term in burger_terms)

            # Main burger dishes and burger-containing specials are the default
            # result; raw ingredients and bread are included only when requested.
            is_allowed_category = category in {
                normalize_text(value) for value in CATALOG_BURGER_CATEGORIES
            }
            if contains_burger and (
                is_allowed_category or explicitly_requests_extras
            ):
                burger_candidates.append(document)
        candidates = burger_candidates

    # Keep the original document order for complete catalog listings.
    return [document.copy() for document in candidates]


def search_restaurant_info(documents: list[dict]) -> list[dict]:
    """Return the dedicated restaurant-information documents."""
    # Restaurant address, phone, hours, and food types are stored separately
    # from menu items, so these questions should not depend on ranking scores.
    return [
        document.copy()
        for document in documents
        if document.get("metadata", {}).get("document_type")
        == "restaurant_information"
    ]


def search_documents(
    query: str,
    model: SentenceTransformer,
    documents: list[dict],
    embeddings: np.ndarray,
    top_k: int = TOP_K,
) -> list[dict]:
    """Rank documents using semantic and keyword search."""
    # Encode only the current query; document embeddings were precomputed.
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    # Normalized vectors make the dot product equivalent to cosine similarity.
    semantic_scores = embeddings @ query_embedding

    hybrid_scores = []
    for index, document in enumerate(documents):
        keyword_score = calculate_keyword_score(query, document)

        # Combine semantic meaning with exact keyword overlap for hybrid ranking.
        final_score = (
            0.7 * float(semantic_scores[index])
            + 0.3 * keyword_score
        )

        hybrid_scores.append(final_score)

    # Return only the highest-ranked documents to keep the LLM context small.
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    results = []
    for index in top_indices:
        result = documents[index].copy()
        result["semantic_score"] = float(semantic_scores[index])
        result["hybrid_score"] = float(hybrid_scores[index])
        results.append(result)

    return results


def main() -> None:
    """Run a hybrid semantic and keyword search."""
    # Import lazily to avoid a circular import during normal module loading.
    from app.rag.retrieval_resources import get_retrieval_resources

    model, documents, embeddings = get_retrieval_resources()

    query = input("Enter your question: ").strip()

    results = search_documents(
        query=query,
        model=model,
        documents=documents,
        embeddings=embeddings,
    )

    for rank, result in enumerate(results, start=1):
        print(f"\nResult {rank}")
        print(f"Hybrid score: {result['hybrid_score']:.4f}")
        print(f"Semantic score: {result['semantic_score']:.4f}")
        print(result["content"])


if __name__ == "__main__":
    main()
