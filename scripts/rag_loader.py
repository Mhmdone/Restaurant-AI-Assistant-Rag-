"""Convert restaurant JSON data into searchable RAG documents."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = PROJECT_ROOT / "data/offonburg_clean_rag.json"
OUTPUT_PATH = PROJECT_ROOT / "data/offonburg_documents.json"


@dataclass
class Document:
    """Represent one searchable document in the RAG pipeline."""

    # Content is embedded and sent to the LLM as retrieval context.
    content: str
    # Metadata is used for filtering, citations, and API source information.
    metadata: dict[str, Any]


def load_restaurant_data(path: Path) -> dict[str, Any]:
    """Load and parse the restaurant JSON file."""
    # UTF-8 preserves Persian menu content during the JSON-to-document step.
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_restaurant_document(data: dict[str, Any]) -> Document:
    """Convert general restaurant information into one document."""
    restaurant = data["restaurant"]

    # Restaurant-level facts are kept in one document for broad questions.
    base_content = f"""
Restaurant name: {restaurant.get("name", "")}
Address: {restaurant.get("address", "")}
Phone: {restaurant.get("phone", "")}
Food types: {", ".join(restaurant.get("food_types", []))}
Service hours: {restaurant.get("service_hours", {}).get("raw", "")}
""".strip()

    # This field stores free-form background information about the restaurant.
    profile_text = restaurant.get("profile_text", "").strip()
    content_parts = [base_content]
    if profile_text:
        content_parts.append(
            f"Additional restaurant information:\n{profile_text}"
        )

    content = "\n\n".join(content_parts)

    # Metadata is deliberately separate from text so the API can expose safe
    # source fields without returning the full internal context.
    metadata = {
        "source": "offonburg_clean_rag.json",
        "document_type": "restaurant_information",
    }

    return Document(content=content, metadata=metadata)


def build_menu_document(menu_item: dict[str, Any]) -> Document:
    """Convert one menu item into one searchable document."""
    ingredients = menu_item.get("ingredients", [])
    # Store each ingredient on its own line to improve keyword retrieval.
    ingredients_text = "\n".join(
        f"- {ingredient}" for ingredient in ingredients
    )

    price = menu_item.get("price_toman")
    price_text = f"{price} toman" if price is not None else "Not available"

    # One menu item per document prevents unrelated menu items from mixing in
    # the same retrieval result.
    content = f"""
Restaurant: مجموعه غذایی آفان
Category: {menu_item.get("category", "")}
Item name: {menu_item.get("name", "")}
Ingredients:
{ingredients_text or "- Not available"}
Price: {price_text}
Availability: {menu_item.get("available", "unknown")}
Description: {menu_item.get("source_description", "")}
""".strip()

    metadata = {
        "source": "offonburg_clean_rag.json",
        "document_type": "menu_item",
        "item_name": menu_item.get("name"),
        "category": menu_item.get("category"),
        "price_toman": price,
        "availability": menu_item.get("available"),
    }

    return Document(content=content, metadata=metadata)


def build_documents(data: dict[str, Any]) -> list[Document]:
    """Build restaurant and menu documents from the source JSON."""
    # Start with restaurant-level information, then append one document per
    # menu item so both broad and item-specific questions are supported.
    documents = [
        build_restaurant_document(data),
    ]

    for menu_item in data.get("menu", []):
        documents.append(build_menu_document(menu_item))

    return documents


def save_documents(documents: list[Document], path: Path) -> None:
    """Save documents in a JSON format for later embedding."""
    # Dataclasses are converted into plain dictionaries before JSON encoding.
    serialized_documents = [asdict(document) for document in documents]

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            serialized_documents,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    """Load the source JSON, create documents, and save the result."""
    data = load_restaurant_data(JSON_PATH)
    documents = build_documents(data)
    save_documents(documents, OUTPUT_PATH)

    print(f"Created {len(documents)} documents.")
    print(f"Saved documents to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
