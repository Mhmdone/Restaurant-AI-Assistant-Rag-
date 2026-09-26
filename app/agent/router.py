"""LLM-based question classification for deterministic workflow routing."""

import json
import logging
import re
import time
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ValidationError

from app.core.llm_client import request_completion
from app.rag.retrieve import load_documents, normalize_text


# These retries handle malformed model output, not network failures.
MAX_ROUTER_RETRIES = 3
ROUTER_RETRY_DELAY = 1

logger = logging.getLogger(__name__)


class QuestionRoute(BaseModel):
    """Represent the validated query plan returned by the router."""

    # Pydantic rejects categories that are not supported by the graph.
    category: Literal[
        "restaurant_specific",
        "general_food",
        "out_of_scope",
        "ambiguous",
    ]
    operation: Literal[
        "single_item",
        "list",
        "comparison",
        "filter",
        "aggregate",
    ] = "single_item"
    domain: Literal[
        "menu",
        "ingredients",
        "nutrition",
        "price",
        "availability",
        "general_food",
        "restaurant_info",
    ] = "menu"
    menu_category: str | None = None
    target_item: str | None = None
    requires_full_catalog: bool = False
    calorie_mode: Literal[
        "not_requested",
        "exact_only",
        "estimate_if_needed",
    ] = "not_requested"


ROUTER_PROMPT = """
You are a query-planning classifier for a restaurant assistant.

The database contains restaurant menu records with these fields:
- item_name
- category
- ingredients
- price
- availability
- description

The catalog categories and item names are provided in the user message.
Use them to resolve spelling variations and identify the relevant category.

Your task is only to classify the question and create a retrieval plan.
Do not answer the question.
Do not invent restaurant facts.
Do not generate SQL.

Categories:
- restaurant_specific: about this restaurant, its menu, or its public
  restaurant information such as address, location, phone number, opening
  hours, and food types.
- general_food: about food in general and not about this restaurant.
- out_of_scope: unrelated to food or the restaurant.
- ambiguous: unclear and requiring clarification.

Operations:
- single_item: about one identifiable menu item.
- list: asks for multiple or all matching menu items.
- comparison: asks for the best, cheapest, most expensive, lightest,
  lowest-calorie, or highest-calorie option.
- filter: asks for items matching a condition.
- aggregate: asks for a total, count, or combined value.

Domains:
- menu, ingredients, nutrition, price, availability, general_food,
  restaurant_info.

Rules:
- Set requires_full_catalog to true for list, comparison, filter, or aggregate.
- A lowest-calorie or highest-calorie request is a nutrition comparison.
- Use calorie_mode=exact_only when exact calorie data is required.
- Use calorie_mode=estimate_if_needed when the user requests an estimate or
  when a calorie comparison cannot be answered without estimation.
- If a spelling variation clearly matches a known item, return its canonical
  item name in target_item.
- Return menu_category only when it is supported by the catalog description.
- Questions such as "Where is the restaurant?", "What is the address?",
  "What is the phone number?", and "What are the opening hours?" are
  restaurant_specific with domain=restaurant_info, not out_of_scope.

Return only valid JSON in this format:
{
  "category": "restaurant_specific",
  "operation": "single_item",
  "domain": "menu",
  "menu_category": null,
  "target_item": null,
  "requires_full_catalog": false,
  "calorie_mode": "not_requested"
}
"""


@lru_cache(maxsize=1)
def build_catalog_summary() -> str:
    """Build a compact catalog description for the router prompt."""
    documents = load_documents()
    grouped_items: dict[str, list[str]] = {}

    for document in documents:
        metadata = document.get("metadata", {})
        category = metadata.get("category") or "unknown"
        item_name = metadata.get("item_name") or "unnamed item"
        grouped_items.setdefault(category, []).append(item_name)

    lines = ["Catalog categories and known item names:"]
    for category, item_names in grouped_items.items():
        lines.append(f"- {category}: {', '.join(item_names)}")

    return "\n".join(lines)


def normalize_query_plan(
    plan: QuestionRoute,
    question: str,
) -> QuestionRoute:
    """Enforce safe plan invariants after validating the model response."""
    normalized_question = normalize_text(question)
    full_catalog_operations = {"list", "comparison", "filter", "aggregate"}
    requires_full_catalog = (
        plan.requires_full_catalog
        or plan.operation in full_catalog_operations
    )

    # Keep the plan consistent when the user explicitly asks about restaurant
    # location or contact details. This corrects an overly strict LLM route
    # while still keeping the LLM as the primary classifier.
    restaurant_info_terms = (
        "ادرس",
        "نشانی",
        "کجاست",
        "کجا قرار دارد",
        "کجا قرار گرفته",
        "مکان",
        "موقعیت",
        "لوکیشن",
        "محله",
        "منطقه",
        "چطور برسم",
        "تلفن",
        "شماره تماس",
        "شماره",
        "ساعت کاری",
        "ساعت فعالیت",
        "ساعت باز",
        "باز هست",
        "روزهای کاری",
        "چه ساعتی",
        "نوع غذا",
        "address",
        "where",
        "where is the restaurant",
        "located",
        "restaurant location",
        "location",
        "phone",
        "contact",
        "opening hours",
        "hours",
    )
    is_restaurant_info_question = any(
        term in normalized_question for term in restaurant_info_terms
    )

    category = plan.category
    operation = plan.operation
    domain = plan.domain
    if is_restaurant_info_question:
        category = "restaurant_specific"
        operation = "single_item"
        domain = "restaurant_info"

    # Keep the plan consistent when the user explicitly mentions calories.
    calorie_terms = (
        "کالری",
        "کم کالری",
        "پرکالری",
        "ارزش غذایی",
        "calorie",
        "calories",
        "low calorie",
        "high calorie",
    )
    if any(term in normalized_question for term in calorie_terms):
        domain = "nutrition"

    calorie_mode = plan.calorie_mode
    estimation_terms = (
        "تقریبی",
        "تخمین",
        "حدودی",
        "حدود",
        "estimate",
        "estimated",
        "approximate",
    )
    if any(term in normalized_question for term in estimation_terms):
        calorie_mode = "estimate_if_needed"
    elif domain == "nutrition" and calorie_mode == "not_requested":
        # A direct nutrition question should require an exact stored value by
        # default; estimation must be explicitly requested or necessary for a
        # comparison such as the lowest-calorie item.
        calorie_mode = "exact_only"

    if domain == "nutrition" and plan.operation == "comparison":
        # A ranking question cannot be answered without comparing values.
        calorie_mode = "estimate_if_needed"

    return plan.model_copy(
        update={
            "category": category,
            "operation": operation,
            "domain": domain,
            "requires_full_catalog": requires_full_catalog,
            "calorie_mode": calorie_mode,
        }
    )


def classify_question(
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> QuestionRoute:
    """Classify a question before selecting an answer strategy."""
    last_error: Exception | None = None
    # History helps resolve follow-up questions such as "what is its price?".
    history_text = format_conversation_history(conversation_history or [])
    normalized_question = normalize_text(question)
    user_content = (
        f"{build_catalog_summary()}\n\n"
        f"Conversation history:\n{history_text}\n\n"
        f"Original question:\n{question}\n\n"
        f"Normalized question for matching:\n{normalized_question}"
    )

    for attempt in range(MAX_ROUTER_RETRIES):
        try:
            # The model is instructed to return JSON; validation happens below.
            response = request_completion(
                messages=[
                    {"role": "system", "content": ROUTER_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )

            raw_answer = response.choices[0].message.content
            if not raw_answer or not raw_answer.strip():
                raise ValueError("The router returned an empty response.")

            answer = raw_answer.strip()

            # Remove Markdown code fences if the model wraps the JSON in them.
            answer = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                answer,
            ).strip()

            # Parse and validate before allowing the value to control routing.
            plan = QuestionRoute.model_validate(json.loads(answer))
            return normalize_query_plan(plan, question)

        except (json.JSONDecodeError, ValidationError, ValueError) as error:
            last_error = error
            logger.warning(
                "Invalid router response on attempt %s/%s: %s",
                attempt + 1,
                MAX_ROUTER_RETRIES,
                error,
            )

            if attempt < MAX_ROUTER_RETRIES - 1:
                # A short delay gives a transient provider response time to recover.
                time.sleep(ROUTER_RETRY_DELAY)

    raise ValueError(
        "The router failed to return a valid category after retries."
    ) from last_error


def format_conversation_history(
    conversation_history: list[dict[str, str]],
) -> str:
    """Format stored messages for a prompt without exposing database details."""
    if not conversation_history:
        return "No previous messages."

    # Keep the role labels so the model can distinguish user input from answers.
    return "\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in conversation_history
    )


if __name__ == "__main__":
    user_question = input("Enter your question: ").strip()
    route = classify_question(user_question)
    print(route.model_dump_json(indent=2))
