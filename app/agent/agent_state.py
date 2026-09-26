"""Shared state definitions used by the LangGraph workflow."""

from typing import Any, Literal, TypedDict


# The router must return one of these explicit categories so the graph can
# select a deterministic downstream path.
RouteCategory = Literal[
    "restaurant_specific",
    "general_food",
    "out_of_scope",
    "ambiguous",
]


class AgentState(TypedDict, total=False):
    """Store data shared between workflow nodes."""

    # The current user input enters the graph through this field.
    question: str
    # Recent messages are loaded from SQLite before the graph starts.
    conversation_history: list[dict[str, str]]
    # The router writes this value and conditional edges read it.
    route: RouteCategory
    # The validated router plan controls the retrieval strategy.
    query_plan: dict[str, Any]
    # Retrieved text is kept separate from the final generated answer.
    context: str
    # Retrieval metadata is later exposed as safe public source information.
    retrieved_documents: list[dict[str, Any]]
    answer: str
    # Nodes can use this field for controlled error propagation if needed.
    error: str | None
