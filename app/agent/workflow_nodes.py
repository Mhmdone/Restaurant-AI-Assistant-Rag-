"""LangGraph node functions for routing, retrieval, and answer generation."""

import logging
import time

from app.agent.agent_state import AgentState
from app.agent.router import classify_question, format_conversation_history
from app.core.llm_client import request_completion
from app.rag.rag_qa import (
    answer_general_food_question,
    build_rag_messages,
    build_context,
    build_restaurant_profile,
)
from app.rag.retrieve import (
    search_catalog,
    search_documents,
    search_restaurant_info,
)
from app.rag.retrieval_resources import get_retrieval_resources


logger = logging.getLogger(__name__)


def route_question_node(state: AgentState) -> AgentState:
    """Classify the user question and store the selected route."""
    # The route is written to state and consumed by workflow.py conditional edges.
    plan = classify_question(
        state["question"],
        state.get("conversation_history", []),
    )
    return {
        **state,
        "route": plan.category,
        "query_plan": plan.model_dump(),
    }


def retrieve_context_for_state(
    state: AgentState,
) -> tuple[list[dict], str]:
    """Retrieve documents and build context for a restaurant question."""
    started_at = time.perf_counter()
    # This call returns cached resources after the first application load.
    model, documents, embeddings = get_retrieval_resources()

    history_text = format_conversation_history(
        state.get("conversation_history", [])
    )
    # Include recent history so follow-up questions can be resolved semantically.
    search_query = (
        f"Conversation history:\n{history_text}\n\n"
        f"Current question:\n{state['question']}"
    )
    query_plan = state.get("query_plan", {})
    if query_plan.get("domain") == "restaurant_info":
        # Public restaurant details have a dedicated document and do not need
        # approximate semantic ranking against individual menu items.
        results = search_restaurant_info(documents)
    elif query_plan.get("requires_full_catalog"):
        # List and comparison plans must inspect the complete relevant catalog.
        results = search_catalog(
            query=state["question"],
            documents=documents,
            menu_category=query_plan.get("menu_category"),
        )
    else:
        # Normal questions use bounded hybrid retrieval to keep context compact.
        results = search_documents(
            query=search_query,
            model=model,
            documents=documents,
            embeddings=embeddings,
        )
    duration_ms = (time.perf_counter() - started_at) * 1000
    top_score = results[0].get("hybrid_score") if results else None
    logger.info(
        "retrieval_completed duration_ms=%.2f results=%s top_hybrid_score=%s",
        duration_ms,
        len(results),
        f"{top_score:.4f}" if top_score is not None else "none",
    )

    return results, build_context(
        results,
        restaurant_profile=build_restaurant_profile(documents),
    )


def retrieve_context_node(state: AgentState) -> AgentState:
    """Retrieve restaurant documents for a restaurant-specific question."""
    results, context = retrieve_context_for_state(state)

    return {
        **state,
        "retrieved_documents": results,
        "context": context,
    }


def generate_rag_answer_node(state: AgentState) -> AgentState:
    """Generate a grounded answer from the retrieved context."""
    # Only the system prompt, retrieved context, history, and current question
    # are sent to the chat model for this branch.
    response = request_completion(
        messages=build_rag_messages(
            question=state["question"],
            context=state["context"],
            conversation_history=state.get("conversation_history", []),
            query_plan=state.get("query_plan"),
        ),
    )
    answer = response.choices[0].message.content or ""
    return {**state, "answer": answer}


def generate_general_answer_node(state: AgentState) -> AgentState:
    """Generate an answer using general food knowledge."""
    # This branch intentionally does not include restaurant retrieval context.
    answer = answer_general_food_question(
        state["question"],
        state.get("conversation_history", []),
    )
    return {**state, "answer": answer}


def handle_out_of_scope_node(state: AgentState) -> AgentState:
    """Return a controlled response for an unrelated question."""
    # Keep unrelated requests within the supported product scope.
    return {
        **state,
        "answer": (
            "I am the OffOn restaurant assistant chatbot. "
            "I can help with food and restaurant questions."
        ),
    }


def handle_ambiguous_node(state: AgentState) -> AgentState:
    """Ask the user to clarify an ambiguous question."""
    # Asking for clarification is safer than guessing the user's intent.
    return {
        **state,
        "answer": (
            "I am the OffOn restaurant assistant chatbot. "
            "Could you clarify your question?"
        ),
    }
