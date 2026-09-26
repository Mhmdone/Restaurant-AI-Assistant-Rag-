"""LangGraph definition and runtime entry point for the restaurant assistant."""

import logging
import time
from typing import Any, Iterator

from langgraph.graph import END, START, StateGraph

from app.agent.agent_state import AgentState
from app.agent.router import classify_question
from app.agent.workflow_nodes import (
    generate_general_answer_node,
    generate_rag_answer_node,
    handle_ambiguous_node,
    handle_out_of_scope_node,
    retrieve_context_node,
    retrieve_context_for_state,
    route_question_node,
)
from app.core.llm_client import request_completion_stream
from app.memory.conversation_memory import (
    load_conversation,
    save_conversation_turn,
)
from app.rag.rag_qa import build_general_messages, build_rag_messages


logger = logging.getLogger(__name__)


def route_from_state(state: AgentState) -> str:
    """Select the next node using the route stored in the state."""
    # Conditional edges use this value to choose the next node.
    return state["route"]


def build_workflow():
    """Build and compile the restaurant question-answering workflow."""
    workflow = StateGraph(AgentState)

    # Each node performs one focused operation and returns an updated state.
    workflow.add_node("route_question", route_question_node)
    workflow.add_node("retrieve_context", retrieve_context_node)
    workflow.add_node("generate_rag_answer", generate_rag_answer_node)
    workflow.add_node("generate_general_answer", generate_general_answer_node)
    workflow.add_node("handle_out_of_scope", handle_out_of_scope_node)
    workflow.add_node("handle_ambiguous", handle_ambiguous_node)

    # Every execution starts with classification before selecting a branch.
    workflow.add_edge(START, "route_question")
    workflow.add_conditional_edges(
        "route_question",
        route_from_state,
        {
            "restaurant_specific": "retrieve_context",
            "general_food": "generate_general_answer",
            "out_of_scope": "handle_out_of_scope",
            "ambiguous": "handle_ambiguous",
        },
    )
    # Restaurant-specific questions retrieve context before answer generation.
    workflow.add_edge("retrieve_context", "generate_rag_answer")
    # These terminal edges prevent a branch from continuing into another path.
    workflow.add_edge("generate_rag_answer", END)
    workflow.add_edge("generate_general_answer", END)
    workflow.add_edge("handle_out_of_scope", END)
    workflow.add_edge("handle_ambiguous", END)

    # Compilation turns the declarative graph into an invokable application.
    return workflow.compile()


app = build_workflow()


def run_workflow(
    question: str,
    session_id: str = "cli-default",
) -> AgentState:
    """Run the workflow with session memory and persist the completed turn."""
    started_at = time.perf_counter()

    memory_started_at = time.perf_counter()
    # Load only the configured recent messages before invoking the graph.
    history = load_conversation(session_id)
    memory_load_ms = (time.perf_counter() - memory_started_at) * 1000

    logger.info(
        "workflow_started session_id=%s history_messages=%s memory_load_ms=%.2f",
        session_id,
        len(history),
        memory_load_ms,
    )

    graph_started_at = time.perf_counter()
    # The initial dictionary becomes the first AgentState seen by START.
    result = app.invoke(
        {
            "question": question,
            "conversation_history": history,
        }
    )
    graph_duration_ms = (time.perf_counter() - graph_started_at) * 1000

    answer = result.get("answer")
    memory_save_ms = 0.0
    if answer:
        # Persist only completed turns so failed requests do not pollute memory.
        memory_save_started_at = time.perf_counter()
        save_conversation_turn(session_id, question, answer)
        memory_save_ms = (time.perf_counter() - memory_save_started_at) * 1000

    total_duration_ms = (time.perf_counter() - started_at) * 1000
    logger.info(
        "workflow_completed session_id=%s route=%s graph_ms=%.2f "
        "memory_save_ms=%.2f total_ms=%.2f",
        session_id,
        result.get("route"),
        graph_duration_ms,
        memory_save_ms,
        total_duration_ms,
    )

    return result


def build_stream_sources(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert retrieval results into safe metadata for stream events."""
    sources = []
    for document in documents:
        metadata = document.get("metadata", {})
        # Keep the stream response compact and avoid exposing full document text.
        sources.append(
            {
                "item_name": metadata.get("item_name"),
                "document_type": metadata.get("document_type"),
                "hybrid_score": document.get("hybrid_score"),
            }
        )
    return sources


def stream_workflow(
    question: str,
    session_id: str = "cli-default",
) -> Iterator[dict[str, Any]]:
    """Yield workflow events while the final answer is generated."""
    started_at = time.perf_counter()
    history = load_conversation(session_id)

    # This event lets the client render a loading state immediately.
    yield {"type": "start"}

    query_plan = classify_question(question, history)
    route = query_plan.category
    yield {
        "type": "route",
        "route": route,
        "operation": query_plan.operation,
        "domain": query_plan.domain,
    }

    state: AgentState = {
        "question": question,
        "conversation_history": history,
        "query_plan": query_plan.model_dump(),
    }
    sources: list[dict[str, Any]] = []

    if route == "restaurant_specific":
        results, context = retrieve_context_for_state(state)
        sources = build_stream_sources(results)
        yield {"type": "sources", "sources": sources}
        messages = build_rag_messages(
            question=question,
            context=context,
            conversation_history=history,
            query_plan=query_plan.model_dump(),
        )
    elif route == "general_food":
        messages = build_general_messages(question, history)
    else:
        # These branches do not need another LLM request.
        static_answers = {
            "out_of_scope": (
                "I am the OffOn restaurant assistant chatbot. "
                "I can help with food and restaurant questions."
            ),
            "ambiguous": (
                "I am the OffOn restaurant assistant chatbot. "
                "Could you clarify your question?"
            ),
        }
        answer = static_answers[route]
        yield {"type": "token", "text": answer}
        save_conversation_turn(session_id, question, answer)
        yield {
            "type": "done",
            "route": route,
            "answer": answer,
            "sources": sources,
        }
        return

    answer_parts = []
    # Each yielded fragment is sent to the browser as soon as it arrives.
    for text_fragment in request_completion_stream(messages=messages):
        answer_parts.append(text_fragment)
        yield {"type": "token", "text": text_fragment}

    answer = "".join(answer_parts)
    if not answer:
        raise RuntimeError("The model returned an empty streaming response.")

    save_conversation_turn(session_id, question, answer)
    duration_ms = (time.perf_counter() - started_at) * 1000
    logger.info(
        "stream_workflow_completed session_id=%s route=%s total_ms=%.2f",
        session_id,
        route,
        duration_ms,
    )
    yield {
        "type": "done",
        "route": route,
        "answer": answer,
        "sources": sources,
    }


def main() -> None:
    """Run the workflow from the command line."""
    question = input("Enter your question: ").strip()
    if not question:
        print("The question cannot be empty.")
        return

    result = run_workflow(question)
    print("\nAnswer:")
    print(result.get("answer", ""))


if __name__ == "__main__":
    main()
