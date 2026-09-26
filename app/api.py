"""FastAPI entry point for the routed restaurant question-answering service."""

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.agent.workflow import run_workflow, stream_workflow
from app.core.embedding_model import get_embedding_model
from app.core.request_context import request_id_context
from app.rag.retrieval_resources import get_retrieval_resources


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

# This logger records lifecycle, request, workflow, and error timing events.
logger = logging.getLogger(__name__)


RouteCategory = Literal[
    "restaurant_specific",
    "general_food",
    "out_of_scope",
    "ambiguous",
]


class AskRequest(BaseModel):
    """Request body for the question-answering endpoint."""

    # The session ID isolates one user's short-term memory from another user's.
    session_id: str = Field(
        min_length=1,
        max_length=100,
        description="Conversation identifier used to isolate memory.",
    )
    # The length limit protects the endpoint from unexpectedly large prompts.
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="The user's question.",
    )

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        """Trim whitespace and reject whitespace-only questions."""
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("The question cannot be empty.")
        return normalized_value

    @field_validator("session_id")
    @classmethod
    def normalize_session_id(cls, value: str) -> str:
        """Trim whitespace and reject whitespace-only session identifiers."""
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("The session_id cannot be empty.")
        return normalized_value


class SourceItem(BaseModel):
    """Public source metadata returned for restaurant-specific answers."""

    # Only safe, compact metadata is exposed to the API client.
    item_name: str | None = None
    document_type: str | None = None
    hybrid_score: float | None = None


class AskResponse(BaseModel):
    """Response body returned by the question-answering endpoint."""

    question: str
    route: RouteCategory
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load shared retrieval resources before serving requests."""
    started_at = time.perf_counter()
    logger.info("application_startup_started")
    # Fail during startup if required local RAG resources are unavailable,
    # instead of failing unpredictably on the first user request.
    get_retrieval_resources()
    duration_ms = (time.perf_counter() - started_at) * 1000
    model_cache_info = get_embedding_model.cache_info()
    resources_cache_info = get_retrieval_resources.cache_info()
    logger.info(
        "application_startup_completed retrieval_load_ms=%.2f "
        "model_cache_hits=%s model_cache_misses=%s resources_cache_hits=%s "
        "resources_cache_misses=%s",
        duration_ms,
        model_cache_info.hits,
        model_cache_info.misses,
        resources_cache_info.hits,
        resources_cache_info.misses,
    )
    yield
    logger.info("application_shutdown_started")


app = FastAPI(
    lifespan=lifespan,
    title="Restaurant AI Assistant",
    version="1.0.2",
    description="A routed RAG question-answering API for restaurant information.",
)


@app.middleware("http")
async def log_request_timing(request: Request, call_next):
    """Log end-to-end server timing and return a correlation ID header."""
    # A new ID lets us correlate every log line belonging to this HTTP request.
    request_id = str(uuid4())
    context_token = request_id_context.set(request_id)
    started_at = time.perf_counter()

    logger.info(
        "request_started request_id=%s method=%s path=%s",
        request_id,
        request.method,
        request.url.path,
    )

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started_at) * 1000
        # These headers let an external client compare server time with round-trip time.
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Server-Time-Ms"] = f"{duration_ms:.2f}"
        logger.info(
            "request_completed request_id=%s status_code=%s server_time_ms=%.2f",
            request_id,
            response.status_code,
            duration_ms,
        )
        return response
    except Exception:
        duration_ms = (time.perf_counter() - started_at) * 1000
        logger.exception(
            "request_failed request_id=%s server_time_ms=%.2f",
            request_id,
            duration_ms,
        )
        raise
    finally:
        request_id_context.reset(context_token)


def build_sources(state: dict[str, Any]) -> list[SourceItem]:
    """Expose safe retrieval metadata without returning the full context."""
    sources = []
    for document in state.get("retrieved_documents", []):
        # Metadata is returned instead of the full internal document content.
        metadata = document.get("metadata", {})
        sources.append(
            SourceItem(
                item_name=metadata.get("item_name"),
                document_type=metadata.get("document_type"),
                hybrid_score=document.get("hybrid_score"),
            )
        )
    return sources


def encode_sse_event(event: dict[str, Any]) -> str:
    """Encode one workflow event using the Server-Sent Events format."""
    event_type = event.get("type", "message")
    event_data = json.dumps(event, ensure_ascii=False)
    return f"event: {event_type}\ndata: {event_data}\n\n"


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return service health status."""
    # This endpoint does not call the LLM and is suitable for process probes.
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """Run the LangGraph workflow for a user question."""
    started_at = time.perf_counter()
    request_id = request_id_context.get()
    logger.info(
        "ask_workflow_started request_id=%s session_id=%s",
        request_id,
        request.session_id,
    )

    try:
        # The workflow owns routing, retrieval, generation, and memory updates.
        state = run_workflow(request.question, request.session_id)
        route = state.get("route")
        answer = state.get("answer")

        # Do not expose an incomplete internal state as a successful response.
        if route is None or not answer:
            logger.error("Workflow returned an incomplete state: %s", state)
            raise HTTPException(
                status_code=502,
                detail="The workflow returned an incomplete response.",
            )

        response = AskResponse(
            question=request.question,
            route=route,
            answer=answer,
            sources=build_sources(state),
        )
        duration_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "ask_workflow_completed request_id=%s route=%s workflow_time_ms=%.2f",
            request_id,
            route,
            duration_ms,
        )
        return response

    except HTTPException:
        raise
    except (RuntimeError, ValueError) as error:
        # Convert expected application failures into a safe public HTTP error.
        duration_ms = (time.perf_counter() - started_at) * 1000
        logger.error(
            "ask_workflow_failed request_id=%s workflow_time_ms=%.2f error=%s",
            request_id,
            duration_ms,
            error,
        )
        raise HTTPException(
            status_code=502,
            detail="The AI service could not complete the request.",
        ) from error
    except Exception as error:
        duration_ms = (time.perf_counter() - started_at) * 1000
        logger.exception(
            "ask_workflow_unexpected_error request_id=%s workflow_time_ms=%.2f",
            request_id,
            duration_ms,
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected server error occurred.",
        ) from error


@app.post("/ask/stream")
def stream_question(request: AskRequest) -> StreamingResponse:
    """Stream answer fragments as Server-Sent Events."""
    request_id = request_id_context.get()

    def event_generator():
        """Convert workflow events into SSE messages for the HTTP client."""
        context_token = request_id_context.set(request_id)
        started_at = time.perf_counter()
        first_token_at: float | None = None

        try:
            for event in stream_workflow(request.question, request.session_id):
                if event.get("type") == "token" and first_token_at is None:
                    first_token_at = time.perf_counter()
                    logger.info(
                        "stream_first_token request_id=%s time_to_first_token_ms=%.2f",
                        request_id,
                        (first_token_at - started_at) * 1000,
                    )
                yield encode_sse_event(event)

            total_duration_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "stream_request_completed request_id=%s total_ms=%.2f",
                request_id,
                total_duration_ms,
            )
        except (RuntimeError, ValueError) as error:
            logger.error(
                "stream_request_failed request_id=%s error=%s",
                request_id,
                error,
            )
            yield encode_sse_event(
                {
                    "type": "error",
                    "detail": "The AI service could not complete the request.",
                }
            )
        except Exception:
            logger.exception(
                "stream_request_unexpected_error request_id=%s",
                request_id,
            )
            yield encode_sse_event(
                {
                    "type": "error",
                    "detail": "An unexpected server error occurred.",
                }
            )
        finally:
            request_id_context.reset(context_token)

    # These headers prevent proxy buffering so fragments reach the client promptly.
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    """Return a safe response for unexpected errors."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred."},
    )
