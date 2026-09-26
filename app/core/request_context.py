"""Request-scoped context used to correlate logs across the workflow."""

from contextvars import ContextVar


# ContextVar keeps the ID isolated when multiple requests run concurrently.
request_id_context: ContextVar[str] = ContextVar(
    "request_id",
    default="-",
)


def get_request_id() -> str:
    """Return the request ID associated with the current execution context."""
    return request_id_context.get()
