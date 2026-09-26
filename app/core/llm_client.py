"""Shared OpenAI-compatible client with retries and request instrumentation."""

import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

from app.core.request_context import get_request_id


# These defaults keep the current OpenRouter setup working when environment
# variables are not configured yet.
DEFAULT_LLM_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLM_MODEL_NAME = "openrouter/free"
# Network reliability settings are centralized so every caller behaves equally.
MAX_RETRIES = 3
BASE_RETRY_DELAY = 2
REQUEST_TIMEOUT = 30.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm_settings() -> dict[str, str]:
    """Load and cache provider settings for the current process."""
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL).strip()
    model_name = os.getenv("LLM_MODEL_NAME", DEFAULT_LLM_MODEL_NAME).strip()

    if not api_key:
        raise RuntimeError("API_KEY was not found. Define it in the .env file.")
    if not base_url:
        raise RuntimeError("LLM_BASE_URL cannot be empty.")
    if not model_name:
        raise RuntimeError("LLM_MODEL_NAME cannot be empty.")

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model_name": model_name,
    }


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Create an OpenAI-compatible client with a request timeout."""
    settings = get_llm_settings()

    # Reuse one client instance instead of creating a new connection wrapper
    # for every router or answer request.
    return OpenAI(
        base_url=settings["base_url"],
        api_key=settings["api_key"],
        timeout=REQUEST_TIMEOUT,
    )


def log_token_usage(completion: Any) -> None:
    """Log token usage when the provider returns usage information."""
    # Providers may omit usage, so logging must not assume it is always present.
    usage = completion.usage
    if usage is None:
        logger.warning("Token usage was not returned by the provider.")
        return

    logger.info(
        "Token usage - prompt: %s, completion: %s, total: %s",
        usage.prompt_tokens or 0,
        usage.completion_tokens or 0,
        usage.total_tokens or 0,
    )


def request_completion(
    messages: list[dict[str, Any]],
    **options: Any,
) -> Any:
    """Call the model with retries for transient API failures."""
    client = get_client()
    model_name = get_llm_settings()["model_name"]

    for attempt in range(MAX_RETRIES):
        started_at = time.perf_counter()
        request_id = get_request_id()
        try:
            logger.info(
                "llm_request_started request_id=%s attempt=%s/%s",
                request_id,
                attempt + 1,
                MAX_RETRIES,
            )
            # The duration measures provider processing plus network latency.
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **options,
            )
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "llm_request_completed request_id=%s attempt=%s duration_ms=%.2f",
                request_id,
                attempt + 1,
                duration_ms,
            )
            log_token_usage(completion)
            return completion

        except (APITimeoutError, APIConnectionError, RateLimitError) as error:
            duration_ms = (time.perf_counter() - started_at) * 1000
            if attempt == MAX_RETRIES - 1:
                logger.error(
                    "llm_request_failed request_id=%s attempt=%s duration_ms=%.2f "
                    "error=%s",
                    request_id,
                    attempt + 1,
                    duration_ms,
                    error,
                )
                raise RuntimeError(
                    "The request failed after multiple attempts."
                ) from error

            # Exponential backoff avoids repeatedly hitting a temporary outage.
            delay = BASE_RETRY_DELAY * (2**attempt)
            logger.warning(
                "llm_request_retry request_id=%s attempt=%s/%s duration_ms=%.2f "
                "retry_in_seconds=%s error=%s",
                request_id,
                attempt + 1,
                MAX_RETRIES,
                duration_ms,
                delay,
                error,
            )
            time.sleep(delay)

        except Exception as error:
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.exception(
                "llm_request_unexpected_error request_id=%s duration_ms=%.2f",
                request_id,
                duration_ms,
            )
            raise RuntimeError("An unexpected API error occurred.") from error

    raise RuntimeError("No completion was returned by the API.")


def request_completion_stream(
    messages: list[dict[str, Any]],
    **options: Any,
) -> Iterator[str]:
    """Yield generated text fragments as they arrive from the provider."""
    client = get_client()
    model_name = get_llm_settings()["model_name"]

    for attempt in range(MAX_RETRIES):
        started_at = time.perf_counter()
        request_id = get_request_id()
        emitted_text = False
        usage = None

        try:
            logger.info(
                "llm_stream_started request_id=%s attempt=%s/%s",
                request_id,
                attempt + 1,
                MAX_RETRIES,
            )
            stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True,
                **options,
            )

            for chunk in stream:
                # Some providers include usage only in the final stream chunk.
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = chunk_usage

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None) or ""
                if text:
                    emitted_text = True
                    yield text

            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "llm_stream_completed request_id=%s attempt=%s duration_ms=%.2f",
                request_id,
                attempt + 1,
                duration_ms,
            )
            if usage is not None:
                logger.info(
                    "Stream token usage - prompt: %s, completion: %s, total: %s",
                    usage.prompt_tokens or 0,
                    usage.completion_tokens or 0,
                    usage.total_tokens or 0,
                )
            return

        except (APITimeoutError, APIConnectionError, RateLimitError) as error:
            duration_ms = (time.perf_counter() - started_at) * 1000

            # Retrying after partial output would duplicate text in the client.
            if emitted_text or attempt == MAX_RETRIES - 1:
                logger.error(
                    "llm_stream_failed request_id=%s attempt=%s duration_ms=%.2f "
                    "partial_output=%s error=%s",
                    request_id,
                    attempt + 1,
                    duration_ms,
                    emitted_text,
                    error,
                )
                raise RuntimeError(
                    "The streaming request failed after partial output."
                    if emitted_text
                    else "The streaming request failed after multiple attempts."
                ) from error

            delay = BASE_RETRY_DELAY * (2**attempt)
            logger.warning(
                "llm_stream_retry request_id=%s attempt=%s/%s duration_ms=%.2f "
                "retry_in_seconds=%s error=%s",
                request_id,
                attempt + 1,
                MAX_RETRIES,
                duration_ms,
                delay,
                error,
            )
            time.sleep(delay)

        except Exception as error:
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.exception(
                "llm_stream_unexpected_error request_id=%s duration_ms=%.2f",
                request_id,
                duration_ms,
            )
            raise RuntimeError("An unexpected streaming error occurred.") from error

    raise RuntimeError("No streaming completion was returned by the API.")
