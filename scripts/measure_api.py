"""Small client utility for comparing API round-trip and server timings."""

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any


def measure_request(url: str, session_id: str, question: str) -> None:
    """Measure client round-trip time and compare it with server timing."""
    # The client timer starts before serialization and ends after the full body
    # is received, so it represents the user-visible round-trip duration.
    payload = json.dumps(
        {"session_id": session_id, "question": question}
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started_at = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_body = response.read()
            status_code = response.status
            server_time_ms = response.headers.get("X-Server-Time-Ms")
            request_id = response.headers.get("X-Request-ID")
    except urllib.error.HTTPError as error:
        response_body = error.read()
        status_code = error.code
        server_time_ms = error.headers.get("X-Server-Time-Ms")
        request_id = error.headers.get("X-Request-ID")

    client_round_trip_ms = (time.perf_counter() - started_at) * 1000
    try:
        response_data: Any = json.loads(response_body.decode("utf-8"))
    except json.JSONDecodeError:
        response_data = response_body.decode("utf-8", errors="replace")

    print(f"status_code: {status_code}")
    print(f"request_id: {request_id or 'not returned'}")
    print(f"client_round_trip_ms: {client_round_trip_ms:.2f}")
    print(f"server_time_ms: {server_time_ms or 'not returned'}")

    if server_time_ms is not None:
        # This difference is an approximation of transport and client overhead;
        # it is not a pure measurement of internet latency.
        network_and_client_overhead_ms = (
            client_round_trip_ms - float(server_time_ms)
        )
        print(
            "network_and_client_overhead_ms: "
            f"{network_and_client_overhead_ms:.2f}"
        )

    print("response:")
    print(json.dumps(response_data, ensure_ascii=False, indent=2))


def main() -> None:
    """Run one timed request against the FastAPI endpoint."""
    # argparse keeps the timing tool reusable for different environments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/ask",
        help="The FastAPI endpoint URL.",
    )
    parser.add_argument(
        "--session-id",
        default="timing-test-session",
        help="The conversation session identifier.",
    )
    parser.add_argument("question", help="The question to send.")
    arguments = parser.parse_args()

    measure_request(
        arguments.url,
        arguments.session_id,
        arguments.question,
    )


if __name__ == "__main__":
    main()
