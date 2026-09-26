"""SQLite-backed short-term memory for user conversations."""

import os
import sqlite3
from pathlib import Path
from typing import TypedDict


# The limit is measured in messages, so ten messages represent five turns.
DEFAULT_MAX_MESSAGES = 10
MIN_MAX_MESSAGES = 2
# Keep the database at the project root even after this module is packaged.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEMORY_DATABASE_PATH = PROJECT_ROOT / "memory.db"


class ConversationMessage(TypedDict):
    """Represent one message stored in conversation memory."""

    role: str
    content: str


def get_memory_limit() -> int:
    """Return the configured message limit and normalize it to an even number."""
    raw_limit = os.getenv("MEMORY_MAX_MESSAGES", str(DEFAULT_MAX_MESSAGES))

    try:
        limit = int(raw_limit)
    except ValueError:
        # Invalid configuration should not prevent the service from starting.
        limit = DEFAULT_MAX_MESSAGES

    # Keeping an even value preserves complete user/assistant message pairs.
    limit = max(limit, MIN_MAX_MESSAGES)
    return limit if limit % 2 == 0 else limit - 1


def get_connection() -> sqlite3.Connection:
    """Create a SQLite connection for conversation memory."""
    connection = sqlite3.connect(MEMORY_DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    # WAL allows readers and writers to work with less locking contention.
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def initialize_memory_database() -> None:
    """Create the conversation memory table and its lookup index."""
    with get_connection() as connection:
        # IF NOT EXISTS makes initialization safe to call on every request.
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversation_messages_session
            ON conversation_messages (session_id, id)
            """
        )


def load_conversation(session_id: str) -> list[ConversationMessage]:
    """Load the newest messages for a session in chronological order."""
    initialize_memory_database()
    limit = get_memory_limit()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT role, content
            FROM conversation_messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()

    # The query reads newest-first for efficiency; reverse the result before
    # sending it to the model so the conversation remains chronological.
    return [
        {"role": row["role"], "content": row["content"]}
        for row in reversed(rows)
    ]


def save_conversation_turn(
    session_id: str,
    question: str,
    answer: str,
) -> None:
    """Store one user/assistant turn and remove messages beyond the limit."""
    initialize_memory_database()
    limit = get_memory_limit()

    with get_connection() as connection:
        # Store the question and answer in one transaction so a turn is atomic.
        connection.execute(
            """
            INSERT INTO conversation_messages (session_id, role, content)
            VALUES (?, 'user', ?)
            """,
            (session_id, question),
        )
        connection.execute(
            """
            INSERT INTO conversation_messages (session_id, role, content)
            VALUES (?, 'assistant', ?)
            """,
            (session_id, answer),
        )
        # Keep only the newest messages for this session and prune old turns.
        connection.execute(
            """
            DELETE FROM conversation_messages
            WHERE session_id = ?
              AND id NOT IN (
                  SELECT id
                  FROM conversation_messages
                  WHERE session_id = ?
                  ORDER BY id DESC
                  LIMIT ?
              )
            """,
            (session_id, session_id, limit),
        )
