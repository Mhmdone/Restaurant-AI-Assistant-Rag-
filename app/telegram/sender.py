"""Telegram message streaming helpers for the synchronous TeleBot client."""

import logging
import os
import time

import telebot
from telebot.types import Message


logger = logging.getLogger(__name__)

TELEGRAM_MESSAGE_LIMIT = 4096
DEFAULT_EDIT_INTERVAL_SECONDS = 0.8


def get_edit_interval_seconds() -> float:
    """Read and normalize the interval between Telegram message edits."""
    raw_value = os.getenv(
        "TELEGRAM_STREAM_EDIT_INTERVAL_SECONDS",
        str(DEFAULT_EDIT_INTERVAL_SECONDS),
    )

    try:
        interval = float(raw_value)
    except ValueError:
        interval = DEFAULT_EDIT_INTERVAL_SECONDS

    return max(interval, 0.2)


def split_telegram_text(text: str) -> list[str]:
    """Split plain text into chunks accepted by Telegram."""
    if not text:
        return []

    return [
        text[index : index + TELEGRAM_MESSAGE_LIMIT]
        for index in range(0, len(text), TELEGRAM_MESSAGE_LIMIT)
    ]


class TelegramStreamWriter:
    """Render a growing answer by editing Telegram messages periodically."""

    def __init__(
        self,
        bot: telebot.TeleBot,
        chat_id: int,
        initial_message: Message,
    ) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._messages: list[Message] = [initial_message]
        self._rendered_parts: list[str] = [initial_message.text or ""]
        self._full_text = ""
        self._last_edit_at = 0.0
        self._edit_interval = get_edit_interval_seconds()

    def append(self, text: str) -> None:
        """Append a fragment and render it when the edit interval has elapsed."""
        if not text:
            return

        self._full_text += text
        now = time.monotonic()
        if now - self._last_edit_at >= self._edit_interval:
            self._render(force=False)

    def finish(self, answer: str | None = None) -> None:
        """Render the final answer immediately, splitting long answers safely."""
        if answer is not None:
            self._full_text = answer
        self._render(force=True)

    def _render(self, force: bool) -> None:
        """Synchronize Telegram messages with the current complete answer."""
        if not force and not self._full_text:
            return

        parts = split_telegram_text(self._full_text)
        if not parts:
            return

        for index, part in enumerate(parts):
            if index == 0:
                message = self._messages[0]
            elif index < len(self._messages):
                message = self._messages[index]
            else:
                message = self._bot.send_message(
                    chat_id=self._chat_id,
                    text=part,
                )
                self._messages.append(message)
                self._rendered_parts.append(part)
                continue

            if self._rendered_parts[index] == part:
                continue

            self._bot.edit_message_text(
                text=part,
                chat_id=self._chat_id,
                message_id=message.message_id,
            )
            self._rendered_parts[index] = part

        self._last_edit_at = time.monotonic()

