"""Telegram command and message handlers for the synchronous TeleBot client."""

import logging
from uuid import uuid4

import telebot
from telebot.types import Message

from app.agent.workflow import stream_workflow
from app.core.request_context import request_id_context
from app.telegram.sender import TelegramStreamWriter


logger = logging.getLogger(__name__)


def build_session_id(message: Message) -> str:
    """Create a stable, isolated memory key from Telegram identifiers."""
    chat = message.chat
    user = message.from_user

    # Private chats naturally belong to one user, so the chat ID is sufficient.
    if chat.type == "private":
        return f"telegram:chat:{chat.id}"

    if user is None:
        return f"telegram:chat:{chat.id}"

    # Group chats use a per-user key to prevent users from sharing memory.
    return f"telegram:chat:{chat.id}:user:{user.id}"


def register_handlers(bot: telebot.TeleBot) -> None:
    """Register commands and text-message handlers on the TeleBot instance."""

    @bot.message_handler(commands=["start"])
    def start_command(message: Message) -> None:
        """Explain how users can interact with the assistant."""
        bot.reply_to(
            message,
            "Send a food or restaurant question and I will help you.",
        )

    @bot.message_handler(commands=["help"])
    def help_command(message: Message) -> None:
        """Return a short usage guide."""
        bot.reply_to(
            message,
            "You can ask about menu items, ingredients, prices, and the restaurant.",
        )

    @bot.message_handler(
        content_types=["text"],
        func=lambda message: not (message.text or "").startswith("/"),
    )
    def handle_text_message(message: Message) -> None:
        """Process one Telegram text message through the shared AI workflow."""
        question = (message.text or "").strip()
        if not question:
            bot.reply_to(message, "The question cannot be empty.")
            return

        session_id = build_session_id(message)
        request_id = f"telegram-{uuid4()}"
        context_token = request_id_context.set(request_id)
        writer: TelegramStreamWriter | None = None

        try:
            # Send one placeholder message that will be updated during generation.
            placeholder = bot.reply_to(message, "Thinking...")
            bot.send_chat_action(message.chat.id, "typing")

            writer = TelegramStreamWriter(
                bot=bot,
                chat_id=message.chat.id,
                initial_message=placeholder,
            )

            completed = False
            # TeleBot handlers are synchronous, so the existing synchronous
            # workflow can be consumed directly without an async bridge.
            for event in stream_workflow(question, session_id):
                event_type = event.get("type")

                if event_type == "token":
                    writer.append(str(event.get("text", "")))
                elif event_type == "done":
                    writer.finish(str(event.get("answer", "")))
                    completed = True
                elif event_type == "error":
                    writer.finish(
                        "I could not complete the request. Please try again."
                    )
                    completed = True

            if not completed:
                writer.finish(
                    "The request ended before a complete answer was generated."
                )

        except (RuntimeError, ValueError) as error:
            logger.error(
                "telegram_request_failed request_id=%s session_id=%s error=%s",
                request_id,
                session_id,
                error,
            )
            if writer is not None:
                writer.finish("I could not complete the request. Please try again.")
            else:
                bot.reply_to(
                    message,
                    "I could not complete the request. Please try again.",
                )
        except Exception:
            logger.exception(
                "telegram_request_unexpected_error request_id=%s session_id=%s",
                request_id,
                session_id,
            )
            if writer is not None:
                writer.finish("An unexpected error occurred. Please try again.")
            else:
                bot.reply_to(
                    message,
                    "An unexpected error occurred. Please try again.",
                )
        finally:
            request_id_context.reset(context_token)

