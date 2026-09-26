"""Telegram bot entry point using the synchronous pyTelegramBotAPI client."""

import logging
import os
from pathlib import Path

import telebot
from dotenv import load_dotenv

from app.rag.retrieval_resources import get_retrieval_resources
from app.telegram.handlers import register_handlers


PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def get_telegram_token() -> str:
    """Load the Telegram bot token without exposing it in logs."""
    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN was not found. Define it in the .env file."
        )
    return token


def preload_retrieval_resources() -> None:
    """Load the embedding model and documents before accepting updates."""
    logger.info("telegram_retrieval_startup_started")
    get_retrieval_resources()
    logger.info("telegram_retrieval_startup_completed")


def build_bot() -> telebot.TeleBot:
    """Build the TeleBot instance and register all handlers."""
    token = get_telegram_token()
    bot = telebot.TeleBot(
        token,
        parse_mode=None,
        threaded=True,
        num_threads=4,
    )

    # Load shared RAG resources once before the polling loop starts.
    preload_retrieval_resources()
    register_handlers(bot)
    return bot


def main() -> None:
    """Start the Telegram bot using long polling."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    bot = build_bot()
    logger.info("telegram_bot_started")
    bot.infinity_polling(skip_pending=True)


if __name__ == "__main__":
    main()

