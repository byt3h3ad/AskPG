#!/usr/bin/env python
"""
A Telegram bot that answers questions using Paul Graham's essays via RAG.
Send any message to get an answer based on PG's essays.
"""

import logging
import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from model import ask_pg

load_dotenv("../.env")
TOKEN = os.getenv("TOKEN")

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    welcome_message = (
        "Hello! I'm AskPG Bot 🤖\n\n"
        "I can answer questions based on Paul Graham's essays. "
        "Just send me any question about startups, entrepreneurship, "
        "programming, or life advice, and I'll search through PG's essays "
        "to give you an answer.\n\n"
        "Go ahead, ask me anything!"
    )
    await update.message.reply_text(welcome_message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    help_text = (
        "I'm here to help you find answers from Paul Graham's essays!\n\n"
        "Just send me any question and I'll search through PG's essays "
        "to provide you with relevant insights.\n\n"
        "Commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n\n"
        "Simply type your question and I'll do my best to answer it!"
    )
    await update.message.reply_text(help_text)


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle user questions and respond with RAG-generated answers."""
    user_question = update.message.text
    user_id = update.effective_user.id
    

    # Send typing action to show bot is processing
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    try:
        # Get answer from RAG model
        answer = ask_pg(user_question, user_id)

        # Format the response
        response = f"💡 {answer}"

        await update.message.reply_text(response, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        error_message = (
            "Sorry, I encountered an error while processing your question. "
            "Please try again or rephrase your question."
        )
        await update.message.reply_text(error_message)


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle unknown commands."""
    await update.message.reply_text(
        "Sorry, I don't understand that command. "
        "Just send me a question and I'll try to answer it based on Paul Graham's essays!"
    )


def main() -> None:
    """Start the bot."""
    if not TOKEN:
        raise ValueError("TOKEN environment variable is not set")

    # Create the Application
    application = Application.builder().token(TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # Handle all text messages as questions
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question)
    )

    # Handle unknown commands
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))

    # Run the bot
    logger.info("Starting AskPG Bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
