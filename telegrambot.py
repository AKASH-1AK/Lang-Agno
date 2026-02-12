print("🚀 Telegram bot script started (DEBUG MODE)")

import os
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
)

from rag.query import retrieve
from llm.llm import generate_answer

# 🔐 Load environment variables
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")

if not BOT_TOKEN:
    raise ValueError("❌ TELEGRAM_TOKEN not found in .env file")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text

    print("\n==============================")
    print("📩 USER QUESTION:")
    print(user_question)

    await update.message.reply_text("🔍 Searching document...")

    context_text = retrieve(user_question)

    print("\n📄 RETRIEVED CONTEXT:")
    print(context_text if context_text else "[EMPTY CONTEXT]")

    answer = generate_answer(context_text, user_question)

    print("\n🤖 GENERATED ANSWER:")
    print(answer)
    print("==============================\n")

    await update.message.reply_text(answer)


def main():
    try:
        app = ApplicationBuilder().token(BOT_TOKEN).build()

        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
        )

        print("🤖 Telegram bot is running (DEBUG MODE)...")
        app.run_polling()

    except Exception as e:
        print("❌ Bot crashed:", e)


if __name__ == "__main__":
    main()
