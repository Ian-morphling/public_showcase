"""
telegram_bot/bot.py
 
Telegram bot for the EU AI Act Navigator.
chat_id is mapped to thread_id for multi-turn context management.
 
Commands: /start, /help, /reset
Usage: python -m telegram_bot.bot
"""
 
import logging
from uuid import uuid4
import httpx
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from telegram_bot.config import TELEGRAM_BOT_TOKEN, RAG_ENDPOINT
 
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
 
WELCOME = (
    " *EU AI Act Navigator*\n\n"
    "Ask me anything about the EU AI Act — I'll return grounded answers with citations.\n\n"
    "/reset — clear conversation history\n"
    "/help — command reference"
)
 
HELP = (
    "*Commands*\n"
    "/start — welcome message\n"
    "/help — this reference\n"
    "/reset — clear context and start fresh\n\n"
    "_Example: What are the obligations for high-risk AI systems?_"
)
 
def get_thread_id(ctx: ContextTypes.DEFAULT_TYPE) -> str:
    if "thread_id" not in ctx.chat_data:
        ctx.chat_data["thread_id"] = str(uuid4())
    return ctx.chat_data["thread_id"]
 
def format_reply(data: dict) -> str:
    answer = data.get("answer", "No answer returned.")
    citations = "\n".join(f"• [{c['label']}]({c['url']})" for c in data.get("citations", []))
    return f"{answer}\n\n📎 *Sources*\n{citations}" if citations else answer
 
async def send(update: Update, text: str) -> None:
    if len(text) <= 4096:
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    else:
        split = text.rfind("\n\n📎", 0, 4000) or 4000
        await update.message.reply_text(text[:split], parse_mode=ParseMode.MARKDOWN)
        await update.message.reply_text(text[split:], parse_mode=ParseMode.MARKDOWN)
 
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    get_thread_id(ctx)
    await send(update, WELCOME)
 
async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await send(update, HELP)
 
async def reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.chat_data["thread_id"] = str(uuid4())
    await update.message.reply_text(" *Conversation reset.*", parse_mode=ParseMode.MARKDOWN)
 
async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.strip()
    if not query:
        return
 
    await update.message.chat.send_action(ChatAction.TYPING)
 
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                RAG_ENDPOINT,
                json={"query": query, "mode": "final"},
                params={"thread_id": get_thread_id(ctx)},
            )
            r.raise_for_status()
            data = r.json()
 
        if tid := data.get("thread_id"):
            ctx.chat_data["thread_id"] = tid
 
        await send(update, format_reply(data))
 
    except httpx.TimeoutException:
        await update.message.reply_text(" Request timed out. Please try again shortly.")
    except httpx.HTTPStatusError as e:
        msg = " Rate limit reached. Please wait a moment." if e.response.status_code == 429 \
            else f" Backend error ({e.response.status_code}). Please try again."
        await update.message.reply_text(msg)
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        await update.message.reply_text(" Unexpected error. Try /reset and ask again.")
 
def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Bot running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
 
if __name__ == "__main__":
    main()