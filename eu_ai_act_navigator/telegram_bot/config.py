"""
telegram_bot/config.py
 
Loads environment variables for the Telegram bot.
"""
 
import os
from dotenv import load_dotenv
 
load_dotenv()
 
TELEGRAM_BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]
FASTAPI_BASE_URL: str = os.environ.get("FASTAPI_BASE_URL", "http://localhost:8000")
RAG_ENDPOINT: str = f"{FASTAPI_BASE_URL}/rag/query"
 
