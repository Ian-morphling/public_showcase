# backend/context_manager.py
"""
Conversation context management for multi-turn RAG with LLM summarization using Groq.
- Tracks conversation history per thread
- Summarizes older turns to bound context size
- Builds conversation-aware query for LangGraph
"""

from typing import Dict, List
from dataclasses import dataclass
import asyncio
import logging
import os
from groq import Client
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# --- Groq Client setup ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in .env")

groq_client = Client(api_key=GROQ_API_KEY)


@dataclass
class Turn:
    user: str
    assistant: str


class ContextManager:
    def __init__(self, max_turns: int = 6, model: str = "llama-3.1-8b-instant"):
        self.max_turns = max_turns
        self.model = model
        self._store: Dict[str, List[Turn]] = {}
        self._summary_store: Dict[str, str] = {}  # summarized context per thread

    def get_history(self, thread_id: str) -> List[Turn]:
        return self._store.get(thread_id, [])
    async def append_turn(self, thread_id: str, user: str, assistant: str):
        history = self._store.setdefault(thread_id, [])
        history.append(Turn(user=user, assistant=assistant))

        # Trim history and optionally summarize older turns
        if len(history) > self.max_turns:
            to_summarize = history[:-2]  # all but last 2 recent turns
            summary_text = await self.summarize_turns(thread_id, to_summarize)

            # Replace old turns with a single summarized turn
            self._store[thread_id] = [Turn(user="summary", assistant=summary_text)] + history[-2:]

    async def summarize_turns(self, thread_id: str, turns: List[Turn]) -> str:
        if not turns:
            return ""

        prompt_lines = []
        for turn in turns:
            prompt_lines.append(f"User: {turn.user}")
            prompt_lines.append(f"Assistant: {turn.assistant}")

        prompt = (
            "Summarize the following conversation between user and assistant "
            "into a concise context suitable for continuing a RAG query:\n\n"
            + "\n".join(prompt_lines)
        )

        try:
            def groq_sync_call():
                response = groq_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                )
                return response.choices[0].message.content.strip()

            summary = await asyncio.to_thread(groq_sync_call)
        except Exception as e:
            logger.warning("LLM summarization failed: %s", e)
            summary = " ".join([t.user + " " + t.assistant for t in turns])

        # Cache summary
        self._summary_store[thread_id] = summary
        return summary

    def build_query(self, thread_id: str, user_query: str) -> str:
        """
        Build a conversation-aware query including summary of older turns.
        """
        history = self.get_history(thread_id)
        context_lines = []

        # Include summarized context if exists
        if self._summary_store.get(thread_id):
            context_lines.append(f"Context Summary: {self._summary_store[thread_id]}")

        # Include last 2 turns verbatim
        for turn in history[-2:]:
            context_lines.append(f"User: {turn.user}")
            context_lines.append(f"Assistant: {turn.assistant}")

        context_text = "\n".join(context_lines)
        return f"You are continuing a conversation.\n\n{context_text}\n\nUser: {user_query}"
