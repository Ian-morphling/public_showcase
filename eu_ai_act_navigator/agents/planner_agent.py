# agents/planner_agent.py
"""
PlannerAgent for Agentic RAG using Groq LLM with seen-doc filtering.

Features:
- Each hop uses only NEW documents for planning
- Prevents duplicates in final output
- Ready for summarization pipelines
"""

import os
import asyncio
from typing import List, Tuple
from agents.retriever_agent import RetrievedDocument
from groq import Client

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in .env")

groq_client = Client(api_key=GROQ_API_KEY)


class PlannerAgent:
    def __init__(self, max_hops: int = 3, enable_sufficiency: bool = True):
        self.intent: str | None = None
        self.seen_doc_ids: set[str] = set()
        self.max_hops = max_hops
        self.enable_sufficiency = enable_sufficiency

    async def classify_intent(self, user_query: str) -> str:
        """Infer user intent from query (placeholder)."""
        self.intent = "general_inquiry"
        return self.intent

    async def check_answer_sufficiency(
        self, original_query: str, recent_docs: list[RetrievedDocument]
    ) -> bool:
        """Ask LLM if the retrieved snippets sufficiently answer the user's question."""
        snippets = "\n".join(f"{i+1}. {doc.content[:200]}" for i, doc in enumerate(recent_docs))
        prompt = f"""
User question:
{original_query}

Retrieved information so far:
{snippets}

QUESTION:
Based ONLY on the information above, has the user's question been sufficiently answered?

- Respond with YES if the question is fully answered.
- Respond with NO if important information might still be missing.
- Output ONLY YES or NO (avoid explanations).
"""

        def groq_sync_call():
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50
            )
            return response.choices[0].message.content.strip()

        result = await asyncio.to_thread(groq_sync_call)
        return "YES" in result.upper()

    async def plan_next_query(
        self,
        original_query: str,
        current_query: str,
        retrieved_docs: list[RetrievedDocument],
        hop: int,
        previous_queries: list[str],
        top_k: int = 5,
    ) -> tuple[str | None, str | None, list[RetrievedDocument]]:

        # --- Filter out already seen documents ---
        new_docs = [doc for doc in retrieved_docs if doc.id not in self.seen_doc_ids]
        self.seen_doc_ids.update(doc.id for doc in new_docs)

        # --- Stop if max hops reached ---
        if hop >= self.max_hops or not new_docs:
            return None, f"Stop: max hops or no new docs", new_docs

        # --- Stop if new information relevance degrades ---
        if self.enable_sufficiency and new_docs:
            last_docs = new_docs[-top_k:]
            avg_last_similarity = sum(doc.similarity for doc in last_docs) / len(last_docs)
            if avg_last_similarity < 0.55:
                return None, "Stop: retrieval relevance too low", new_docs

        # --- LLM sufficiency check after hop >= 2 ---
        if self.enable_sufficiency and hop >= 2:
            recent_docs = new_docs[-top_k:]
            done = await self.check_answer_sufficiency(original_query, recent_docs)
            if done:
                return None, "Stop: answer sufficiently covered", new_docs

        # --- Prepare recent snippets for next query generation ---
        recent_docs = new_docs[-top_k:]
        doc_snippets = "\n".join(f"{i+1}. {doc.content[:200]}" for i, doc in enumerate(recent_docs))

        # --- Compose Groq prompt for next query ---
        prompt = f"""
You are an agentic retrieval planner for the EU AI Act.

User query: {original_query}
User intent: {self.intent}

Previously retrieved document snippets (NEW info only):
{doc_snippets}

TASK:
- Generate ONE concise follow-up search query (max 15 words).
- The query MUST retrieve new information not already covered.
- If no meaningful new information remains, respond with exactly: STOP

RULES:
- Do NOT explain your reasoning.
- Do NOT include quotes.
- Do NOT include multiple queries.
- Output ONLY the query text or STOP.
"""

        # --- Call Groq LLM asynchronously ---
        def groq_sync_call():
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            return response.choices[0].message.content.strip()

        next_query = await asyncio.to_thread(groq_sync_call)

        # --- Stop if LLM indicates 'STOP' or repeats previous query ---
        if next_query.upper() == "STOP" or next_query in previous_queries:
            return None, "LLM indicated stop or repeated query", new_docs

        return next_query, None, new_docs
