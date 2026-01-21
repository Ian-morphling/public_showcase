# agents/explainer_agent.py
"""
ExplainerAgent for Agentic RAG.

Responsibilities:
- Synthesize retrieved EU AI Act documents into a structured answer
- Enforce strict citation discipline (Articles / Recitals / Annexes)
- Never retrieve or plan — reasoning only
"""

import os
import asyncio
from typing import List, Dict
from groq import Client
from agents.retriever_agent import RetrievedDocument

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in .env")

groq_client = Client(api_key=GROQ_API_KEY)


class ExplainerAgent:
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.model = model

    async def explain(
        self,
        query: str,
        docs: List[RetrievedDocument],
    ) -> Dict:
        """
        Generate a citation-aware explanation based ONLY on retrieved documents.
        The output is human-readable and structured, no JSON formatting.
        """

        if not docs:
            return {
                "answer": "No relevant provisions were retrieved to answer this question.",
                "sources": [],
            }

        # --- Prepare labeled excerpts and source list ---
        excerpts = []
        sources = []
        for doc in docs:
            label = f"{doc.section_type} {doc.section_title}"
            excerpts.append(f"[{label}]\n{doc.content.strip()}")
            sources.append({"label": label, "url": doc.url})

        excerpts_text = "\n\n".join(excerpts)

        # --- Prompt with strict citation scoping ---
        prompt = f"""
You are a legal AI assistant specializing in the EU Artificial Intelligence Act.

USER QUESTION:
{query}

Below are excerpts from the EU AI Act retrieved for this query. Each excerpt is labeled with its legal source.

EXCERPTS:
{excerpts_text}

TASK:
- Answer the question using ONLY the excerpts provided.
- Do NOT use or reference any external knowledge.
- Cite sources exactly as labeled above (e.g., "Article 9", "Recital 52").
- Structure the answer clearly:
  1. Concise summary paragraph.
  2. Structured explanation using bullet points with explicit citations.
  3. Sources section listing all cited Articles/Recitals/Annexes with label + URL.
- Skip any points not covered in the excerpts; do not hallucinate.
- At the end, list Sources as bullet points showing label and URL, no extra commentary.
"""

        # --- LLM call ---
        def groq_sync_call():
            response = groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=700,
            )
            return response.choices[0].message.content.strip()

        answer_text = await asyncio.to_thread(groq_sync_call)

        # Return human-readable answer with clean source list
        return {
            "answer": answer_text,
            "sources": sources,
        }
