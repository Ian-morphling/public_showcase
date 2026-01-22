# agents/explainer_agent.py
"""
ExplainerAgent for Agentic RAG.

Responsibilities:
- Synthesize retrieved EU AI Act documents into a structured answer
- Enforce strict citation discipline (Articles / Recitals / Annexes)
- Never retrieve, plan, or select sources — reasoning only
"""

import os
import asyncio
from typing import List
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
    ) -> str:
        """
        Generate a citation-bound explanation based ONLY on retrieved documents.

        IMPORTANT:
        - The LLM must NOT generate URLs or a sources list
        - Citations must reference ONLY the provided labels (e.g., Article 9)
        """

        if not docs:
            return "No relevant provisions were retrieved to answer this question."

        # --- Prepare labeled excerpts (labels are the ONLY allowed citations) ---
        excerpts = []
        for doc in docs:
            label = f"{doc.section_type} {doc.section_title}"
            excerpts.append(
                f"[{label}]\n{doc.content.strip()}"
            )

        excerpts_text = "\n\n".join(excerpts)

        # --- Strict citation-bound prompt ---
        prompt = f"""
You are a legal AI assistant specializing in the EU Artificial Intelligence Act.

USER QUESTION:
{query}

Below are excerpts from the EU AI Act retrieved for this query.
Each excerpt is labeled with its legal source.

EXCERPTS:
{excerpts_text}

TASK:
- Answer the question using ONLY the excerpts provided.
- Do NOT use or reference any external knowledge.
- Do NOT invent or guess Articles, Recitals, or Annexes.
- Cite sources ONLY using the provided labels (e.g., Article 9, Recital 52).
- Do NOT include URLs.
- Do NOT include a sources list.

OUTPUT STRUCTURE:
1. Concise summary paragraph.
2. Structured explanation using bullet points.
   - Each bullet must include inline citation labels where relevant.
- Skip any points not supported by the excerpts.
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

        return answer_text
