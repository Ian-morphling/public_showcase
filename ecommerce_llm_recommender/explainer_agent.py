import os
import requests
from typing import List, Dict


class ExplainerAgent:
    def __init__(self, groq_api_key: str):
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY must be provided")
        self.api_key = groq_api_key
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def build_rag_prompt(self, query: str, retrieved_docs: List[Dict]) -> str:
        context = ""
        for i, doc in enumerate(retrieved_docs):
            context += f"Review {i+1} (Similarity: {doc['similarity']:.4f}):\n"
            if "summary" in doc:
                context += f"Summary: {doc['summary']}\n"
            context += f"Rating: {doc.get('rating', 'N/A')} | Verified: {doc.get('verified', 'N/A')}\n"
            context += f"Helpful Votes: {doc.get('votes', 'N/A')}\n"
            context += f"Date: {doc.get('date', 'N/A')} | Reviewer: {doc.get('reviewer', 'N/A')}\n"
            context += f"Text: {doc.get('text', '')}\n\n"

        prompt = f"""You are a helpful e-commerce assistant. Use the following product reviews to answer the user's question.
The reviews are ranked by similarity to the query.

--- START OF REVIEWS ---
{context.strip()}
--- END OF REVIEWS ---

User question:
{query}

Answer concisely, referencing the relevant reviews if needed. If unclear, say so honestly."""
        return prompt

    def call_groq_llm(
        self, prompt: str, model: str = "llama3-70b-8192", temperature: float = 0.7, max_tokens: int = 512
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful e-commerce assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = requests.post(self.endpoint, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def generate_answer(self, query: str, retrieved_docs: List[Dict]) -> str:
        prompt = self.build_rag_prompt(query, retrieved_docs)
        return self.call_groq_llm(prompt)
