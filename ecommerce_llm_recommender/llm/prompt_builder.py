from typing import List, Dict

def build_rag_prompt(query: str, retrieved_docs: List[Dict]) -> str:
    context = ""
    for i, doc in enumerate(retrieved_docs):
        r = doc["review"]
        context += f"Review {i+1} (Similarity: {doc['similarity']:.4f}):\n"
        if "summary" in r:
            context += f"Summary: {r['summary']}\n"
        context += f"Rating: {r.get('rating', 'N/A')} | Verified: {r.get('verified', 'N/A')}\n"
        context += f"Helpful Votes: {r.get('votes', 'N/A')}\n"
        context += f"Date: {r.get('date', 'N/A')} | Reviewer: {r.get('reviewer', 'N/A')}\n"
        context += f"Text: {r.get('text', '')}\n\n"

    prompt = f"""You are a helpful e-commerce assistant. Use the following product reviews to answer the user's question.
The reviews are ranked by similarity to the query.

--- START OF REVIEWS ---
{context.strip()}
--- END OF REVIEWS ---

User question:
{query}

Answer concisely, referencing the relevant reviews if needed. If unclear, say so honestly."""
    return prompt
