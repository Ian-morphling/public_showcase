# python -m scripts.run_rag --query "I want a durable laptop with long battery"

import os
import argparse
from agents.retriever_agent import RetrieverAgent
from agents.explainer_agent import ExplainerAgent
from dotenv import load_dotenv


def display_retrieval_results(retrieved_reviews):
    print(f"\n=== Retrieved {len(retrieved_reviews)} Reviews ===\n")
    for i, r in enumerate(retrieved_reviews):
        print(f"Review {i+1} (Similarity: {r.get('similarity', 0):.4f}):")
        print(f"  Summary: {r.get('summary', 'N/A')}")
        print(f"  Rating: {r.get('rating', 'N/A')} stars | Verified: {r.get('verified', 'N/A')}")
        print(f"  Helpful Votes: {r.get('votes', 'N/A')}")
        print(f"  Date: {r.get('date', 'N/A')} | Reviewer: {r.get('reviewer', 'N/A')}")
        print(f"  Text: {r.get('text', '')[:300]}...\n")
        print("-" * 60)

def build_rag_prompt(query, retrieved_reviews):
    context = ""
    for i, r in enumerate(retrieved_reviews):
        context += f"Review {i+1} (Similarity: {r.get('similarity', 0.0):.4f}):\n"
        if r.get('summary'):
            context += f"Summary: {r['summary']}\n"
        context += f"Rating: {r.get('rating', 'N/A')} stars | Verified: {r.get('verified', 'N/A')}\n"
        context += f"Helpful Votes: {r.get('votes', 'N/A')}\n"
        context += f"Date: {r.get('date', 'N/A')} | Reviewer: {r.get('reviewer', 'N/A')}\n"
        context += f"Text: {r.get('text', '')}\n\n"

    prompt = f"""
You are a helpful e-commerce assistant. Use the following product reviews to answer the user's question.
The reviews are ranked by similarity to your query (higher similarity = more relevant).

--- START OF REVIEWS ---
{context.strip()}
--- END OF REVIEWS ---

User question:
{query}

Answer concisely, referencing the relevant reviews if needed. If the answer isn't clear, say so honestly.
""".strip()
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Run RAG e-commerce recommender")
    parser.add_argument("--query", "-q", type=str, required=True, help="User query string")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results to retrieve")
    args = parser.parse_args()

    # Set paths and env vars here (adjust as needed)
    load_dotenv()
    INDEX_PATH = os.getenv("INDEX_PATH", "index/product.index")
    MAPPING_PATH = os.getenv("MAPPING_PATH", "index/id_to_filename.pkl")
    DOCS_DIR = os.getenv("DOCS_DIR", "outputs")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    retriever = RetrieverAgent(INDEX_PATH, MAPPING_PATH, DOCS_DIR)
    retriever.initialize()

    explainer = ExplainerAgent(GROQ_API_KEY)

    # Retrieve
    raw_results = retriever.query(args.query, top_k=args.top_k)

    # Print raw retrieved reviews with similarity scores and metadata
    display_retrieval_results(raw_results)

    # Build RAG prompt to feed to LLM
    rag_prompt = build_rag_prompt(args.query, raw_results)

    # Generate answer using LLM
    answer = explainer.generate_answer(args.query, raw_results)  # pass query, docs

    print(f"\nFinal answer:\n{answer}")

if __name__ == "__main__":
    main()
