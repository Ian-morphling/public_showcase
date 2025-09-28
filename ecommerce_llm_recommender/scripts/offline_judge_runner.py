# python -m scripts.offline_judge_runner
import os
import time
import json
from pathlib import Path
from langgraph_nodes import build_graph

# --- Config paths ---
INDEX_PATH = "index/product.index"
MAPPING_PATH = "index/id_to_filename.pkl"
DOCS_DIR = Path("outputs")
USER_PROFILES_DIR = DOCS_DIR
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Test Queries ---
TEST_QUERIES = [
    "Best wireless headphones under $200?",
    "Top-rated Bluetooth speakers for outdoor use?",
    "Laptops with longest battery life under $1000?",
]

# --- Load LangGraph with Judge enabled ---
graph = build_graph(
    index_path=INDEX_PATH,
    mapping_path=MAPPING_PATH,
    docs_dir=DOCS_DIR,
    user_profiles_dir=USER_PROFILES_DIR,
    groq_api_key=GROQ_API_KEY,
    use_judge_in_app=True  # <-- enable judge
)

# --- Ensure output directory exists ---
OUT_DIR = Path("judge_outputs")
OUT_DIR.mkdir(exist_ok=True)

# --- Run all queries ---
for i, query in enumerate(TEST_QUERIES, 1):
    print(f"\n=== Running query {i}/{len(TEST_QUERIES)} ===")
    print(f"Query: {query}")

    state = {"query": query, "top_k": 5, "reviewer_id": None}

    try:
        result = graph.invoke(state)

        # Extract answer, docs, and judge_scores from the result
        answer = result.get("answer", "")
        retrieved_docs = result.get("retrieved_docs", [])
        judge_scores = result.get("judge_scores", {})

        # Prepare timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        out_file = OUT_DIR / f"offline_judge_{i}_{timestamp}.json"

        # --- Build payload ---
        payload = {
            "query": query,
            "answer": answer,
            "judge_scores": judge_scores,
            "docs_used": [d.get("text", "") for d in retrieved_docs],
        }

        # Save offline JSON
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        # --- Print concise judge summary only ---
        if judge_scores:
            summary = {
                "Relevance": judge_scores.get("Relevance"),
                "Groundedness": judge_scores.get("Groundedness"),
                "Balance": judge_scores.get("Balance"),
            }
            print(f"Judge results saved to {out_file}")
            print(f"Judge summary: {summary}")
        else:
            print(f"Judge results saved to {out_file} (no scores returned)")

    except Exception as e:
        print(f"Error processing query '{query}': {e}")
