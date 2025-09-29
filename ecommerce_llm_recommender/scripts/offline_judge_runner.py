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
    use_judge_in_app=True  # enable judge
)

# --- Output directory ---
OUT_DIR = Path("judge_outputs")
OUT_DIR.mkdir(exist_ok=True)

# --- Run all queries ---
for i, query in enumerate(TEST_QUERIES, 1):
    print(f"\n=== Running query {i}/{len(TEST_QUERIES)} ===")
    print(f"Query: {query}")

    state = {"query": query, "top_k": 5, "reviewer_id": None}

    try:
        result = graph.invoke(state)

        # Extract fields
        answer = result.get("answer", "")
        judge_scores = result.get("judge_scores", {})

        # --- Find latest judge JSON written by node_judge ---
        all_files = sorted(OUT_DIR.glob("judge_run_*.json"))
        if all_files:
            latest_file = all_files[-1]  # the most recent file
            with open(latest_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            print(f"Using existing judge JSON: {latest_file}")
        else:
            # fallback if node_judge did not write
            payload = {
                "query": query,
                "answer": answer,
                "judge_scores": judge_scores,
                "docs_used": [d.get("text", "") for d in result.get("retrieved_docs", [])],
            }
            print("No existing judge JSON found; using fallback payload.")

        # --- Print concise judge summary ---
        if judge_scores:
            summary = {
                "Relevance": judge_scores.get("Relevance"),
                "Groundedness": judge_scores.get("Groundedness"),
                "Balance": judge_scores.get("Balance"),
            }
            print(f"Judge summary: {summary}")
        else:
            print("Judge summary not available")

    except Exception as e:
        print(f"Error processing query '{query}': {e}")