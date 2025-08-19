import pandas as pd
from pathlib import Path
import json
import glob

def generate_ground_truth_from_queries(queries_file, docs_dir, output_json):
    """
    Build ground truth automatically using metadata from preprocessed Parquet docs.
    For each query, find documents whose text or summary contains any keyword from the query.
    """
    docs_dir = Path(docs_dir)
    parquet_files = sorted(glob.glob(str(docs_dir / "documents_part_*.parquet")))
    if not parquet_files:
        raise ValueError(f"No documents found in {docs_dir}")

    # Load all docs
    all_docs = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        for _, row in df.iterrows():
            all_docs.append({
                "asin": row["asin"],
                "title": row.get("summary", ""),   # use summary as title
                "text": row.get("text", "")
            })

    # Read queries
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    ground_truth = {}
    for query in queries:
        query_keywords = set(query.lower().split())
        matched_ids = []

        for doc in all_docs:
            # Combine summary/title and review text for matching
            text = " ".join([
                str(doc.get("title", "")),
                str(doc.get("text", ""))
            ]).lower()

            if any(word in text for word in query_keywords):
                matched_ids.append(doc["asin"])

        ground_truth[query] = matched_ids

    # Save ground truth JSON
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Generated ground truth for {len(queries)} queries. Saved to {output_json}")