#python -m recommender.retrieval.query_engine "I want a durable laptop with long battery"

import sys
import os
import re
import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer

from recommender.retrieval.faiss_index import load_index


def compute_chunk_sizes_and_offsets(id_to_filename: Dict[int, str], documents_dir: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Compute number of rows in each parquet chunk and cumulative offsets.
    """
    chunk_sizes = {}
    file_offsets = {}
    sorted_files = sorted(set(id_to_filename.values()))
    offset = 0

    for filename in sorted_files:
        path = os.path.join(documents_dir, filename)
        df = pd.read_parquet(path, columns=["text"])
        size = len(df)
        chunk_sizes[filename] = size
        file_offsets[filename] = offset
        offset += size

    total_vectors = sum(chunk_sizes.values())
    assert total_vectors == len(id_to_filename), f"Mismatch between parquet rows ({total_vectors}) and FAISS vectors ({len(id_to_filename)})"
    return chunk_sizes, file_offsets


def find_file_and_local_idx(global_idx: int, chunk_sizes: Dict[str, int], file_offsets: Dict[str, int]) -> Tuple[str, int]:
    """
    Given a global index from FAISS, find the corresponding parquet file and local row index.
    """
    for filename in chunk_sizes:
        start = file_offsets[filename]
        end = start + chunk_sizes[filename]
        if start <= global_idx < end:
            local_idx = global_idx - start
            return filename, local_idx
    raise IndexError(f"Global index {global_idx} not found in any parquet chunk")


def extract_summary(text: str) -> str:
    match = re.search(r"Summary:\s*(.*)", text)
    return match.group(1).strip() if match else ""


def remove_embedded_summary(text: str) -> str:
    return re.sub(r"Summary:\s*.*", "", text).strip()


def get_review_by_global_id(global_idx: int, id_to_filename: Dict[int, str], chunk_sizes: Dict[str, int], file_offsets: Dict[str, int], documents_dir: str) -> pd.Series:
    """
    Retrieve the review row from parquet corresponding to a global index from FAISS.
    """
    filename, local_idx = find_file_and_local_idx(global_idx, chunk_sizes, file_offsets)
    path = os.path.join(documents_dir, filename)
    df = pd.read_parquet(path)
    if local_idx >= len(df):
        raise IndexError(f"Local index {local_idx} out of bounds for file {filename}")
    return df.iloc[local_idx]


class Retriever:
    def __init__(
        self,
        index_path: str = "index/product.index",
        mapping_path: str = "index/id_to_filename.pkl",
        documents_dir: str = "outputs",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        print("Loading FAISS index and mappings...")
        self.index, self.id_to_filename = load_index()
        self.documents_dir = documents_dir

        self.chunk_sizes, self.file_offsets = compute_chunk_sizes_and_offsets(self.id_to_filename, self.documents_dir)

        print(f"FAISS index loaded with {self.index.ntotal} vectors.")
        print(f"Loaded metadata for {len(self.chunk_sizes)} parquet chunks.")

        # Load embedding model (GPU if available)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model on {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.model.eval()

    def embed_query(self, query_text: str) -> np.ndarray:
        embedding = self.model.encode(query_text, normalize_embeddings=True)
        return embedding.astype(np.float32).reshape(1, -1)

    def query(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        distances, indices = self.index.search(query_embedding, top_k)
        results = []

        for dist, global_id in zip(distances[0], indices[0]):
            try:
                review_row = get_review_by_global_id(global_id, self.id_to_filename, self.chunk_sizes, self.file_offsets, self.documents_dir)
                review_dict = review_row.to_dict()

                raw_text = review_dict.get("text", "")
                summary = extract_summary(raw_text)
                review_text = remove_embedded_summary(raw_text)

                metadata = review_dict.get("metadata", {})

                results.append({
                    "global_id": int(global_id),
                    "similarity": float(dist),
                    "asin": metadata.get("asin", "N/A"),
                    "rating": metadata.get("overall", "N/A"),
                    "verified": metadata.get("verified", "N/A"),
                    "votes": metadata.get("votes", "N/A"),
                    "vote_bin": metadata.get("vote_bin", "N/A"),
                    "reviewer": metadata.get("reviewerName", "N/A"),
                    "date": metadata.get("reviewTime", "N/A"),
                    "summary": summary,
                    "review_text": review_text,
                })
            except Exception as e:
                print(f"Error retrieving review for global id {global_id}: {e}")
                continue

        return results


def print_results(results: List[Dict[str, Any]]) -> None:
    for i, r in enumerate(results, 1):
        print("=" * 80)
        print(f"Rank {i} | Similarity Score: {r['similarity']:.4f}")
        print(f"ASIN: {r['asin']}")
        print(f"Rating: {r['rating']} stars | Verified: {r['verified']} | Helpful Votes: {r['votes']} ({r['vote_bin']})")
        print(f"Reviewer: {r['reviewer']} | Date: {r['date']}")
        if r['summary']:
            print(f"Summary: {r['summary']}")
        print(f"Review Text: {r['review_text']}")
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m recommender.retrieval.query_engine \"your query here\"")
        sys.exit(1)

    query_str = sys.argv[1]
    retriever = Retriever()
    query_emb = retriever.embed_query(query_str)
    results = retriever.query(query_emb, top_k=5)
    print(f"Results for query: '{query_str}'\n")
    print_results(results)


if __name__ == "__main__":
    main()
