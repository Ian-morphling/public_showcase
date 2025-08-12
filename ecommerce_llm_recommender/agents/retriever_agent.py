import os
import pickle
from typing import List, Dict, Any, Tuple
import faiss
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import re


class RetrieverAgent:
    def __init__(
        self,
        index_path: str,
        mapping_path: str,
        documents_dir: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.documents_dir = documents_dir
        self.model_name = model_name

        self.index = None
        self.id_to_file = None
        self.chunk_sizes = {}
        self.file_offsets = {}
        self.sorted_files = []
        self.model = None

    def initialize(self):
        self._load_index()
        self._load_model()
        self._build_mappings()

    def _load_index(self):
        print(f"Loading FAISS index from {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        with open(self.mapping_path, "rb") as f:
            self.id_to_file = pickle.load(f)
        # Convert embedding filenames to parquet chunk filenames if needed
        self.id_to_file = {
            k: self._embedding_to_parquet(v) for k, v in self.id_to_file.items()
        }

    def _embedding_to_parquet(self, filename: str) -> str:
        # Example: embeddings_part_0001.npy -> documents_part_0001.parquet
        if filename.endswith(".npy"):
            idx = filename.split("_")[-1].split(".")[0]
            return f"documents_part_{int(idx):04d}.parquet"
        return filename

    def _load_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading sentence transformer model on device: {device}")
        self.model = SentenceTransformer(self.model_name, device=device)

    def _build_mappings(self):
        print("Building chunk file mappings and offsets...")
        self.sorted_files = sorted(set(self.id_to_file.values()))
        offset = 0
        for file in self.sorted_files:
            file_path = os.path.join(self.documents_dir, file)
            df = pd.read_parquet(file_path)
            size = len(df)
            self.chunk_sizes[file] = size
            self.file_offsets[file] = offset
            offset += size
        total = sum(self.chunk_sizes.values())
        assert total == self.index.ntotal, f"Index size mismatch: {total} vs {self.index.ntotal}"
        print(f"Loaded {len(self.sorted_files)} chunks totaling {total} documents.")

    def _find_file_local_idx(self, global_idx: int) -> Tuple[str, int]:
        for file in self.sorted_files:
            start = self.file_offsets[file]
            end = start + self.chunk_sizes[file]
            if start <= global_idx < end:
                return file, global_idx - start
        raise IndexError(f"Global idx {global_idx} not found in chunks.")

    def embed_query(self, query: str):
        emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype("float32")

    def extract_summary(self, text: str) -> str:
        match = re.search(r"Summary:\s*(.*)", text)
        return match.group(1).strip() if match else ""

    def remove_embedded_summary(self, text: str) -> str:
        return re.sub(r"Summary:\s*.*", "", text).strip()

    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_emb = self.embed_query(query)
        D, I = self.index.search(query_emb, top_k)

        results = []
        for dist, idx in zip(D[0], I[0]):
            file, local_idx = self._find_file_local_idx(idx)
            df = pd.read_parquet(os.path.join(self.documents_dir, file))
            row = df.iloc[local_idx]
            
            # Extract text and metadata
            raw_text = row.get("text", "")
            summary = self.extract_summary(raw_text)
            clean_text = self.remove_embedded_summary(raw_text)

            metadata = row.get("metadata", {}) if "metadata" in row else {}

            # For normalized vectors with METRIC_INNER_PRODUCT, 'dist' is cosine similarity
            similarity_score = dist

            results.append(
                {
                    "similarity": similarity_score, 
                    "summary": summary,
                    "rating": metadata.get("overall", "N/A"),
                    "votes": metadata.get("votes", "N/A"),
                    "verified": metadata.get("verified", "N/A"),
                    "date": metadata.get("reviewTime", "N/A"),
                    "reviewer": metadata.get("reviewerName", "N/A"),
                    "text": clean_text,
                }
            )
        return results
