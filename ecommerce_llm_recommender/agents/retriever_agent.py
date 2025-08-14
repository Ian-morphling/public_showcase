import os
import pickle
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

class RetrieverAgent:
    def __init__(
        self,
        index_path: str,
        mapping_path: str,
        docs_dir: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.index_path = Path(index_path)
        self.mapping_path = Path(mapping_path)
        self.docs_dir = Path(docs_dir)
        self.model_name = model_name

        # Automatically select device
        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"GPU detected. Using device: {self.device}")
        else:
            self.device = "cpu"
            print(f"No GPU detected. Using device: {self.device}")

        # Validate files
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not self.mapping_path.exists():
            raise FileNotFoundError(f"ID-to-file mapping not found: {mapping_path}")
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

        print("Loading FAISS index (memory-mapped)...")
        self.index = faiss.read_index(str(self.index_path), faiss.IO_FLAG_MMAP)

        print("Loading ID-to-file mapping...")
        with open(self.mapping_path, "rb") as f:
            self.id_to_file = pickle.load(f)

        print(f"Loading embedding model on device: {self.device}")
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:
            print(f"Failed to load model on {self.device}: {e}")
            print("Falling back to CPU")
            self.device = "cpu"
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def _load_doc(self, doc_id: int) -> Dict:
        """Load the document safely from chunked parquet file."""
        if doc_id not in self.id_to_file:
            return {"text": "", "metadata": {}, "overall": None, "verified": None}

        file_name = self.id_to_file[doc_id]
        file_path = self.docs_dir / file_name
        if not file_path.exists():
            return {"text": "", "metadata": {}, "overall": None, "verified": None}

        try:
            df = pd.read_parquet(file_path, engine="pyarrow")
            row_idx = doc_id % len(df) if len(df) > 0 else 0
            row = df.iloc[row_idx]
            return {
                "text": row.get("text",""),
                "metadata": row.get("metadata", {}),
                "overall": row.get("metadata", {}).get("overall", None),
                "verified": row.get("metadata", {}).get("verified", None)
            }
        except Exception as e:
            print(f"Warning loading doc {doc_id}: {e}")
            return {"text": "", "metadata": {}, "overall": None, "verified": None}

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Return top_k FAISS results, safe against failures."""
        try:
            query_vec = self.model.encode([query], normalize_embeddings=True)
            query_vec = np.array(query_vec).astype("float32")
        except Exception as e:
            print(f"Error embedding query: {e}")
            return []

        try:
            D, I = self.index.search(query_vec, top_k)
            D = D.flatten()
            I = I.flatten()
        except Exception as e:
            print(f"FAISS search error: {e}")
            return []

        results = []
        for doc_id, sim in zip(I, D):
            doc = self._load_doc(int(doc_id))
            doc["similarity"] = float(sim)  # renamed for clarity
            results.append(doc)

        return results
