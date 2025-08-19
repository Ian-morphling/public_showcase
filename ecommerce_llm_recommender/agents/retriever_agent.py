import re
import pickle
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


class RetrieverAgent:
    def __init__(
        self,
        index_path: str,
        mapping_path: str,
        docs_dir: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.index_path = Path(index_path)
        self.mapping_path = Path(mapping_path)
        self.docs_dir = Path(docs_dir)
        self.model_name = model_name

        # ---- Device selection ----
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # ---- Validate paths ----
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.mapping_path.exists():
            raise FileNotFoundError(f"ID-to-file mapping not found: {self.mapping_path}")
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {self.docs_dir}")

        # ---- Load FAISS index (memory-mapped) ----
        print("Loading FAISS index (memory-mapped)...")
        self.index = faiss.read_index(str(self.index_path), faiss.IO_FLAG_MMAP)

        # ---- Load ID -> parquet filename mapping ----
        print("Loading ID-to-file mapping...")
        with open(self.mapping_path, "rb") as f:
            self.id_to_file: Dict[int, str] = pickle.load(f)

        # ---- Load embedding model ----
        print(f"Loading embedding model on device: {self.device}")
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:
            print(f"Failed to load model on {self.device}: {e}")
            print("Falling back to CPU.")
            self.device = "cpu"
            self.model = SentenceTransformer(self.model_name, device=self.device)

        # Precompile summary regex (extracts everything after 'Summary:')
        self._summary_re = re.compile(r"Summary:\s*(.*)", flags=re.IGNORECASE | re.DOTALL)

    # -------- Internal helpers --------
    def _extract_title_from_text(self, text: str) -> str | None:
        if not isinstance(text, str):
            return None
        m = self._summary_re.search(text)
        if m:
            title = m.group(1).strip()
            return title if title else None
        return None

    def _load_doc(self, doc_id: int) -> Dict:
        """
        Load a single row from the mapped parquet file.
        Parquet schema (from preprocessor.py):
        ['id','text','asin','overall','rating_label','review_length',
         'verified','reviewTime','reviewerID','reviewerName','has_votes',
         'votes','vote_bin']
        """
        # Safe defaults
        empty = {
            "asin": None,
            "title": None,
            "text": None,
            "overall": None,
            "verified": None,
            "reviewerID": None,
            "reviewerName": None,
        }

        file_name = self.id_to_file.get(doc_id)
        if file_name is None:
            return empty

        file_path = self.docs_dir / file_name
        if not file_path.exists():
            return empty

        try:
            df = pd.read_parquet(file_path, engine="pyarrow")
            if len(df) == 0:
                return empty

            # NOTE: Because build_index doesn't store row offsets,
            # doc_ids for this file are sequential. We modulo into the file.
            row_idx = doc_id % len(df)
            row = df.iloc[row_idx]

            text_val = row.get("text", None)
            title_val = self._extract_title_from_text(text_val) if isinstance(text_val, str) else None

            return {
                "asin": row.get("asin", None),
                "title": title_val,
                "text": text_val,
                "overall": row.get("overall", None),
                "verified": row.get("verified", None),
                "reviewerID": row.get("reviewerID", None),
                "reviewerName": row.get("reviewerName", None),
            }
        except Exception as e:
            print(f"Warning loading doc {doc_id} from {file_path}: {e}")
            return empty

    # -------- Public API --------
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Embed the string query and return top_k documents.
        Returns a list of dicts with: asin, title, text, overall, verified, reviewerID, reviewerName, score(similarity).
        """
        # --- Embed query (normalized for cosine/IP) ---
        try:
            qvec = self.model.encode([query], normalize_embeddings=True)
            qvec = np.asarray(qvec, dtype="float32")
        except Exception as e:
            print(f"Error embedding query: {e}")
            return []

        # --- Search FAISS ---
        try:
            D, I = self.index.search(qvec, top_k)
            D = D.flatten()
            I = I.flatten()
        except Exception as e:
            print(f"FAISS search error: {e}")
            return []

        # --- Gather & format results ---
        results: List[Dict] = []
        for doc_id, sim in zip(I, D):
            doc = self._load_doc(int(doc_id))
            # IP on L2-normalized vectors equals cosine similarity
            doc["score"] = float(sim)
            doc["similarity"] = float(sim)  # alias for app UIs that expect 'similarity'
            results.append(doc)

        return results
