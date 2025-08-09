import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pathlib import Path
import glob

# Config
DATA_DIR = "outputs/clean_reviews"   # <-- ensure this matches load_reviews.py output_dir
EMBEDDINGS_DIR = "data/embeddings"
CHUNK_SIZE = 100_000
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

Path(EMBEDDINGS_DIR).mkdir(parents=True, exist_ok=True)

# Load model once
model = SentenceTransformer(MODEL_NAME)

# Track embedding chunk index
chunk_id = 0
buffer_texts = []

def embed_and_save(texts, chunk_id):
    print(f"Embedding chunk {chunk_id} with {len(texts)} texts...")
    embeddings = model.encode(
        texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True
    )
    out_path = os.path.join(EMBEDDINGS_DIR, f"product_embeddings_minilm_chunk_{chunk_id:03d}.npy")
    np.save(out_path, embeddings)
    print(f"Saved {out_path}")

# Load all .parquet files
parquet_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
print(f"Found {len(parquet_files)} cleaned .parquet files.")

for file_path in parquet_files:
    print(f"Processing file: {file_path}")
    df = pd.read_parquet(file_path)
    texts = df["reviewText"].dropna().astype(str).tolist()

    for text in texts:
        buffer_texts.append(text)
        if len(buffer_texts) >= CHUNK_SIZE:
            embed_and_save(buffer_texts, chunk_id)
            chunk_id += 1
            buffer_texts = []

# Save final chunk
if buffer_texts:
    embed_and_save(buffer_texts, chunk_id)
    print(f"Finished final chunk {chunk_id}.")


