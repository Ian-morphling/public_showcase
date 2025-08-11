import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pathlib import Path
import glob

DATA_DIR = "outputs"  
EMBEDDINGS_DIR = "data/embeddings"
CHUNK_SIZE = 100_000
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

Path(EMBEDDINGS_DIR).mkdir(parents=True, exist_ok=True)

# Load model once
model = SentenceTransformer(MODEL_NAME)

def embed_and_save(texts, chunk_id):
    print(f"Embedding chunk {chunk_id} with {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    out_path = os.path.join(
        EMBEDDINGS_DIR,
        f"product_embeddings_minilm_chunk_{chunk_id:03d}.npy"
    )
    np.save(out_path, embeddings)
    print(f"Saved {out_path}")

def main():
    chunk_id = 0
    buffer_texts = []

    # Only pick up the preprocessed "documents_part" parquet files
    parquet_files = sorted(
        glob.glob(os.path.join(DATA_DIR, "documents_part_*.parquet"))
    )
    print(f"Found {len(parquet_files)} preprocessed .parquet files.")

    for file_path in tqdm(parquet_files, desc="Processing documents"):
        try:
            # Read with pyarrow backend 
            df = pd.read_parquet(file_path, engine="pyarrow")
            # Use the combined review + summary text from preprocessor.py
            texts = df["text"].dropna().astype(str).tolist()

            for text in texts:
                buffer_texts.append(text)
                if len(buffer_texts) >= CHUNK_SIZE:
                    embed_and_save(buffer_texts, chunk_id)
                    chunk_id += 1
                    buffer_texts = []
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if buffer_texts:
        embed_and_save(buffer_texts, chunk_id)
        print(f"Finished final chunk {chunk_id}.")

if __name__ == "__main__":
    main()
