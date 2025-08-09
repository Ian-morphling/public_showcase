import os
import numpy as np
import faiss
import pickle
import re

# === Config ===
EMBEDDING_DIR = "/home/ianli/homl-self/public_showcase/ecommerce_llm_recommender/data/embeddings"
DOCUMENTS_DIR = "/home/ianli/homl-self/public_showcase/ecommerce_llm_recommender/outputs"
SAVE_DIR = "/home/ianli/homl-self/public_showcase/ecommerce_llm_recommender/index"
INDEX_FILE = os.path.join(SAVE_DIR, "product.index")
MAPPING_FILE = os.path.join(SAVE_DIR, "id_to_filename.pkl")

EMBEDDING_DIM = 384
NLIST = 100
SAVE_INTERVAL = 5

os.makedirs(SAVE_DIR, exist_ok=True)

# Get sorted embedding files
embedding_files = sorted([
    os.path.join(EMBEDDING_DIR, f)
    for f in os.listdir(EMBEDDING_DIR)
    if f.endswith(".npy")
])

print(f"Found {len(embedding_files)} embedding files")

# Train index on first file
vecs = np.load(embedding_files[0])
faiss.normalize_L2(vecs)

quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
index = faiss.IndexIVFFlat(quantizer, EMBEDDING_DIM, NLIST, faiss.METRIC_INNER_PRODUCT)

assert not index.is_trained, "Index should not be trained yet"
index.train(vecs)
print("Index trained.")

id_to_filename = {}
next_id = 0

def embedding_to_parquet(embedding_filename):
    """
    Convert embedding filename like
    product_embeddings_minilm_chunk_000.npy
    to document parquet filename like
    documents_part_0000.parquet
    """
    m = re.search(r"chunk_(\d+)", embedding_filename)
    if m:
        chunk_num = int(m.group(1))
        return f"documents_part_{chunk_num:04d}.parquet"
    else:
        raise ValueError(f"Cannot parse chunk number from {embedding_filename}")

for i, fpath in enumerate(embedding_files):
    try:
        vecs = np.load(fpath)
        faiss.normalize_L2(vecs)
        index.add(vecs)
        print(f" Added chunk {i}, total vectors: {index.ntotal}")

        parquet_file = embedding_to_parquet(os.path.basename(fpath))
        for _ in range(vecs.shape[0]):
            id_to_filename[next_id] = parquet_file
            next_id += 1

        if (i + 1) % SAVE_INTERVAL == 0:
            faiss.write_index(index, INDEX_FILE)
            with open(MAPPING_FILE, "wb") as f:
                pickle.dump(id_to_filename, f)
            print(f" Checkpoint saved at chunk {i + 1}")

    except Exception as e:
        print(f" Error at chunk {i} ({fpath}): {e}")

faiss.write_index(index, INDEX_FILE)
with open(MAPPING_FILE, "wb") as f:
    pickle.dump(id_to_filename, f)

print(f" Finished! Total vectors: {index.ntotal}")
print(f"FAISS index saved to: {INDEX_FILE}")
print(f"ID mapping saved to: {MAPPING_FILE}")
