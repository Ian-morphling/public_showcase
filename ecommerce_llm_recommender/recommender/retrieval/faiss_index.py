import os
import pickle
import faiss
import pandas as pd
from pathlib import Path
from functools import lru_cache
from typing import Tuple, Dict

# Update these paths to your project config or import from recommender.config
INDEX_PATH = Path("/home/ianli/homl-self/public_showcase/ecommerce_llm_recommender/index/product.index")
MAPPING_PATH = Path("/home/ianli/homl-self/public_showcase/ecommerce_llm_recommender/index/id_to_filename.pkl")
DOCUMENTS_DIR = Path("/home/ianli/homl-self/public_showcase/ecommerce_llm_recommender/outputs")

def load_index() -> Tuple[faiss.Index, Dict[int, str]]:
    """
    Load the FAISS index and ID to parquet filename mapping.

    Returns:
        index (faiss.Index): Loaded FAISS index.
        id_to_filename (dict): Maps global vector ID to parquet filename.

    Raises:
        FileNotFoundError: If index or mapping files do not exist.
        ValueError: If mapping size doesn't match index ntotal.
    """
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index file not found at {INDEX_PATH}")
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(f"ID mapping file not found at {MAPPING_PATH}")

    index = faiss.read_index(str(INDEX_PATH))
    with open(MAPPING_PATH, "rb") as f:
        id_to_filename = pickle.load(f)

    if len(id_to_filename) != index.ntotal:
        raise ValueError(f"Mapping length {len(id_to_filename)} does not match index ntotal {index.ntotal}")

    return index, id_to_filename

def compute_chunk_sizes_and_offsets(id_to_filename: Dict[int, str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Compute chunk sizes (number of rows) and cumulative offsets for each parquet file.

    Args:
        id_to_filename (dict): Mapping from global vector ID to parquet filename.

    Returns:
        chunk_sizes (dict): parquet filename → number of rows in that file.
        file_offsets (dict): parquet filename → starting global index offset.

    Raises:
        ValueError: If sum of chunk sizes doesn't match length of mapping.
    """
    sorted_files = sorted(set(id_to_filename.values()))
    chunk_sizes = {}
    file_offsets = {}

    offset = 0
    for filename in sorted_files:
        file_path = DOCUMENTS_DIR / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")

        # Only load minimal columns for speed
        df = pd.read_parquet(file_path, columns=["text"])
        size = len(df)
        chunk_sizes[filename] = size
        file_offsets[filename] = offset
        offset += size

    total_vectors = sum(chunk_sizes.values())
    if total_vectors != len(id_to_filename):
        raise ValueError(f"Sum of chunk sizes {total_vectors} does not equal mapping length {len(id_to_filename)}")

    return chunk_sizes, file_offsets

@lru_cache(maxsize=8)
def load_parquet_cached(filename: str) -> pd.DataFrame:
    """
    Load parquet file from DOCUMENTS_DIR with LRU cache to speed repeated access.

    Args:
        filename (str): Parquet filename.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    path = DOCUMENTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path)

def get_file_and_local_idx(global_idx: int,
                           chunk_sizes: Dict[str, int],
                           file_offsets: Dict[str, int]) -> Tuple[str, int]:
    """
    Given a global FAISS vector index, find the parquet file and local index inside that file.

    Args:
        global_idx (int): Global FAISS vector index.
        chunk_sizes (dict): parquet filename → number of rows.
        file_offsets (dict): parquet filename → global offset.

    Returns:
        (filename, local_idx): filename of parquet and local row index inside parquet.

    Raises:
        IndexError: If global index is out of bounds.
    """
    for filename in sorted(chunk_sizes.keys()):
        start = file_offsets[filename]
        end = start + chunk_sizes[filename]
        if start <= global_idx < end:
            return filename, global_idx - start
    raise IndexError(f"Global index {global_idx} not found in any chunk file.")

def get_review_by_global_id(global_idx: int,
                            chunk_sizes: Dict[str, int],
                            file_offsets: Dict[str, int]) -> pd.Series:
    """
    Retrieve the review row (as pandas Series) from the parquet file by global FAISS vector index.

    Args:
        global_idx (int): Global FAISS vector index.
        chunk_sizes (dict): parquet filename → number of rows.
        file_offsets (dict): parquet filename → global offset.

    Returns:
        pd.Series: The review row.

    Raises:
        IndexError: If local index is out of bounds in the parquet file.
    """
    filename, local_idx = get_file_and_local_idx(global_idx, chunk_sizes, file_offsets)
    df = load_parquet_cached(filename)

    if local_idx >= len(df):
        raise IndexError(f"Local index {local_idx} out of range for file {filename}")

    return df.iloc[local_idx]

if __name__ == "__main__":
    print("Testing faiss_index.py module...")

    try:
        index, id_to_filename = load_index()
        print(f"Loaded FAISS index with {index.ntotal} vectors")
        
        chunk_sizes, file_offsets = compute_chunk_sizes_and_offsets(id_to_filename)
        print(f"Computed {len(chunk_sizes)} chunk sizes and offsets")

        # Test a few random global IDs (try 0, middle, last)
        test_ids = [0, index.ntotal // 2, index.ntotal - 1]

        for gid in test_ids:
            filename, local_idx = get_file_and_local_idx(gid, chunk_sizes, file_offsets)
            print(f"Global ID {gid} -> File: {filename}, Local ID: {local_idx}")

            review = get_review_by_global_id(gid, chunk_sizes, file_offsets)
            print(f"Sample review text (first 100 chars): {review.get('text', '')[:100]}")

        print("All tests passed!")

    except Exception as e:
        print(f"Error during test: {e}")
