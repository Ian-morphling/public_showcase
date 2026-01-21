import os
import json
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

# Load .env
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_HOST")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load MiniLM model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
print(f"Loading embedding model {MODEL_NAME}...")
embed_model = SentenceTransformer(MODEL_NAME)

# Data folder
RAW_FOLDER = Path("data/raw")

def embed_text(text: str):
    """Generate embedding using MiniLM."""
    return embed_model.encode(text).tolist()

def insert_chunks_to_supabase(chunks, section_type):
    for i, chunk in enumerate(chunks, 1):
        payload = {
            "section_type": section_type,
            "section_title": chunk.get("section_title", ""),
            "content": chunk.get("content", ""),
            "url": chunk.get("url", ""),
            "embedding": embed_text(chunk.get("content", "")),
        }
        try:
            response = supabase.table("documents").insert(payload).execute()
        except Exception as e:
            print(f"[Error] Chunk {i}: {e}")

# Process all JSON files in RAW_FOLDER
json_files = sorted(RAW_FOLDER.glob("*.json"))
print(f"Found {len(json_files)} JSON files: {[f.name for f in json_files]}")

for json_file in json_files:
    print(f"\nProcessing {json_file.name} ...")
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Ensure data is a list of chunks
    if isinstance(data, dict):
        chunks = data.get("chunks", [])
    else:
        chunks = data

    print(f"Loaded {len(chunks)} chunks.")
    
    # Determine section_type from file name
    if "article" in json_file.name.lower():
        section_type = "Article"
    elif "recital" in json_file.name.lower():
        section_type = "Recital"
    elif "annex" in json_file.name.lower():
        section_type = "Annex"
    else:
        section_type = "Unknown"

    insert_chunks_to_supabase(chunks, section_type)

print("\nAll JSON files processed successfully.")
