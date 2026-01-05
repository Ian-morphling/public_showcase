# backend/api.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph_nodes import build_graph
import os
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pathlib import Path
from dotenv import load_dotenv
import traceback
import numpy as np

# Config Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT_DIR / "index" / "product.index"
MAPPING_PATH = ROOT_DIR / "index" / "id_to_filename.pkl"
CHUNKS_DIR = ROOT_DIR / "outputs"
USER_PROFILES_DIR = CHUNKS_DIR
load_dotenv(dotenv_path=ROOT_DIR / ".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Helper: Convert NumPy to native Python
def to_native(obj):
    """Recursively convert NumPy types to native Python types."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()  # numpy.int64, numpy.bool_, etc.
    else:
        return obj

# -------------------------
# FastAPI + Rate Limiter
# -------------------------
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="E-Commerce Recommender API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Allow CORS for local testing with Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Schema
class RecommendRequest(BaseModel):
    query: str
    top_k: int = 5
    reviewer_id: str | None = None

# Custom Exceptions
class TransientGraphError(Exception):
    """Retryable error (e.g., temporary network or API issues)"""
    pass

class NonTransientGraphError(Exception):
    """Non-retryable error (e.g., invalid input or missing data)"""
    pass

# Load LangGraph once
graph = build_graph(
    index_path=INDEX_PATH,
    mapping_path=MAPPING_PATH,
    docs_dir=CHUNKS_DIR,
    user_profiles_dir=USER_PROFILES_DIR,
    groq_api_key=GROQ_API_KEY
)

# Retry Decorator for transient failures only
@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(TransientGraphError),
    reraise=True
)
def invoke_graph_with_retry(state: dict):
    """Invoke LangGraph workflow, retrying only transient errors."""
    try:
        result = graph.invoke(state)
        return result
    except TransientGraphError:
        raise
    except Exception as e:
        # Any other error is non-transient
        raise NonTransientGraphError(str(e)) from e

# API Endpoint
@app.post("/recommend")
@limiter.limit("5/minute")
async def recommend(req: RecommendRequest, request: Request):
    state = {
        "query": req.query,
        "top_k": req.top_k,
        "reviewer_id": req.reviewer_id
    }
    try:
        result = invoke_graph_with_retry(state)
        response = to_native({
            "answer": result.get("answer"),
            "retrieved_docs": result.get("retrieved_docs"),
            "user_profile_summary": result.get("user_profile_summary")
        })
        return response

    except NonTransientGraphError as e:
        # Client error or permanent failure: return 400
        raise HTTPException(
            status_code=400,
            detail=f"Non-transient error: {str(e)}"
        )

    except Exception as e:
        # Transient/internal error after retries: return 500
        print("Graph execution failed:", e)
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Transient/internal error: {str(e)}"
        )
