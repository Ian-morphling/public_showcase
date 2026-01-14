# backend/api.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph_nodes import build_graph
import os
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pathlib import Path
from dotenv import load_dotenv
import traceback
import numpy as np
import logging
from contextlib import asynccontextmanager 

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

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

# FastAPI + Rate Limiter
limiter = Limiter(key_func=get_remote_address)

# FastAPI Lifespan #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan:
    - Initialize LangGraph + agents once at startup
    - Fail fast if index/model loading fails
    """
    try:
        logger.info("Starting up: building LangGraph recommender")

        app.state.graph = build_graph(
            index_path=INDEX_PATH,
            mapping_path=MAPPING_PATH,
            docs_dir=CHUNKS_DIR,
            user_profiles_dir=USER_PROFILES_DIR,
            groq_api_key=GROQ_API_KEY
        )

        app.state.ready = True
        logger.info("Startup complete: LangGraph ready")

        yield

    except Exception as e:
        logger.exception("Startup failed: LangGraph initialization error")
        app.state.ready = False
        raise e

    finally:
        logger.info("Shutting down API")

# Attach lifespan to FastAPI app
app = FastAPI(
    title="E-Commerce Recommender API",
    lifespan=lifespan
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
    """Retryable error (temporary, safe to retry)"""
    pass

class NonTransientGraphError(Exception):
    """Non-retryable error (client or permanent)"""
    pass

# Graph Invocation with retry #
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=6),
    retry=retry_if_exception_type(TransientGraphError),
    reraise=True
)
def invoke_graph_with_retry(graph, state: dict): 
    """
    Invoke LangGraph workflow.
    Retry only on transient errors.
    """
    try:
        result = graph.invoke(state)

        # Detect empty retrievals as transient
        if not result.get("retrieved_docs"):
            logger.warning("Transient: Empty retrievals for query: %s", state["query"])
            raise TransientGraphError("Empty retrieval results")

        return result

    except RuntimeError as e:
        msg = str(e)
        if "CUDA out of memory" in msg:
            logger.error("Transient: GPU OOM: %s", msg)
            raise TransientGraphError("GPU out-of-memory") from e
        elif "cannot mmap" in msg or "FAISS" in msg:
            logger.error("Transient: FAISS index load failure: %s", msg)
            raise TransientGraphError("FAISS index loading issue") from e
        elif "Groq API" in msg:
            logger.error("Transient: Groq API failure: %s", msg)
            raise TransientGraphError("Groq API failure") from e
        else:
            logger.error("Non-transient runtime error: %s", msg)
            raise NonTransientGraphError(msg) from e

    except TransientGraphError:
        raise

    except Exception as e:
        logger.error("Non-transient unknown error: %s", e)
        raise NonTransientGraphError(str(e)) from e

# /recommend Endpoint #
@app.post("/recommend")
@limiter.limit("5/minute")
async def recommend(req: RecommendRequest, request: Request):
    state = {
        "query": req.query,
        "top_k": req.top_k,
        "reviewer_id": req.reviewer_id
    }

    try:
        graph = request.app.state.graph 
        result = invoke_graph_with_retry(graph, state)
        response = to_native({
            "answer": result.get("answer"),
            "retrieved_docs": result.get("retrieved_docs"),
            "user_profile_summary": result.get("user_profile_summary")
        })
        return response

    except NonTransientGraphError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Non-transient error: {str(e)}"
        )

    except Exception as e:
        logger.error("Graph execution failed after retries: %s", e)
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Transient/internal error: {str(e)}"
        )

# /health Endpoint
@app.get("/health")
async def health():
    """
    Health / readiness check.
    Suitable for container orchestration.
    """
    if getattr(app.state, "ready", False) and hasattr(app.state, "graph"):
        return {
            "status": "healthy",
            "service": "langgraph-recommender"
        }

    return {
        "status": "unhealthy",
        "service": "langgraph-recommender"
    }