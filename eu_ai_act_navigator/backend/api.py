# backend/api.py
"""
FastAPI backend for EU AI Act Navigator

Exposes a single RAG endpoint with:
- final mode: answer + citations
- full mode: agentic trace + answer + citations (includes doc snippets for multi-hop reasoning)

Retries only on clearly transient errors (network, timeout, or LLM/API issues).
"""

from typing import List, Optional, Literal, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import asyncio
from textwrap import shorten
import logging

from langgraph_graph import build_graph

# --- Retry & Rate limiting imports ---
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception,
    RetryError,
)
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# --- FastAPI app ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="EU AI Act Navigator API",
    description="Agentic RAG API for grounded EU AI Act regulatory queries",
    version="0.1.0",
)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Try again later."},
    )

# --- Request / Response Schemas ---
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query about the EU AI Act")
    mode: Literal["final", "full"] = Field(
        "final",
        description="Response mode: final (answer + citations) or full (agentic trace)",
    )

class Citation(BaseModel):
    label: str
    url: str

class HopDoc(BaseModel):
    label: str
    url: str
    similarity: float
    snippet: str

class HopTrace(BaseModel):
    hop_number: int
    query_used: str
    stop_reason: Optional[str] = None
    new_docs: List[HopDoc]

class FinalResponse(BaseModel):
    answer: str
    citations: List[Citation]

class FullResponse(FinalResponse):
    hops: List[HopTrace]

# --- Graph initialization ---
graph = build_graph()

# --- Helper functions ---
def preview(text: str, max_chars: int = 220) -> str:
    if not text:
        return ""
    return shorten(text.replace("\n", " "), width=max_chars, placeholder="...")

def format_citations(sources: List[Dict[str, str]]) -> List[Citation]:
    seen = set()
    citations = []
    for src in sources:
        key = src["url"]
        if key in seen:
            continue
        seen.add(key)
        citations.append(Citation(label=src["label"], url=src["url"]))
    return citations

def format_hops(hops: List[Dict[str, Any]]) -> List[HopTrace]:
    formatted = []
    for hop in hops:
        hop_docs = []
        for doc in hop.get("new_docs", []):
            snippet = preview(getattr(doc, "content", None))
            hop_docs.append(
                HopDoc(
                    label=f"{doc.section_type} {doc.section_title}",
                    url=doc.url,
                    similarity=round(doc.similarity, 4),
                    snippet=snippet,
                )
            )
        formatted.append(
            HopTrace(
                hop_number=hop["hop_number"],
                query_used=hop["query_used"],
                stop_reason=hop.get("stop_reason"),
                new_docs=hop_docs,
            )
        )
    return formatted

# --- Helper: classify transient exceptions ---
def is_transient_error(e: Exception) -> bool:
    """
    Detect likely transient errors based on type and message.
    Only retry for clearly transient issues.
    """
    msg = str(e).lower()

    # Built-in network / timeout issues
    if isinstance(e, (TimeoutError, ConnectionError)):
        logger.warning("Transient: built-in network/timeout error: %s", msg)
        return True

    # LLM / API transient messages
    llm_keywords = ["groq", "rate limit", "timeout", "503", "502"]
    if any(k in msg for k in llm_keywords):
        logger.warning("Transient: LLM/API issue: %s", msg)
        return True

    return False

# --- Graph Invocation with Retry ---
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=6),
    retry=retry_if_exception(is_transient_error),
    reraise=True,
)
async def invoke_graph_with_retry(initial_state: dict) -> dict:
    """
    Invoke the agentic RAG graph with retry on transient errors.
    """
    return await graph.ainvoke(initial_state)

# --- API Endpoint with Rate Limiting ---
@app.post("/rag/query", response_model=FinalResponse | FullResponse)
@limiter.limit("5/minute")
async def query_rag(request: Request, body: QueryRequest):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    initial_state = {"user_query": body.query.strip()}

    try:
        final_state = await invoke_graph_with_retry(initial_state)
    except RetryError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Graph failed after retries: {str(e)}",
        )

    answer = final_state.get("answer")
    sources = final_state.get("sources", [])
    hops = final_state.get("hops", [])

    if not answer:
        raise HTTPException(status_code=500, detail="No answer generated by RAG pipeline")

    citations = format_citations(sources)

    if body.mode == "final":
        return FinalResponse(answer=answer, citations=citations)
    return FullResponse(answer=answer, citations=citations, hops=format_hops(hops))

# --- Local dev entrypoint ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
