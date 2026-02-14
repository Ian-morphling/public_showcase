# backend/api.py
"""
FastAPI backend for EU AI Act Navigator

Exposes a single RAG endpoint with:
- final mode: answer + citations
- full mode: agentic trace + answer + citations (includes doc snippets for multi-hop reasoning)

All citations are strictly derived from retrieved RAG documents.
"""

from typing import List, Optional, Literal, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import asyncio
from textwrap import shorten

from langgraph_graph import build_graph

# --- Retry & Rate limiting imports ---
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
    RetryError,
)
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse

# --- Custom Exception ---
class TransientGraphError(Exception):
    """Custom exception for retrying transient graph failures."""

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
    """Short preview snippet for a document."""
    if not text:
        return ""
    return shorten(text.replace("\n", " "), width=max_chars, placeholder="...")

def format_citations(sources: List[Dict[str, str]]) -> List[Citation]:
    """Convert internal sources into API-safe citation objects."""
    seen = set()
    citations = []

    for src in sources:
        key = src["url"]
        if key in seen:
            continue
        seen.add(key)

        citations.append(
            Citation(
                label=src["label"],
                url=src["url"],
            )
        )

    return citations

def format_hops(hops: List[Dict[str, Any]]) -> List[HopTrace]:
    """
    Convert internal hop trace into frontend-safe structure,
    including preview snippets for each document.
    """
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

# --- Graph Invocation with Retry ---
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=6),
    retry=retry_if_exception_type(TransientGraphError),
    reraise=True,
)
async def invoke_graph_with_retry(initial_state: dict) -> dict:
    """
    Invoke the agentic RAG graph with retry on transient errors.
    Raises RetryError if all attempts fail.
    """
    try:
        return await graph.ainvoke(initial_state)
    except Exception as e:
        raise TransientGraphError(str(e))

# --- API Endpoint with Rate Limiting ---
@app.post("/rag/query", response_model=FinalResponse | FullResponse)
@limiter.limit("5/minute")  # rate limiting per IP
async def query_rag(request: Request, body: QueryRequest):
    """
    Execute agentic RAG over the EU AI Act.
    Rate-limited and retried on transient graph errors.
    """
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
        raise HTTPException(
            status_code=500,
            detail="No answer generated by RAG pipeline",
        )

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
