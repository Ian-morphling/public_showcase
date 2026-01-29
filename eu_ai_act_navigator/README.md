# EU AI Act Navigator

An agentic Retrieval-Augmented Generation (RAG) system that performs **planner-driven, multi-hop retrieval** over the EU Artificial Intelligence Act.  
The project is designed to demonstrate **production-oriented agentic search and reasoning**.

---

## Objective

Most RAG systems treat retrieval as a fixed step: embed -> search -> generate.

In real production systems (search, compliance, enterprise AI), retrieval itself is a **reasoning problem**:
- What information is needed?
- Is one retrieval hop sufficient?
- Should the query be reformulated?
- How do we avoid redundant searches?
- How do we guarantee grounded, explainable outputs?

This project implements **planner-driven retrieval**, where an explicit planner agent decides:
- single-hop vs multi-hop retrieval
- query reformulation across hops
- when enough evidence has been gathered

The result is an **agentic search system** that mirrors real-world backend and IR architectures.

**Why the EU AI Act?**

The EU AI Act was chosen as a domain due to its dense cross-referencing structure, where key obligations are distributed across multiple articles and annexes. Answering real compliance questions often requires reasoning across definitions, scope, and obligations, making it a natural fit for planner-driven, multi-hop retrieval.

---

## High-Level Architecture

```text
User Query
   |
   v
+---------------------------+
| Planner Agent             |
| - Decide single/multi-hop |
| - Reformulate queries     |
| - Stop when evidence      |
|   is sufficient           |
+---------------------------+
   |
   v
+---------------------------+
| Retriever Agent           |
| - Search vector DB        |
|   (pgvector)              |
| - Return structured chunks|
+---------------------------+
   |
   v
+---------------------------+
| Explainer Agent           |
| - Synthesize answer       |
| - Strictly grounded       |
| - Generate citations      |
|   (Article / Annex)       |
+---------------------------+
   |
   v
Final Answer (Grounded & Explainable)
```

Each agent has a **strict, non-overlapping responsibility**, enabling observability, testing, and future scaling.

---

## Core Design Principles

### 1. Planner-Driven Retrieval
Retrieval is treated as a decision-making process, not a fixed pipeline.  
The planner determines whether additional hops or query reformulation are required, explicitly avoiding repeated or redundant vector searches.

### 2. Deterministic Agent Responsibilities
Each agent performs a single role:
- **Planner Agent**: reasoning, control flow, and retrieval planning
- **Retriever Agent**: data access only (no synthesis or reasoning)
- **Explainer Agent**: answer synthesis strictly grounded in retrieved evidence

This separation mirrors production service boundaries.

### 3. Strict Grounding and Citation Discipline
The explainer can only reference documents returned by the retriever, using structured labels (e.g., *Article 6*, *Annex III*).  
Hallucinated sources are structurally prevented.

### 4. Production-Oriented Implementation
- Async Python modules
- LangGraph used as a deterministic execution graph
- Designed to be embedded into FastAPI or other backend services
- State explicitly passed and inspected across steps

---

## Retrieval & Data Layer

- Data source:  
  https://artificialintelligenceact.eu/ai-act-explorer/

### Storage & Indexing
- EU AI Act text segmented by Articles, Recitals and Annexes
- Chunked and embedded using sentence-transformers
- Stored in PostgreSQL with **pgvector** (Supabase free tier)
- Metadata preserved for filtering, labeling, and citation

---

## LangGraph Execution Model

The system is implemented as a deterministic, stateful execution graph rather than an ad-hoc agent loop, enabling inspection, testing, and production-style reasoning control.

Key characteristics:
- Explicit state transitions
- Async execution via `graph.ainvoke()`
- Planner decisions and retrieved documents are inspectable
- Easy to add new tools or retrieval strategies

---

## Testing

The project includes focused tests to validate planner behavior and agent control flow.

### Unit Tests (pytest)

- `tests/test_planner_agent.py` validates the **PlannerAgent** in isolation
- External LLM calls (Groq) are mocked using `pytest.monkeypatch`
- Tests cover:
  - LLM-based sufficiency checks
  - Multi-hop stop conditions
  - Document deduplication across hops
  - Max-hop termination
  - Similarity-based early stopping

This demonstrates isolation of external LLM dependencies in agentic systems.

Example (clean pytest run):

```text
tests/test_planner_agent.py::test_check_answer_sufficiency PASSED
tests/test_planner_agent.py::test_plan_next_query_stop_on_stop PASSED
tests/test_planner_agent.py::test_plan_next_query_filters_seen_docs PASSED
tests/test_planner_agent.py::test_plan_next_query_stops_on_max_hops PASSED
tests/test_planner_agent.py::test_plan_next_query_stops_on_low_similarity PASSED
```
Below is a sample run of the pytest script:
File: [test_outputs/pytest_output_clen.txt](test_outputs/pytest_output_clean.txt)
---

## Testing & Introspection

- End-to-end execution tested via `test_langgraph_flow.py`
- Intermediate outputs exposed:
  - Planner decisions
  - Reformulated queries
  - Retrieved documents
- Designed to support future regression testing and retrieval evaluation

### EU AI Act Navigator Example

Below is a sample run of the agentic RAG workflow, showing multi-hop retrieval, explainable final answer, and deduplicated sources.

File: [examples/test_run.txt](examples/test_run.txt)

---

## FastAPI Backend

The project exposes a production-style FastAPI endpoint (/rag/query) that runs an agentic RAG workflow over the EU AI Act.

- final mode: grounded answer + citations

- full mode: multi-hop reasoning trace + answer + citations

All citations are strictly derived from retrieved EU AI Act articles.

The API is designed to be consumed by downstream applications (e.g. web frontends, internal tools, compliance workflows).

[screenshot](screenshots/eu_fastapi_final.png)

---

## Demo Frontend (Streamlit)

A lightweight Streamlit application is included as a **reference frontend** for interacting with the FastAPI backend.

The Streamlit app:
- Calls the FastAPI `/rag/query` endpoint
- Allows users to select between:
  - **Final mode**: concise answer with citations
  - **Explainable mode**: multi-hop retrieval trace with document snippets
- Renders answers, citations, and (optionally) retrieval hops in a readable UI

The frontend is intentionally thin and stateless, serving as a demonstration of how the backend API can be consumed by downstream applications.

### Streamlit Demo (Final Mode)

Below is an example of the Streamlit UI using **final mode**, showing a grounded answer and citations:

[screenshot](screenshots/eu_streamlit_final.png)

> Multi-hop reasoning and retrieval traces are demonstrated separately via the test script output (`test_langgraph_flow.py`) and API `full` mode.

---

## What This Project Demonstrates

This project intentionally focuses on agentic system design.

It demonstrates:
- Agentic search and multi-step reasoning
- Planner-driven query processing
- Production-style RAG architecture
- Clear separation of concerns
- Grounded, explainable LLM outputs
- Backend-first AI engineering mindset
- Clean API contracts consumable by frontend applications

---

## Tech Stack

- **Language**: Python
- **Agent Framework**: LangGraph
- **Embeddings**: sentence-transformers
- **Vector Store**: PostgreSQL + pgvector (Supabase-hosted)
- **Async Runtime**: asyncio
- **Testing**: Pytest (unit-level, isolated agent tests)
- **Backend API**: FastAPI
- **Demo Frontend**: Streamlit
---

## Disclaimer

This project is intended as a **technical showcase** for agentic AI and search system design.  
It is not legal advice and should not be used for regulatory compliance decisions.
