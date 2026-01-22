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

This project explores **planner-driven retrieval**, where an explicit planner agent decides:
- single-hop vs multi-hop retrieval
- query reformulation across hops
- when enough evidence has been gathered

The result is an **agentic search system** that mirrors real-world backend and IR architectures.

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

## Testing & Introspection

- End-to-end execution tested via `test_langgraph_flow.py`
- Intermediate outputs exposed:
  - Planner decisions
  - Reformulated queries
  - Retrieved documents
- Designed to support future regression testing and retrieval evaluation

---

## What This Project Demonstrates

This project intentionally focuses on system design.

It demonstrates:
- Agentic search and multi-step reasoning
- Planner-driven query processing
- Production-style RAG architecture
- Clear separation of concerns
- Grounded, explainable LLM outputs
- Backend-first AI engineering mindset

---

## Tech Stack

- **Language**: Python 3.11
- **Agent Framework**: LangGraph
- **Embeddings**: sentence-transformers
- **Vector Store**: PostgreSQL + pgvector (Supabase)
- **Async Runtime**: asyncio

---

## Disclaimer

This project is intended as a **technical showcase** for agentic AI and search system design.  
It is not legal advice and should not be used for regulatory compliance decisions.
