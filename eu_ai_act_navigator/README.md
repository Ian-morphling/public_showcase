# EU AI Act Navigator

Agentic Retrieval-Augmented Generation (RAG) system built with LangGraph.
Designed for regulatory reasoning over the EU AI Act using planner-driven
multi-hop retrieval and structured search steps, with an emphasis on
explainable responses.

Current status:
- Planner → Retriever agent flow implemented
- pgvector (Supabase/Postgres) used for production-style vector search
- Planner-driven retrieval behavior and grounding prioritized ahead of UI or API layers

Tech stack:
LangGraph, pgvector (Postgres/Supabase)
