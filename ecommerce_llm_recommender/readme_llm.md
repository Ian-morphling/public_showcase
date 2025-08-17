# Multi-Agent LLM Recommender with RAG & Explainability for E-Commerce

A showcase project demonstrating a **Retrieval-Augmented Generation (RAG)** pipeline combined with multi-agent orchestration for personalized, explainable product recommendations based on Amazon Electronics 5-core reviews.

---

## Project Overview

This project builds an end-to-end recommender system using:

- Cleaned and enriched Amazon Electronics 5-core review data
- MiniLM embeddings stored in a FAISS IVF index for efficient similarity search
- Multi-agent architecture (**RetrieverAgent**, **ExplainerAgent**, **UserProfilerAgent**, **RecommenderAgent**) orchestrated via LangGraph
- Explainable recommendations leveraging review metadata like verified purchase status, helpful votes, and sentiment
- [Streamlit UI available](https://ecomm-recommender.streamlit.app/)

The core idea:  
Use a user query + profile to retrieve relevant reviews, then generate concise, contextual answers with explicit rationale referencing trustworthy reviews.

This project now supports two alternative workflows:

Direct RAG pipeline (RetrieverAgent -> ExplainerAgent -> output)

LangGraph Orchestration: Multi-agent graph execution using build_graph in agents/langgraph_nodes.py

---
### Future Improvements / Experimentation

- Implemented LangGraph orchestration with multi-agent nodes

- Originally explored OpenAI’s E5 embeddings for multilingual search (e-commerce in Asia).
Due to GPU constraints, the current demo uses all-MiniLM-L6-v2 for stable performance.
The E5 experiment shows readiness to scale to richer semantic retrieval in production.

## Code Modules

### Data & Embedding Pipeline

- **recommender/data/loader.py**  
  Loads raw Amazon Electronics JSON reviews, validates, and writes cleaned batches to parquet files.

- **recommender/data/preprocess.py**  
  Enriches cleaned reviews with metadata, vote bins, sentiment; creates user profile parquets for personalization.
  Data is stored in multiple parquet files to enable scalable processing of large review datasets.
  
- **recommender/data/embeddings.py**  
  Generates normalized embeddings from cleaned review text in parquet files, chunked for scalability.

- **scripts/build_index.py**  
  Builds and saves the FAISS IVF index from chunked `.npy` embedding files and maps vector IDs to parquet chunk files.

### Retrieval & Querying

- **recommender/retrieval/faiss_index.py**  
  Utilities to load FAISS index, map vector IDs to parquet chunks, cache loaded chunks, and handle index-to-row mapping.

- **recommender/retrieval/query_engine.py**  
  RetrieverAgent implementation: embeds queries, performs similarity search on FAISS index, returns enriched review results.

### LLM Integration & Agents

- **agents/explainer_agent.py**  
  Builds RAG prompts and calls the Groq LLM to generate final, explainable answers referencing review metadata.

- **agents/retriever_agent.py**  
  Loads FAISS index & metadata, embeds queries, executes similarity search, and returns detailed enriched review results.


- **agents/userprofile_agent.py**
Manages user-specific data for personalized recommendations, Provides summary statistics: total reviews, average rating, verified purchase count, rating distribution
Enables the recommender to tailor retrieved reviews and AI-generated explanations based on a reviewer’s historical preferences

- **agents/langgraph_nodes.py**
Defines LangGraph-compatible nodes for each agent (UserProfileNode, RetrieverNode, RecommenderNode, ExplainerNode) and build_graph() to chain them together.
Enables flexible, multi-step orchestration where each agent’s output feeds into the next node.

###Workflows

Workflow A: Direct RAG Pipeline

Runs retrieval + explanation sequentially via scripts/run_rag.py.

Suitable for debugging or lightweight mode

Flow: RetrieverAgent -> ExplainerAgent

Workflow B: LangGraph Orchestration

Uses build_graph() from agents/langgraph_nodes.py to orchestrate multiple agents in a graph execution model.

Agents act as nodes (UserProfileNode, RetrieverNode, RecommenderNode, ExplainerNode).

Provides potential for future expansion in additional reasoning, or personalization nodes.

Invoked via app.py (Streamlit demo)



###Orchestration

- **scripts/run_rag.py**  
  Runs the full retrieval + explanation pipeline end-to-end, displaying retrieved reviews and LLM-generated answers.
  
- **app.py**
Streamlit front-end that lets users run queries with or without reviewerid

Quick Start Guide
### Workflow A: Direct RAG pipeline
python scripts/run_rag.py --query "Best headphones under $100"

### Workflow B: LangGraph orchestration [Streamlit UI available](https://ecomm-recommender.streamlit.app/)
streamlit run app.py


###LangGraph Orchestration

With today’s improvements, the recommender supports two alternative workflows via agents/langgraph_nodes.py:

1. Retrieval -> Explanation (Q&A mode)

User asks a question -> System retrieves relevant reviews -> LLM explains answer.

2. Profile -> Retrieval -> Recommendation + Explanation (Personalized mode)

User profile guides retrieval -> Reviews retrieved -> Personalized explanation generated.
