# Multi-Agent LLM Recommender with RAG & Explainability for E-Commerce

A showcase project demonstrating a **Retrieval-Augmented Generation (RAG)** pipeline combined with multi-agent orchestration for personalized, explainable product recommendations based on Amazon Electronics 5-core reviews.

---

## Project Overview

This project builds an end-to-end recommender system using:

- Cleaned and enriched Amazon Electronics 5-core review data
- MiniLM embeddings stored in a FAISS IVF index for efficient similarity search
- Multi-agent architecture (**RetrieverAgent**, **ExplainerAgent**, **UserProfilerAgent**, **RecommenderAgent**) orchestrated via LangChain/LangGraph (planned)
- Explainable recommendations leveraging review metadata like verified purchase status, helpful votes, and sentiment
- Streamlit UI available for local and in-person demonstrations; public deployment planned.

The core idea:  
Use a user query + profile to retrieve relevant reviews, then generate concise, contextual answers with explicit rationale referencing trustworthy reviews.

---

## Code Modules

### Data & Embedding Pipeline

- **build_index.py**  
  Builds and saves the FAISS IVF index from chunked `.npy` embedding files and maps vector IDs to parquet chunk files.

- **recommender/data/embeddings.py**  
  Generates normalized embeddings from cleaned review text in parquet files, chunked for scalability.

- **recommender/data/loader.py**  
  Loads raw Amazon Electronics JSON reviews, validates, and writes cleaned batches to parquet files.

- **recommender/data/preprocess.py**  
  Enriches cleaned reviews with metadata, vote bins, sentiment; creates user profile JSON for personalization.

### Retrieval & Querying

- **recommender/retrieval/faiss_index.py**  
  Utilities to load FAISS index, map vector IDs to parquet chunks, cache loaded chunks, and handle index-to-row mapping.

- **recommender/retrieval/query_engine.py**  
  RetrieverAgent implementation: embeds queries, performs similarity search on FAISS index, returns enriched review results.

### LLM Integration & Agents

- **llm/groq_api.py**  
  Simple wrapper class to call the Groq LLM API with chat completions using your prompt and model parameters.

- **llm/prompt_builder.py**  
  Builds structured RAG prompts combining retrieved reviews and user query for input to the LLM.

- **agents/explainer_agent.py**  
  Builds RAG prompts and calls the Groq LLM to generate final, explainable answers referencing review metadata.

- **agents/retriever_agent.py**  
  Loads FAISS index & metadata, embeds queries, executes similarity search, and returns detailed enriched review results.

### Orchestration

- **scripts/run_rag.py**  
  Runs the full retrieval + explanation pipeline end-to-end, displaying retrieved reviews and LLM-generated answers.
