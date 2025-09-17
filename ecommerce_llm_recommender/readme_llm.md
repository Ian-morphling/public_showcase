# Multi-Agent LLM Recommender with RAG & Explainability for E-Commerce

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline combined with multi-agent orchestration for personalized and explainable product recommendations based on Amazon Electronics 5-core reviews.

The system is designed for technical scalability, modularity, and explainability, demonstrating LLM integration, vector search, and agentic AI workflows

[Streamlit UI available](https://ecomm-recommender.streamlit.app/)

---

## Project Overview

This project builds an end-to-end recommender system using:

- Cleaned and enriched Amazon Electronics 5-core review data
- MiniLM embeddings stored in a FAISS IVF index for efficient similarity search
- Multi-agent architecture (**RetrieverAgent**, **ExplainerAgent**, **UserProfilerAgent**, **QualityAnalysisAgent**) orchestrated via LangGraph
- Explainable recommendations leveraging review metadata like verified purchase status, helpful votes, sentiment, and review quality scoring
- Optional personalized recommendations via reviewer ID

## Key Features

- Two Operational Workflows:
1. Direct RAG Pipeline: Sequential retrieval + explanation (lightweight, debugging-friendly)
2. LangGraph Orchestration: Multi-agent graph execution with conditional personalization and function/tool routing

- Personalized Recommendations: Tailor results based on reviewer ID, including user stats like total reviews, average rating, verified purchases, and 5-star distribution.

- Quality Scoring: Each review now includes a quality_label and quality_score to indicate reliability.

- Advanced RAG Features: Supports function calling and tool routing via LangGraph agents.

- Explainable Recommendations: Uses review metadata (verified purchases, helpful votes, ratings, sentiment) to produce transparent reasoning

- Scalable Design: Parquet-based storage and chunked embeddings allow processing of large number of reviews

---

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


### LLM Integration & Agents

- **agents/retriever_agent.py**  
  Loads FAISS index & metadata, embeds queries, executes similarity search, and returns detailed enriched review results.
  
- **agents/explainer_agent.py**  
  Builds RAG prompts and calls the Groq LLM to generate final, explainable answers referencing review metadata.

- **agents/userprofile_agent.py**
Manages user-specific data for personalized recommendations, Provides summary statistics: total reviews, average rating, verified purchase count, rating distribution
Enables the recommender to tailor retrieved reviews and AI-generated explanations based on a reviewer’s historical preferences

- **agents/quality_analysis_agent.py**
Evaluates retrieved reviews for reliability and usefulness. Produces quality_label and quality_score for each review to help the system prioritize high-quality content in explanations and recommendations.

- **langgraph_nodes.py**
Defines LangGraph-compatible nodes for each agent (UserProfilerNode, RetrieverNode, RecommenderNode, ExplainerNode, QualityAnalysisNode) and build_graph() to chain them together.
Enables flexible orchestration where each agent’s output feeds into the next node and supports function calling and tool routing.

### Workflows

Workflow A: Direct RAG Pipeline

Runs retrieval + explanation sequentially via scripts/run_rag.py.

Suitable for debugging or lightweight mode

RetrieverAgent -> ExplainerAgent


Workflow B: LangGraph Orchestration

Uses build_graph() from agents/langgraph_nodes.py to orchestrate multiple agents in a graph execution model.

Supports personalized mode with reviewer profile (example ID: A3QVAKVRAH657N)

Run via Streamlit front-end (app.py)

UserProfilerNode -> RetrieverNode -> QualityAnalysisNode -> ExplainerNode

### Orchestration

- **scripts/run_rag.py**  
  Runs retrieval + explanation pipeline end-to-end, displaying retrieved reviews and LLM-generated answers.
  
- **app.py**
Streamlit front-end that lets users run queries with or without reviewerid

### LangGraph Orchestration

the recommender supports two alternative workflows via agents/langgraph_nodes.py:

1. Retrieval -> Explanation (Q&A mode)

User asks a question -> System retrieves relevant reviews -> LLM explains answer.

2. Profile -> Retrieval -> Recommendation + Explanation (Personalized mode) (sample reviewer id: A3QVAKVRAH657N	)

User profile guides retrieval -> Reviews retrieved -> Personalized explanation generated.


## Quick Start Guide

pip install -r requirements.txt

### Workflow A: Direct RAG pipeline
python -m scripts.run_rag --query "I want a durable laptop with long battery"

### Workflow B: LangGraph orchestration [Streamlit UI available](https://ecomm-recommender.streamlit.app/)
streamlit run app.py




