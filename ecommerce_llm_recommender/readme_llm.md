# Multi-Agent LLM Recommender with RAG & Explainability for E-Commerce

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline combined with multi-agent orchestration for personalized and explainable product recommendations based on Amazon Electronics 5-core reviews.

The system is designed for modularity, scalability, and explainability, demonstrating LLM integration, vector search, and agentic AI workflows

[Streamlit UI available](https://ecomm-recommender.streamlit.app/)

---

## Project Overview

This project builds an end-to-end recommender system using:

- Cleaned and enriched Amazon Electronics 5-core review data
- MiniLM embeddings stored in a FAISS IVF index for efficient similarity search
- Multi-agent architecture (**RetrieverAgent**, **ExplainerAgent**, **UserProfilerAgent**, **QualityAnalysisAgent**, **JudgeAgent**) orchestrated via LangGraph
- Explainable recommendations leveraging review metadata like verified purchase status, helpful votes, sentiment, and review quality scoring
- Optional personalized recommendations via reviewer ID

## Key Features

- Two Operational Workflows:
1. Direct RAG Pipeline: Sequential retrieval + explanation (lightweight, debugging-friendly)
2. LangGraph Orchestration: Multi-agent graph execution with conditional personalization and function/tool routing

- Offline Evaluation Workflow (CLI): Evaluate AI explanations with LLM as a Judge and save JSON results (see Workflow C below)

- Personalized Recommendations: Tailor results based on reviewer ID, including user stats like total reviews, average rating, verified purchases, and 5-star distribution.

- Quality Scoring: Each review now includes a quality_label and quality_score to indicate reliability.

- Advanced RAG Features: Supports function calling and tool routing via LangGraph agents.

- Explainable Recommendations: Transparent reasoning using review metadata, highlighting pros, cons, and evidence from reviews

- Scalable Design: Parquet-based storage and chunked embeddings allow processing of large datasets

---

## Code Modules

### Data & Embedding Pipeline

- **recommender/data/loader.py**  
  Loads raw Amazon Electronics JSON reviews, validates, and writes cleaned batches to parquet files.

- **recommender/data/preprocess.py**  
  Enriches reviews with metadata, vote bins, sentiment, and user profiles
  
- **recommender/data/embeddings.py**  
  Generates normalized embeddings from review text, chunked for scalability

- **scripts/build_index.py**  
  Builds and saves the FAISS IVF index from chunked `.npy` embedding files and maps vector IDs to parquet chunks.


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

- **agents/judge_agent.py**
Evaluates AI explanations for Relevance, Groundedness, and Balance

- **langgraph_nodes.py**
Defines LangGraph-compatible nodes for each agent and build_graph() to chain them together.
Enables flexible orchestration where each agent’s output feeds into the next node and supports function calling and tool routing.

## Workflows

### Workflow A: Direct RAG Pipeline

Runs retrieval + explanation sequentially via scripts/run_rag.py.

Suitable for debugging or lightweight mode

RetrieverAgent -> ExplainerAgent

Run command:

python -m scripts.run_rag --query "I want a durable laptop with long battery"

### Workflow B: LangGraph Orchestration [Streamlit UI](https://ecomm-recommender.streamlit.app/)

Uses build_graph() from agents/langgraph_nodes.py to orchestrate multiple agents in a graph execution model.

Supports personalized mode with reviewer profile (example ID: A3QVAKVRAH657N)

UserProfilerNode -> RetrieverNode -> QualityAnalysisNode -> ExplainerNode

Run command:

```bash
streamlit run app.py
```

Run via Streamlit front-end (app.py)

### Workflow C: Offline LLM Judge Evaluation (CLI)

Evaluate AI explanations automatically without the Streamlit UI using JudgeAgent.

Run command:

```bash
python -m scripts.offline_judge_runner
```

Outputs: JSON files with judge scores are saved under judge_outputs/. Example files already included:

offline_judge_1_20250929-005257.json

offline_judge_2_20250929-005259.json

Purpose: This workflow allows assessment of explanation quality (Relevance, Groundedness, Balance) offline.

## Quick Start Guide

### 1. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Workflow A: Direct RAG pipeline
```bash
python -m scripts.run_rag --query "I want a durable laptop with long battery"
```

### 3. Workflow B: LangGraph orchestration [Streamlit UI available](https://ecomm-recommender.streamlit.app/)
```bash
streamlit run app.py
```

### 4. Workflow C: Offline Judge (CLI)
```bash
python -m scripts.offline_judge_runner
```



