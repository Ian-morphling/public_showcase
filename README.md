#  Public Project Showcase

Welcome! This repository is a showcase of my data science and machine learning projects, designed to demonstrate applications of analytics, modeling, ai engineering and data engineering skills across real-world datasets.

Each folder in this repository contains a self-contained project.

---

##  Featured Projects

### [LLM-powered Multi-Agent Recommender (Ongoing)](./ecommerce_llm_recommender)
- **Objective:** Personalized, explainable e-commerce product recommendations using RAG, FAISS, LLMs, and multi-agent orchestration.
- **Impact / Highlights:**
  - Scalable embedding + **FAISS vector search** for fast semantic retrieval.
  - **Multi-agent workflows:** Direct RAG & LangGraph orchestration.
  - Explainable recommendations using review metadata (verified purchases, helpful votes, ratings)
  - Interactive [Streamlit demo](https://ecomm-recommender.streamlit.app/)
- **Tech Stack:** Python, FAISS, Sentence-Transformers (MiniLM), LangGraph, Streamlit, Groq LLM API.
- **Status:** Early development
- [detailed readme](./ecommerce_llm_recommender/readme_llm.md)

### [Music Review Summarization LLM (QLoRA)](./Music_Review_Summarization_LLM(QLoRA))
- **Objective:** Fine-tune a large language model for summarizing Amazon Digital Music reviews using QLoRA for memory-efficient instruction-following.
- **Impact / Highlights:**
  - Demonstrates instruction-following fine-tuning on a real-world dataset.
  - Efficient 4-bit QLoRA adapters enable fine-tuning on small GPUs (OPT-1.3B).
  - Terminal-based multi-turn chatbot showcases interactive LLM inference.
  - Comparison of base model vs fine-tuned adapter outputs to highlight performance improvement.
- **Tech Stack:** Python, Hugging Face Transformers, PEFT (QLoRA), PyTorch
- **Status:** Demonstration / Portfolio showcase
- [detailed readme](./Music_Review_Summarization_LLM(QLoRA)/readme.md)

### [Credit Card Fraud Detection](./Credit_fraud)
- **Objective:** Detect fraudulent transactions in a highly imbalanced dataset.
- **Highlights:** Achieved **F1 score of 0.86** using Random Forest with PCA and threshold tuning. Conducted a comparative analysis of **SMOTE vs. no-SMOTE approaches** across two notebooks to evaluate the impact on model performance.
- **Tech Stack:** Python, scikit-learn, imbalanced-learn (SMOTE), matplotlib

###  [M5 Forecasting – Retail Demand](./walmart)
- **Objective:** Forecast daily Walmart sales using time series models.
- **Highlights:** Built Prophet-based models with performance metrics **R²: 0.85**, **MAPE: 4%**.
- **Dashboard:** Deployed interactive Flask dashboard on [[Render.com](https://public-showcase.onrender.com)] to visualize:
  - Forecast vs actuals
  - Holiday impact by date & event type
- **Tech Stack:** Prophet, pandas, Flask, matplotlib, Plotly, Render.com

###  [Geospatial Data Pipeline – Brazil E-Commerce](./br_ecommerce)
- **Objective:** Build an end-to-end data pipeline to uncover trends.
- **Highlights:** Developed an automated **ELT pipeline** using **Dagster**. Transformed and visualized data into **geospatial heatmaps** 
- **Tech Stack:** SQL, python, Dagster, GeoPandas, pandas, DBT, great expectations, meltano

###  [EDA & Visualization – Mexico COVID-19](./mexico-covid)
- **Objective:** Perform exploratory analysis on COVID-19 cases and resource usage in Mexico.
- **Highlights:** Created insightful queries using visualizations to reveal patterns
- **Tech Stack:** Python, seaborn, matplotlib.

## Key Skills Demonstrated Across Projects
- **ML & AI:** Random Forest, PCA, Prophet, RAG, LLM integration, FAISS embeddings
- **Data Engineering:** ELT pipelines, Parquet storage, Dagster
- **Web / Visualization:** Flask, Streamlit, Plotly, interactive dashboards
- **Tools & Libraries:** Python, scikit-learn, pandas, PyArrow, GeoPandas, Sentence-Transformers, LangGraph
- **Deployment & Scaling:** Streamlit, FAISS vector search, multi-agent orchestration
