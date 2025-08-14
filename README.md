#  Public Project Showcase

Welcome! This repository is a showcase of my data science and machine learning projects, designed to demonstrate applications of analytics, modeling, and data engineering skills across real-world datasets.

Each folder in this repository contains a self-contained project.

---

##  Featured Projects

### [Credit Card Fraud Detection](./Credit_fraud)
- **Objective:** Detect fraudulent transactions in a highly imbalanced dataset.
- **Highlights:** Achieved **F1 score of 0.86** using Random Forest with PCA and threshold tuning. Conducted a comparative analysis of **SMOTE vs. no-SMOTE approaches** across two notebooks to evaluate the impact on model performance.
- **Tech Stack:** Python, scikit-learn, imbalanced-learn (SMOTE), matplotlib

###  [M5 Forecasting – Retail Demand](./walmart)
- **Objective:** Forecast daily Walmart sales using time series models.
- **Highlights:** Built Prophet-based models with performance metrics **R²: 0.85**, **MAPE: 4%**.
- **Dashboard:** Deployed interactive Flask dashboard on [[Render.com](https://public-showcase.onrender.com)] to visualize:
  - Forecast vs actuals
  - Holiday impact by date
  - Holiday impact by event type
- **Tech Stack:** Prophet, pandas, Flask, matplotlib, Plotly, Render.com

###  [Geospatial Data Pipeline – Brazil E-Commerce](./br_ecommerce)
- **Objective:** Build an end-to-end data pipeline to uncover trends.
- **Highlights:** Developed an automated **ELT pipeline** using **Dagster**. Transformed and visualized data into **geospatial heatmaps** 
- **Tech Stack:** SQL, python, Dagster, GeoPandas, pandas, DBT, great expectations, meltano

###  [EDA & Visualization – Mexico COVID-19](./mexico-covid)
- **Objective:** Perform exploratory analysis on COVID-19 cases and resource usage in Mexico.
- **Highlights:** Created insightful queries using visualizations to reveal patterns
- **Tech Stack:** Python, seaborn, matplotlib.

### [LLM-powered Multi-Agent Recommender](./ecommerce_llm_recommender) (Ongoing)
- **Objective:**  Build an explainable and personalized product recommendation system using RAG, FAISS, and LLMs.
- **Highlights:**
Scalable embedding + FAISS vector search pipeline
Modular multi-agent architecture (Retriever, UserProfiler, Explainer)
Explainable recommendations via RAG prompt engineering
Interactive Streamlit UI: [Try it here](https://ecomm-recommender.streamlit.app/)
- **Tech Stack:** Python, FAISS, Sentence-Transformers (MiniLM), Pandas / PyArrow, LangGraph / LangChain (planned), Streamlit, Groq LLM API, Parquet

  Status: Early development stage
