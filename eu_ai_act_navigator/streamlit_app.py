import streamlit as st
import requests
from typing import Dict, Any

API_URL = "http://127.0.0.1:8000/rag/query"

st.set_page_config(
    page_title="EU AI Act Navigator",
    page_icon="📘",
    layout="wide",
)

# Helper functions
def call_rag_api(query: str, mode: str) -> Dict[str, Any]:
    payload = {
        "query": query,
        "mode": mode,
    }

    response = requests.post(API_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def render_citations(citations):
    if not citations:
        return

    st.subheader("Sources")
    for i, c in enumerate(citations, 1):
        st.markdown(f"{i}. **{c['label']}**  \n{c['url']}")


def render_hops(hops):
    st.subheader("Multi-Hop Retrieval Trace")

    for hop in hops:
        with st.expander(f"Hop {hop['hop_number']}"):
            st.markdown(f"**Query used:** {hop['query_used']}")

            if hop.get("stop_reason"):
                st.markdown(f"**Stop reason:** {hop['stop_reason']}")

            docs = hop.get("new_docs", [])
            if not docs:
                st.info("No new documents retrieved in this hop.")
                continue

            for doc in docs:
                st.markdown(f"**{doc['label']}**")
                st.markdown(f"- Similarity: `{doc['similarity']}`")
                st.markdown(f"- URL: {doc['url']}")
                st.markdown(f"> {doc['snippet']}")
                st.markdown("---")


# UI
st.title("📘 EU AI Act Navigator")
st.markdown(
    """
Agentic Retrieval-Augmented Generation (RAG) system for **grounded, explainable queries**
over the EU Artificial Intelligence Act.

- Default mode returns a concise, cited answer
- Explainable mode exposes multi-hop retrieval and document snippets
"""
)

with st.sidebar:
    st.header("Settings")

    mode_label = st.radio(
        "Answer mode",
        options=[
            "Final answer (concise)",
            "Explainable (multi-hop trace)",
        ],
        index=0,
        help="Explainable mode shows retrieval hops and document snippets used to generate the answer.",
    )

    api_mode = "full" if mode_label.startswith("Explainable") else "final"

    st.markdown("---")
    st.caption("Backend: FastAPI + LangGraph")

query = st.text_area(
    "Enter your question about the EU AI Act",
    placeholder="e.g. List the Articles defining high-risk AI systems",
    height=100,
)

run = st.button("Run Query", type="primary")

# Execution
if run:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Running agentic RAG pipeline..."):
            try:
                result = call_rag_api(query, api_mode)
            except Exception as e:
                st.error(f"API error: {e}")
            else:
                st.subheader("Answer")
                st.markdown(result["answer"])

                render_citations(result.get("citations", []))

                if api_mode == "full":
                    render_hops(result.get("hops", []))
