import streamlit as st
import requests
from typing import Dict, Any
from uuid import uuid4

API_URL = "http://127.0.0.1:8000/rag/query"

# --- Page configuration ---
st.set_page_config(
    page_title="EU AI Act Navigator",
    page_icon="📘",
    layout="wide",
)

# --- Initialize session state ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # each item: {"query": str, "answer": str}
if "last_result" not in st.session_state:
    st.session_state.last_result = None  # store last API response safely

# --- Helper functions ---
def call_rag_api(query: str, mode: str, thread_id: str = None) -> Dict[str, Any]:
    payload = {"query": query, "mode": mode}
    params = {}
    if thread_id:
        params["thread_id"] = thread_id
    response = requests.post(API_URL, json=payload, params=params, timeout=120)
    response.raise_for_status()
    return response.json()


def render_chat_history():
    """Render chat messages in order (user -> assistant)"""
    if not st.session_state.chat_history:
        return
    st.subheader("Conversation")
    for turn in st.session_state.chat_history:
        st.markdown(f"**You:** {turn['query']}")
        st.markdown(f"**Assistant:** {turn['answer']}")
        st.markdown("---")


def render_citations(citations):
    if not citations:
        return
    st.subheader("Sources")
    for i, c in enumerate(citations, 1):
        st.markdown(f"{i}. **{c['label']}**  \n{c['url']}")


def render_hops(hops):
    if not hops:
        return
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


# --- Sidebar settings ---
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

    # Multi-turn only allowed for final answer mode
    if api_mode == "full":
        multi_turn = False
        st.checkbox(
            "Enable multi-turn context",
            value=False,
            help="Multi-turn context is only available in concise mode.",
            disabled=True,
        )
    else:
        multi_turn = st.checkbox(
            "Enable multi-turn context",
            value=False,
            help="Maintain conversation history within this session for follow-up queries."
        )

    st.markdown("---")
    st.caption("Backend: FastAPI + LangGraph")


# --- App Header ---
st.title("📘 EU AI Act Navigator")
st.markdown(
    """
Agentic Retrieval-Augmented Generation (RAG) system for **grounded, explainable queries**
over the EU Artificial Intelligence Act.

- Default mode returns a concise, cited answer
- Explainable mode exposes multi-hop retrieval and document snippets
- Optional multi-turn context for follow-up questions
"""
)

# --- Chat container ---
chat_container = st.container()

# --- Query input ---
query = st.text_area(
    "Enter your question about the EU AI Act",
    placeholder="e.g. List the Articles defining high-risk AI systems",
    height=100,
)

# --- Handle thread_id for multi-turn ---
if multi_turn:
    if st.session_state.thread_id is None:
        st.session_state.thread_id = str(uuid4())
    thread_id = st.session_state.thread_id
else:
    thread_id = None

# --- Run query ---
if st.button("Run Query", type="primary"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Running agentic RAG pipeline..."):
            try:
                # Call API and store response in session state
                st.session_state.last_result = call_rag_api(query, api_mode, thread_id=thread_id)
                answer = st.session_state.last_result.get("answer", "[No answer returned]")

                # Add to chat history only in final mode or multi-turn allowed
                if api_mode == "final":
                    st.session_state.chat_history.append({
                        "query": query,
                        "answer": answer
                    })
            except Exception as e:
                st.error(f"API error: {e}")

# --- Render chat history and last result ---
with chat_container:
    render_chat_history()
    # Render citations and hops for the last query if available
    if st.session_state.last_result:
        render_citations(st.session_state.last_result.get("citations", []))
        if api_mode == "full":
            render_hops(st.session_state.last_result.get("hops", []))
