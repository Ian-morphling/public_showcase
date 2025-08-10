import os
import streamlit as st
from dotenv import load_dotenv
from agents.retriever_agent import RetrieverAgent
from agents.explainer_agent import ExplainerAgent

load_dotenv()  # loads .env from current working directory

def get_env_var(key: str, default=None):
    is_cloud = os.getenv("IS_STREAMLIT_CLOUD", "false").lower() == "true"
    if is_cloud:
        # For cloud, use secrets or environment (URLs)
        return st.secrets.get(key, os.getenv(key, default))
    else:
        # Locally, prefer *_PATH over *_URL for files
        if key == "INDEX_URL":
            return os.getenv("INDEX_PATH", default)
        if key == "MAPPING_URL":
            return os.getenv("MAPPING_PATH", default)
        return os.getenv(key, default)

@st.cache_resource(show_spinner=False)
def init_agents():
    INDEX_PATH = get_env_var("INDEX_URL")
    MAPPING_PATH = get_env_var("MAPPING_URL")
    DOCS_DIR = get_env_var("DOCS_DIR", "outputs")
    GROQ_API_KEY = get_env_var("GROQ_API_KEY")

    st.write(f"Using INDEX_PATH: {INDEX_PATH}")
    st.write(f"Using MAPPING_PATH: {MAPPING_PATH}")
    st.write(f"Using DOCS_DIR: {DOCS_DIR}")
    st.write(f"GROQ_API_KEY set? {'Yes' if GROQ_API_KEY else 'No'}")

    retriever = RetrieverAgent(INDEX_PATH, MAPPING_PATH, DOCS_DIR)
    retriever.initialize()

    explainer = ExplainerAgent(GROQ_API_KEY)
    return retriever, explainer

retriever, explainer = init_agents()

st.title("E-Commerce Multi-Agent Recommender")

query = st.text_input("Enter your product question or query:", "")

top_k = st.slider("Number of reviews to retrieve:", min_value=1, max_value=10, value=5)

if st.button("Get Recommendations") and query:
    with st.spinner("Retrieving reviews..."):
        retrieved = retriever.query(query, top_k=top_k)
    st.write(f"Retrieved top {len(retrieved)} reviews:")

    for i, r in enumerate(retrieved):
        st.markdown(f"**Review {i+1} (Similarity: {r.get('similarity', 0):.4f})**")
        if r.get("summary"):
            st.markdown(f"*Summary:* {r['summary']}")
        st.markdown(
            f"Rating: {r.get('rating', 'N/A')} | Verified: {r.get('verified', 'N/A')} | Helpful Votes: {r.get('votes', 'N/A')}"
        )
        st.markdown(f"Date: {r.get('date', 'N/A')} | Reviewer: {r.get('reviewer', 'N/A')}")
        st.write(r.get("text", "")[:500] + ("..." if len(r.get("text", "")) > 500 else ""))
        st.markdown("---")

    with st.spinner("Generating explanation and answer..."):
        answer = explainer.generate_answer(query, retrieved)
    st.subheader("AI Explanation & Answer")
    st.write(answer)