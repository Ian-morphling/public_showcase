import os
import requests
import streamlit as st
from agents.retriever_agent import RetrieverAgent
from agents.explainer_agent import ExplainerAgent

# Try to load .env locally, but ignore errors if not present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_secret(key, default=None):
    """
    Return secret/config value from Streamlit secrets or environment variables.
    Priority: Streamlit secrets > os.environ > default
    """
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

def download_if_not_exists(url: str, local_path: str):
    if not os.path.exists(local_path):
        st.info(f"Downloading {os.path.basename(local_path)}...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

@st.cache_resource(show_spinner=False)
def init_agents():
    INDEX_URL = get_secret("INDEX_URL")
    MAPPING_URL = get_secret("MAPPING_URL")
    GROQ_API_KEY = get_secret("GROQ_API_KEY")
    DOCS_DIR = get_secret("DOCS_DIR", "outputs")

    if not INDEX_URL or not MAPPING_URL or not GROQ_API_KEY:
        st.error("Missing essential configuration! Please check your environment variables or Streamlit secrets.")
        st.stop()

    local_index_path = os.path.join(CACHE_DIR, "product.index")
    local_mapping_path = os.path.join(CACHE_DIR, "id_to_filename.pkl")

    download_if_not_exists(INDEX_URL, local_index_path)
    download_if_not_exists(MAPPING_URL, local_mapping_path)

    retriever = RetrieverAgent(local_index_path, local_mapping_path, DOCS_DIR)
    retriever.initialize()

    explainer = ExplainerAgent(GROQ_API_KEY)
    return retriever, explainer

retriever, explainer = init_agents()

st.title("E-Commerce Multi-Agent Recommender")

query = st.text_input("Enter your product question or query:", "")

top_k = st.slider("Number of reviews to retrieve:", 1, 10, 5)

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
