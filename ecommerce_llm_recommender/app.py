import os
import requests
import streamlit as st
from agents.retriever_agent import RetrieverAgent
from agents.explainer_agent import ExplainerAgent
from dotenv import load_dotenv

# Load environment variables from local .env (for local dev)
load_dotenv()

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_config(key, default=None):
    """Get config from Streamlit secrets, then env vars, then default."""
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

def download_if_not_exists(url: str, local_path: str):
    """Download a remote file only if it does not exist locally."""
    if not os.path.exists(local_path):
        st.info(f"Downloading {os.path.basename(local_path)}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success(f"Downloaded and cached: {local_path}")
        except Exception as e:
            st.error(f"Failed to download {url}: {e}")
            st.stop()

@st.cache_resource(show_spinner=False)
def init_agents():
    INDEX_URL = get_config("INDEX_URL")
    MAPPING_URL = get_config("MAPPING_URL")
    DOCS_DIR = get_config("DOCS_DIR", "outputs")
    GROQ_API_KEY = get_config("GROQ_API_KEY")

    st.write("### Config Debug Info:")
    st.write(f"INDEX_URL source: {'streamlit secrets' if 'INDEX_URL' in st.secrets else 'env or default'}")
    st.write(f"MAPPING_URL source: {'streamlit secrets' if 'MAPPING_URL' in st.secrets else 'env or default'}")
    st.write(f"DOCS_DIR: {DOCS_DIR}")
    st.write(f"GROQ_API_KEY set? {'Yes' if GROQ_API_KEY else 'No'}")

    # Download index if remote
    if INDEX_URL.startswith("http"):
        local_index_path = os.path.join(CACHE_DIR, os.path.basename(INDEX_URL))
        download_if_not_exists(INDEX_URL, local_index_path)
    else:
        local_index_path = INDEX_URL

    # Download mapping if remote
    if MAPPING_URL.startswith("http"):
        local_mapping_path = os.path.join(CACHE_DIR, os.path.basename(MAPPING_URL))
        download_if_not_exists(MAPPING_URL, local_mapping_path)
    else:
        local_mapping_path = MAPPING_URL

    # Initialize retriever agent with local paths
    retriever = RetrieverAgent(local_index_path, local_mapping_path, DOCS_DIR)
    try:
        retriever.initialize()
    except Exception as e:
        st.error(f"Failed to initialize RetrieverAgent: {e}")
        st.stop()

    # Initialize explainer agent
    try:
        explainer = ExplainerAgent(GROQ_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize ExplainerAgent: {e}")
        st.stop()

    return retriever, explainer

# Initialize agents
retriever, explainer = init_agents()

st.title("🛒 E-Commerce Multi-Agent Recommender")

query = st.text_input("Enter your product question or query:")
top_k = st.slider("Number of reviews to retrieve:", 1, 10, 5)

if st.button("Get Recommendations") and query.strip():
    with st.spinner("Retrieving reviews..."):
        try:
            retrieved = retriever.query(query, top_k=top_k)
        except Exception as e:
            st.error(f"Retrieval failed: {e}")
            st.stop()

    st.success(f"Retrieved top {len(retrieved)} reviews:")

    for i, r in enumerate(retrieved):
        st.markdown(f"**Review {i+1} (Similarity: {r.get('similarity', 0):.4f})**")
        if r.get("summary"):
            st.markdown(f"*Summary:* {r['summary']}")
        st.markdown(
            f"Rating: {r.get('rating', 'N/A')} | Verified: {r.get('verified', 'N/A')} | Helpful Votes: {r.get('votes', 'N/A')}"
        )
        st.markdown(f"Date: {r.get('date', 'N/A')} | Reviewer: {r.get('reviewer', 'N/A')}")
        text_preview = r.get("text", "")
        st.write(text_preview[:500] + ("..." if len(text_preview) > 500 else ""))
        st.markdown("---")

    with st.spinner("Generating explanation and answer..."):
        try:
            answer = explainer.generate_answer(query, retrieved)
            st.subheader("AI Explanation & Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Failed to generate answer: {e}")
