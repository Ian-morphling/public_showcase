import os
import threading
import requests
import streamlit as st

from agents.retriever_agent import RetrieverAgent
from agents.explainer_agent import ExplainerAgent

# Try loading local .env for local dev
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Utility functions ---
def get_secret(key, default=None):
    """Priority: Streamlit secrets > os.environ > default"""
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

def download_if_not_exists(url: str, local_path: str):
    """Download file from URL if it doesn't exist locally (no st.* calls here)."""
    if not os.path.exists(local_path):
        try:
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            raise RuntimeError(f"Failed to download {url}: {e}")

# --- Background init ---
def init_agents_bg():
    """Initialize agents in a background thread without st.* calls."""
    try:
        INDEX_URL = get_secret("INDEX_URL")
        MAPPING_URL = get_secret("MAPPING_URL")
        GROQ_API_KEY = get_secret("GROQ_API_KEY")
        DOCS_DIR = get_secret("DOCS_DIR", "outputs")

        if not INDEX_URL or not MAPPING_URL or not GROQ_API_KEY:
            raise RuntimeError(
                "Missing essential configuration! "
                "Set in `.env` (local) or Streamlit Secrets (Cloud)."
            )

        local_index_path = os.path.join(CACHE_DIR, "product.index")
        local_mapping_path = os.path.join(CACHE_DIR, "id_to_filename.pkl")

        download_if_not_exists(INDEX_URL, local_index_path)
        download_if_not_exists(MAPPING_URL, local_mapping_path)

        retriever = RetrieverAgent(local_index_path, local_mapping_path, DOCS_DIR)
        retriever.initialize()

        explainer = ExplainerAgent(GROQ_API_KEY)

        st.session_state["agents"] = (retriever, explainer)
        st.session_state["agents_ready"] = True
    except Exception as e:
        st.session_state["agents_error"] = str(e)

# --- UI setup ---
st.set_page_config(page_title="E-Commerce Multi-Agent Recommender", layout="wide")
st.title("🛒 E-Commerce Multi-Agent Recommender")

# Init session state
if "agents_ready" not in st.session_state:
    st.session_state["agents_ready"] = False
    st.session_state["agents_error"] = None
    st.session_state["agents"] = None

    # Start initialization in background
    threading.Thread(target=init_agents_bg, name="init_agents_bg", daemon=True).start()

# --- Show init status ---
if st.session_state["agents_error"]:
    st.error(f" Agent initialization failed: {st.session_state['agents_error']}")
    st.stop()

if not st.session_state["agents_ready"]:
    st.warning("⏳ Loading agents in the background... Please wait.")
    st.stop()

# --- Main app logic ---
retriever, explainer = st.session_state["agents"]

query = st.text_input("Enter your product question or query:")
top_k = st.slider("Number of reviews to retrieve:", 1, 10, 5)

if st.button("Get Recommendations") and query:
    with st.spinner("🔍 Retrieving reviews..."):
        try:
            retrieved = retriever.query(query, top_k=top_k)
        except Exception as e:
            st.error(f" Retrieval failed: {e}")
            st.stop()

    st.success(f" Retrieved top {len(retrieved)} reviews:")
    for i, r in enumerate(retrieved):
        st.markdown(f"**Review {i+1} (Similarity: {r.get('similarity', 0):.4f})**")
        if r.get("summary"):
            st.markdown(f"*Summary:* {r['summary']}")
        st.markdown(
            f"Rating: {r.get('rating', 'N/A')} | "
            f"Verified: {r.get('verified', 'N/A')} | "
            f"Helpful Votes: {r.get('votes', 'N/A')}"
        )
        st.markdown(
            f"Date: {r.get('date', 'N/A')} | Reviewer: {r.get('reviewer', 'N/A')}"
        )
        text_preview = r.get("text", "")
        st.write(text_preview[:500] + ("..." if len(text_preview) > 500 else ""))
        st.markdown("---")

    with st.spinner(" Generating explanation and answer..."):
        try:
            answer = explainer.generate_answer(query, retrieved)
            st.subheader("AI Explanation & Answer")
            st.write(answer)
        except Exception as e:
            st.error(f" Failed to generate answer: {e}")
