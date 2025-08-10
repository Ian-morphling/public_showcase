import os
import requests
import streamlit as st
from agents.retriever_agent import RetrieverAgent
from agents.explainer_agent import ExplainerAgent

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Detect if running on Streamlit Cloud from env var (set in Streamlit secrets or env)
IS_STREAMLIT_CLOUD = os.getenv("IS_STREAMLIT_CLOUD", "false").lower() == "true"

def is_remote_url(path_or_url: str) -> bool:
    return str(path_or_url).startswith("http://") or str(path_or_url).startswith("https://")

def get_secret(key, default=None):
    """Priority: Streamlit secrets > os.environ > default"""
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

def download_if_not_exists(url: str, local_path: str):
    """Download file from remote URL if not exists locally."""
    if is_remote_url(url):
        if not os.path.exists(local_path):
            try:
                st.info(f"Downloading {os.path.basename(local_path)} from {url}...")
                r = requests.get(url, stream=True)
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success(f"Downloaded and cached: {local_path}")
            except Exception as e:
                st.error(f"Failed to download {url}: {e}")
                st.stop()
    else:
        # Local file path assumed to exist
        if not os.path.exists(url):
            st.error(f"Local file not found: {url}")
            st.stop()

@st.cache_resource(show_spinner=False)
def init_agents():
    if IS_STREAMLIT_CLOUD:
        INDEX_URL = get_secret("INDEX_SMALL_URL")
        MAPPING_URL = get_secret("MAPPING_SMALL_URL")
        DOCS_DIR = get_secret("DOCS_DIR", "outputs")
    else:
        INDEX_URL = get_secret("INDEX_URL")
        MAPPING_URL = get_secret("MAPPING_URL")
        DOCS_DIR = get_secret("DOCS_DIR", "outputs")

    GROQ_API_KEY = get_secret("GROQ_API_KEY")

    # Debug info
    st.write("### Config Debug Info:")
    st.write(f"- Running on Streamlit Cloud? {'Yes' if IS_STREAMLIT_CLOUD else 'No'}")
    st.write(f"- INDEX_URL set? {'Yes' if INDEX_URL else 'No'}")
    st.write(f"- MAPPING_URL set? {'Yes' if MAPPING_URL else 'No'}")
    st.write(f"- GROQ_API_KEY set? {'Yes' if GROQ_API_KEY else 'No'}")
    st.write(f"- DOCS_DIR: {DOCS_DIR}")

    if not INDEX_URL or not MAPPING_URL or not GROQ_API_KEY:
        st.error("Missing essential configuration! Please set secrets or environment variables.")
        st.stop()

    local_index_path = os.path.join(CACHE_DIR, os.path.basename(INDEX_URL))
    local_mapping_path = os.path.join(CACHE_DIR, os.path.basename(MAPPING_URL))

    download_if_not_exists(INDEX_URL, local_index_path)
    download_if_not_exists(MAPPING_URL, local_mapping_path)

    try:
        retriever = RetrieverAgent(local_index_path, local_mapping_path, DOCS_DIR)
        retriever.initialize()
    except Exception as e:
        st.error(f"Failed to initialize RetrieverAgent: {e}")
        st.stop()

    try:
        explainer = ExplainerAgent(GROQ_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize ExplainerAgent: {e}")
        st.stop()

    return retriever, explainer

retriever, explainer = init_agents()

st.title("🛒 E-Commerce Multi-Agent Recommender")

query = st.text_input("Enter your product question or query:")
top_k = st.slider("Number of reviews to retrieve:", 1, 10, 5)

if st.button("Get Recommendations") and query:
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
        st.write(text_preview[:500] + ("---" if len(text_preview) > 500 else ""))
        st.markdown("---")

    with st.spinner("Generating explanation and answer..."):
        try:
            answer = explainer.generate_answer(query, retrieved)
            st.subheader("AI Explanation & Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Failed to generate answer: {e}")
