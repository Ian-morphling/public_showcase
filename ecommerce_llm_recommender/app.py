import os
import requests
import streamlit as st
from agents.retriever_agent import RetrieverAgent
from agents.explainer_agent import ExplainerAgent

# Try loading local .env for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_secret(key, default=None):
    """Return config from Streamlit secrets, env variables, or default."""
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

def download_if_not_exists(url: str, local_path: str):
    """Download file from URL if not already present."""
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

@st.cache_resource(show_spinner=False)
def init_agents():
    INDEX_URL = get_secret("INDEX_URL")
    MAPPING_URL = get_secret("MAPPING_URL")
    GROQ_API_KEY = get_secret("GROQ_API_KEY")
    DOCS_DIR = get_secret("DOCS_DIR", "outputs")

    # Debug info for troubleshooting
    st.write("### Config Debug Info:")
    st.write(f"INDEX_URL: {INDEX_URL}")
    st.write(f"MAPPING_URL: {MAPPING_URL}")
    st.write(f"GROQ_API_KEY set? {'Yes' if GROQ_API_KEY else 'No'}")
    st.write(f"DOCS_DIR: {DOCS_DIR}")

    if not INDEX_URL or not MAPPING_URL or not GROQ_API_KEY:
        st.error("Missing essential configuration! Please set INDEX_URL, MAPPING_URL, and GROQ_API_KEY.")
        st.stop()

    local_index_path = os.path.join(CACHE_DIR, os.path.basename(INDEX_URL))
    local_mapping_path = os.path.join(CACHE_DIR, os.path.basename(MAPPING_URL))

    # Only download if INDEX_URL / MAPPING_URL is a remote URL (starts with http)
    if INDEX_URL.startswith("http"):
        download_if_not_exists(INDEX_URL, local_index_path)
    else:
        # Assume local path, just use as is
        local_index_path = INDEX_URL

    if MAPPING_URL.startswith("http"):
        download_if_not_exists(MAPPING_URL, local_mapping_path)
    else:
        local_mapping_path = MAPPING_URL

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

# Initialize agents
retriever, explainer = init_agents()

# UI
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
        st.write(text_preview[:500] + ("..." if len(text_preview) > 500 else ""))
        st.markdown("---")

    with st.spinner("Generating explanation and answer..."):
        try:
            answer = explainer.generate_answer(query, retrieved)
            st.subheader("AI Explanation & Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Failed to generate answer: {e}")
