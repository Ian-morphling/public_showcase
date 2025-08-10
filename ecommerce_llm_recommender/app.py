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
    # First check env vars (including those loaded from .env)
    val = os.getenv(key)
    if val:
        st.write(f"Using env var for {key}: {val}")
        return val
    # Then check Streamlit secrets (used on Streamlit Cloud)
    if key in st.secrets:
        st.write(f"Using Streamlit secret for {key}: {st.secrets[key]}")
        return st.secrets[key]
    # Else use default fallback
    st.write(f"Using default for {key}: {default}")
    return default

def is_local_file(path):
    return path and os.path.isfile(path)

def download_if_not_exists(url: str, local_path: str):
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
    # Get raw values from env or secrets
    raw_index_url = get_secret("INDEX_URL")
    raw_mapping_url = get_secret("MAPPING_URL")
    groq_api_key = get_secret("GROQ_API_KEY")
    docs_dir = get_secret("DOCS_DIR", "outputs")

    # If the path exists locally, use it directly; else treat as URL and download
    if is_local_file(raw_index_url):
        local_index_path = raw_index_url
        st.write(f"Using local index file: {local_index_path}")
    else:
        local_index_path = os.path.join(CACHE_DIR, os.path.basename(raw_index_url))
        download_if_not_exists(raw_index_url, local_index_path)

    if is_local_file(raw_mapping_url):
        local_mapping_path = raw_mapping_url
        st.write(f"Using local mapping file: {local_mapping_path}")
    else:
        local_mapping_path = os.path.join(CACHE_DIR, os.path.basename(raw_mapping_url))
        download_if_not_exists(raw_mapping_url, local_mapping_path)

    if not groq_api_key:
        st.error("Missing GROQ_API_KEY in environment or secrets.")
        st.stop()

    try:
        retriever = RetrieverAgent(local_index_path, local_mapping_path, docs_dir)
        retriever.initialize()
    except Exception as e:
        st.error(f"Failed to initialize RetrieverAgent: {e}")
        st.stop()

    try:
        explainer = ExplainerAgent(groq_api_key)
    except Exception as e:
        st.error(f"Failed to initialize ExplainerAgent: {e}")
        st.stop()

    return retriever, explainer

# Initialize agents once (cached)
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
        st.write(text_preview[:500] + ("..." if len(text_preview) > 500 else ""))
        st.markdown("---")

    with st.spinner("Generating explanation and answer..."):
        try:
            answer = explainer.generate_answer(query, retrieved)
            st.subheader("AI Explanation & Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Failed to generate answer: {e}")