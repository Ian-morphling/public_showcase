import streamlit as st
import os
import requests
from dotenv import load_dotenv
from agents.retriever_agent import RetrieverAgent
from agents.explainer_agent import ExplainerAgent

# Load .env variables
load_dotenv()

# Local cache directory
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def download_file(url: str, local_path: str):
    """Download a file from a URL to local_path if not already cached."""
    if not os.path.exists(local_path):
        st.info(f"Downloading {os.path.basename(local_path)} from Hugging Face... This may take a minute.")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.success(f"Downloaded and cached: {local_path}")
    else:
        # Debug message only in local run
        if os.getenv("STREAMLIT_SERVER_RUNNING") != "true":
            print(f"Using cached file: {local_path}")


@st.cache_resource(show_spinner=True)
def init_agents():
    # Hugging Face direct download URLs (must be set in .env)
    INDEX_URL = os.getenv("INDEX_URL")
    MAPPING_URL = os.getenv("MAPPING_URL")
    DOCS_DIR = os.getenv("DOCS_DIR", "outputs")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    if not INDEX_URL or not MAPPING_URL:
        raise ValueError("INDEX_URL and MAPPING_URL must be set in your .env file")

    index_path = os.path.join(CACHE_DIR, "product.index")
    mapping_path = os.path.join(CACHE_DIR, "id_to_filename.pkl")

    # Download from Hugging Face if not cached
    download_file(INDEX_URL, index_path)
    download_file(MAPPING_URL, mapping_path)

    retriever = RetrieverAgent(index_path, mapping_path, DOCS_DIR)
    retriever.initialize()

    explainer = ExplainerAgent(GROQ_API_KEY)
    return retriever, explainer


# Initialize agents once
retriever, explainer = init_agents()

# ---- UI ----
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
        st.write(r.get("text", "")[:500] + ("---" if len(r.get("text", "")) > 500 else ""))
        st.markdown("---")

    with st.spinner("Generating explanation and answer..."):
        answer = explainer.generate_answer(query, retrieved)

    st.subheader("AI Explanation & Answer")
    st.write(answer)

