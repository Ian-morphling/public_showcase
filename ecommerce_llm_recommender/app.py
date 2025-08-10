import streamlit as st
from agents.retriever_agent import RetrieverAgent
from agents.explainer_agent import ExplainerAgent
from dotenv import load_dotenv
import os
import pickle
import tempfile
import requests

# Load local .env variables
load_dotenv()

# Determine if running in Streamlit Cloud environment
IS_STREAMLIT_CLOUD = os.getenv("IS_STREAMLIT_CLOUD", "false").lower() == "true"

# Set paths depending on environment
if not IS_STREAMLIT_CLOUD:
    # Local (full) index and mapping paths
    INDEX_PATH = os.getenv(
        "INDEX_URL_LOCAL",
        "/home/ianli/homl-self/public_showcase/ecommerce_llm_recommender/index/product.index",
    )
    MAPPING_PATH = os.getenv(
        "MAPPING_URL_LOCAL",
        "/home/ianli/homl-self/public_showcase/ecommerce_llm_recommender/index/id_to_filename.pkl",
    )
else:
    # Streamlit Cloud (small) index and UPDATED mapping paths
    INDEX_PATH = os.getenv("INDEX_URL")  # small index URL or path
    # IMPORTANT: Use updated mapping for parquet chunks on cloud
    MAPPING_PATH = os.getenv(
        "MAPPING_URL",
        "https://your-cdn-or-hosting/id_to_filename_small_updated.pkl"  # fallback URL for updated mapping
    )

# DOCS_DIR: point to small parquet chunk folder on cloud, or local full parquet folder
DOCS_DIR = os.getenv("DOCS_DIR", "outputs")  

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.title("🛒 E-Commerce Multi-Agent Recommender")

# Debug info for environment config
st.text("Config Debug Info:")
st.text(f"Running on Streamlit Cloud: {IS_STREAMLIT_CLOUD}")
st.text(f"INDEX_URL source: {'streamlit secrets' if IS_STREAMLIT_CLOUD else '.env / OS env'}")
st.text(f"INDEX_URL: {INDEX_PATH}")
st.text(f"MAPPING_URL: {MAPPING_PATH}")
st.text(f"DOCS_DIR: {DOCS_DIR}")
st.text(f"GROQ_API_KEY set? {'Yes' if GROQ_API_KEY else 'No'}")

def download_file(url, local_path):
    if not os.path.exists(local_path):
        resp = requests.get(url)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)
        st.text(f"Downloaded {url} to {local_path} ({os.path.getsize(local_path)} bytes)")
    else:
        st.text(f"File {local_path} already exists.")

def prepare_local_file(path_or_url, filename_hint):
    if path_or_url.startswith("http"):
        temp_path = os.path.join(tempfile.gettempdir(), filename_hint)
        download_file(path_or_url, temp_path)
        return temp_path
    else:
        return path_or_url

# Download index and mapping locally if needed
local_index_path = prepare_local_file(INDEX_PATH, "product.index")
local_mapping_path = prepare_local_file(MAPPING_PATH, "id_to_filename.pkl")

st.text(f"Local index path exists? {os.path.exists(local_index_path)}")
st.text(f"Local mapping path exists? {os.path.exists(local_mapping_path)}")

def check_parquet_files_exist(mapping_path, docs_dir):
    missing_files = []
    try:
        if mapping_path.startswith("http"):
            import requests
            from io import BytesIO
            r = requests.get(mapping_path)
            r.raise_for_status()
            mapping = pickle.load(BytesIO(r.content))
        else:
            with open(mapping_path, "rb") as f:
                mapping = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load mapping file: {e}")
        return None, None

    for parquet_file in set(mapping.values()):
        parquet_path = os.path.join(docs_dir, parquet_file)
        if not os.path.isfile(parquet_path):
            missing_files.append(parquet_file)

    return mapping, missing_files

mapping, missing_files = check_parquet_files_exist(local_mapping_path, DOCS_DIR)
if mapping is None:
    st.stop()

if missing_files:
    st.error(f"Missing parquet files in DOCS_DIR ({DOCS_DIR}):")
    for mf in missing_files:
        st.error(f"- {mf}")
    st.stop()
else:
    st.success(f"All {len(mapping)} parquet files referenced in mapping exist in DOCS_DIR.")

@st.cache_resource(show_spinner=False)
def init_agents():
    retriever = RetrieverAgent(local_index_path, local_mapping_path, DOCS_DIR)
    retriever.initialize()

    explainer = ExplainerAgent(GROQ_API_KEY)
    return retriever, explainer

try:
    retriever, explainer = init_agents()
except Exception as e:
    st.error(f"Failed to initialize RetrieverAgent: {e}")
    st.stop()

query = st.text_input("Enter your product question or query:", "")

top_k = st.slider("Number of reviews to retrieve:", min_value=1, max_value=10, value=5)

if st.button("Get Recommendations") and query.strip():
    with st.spinner("Retrieving reviews"):
        try:
            retrieved = retriever.query(query, top_k=top_k)
        except Exception as e:
            st.error(f"Error during retrieval: {e}")
            retrieved = []

    if not retrieved:
        st.warning("No reviews retrieved.")
    else:
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

        with st.spinner("Generating explanation and answer"):
            try:
                answer = explainer.generate_answer(query, retrieved)
                st.subheader("AI Explanation & Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Error during explanation generation: {e}")
