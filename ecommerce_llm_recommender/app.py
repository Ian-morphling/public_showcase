import os
import requests
import streamlit as st
import threading
from agents.retriever_agent import RetrieverAgent
from agents.explainer_agent import ExplainerAgent

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_secret(key, default=None):
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

def download_if_not_exists(url: str, local_path: str):
    if not os.path.exists(local_path):
        st.info(f" Downloading {os.path.basename(local_path)}...")
        try:
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            st.error(f" Failed to download {url}: {e}")
            raise

def init_agents_bg():
    """Run in a background thread so UI can load instantly."""
    INDEX_URL = get_secret("INDEX_URL")
    MAPPING_URL = get_secret("MAPPING_URL")
    GROQ_API_KEY = get_secret("GROQ_API_KEY")
    DOCS_DIR = get_secret("DOCS_DIR", "outputs")

    try:
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

# --- UI loads instantly ---
st.title("🛒 E-Commerce Multi-Agent Recommender")

if "agents_ready" not in st.session_state and "agents_error" not in st.session_state:
    st.session_state["agents_ready"] = False
    threading.Thread(target=init_agents_bg, daemon=True).start()

if st.session_state.get("agents_error"):
    st.error(f" Agent initialization failed: {st.session_state['agents_error']}")
elif not st.session_state["agents_ready"]:
    st.warning(" Loading agents in the background... Please wait.")
else:
    retriever, explainer = st.session_state["agents"]
    query = st.text_input("Enter your product question or query:")
    top_k = st.slider("Number of reviews to retrieve:", 1, 10, 5)

    if st.button("Get Recommendations") and query:
        with st.spinner(" Retrieving reviews"):
            retrieved = retriever.query(query, top_k=top_k)

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

        with st.spinner(" Generating explanation and answer"):
            answer = explainer.generate_answer(query, retrieved)
            st.subheader("AI Explanation & Answer")
            st.write(answer)
