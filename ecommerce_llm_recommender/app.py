import streamlit as st
import os
import gc
import psutil
from pathlib import Path
import requests
from dotenv import load_dotenv
from langgraph_nodes import build_graph

load_dotenv()

# --- Paths & URLs ---
INDEX_PATH = "index/product.index"
MAPPING_PATH = "index/id_to_filename.pkl"
CHUNKS_DIR = Path(os.getenv("DOCS_DIR", "outputs"))
USER_PROFILES_DIR = CHUNKS_DIR

INDEX_URL = os.getenv("INDEX_URL")
MAPPING_URL = os.getenv("MAPPING_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# --- Helper: Download files if missing ---
def download_file(url, target_path):
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not target_path.exists():
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(target_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        st.info(f"Downloaded {target_path.name}")
    else:
        st.info(f"{target_path.name} already exists")
    return target_path


if INDEX_URL and not Path(INDEX_PATH).exists():
    download_file(INDEX_URL, INDEX_PATH)
if MAPPING_URL and not Path(MAPPING_PATH).exists():
    download_file(MAPPING_URL, MAPPING_PATH)


# --- Page Config ---
st.set_page_config(
    page_title="🛒 E-Commerce Multi-Agent Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- System Status & Validation ---
@st.cache_data
def get_memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB


def validate_files():
    required_files = {
        "FAISS Index": INDEX_PATH,
        "ID Mapping": MAPPING_PATH,
        "Documents Directory": CHUNKS_DIR,
        "User Profiles Dir": USER_PROFILES_DIR
    }
    missing = []
    for name, path in required_files.items():
        p = Path(path)
        if name == "User Profiles Dir":
            if not list(p.glob("user_profiles_part_*.parquet")):
                missing.append(f"{name}: No parquet files found in {path}")
        elif not p.exists():
            missing.append(f"{name}: {path}")
    return missing


with st.sidebar:
    st.header("System Status")
    st.metric("Memory Usage", f"{get_memory_usage():.1f} MB")

    missing_files = validate_files()
    if missing_files:
        st.error("Missing Files:")
        for f in missing_files:
            st.text(f"• {f}")
    else:
        st.success("All files found")
    if GROQ_API_KEY:
        st.success("Groq API Key loaded")
    else:
        st.warning("No Groq API Key (fallback mode)")

    st.header("System Controls")
    if st.button("Clear Cache & Free Memory"):
        st.cache_data.clear()
        st.cache_resource.clear()
        gc.collect()
        st.success("Cache cleared!")
        st.rerun()


# --- Main App ---
st.title("🛒 E-Commerce Multi-Agent Recommender")
st.markdown("*Advanced RAG-based product recommendation system with personalization*")

if missing_files:
    st.error("Cannot start application - missing required files. See sidebar for details.")
    st.stop()


# --- Load LangGraph ---
@st.cache_resource
def load_graph():
    return build_graph(
        index_path=INDEX_PATH,
        mapping_path=MAPPING_PATH,
        docs_dir=CHUNKS_DIR,
        user_profiles_dir=USER_PROFILES_DIR,
        groq_api_key=GROQ_API_KEY
    )


graph = load_graph()


# --- UI ---
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Search Query")
    query = st.text_input(
        "Enter your product question:",
        placeholder="e.g., 'best wireless headphones under $200'"
    )

with col2:
    st.subheader("Settings")
    top_k = st.slider("Number of reviews to retrieve:", 1, 20, 5)


st.subheader("Personalization (Optional)")
reviewer_id = st.text_input(
    "Reviewer ID:",
    placeholder="Enter reviewer ID for personalized recommendations"
)


# --- Query Processing ---
if st.button("Get Recommendations", type="primary", use_container_width=True):
    if not query.strip():
        st.error("Please enter a search query!")
        st.stop()

    state = {
        "query": query,
        "top_k": top_k,
        "reviewer_id": reviewer_id.strip() or None
    }

    with st.spinner("Generating recommendations..."):
        try:
            result_state = graph.invoke(state)
        except Exception as e:
            st.error(f"Error during graph execution: {e}")
            st.stop()

    retrieved_docs = result_state.get("retrieved_docs", [])
    answer = result_state.get("answer")
    user_profile_summary = result_state.get("user_profile_summary")

    # --- User Profile Stats ---
    if user_profile_summary:
        st.success(f"✅ Personalized recommendations generated using reviewer ID **{reviewer_id}**")
        st.subheader("Reviewer Stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Reviews", user_profile_summary['total_reviews'])
        c2.metric("Avg Rating", f"{user_profile_summary['avg_rating']:.1f}")
        c3.metric("Verified Purchases", user_profile_summary['verified_purchases'])
        c4.metric("5-Star Reviews", user_profile_summary['rating_distribution']['5_star'])

    # --- Retrieved Reviews ---
    if retrieved_docs:
        st.subheader("Retrieved Reviews")
        with st.expander("View Retrieved Reviews", expanded=True):
            for i, doc in enumerate(retrieved_docs):
                c1, c2 = st.columns([3, 1])
                with c2:
                    rating = doc.get("overall")
                    rating_str = f"{rating:.1f}" if isinstance(rating, (int, float)) else "N/A"
                    st.markdown(f"""
                    **Review {i+1}**
                    - Rating: **{rating_str}**
                    - Quality: **{doc.get("quality_label","N/A")}** (score = {doc.get("quality_score",0):.2f})
                    - Similarity: `{doc.get("similarity",0):.3f}`
                    - Verified: {"✅" if doc.get("verified") else "❌"}
                    """)
                with c1:
                    text = doc.get("text","")
                    if len(text) > 800:
                        st.write(text[:800] + "...")
                        with st.expander("Read full review"):
                            st.write(text)
                    else:
                        st.write(text)
                st.divider()

    # --- AI Analysis ---
    if answer:
        st.subheader("AI Analysis & Recommendations")
        st.markdown("### Recommendation Summary")
        st.write(answer)


# --- Footer ---
st.markdown("---")
st.markdown(
    "** Multi-Agent E-Commerce Recommender** | "
    "*Powered by LangGraph, FAISS, Sentence Transformers, and Groq LLM*"
)