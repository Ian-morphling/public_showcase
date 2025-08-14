import streamlit as st
import os
import gc
import psutil
from pathlib import Path
import requests

from agents.retriever_agent import RetrieverAgent
from agents.userprofile_agent import UserProfileAgent  
from agents.explainer_agent import ExplainerAgent      



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

INDEX_PATH = "index/product.index"
MAPPING_PATH = "index/id_to_filename.pkl"
CHUNKS_DIR = Path(os.getenv("DOCS_DIR", "outputs"))
USER_PROFILES_DIR = CHUNKS_DIR

INDEX_URL = os.getenv("INDEX_URL")
MAPPING_URL = os.getenv("MAPPING_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


if INDEX_URL and not Path(INDEX_PATH).exists():
    download_file(INDEX_URL, INDEX_PATH)

if MAPPING_URL and not Path(MAPPING_PATH).exists():
    download_file(MAPPING_URL, MAPPING_PATH)

# === Page Config ===
st.set_page_config(
    page_title="🛒 E-Commerce Multi-Agent Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Helper Functions ===
@st.cache_data
def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def validate_files():
    """Validate all required files exist"""
    required_files = {
        "FAISS Index": INDEX_PATH,
        "ID Mapping": MAPPING_PATH,
        "Documents Directory": CHUNKS_DIR,
        "User Profiles Dir": USER_PROFILES_DIR
    }
    
    missing_files = []
    for name, path in required_files.items():
        p = Path(path)
        if name == "User Profiles Dir":
            parquet_files = list(p.glob("user_profiles_part_*.parquet"))
            if not parquet_files:
                missing_files.append(f"{name}: No parquet files found in {path}")
        else:
            if not p.exists():
                missing_files.append(f"{name}: {path}")
    
    return missing_files

# === Sidebar Info ===
with st.sidebar:
    st.header("🔧 System Status")
    
    memory_mb = get_memory_usage()
    st.metric("Memory Usage", f"{memory_mb:.1f} MB")
    
    missing_files = validate_files()
    if missing_files:
        st.error(" Missing Files:")
        for file in missing_files:
            st.text(f"• {file}")
    else:
        st.success(" All files found")
    
    if GROQ_API_KEY:
        st.success(" Groq API Key loaded")
    else:
        st.warning(" No Groq API Key (fallback mode)")
    
    st.header(" System Controls")
    if st.button(" Clear Cache & Free Memory"):
        st.cache_data.clear()
        st.cache_resource.clear()
        for key in ['retriever', 'userprofile_agent', 'explainer']:
            if key in st.session_state:
                del st.session_state[key]
        gc.collect()
        st.success("Cache cleared!")
        st.rerun()

# === Main App ===
st.title("🛒 E-Commerce Multi-Agent Recommender")
st.markdown("*Advanced RAG-based product recommendation system with personalization*")

# Early exit if missing critical files
missing_files = validate_files()
if missing_files:
    st.error("Cannot start application - missing required files. See sidebar for details.")
    st.stop()

# === Lazy Agent Loading ===
@st.cache_resource
def load_retriever():
    try:
        return RetrieverAgent(INDEX_PATH, MAPPING_PATH, CHUNKS_DIR)
    except Exception as e:
        st.error(f"Failed to load retriever: {e}")
        return None

@st.cache_resource  
def load_userprofile_agent():
    try:
        return UserProfileAgent(USER_PROFILES_DIR)
    except Exception as e:
        st.error(f"Failed to load user profiles: {e}")
        return None

@st.cache_resource
def load_explainer():
    try:
        return ExplainerAgent(GROQ_API_KEY)
    except Exception as e:
        st.warning(f"LLM agent initialized with warnings: {e}")
        return ExplainerAgent(GROQ_API_KEY)  # fallback

retriever = load_retriever()
userprofile_agent = load_userprofile_agent()
explainer = load_explainer()

if not retriever:
    st.error("Cannot proceed without retriever agent.")
    st.stop()

# === User Interface ===
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(" Search Query")
    query = st.text_input(
        "Enter your product question:",
        placeholder="e.g., 'best wireless headphones under $200'",
        help="Describe what you're looking for or ask specific questions about products"
    )

with col2:
    st.subheader(" Settings")
    top_k = st.slider(
        "Number of reviews to retrieve:", 
        min_value=1, max_value=20, value=5,
        help="More reviews = better context but slower processing"
    )

st.subheader("👤 Personalization (Optional)")
reviewer_id = st.text_input(
    "Reviewer ID:", 
    placeholder="Enter reviewer ID for personalized recommendations",
    help="If provided, recommendations will be tailored based on your review history"
)

if reviewer_id.strip() and userprofile_agent:
    with st.spinner("Checking user profile..."):
        if userprofile_agent.has_user(reviewer_id):
            summary = userprofile_agent.get_user_summary(reviewer_id)
            if summary:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Total Reviews", summary['total_reviews'])
                with c2:
                    st.metric("Avg Rating", f"{summary['avg_rating']:.1f}")
                with c3:
                    st.metric("Verified Purchases", summary['verified_purchases'])
                with c4:
                    st.metric("5-Star Reviews", summary['rating_distribution']['5_star'])
        else:
            st.warning(f" Reviewer ID '{reviewer_id}' not found in database")

# === Main Query Processing ===
if st.button(" Get Recommendations", type="primary", use_container_width=True):
    if not query.strip():
        st.error("Please enter a search query!")
        st.stop()
    
    retrieved_docs = []
    user_profile_data = None
    
    # Phase 1: Retrieve reviews
    with st.spinner(" Searching relevant reviews..."):
        try:
            retrieved_docs = retriever.retrieve(query, top_k=top_k)
            st.success(f" Retrieved {len(retrieved_docs)} relevant reviews")
        except Exception as e:
            st.error(f" Retrieval failed: {e}")
            st.stop()
    
    # Phase 2: Load user profile
    if reviewer_id.strip() and userprofile_agent:
        with st.spinner("👤 Loading user profile..."):
            try:
                if userprofile_agent.has_user(reviewer_id):
                    user_profile_data = userprofile_agent.get_user_stats(reviewer_id)
                    if user_profile_data:
                        st.success(f" Loaded profile with {len(user_profile_data)} reviews")
                    else:
                        st.warning(" User profile exists but no data loaded")
                else:
                    st.warning(f" Reviewer ID '{reviewer_id}' not found")
            except Exception as e:
                st.error(f" User profile loading failed: {e}")
    
    # Phase 3: Display retrieved reviews
    if retrieved_docs:
        st.subheader(" Retrieved Reviews")
        ratings = [doc.get('overall') for doc in retrieved_docs if doc.get('overall')]
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            st.info(f" Average rating of retrieved reviews: {avg_rating:.1f}⭐ ({len(ratings)} reviews)")
        
        with st.expander(" View Retrieved Reviews", expanded=True):
            for i, doc in enumerate(retrieved_docs):
                with st.container():
                    c1, c2 = st.columns([3, 1])
                    
                    with c2:
                        rating = doc.get('overall', 'N/A')
                        verified = doc.get('verified', 'N/A')
                        distance = doc.get('faiss_distance', 0)
                        
                        st.metric(f"Review {i+1}", f"{rating}")
                        st.caption(f"Similarity: {1-distance:.3f}")
                        st.caption(f"Verified: {'✅' if verified else '❌'}")
                    
                    with c1:
                        text = doc.get("text", "")
                        if len(text) > 800:
                            st.write(text[:800] + "...")
                            with st.expander("Read full review"):
                                st.write(text)
                        else:
                            st.write(text)
                    
                    st.divider()
    
    # Phase 4: Generate AI explanation
    if retrieved_docs and explainer:
        st.subheader(" AI Analysis & Recommendations")
        with st.spinner(" Generating intelligent analysis..."):
            try:
                answer = explainer.generate_answer(query, retrieved_docs, user_profile=user_profile_data)
                st.markdown("### Recommendation Summary")
                st.write(answer)
                if user_profile_data:
                    st.success(" This recommendation was personalized based on your review history!")
            except Exception as e:
                st.error(f" AI analysis failed: {e}")
    
    gc.collect()

# === Footer ===
st.markdown("---")
st.markdown(
    "** Multi-Agent E-Commerce Recommender** | "
    "*Powered by FAISS, Sentence Transformers, and Groq LLM*"
)
