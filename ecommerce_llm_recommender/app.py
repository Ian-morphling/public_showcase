
import streamlit as st
from agents.retriever_agent import RetrieverAgent
from agents.explainer_agent import ExplainerAgent
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize agents once and cache them to avoid reload on every interaction
@st.cache_resource(show_spinner=False)
def init_agents():
    INDEX_PATH = os.getenv("INDEX_PATH", "index/product.index")
    MAPPING_PATH = os.getenv("MAPPING_PATH", "index/id_to_filename.pkl")
    DOCS_DIR = os.getenv("DOCS_DIR", "outputs")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    retriever = RetrieverAgent(INDEX_PATH, MAPPING_PATH, DOCS_DIR)
    retriever.initialize()

    explainer = ExplainerAgent(GROQ_API_KEY)
    return retriever, explainer

retriever, explainer = init_agents()

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
        st.write(r.get("text", "")[:500] + ("..." if len(r.get("text", "")) > 500 else ""))
        st.markdown("---")

    with st.spinner("Generating explanation and answer..."):
        answer = explainer.generate_answer(query, retrieved)
    st.subheader("AI Explanation & Answer")
    st.write(answer)
