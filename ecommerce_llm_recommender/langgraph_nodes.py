from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END

from agents.retriever_agent import RetrieverAgent
from agents.userprofile_agent import UserProfileAgent
from agents.explainer_agent import ExplainerAgent

class GraphState(TypedDict, total=False):
    query: str
    top_k: int
    reviewer_id: Optional[str]

    retrieved_docs: List[Dict]
    user_profile: Optional[List[Dict]]
    user_profile_summary: Optional[Dict]

    answer: str

def build_graph(
    index_path: str,
    mapping_path: str,
    docs_dir: str,
    user_profiles_dir: str,
    groq_api_key: Optional[str] = None,
):
    retriever = RetrieverAgent(index_path=index_path, mapping_path=mapping_path, docs_dir=docs_dir)
    userprof = UserProfileAgent(user_profiles_dir=user_profiles_dir)
    explainer = ExplainerAgent(groq_api_key)

    graph = StateGraph(GraphState)

    # --- Nodes ---
    def node_retrieve(state: GraphState) -> GraphState:
        query = state["query"]
        top_k = int(state.get("top_k", 5))
        docs = retriever.retrieve(query, top_k=top_k)
        return {"retrieved_docs": docs}

    def node_user_profile(state: GraphState) -> GraphState:
        reviewer_id = (state.get("reviewer_id") or "").strip()
        if reviewer_id and userprof.has_user(reviewer_id):
            profile = userprof.get_user_stats(reviewer_id)
            summary = userprof.get_user_summary(reviewer_id)
        else:
            profile = None
            summary = None
        return {"user_profile": profile, "user_profile_summary": summary}

    def node_explain(state: GraphState) -> GraphState:
        query = state["query"]
        docs = state.get("retrieved_docs", [])
        profile = state.get("user_profile", None)
        answer = explainer.generate_answer(query, docs, user_profile=profile)
        return {"answer": answer}

    # --- Add nodes to graph ---
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("user_profile", node_user_profile)
    graph.add_node("explain", node_explain)

    # --- Routing ---
    def router_after_retrieve(state: GraphState) -> str:
        reviewer_id = (state.get("reviewer_id") or "").strip()
        if reviewer_id and userprof.has_user(reviewer_id):
            return "user_profile"
        return "explain"

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges(
        "retrieve",
        router_after_retrieve,
        {"user_profile": "user_profile", "explain": "explain"},
    )
    graph.add_edge("user_profile", "explain")
    graph.add_edge("explain", END)

    return graph.compile()
