from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END

from agents.retriever_agent import RetrieverAgent
from agents.userprofile_agent import UserProfileAgent
from agents.explainer_agent import ExplainerAgent
from agents.quality_analysis_agent import QualityAnalysisAgent 
from agents.judge_agent import JudgeAgent

import json
import time
from pathlib import Path

class GraphState(TypedDict, total=False):
    query: str
    top_k: int
    reviewer_id: Optional[str]

    retrieved_docs: List[Dict]
    user_profile: Optional[List[Dict]]
    user_profile_summary: Optional[Dict]

    answer: str
    judge_scores: Optional[Dict]

def build_graph(
    index_path: str,
    mapping_path: str,
    docs_dir: str,
    user_profiles_dir: str,
    groq_api_key: Optional[str] = None,
    use_judge_in_app: bool = False
):
    retriever = RetrieverAgent(index_path=index_path, mapping_path=mapping_path, docs_dir=docs_dir)
    userprof = UserProfileAgent(user_profiles_dir=user_profiles_dir)
    explainer = ExplainerAgent(groq_api_key)
    qa_agent = QualityAnalysisAgent()
    judge_agent = JudgeAgent(groq_api_key)

    graph = StateGraph(GraphState)

    # --- Nodes ---
    def node_retrieve(state: GraphState) -> GraphState:
        query = state["query"]
        top_k = int(state.get("top_k", 5))
        docs = retriever.retrieve(query, top_k=top_k)
        return {"retrieved_docs": docs}

    def node_quality_analysis(state: GraphState) -> GraphState:
        docs = state.get("retrieved_docs", [])
        analyzed_docs = qa_agent.analyze_reviews(docs)
        return {"retrieved_docs": analyzed_docs}

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
    
    def node_judge(state: GraphState) -> GraphState:
        if not use_judge_in_app:
            return state

        query = state["query"]
        explanation = state.get("answer", "")
        docs = state.get("retrieved_docs", [])

        if explanation and docs:
            top_docs_for_judge = docs[:5]
            judge_scores = judge_agent.judge(query, explanation, top_docs_for_judge)

            # --- Save live judge JSON ---
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            out_dir = Path("judge_outputs")
            out_dir.mkdir(exist_ok=True)
            live_file = out_dir / f"judge_run_{timestamp}.json"

            payload_live = {
                "query": query,
                "answer": explanation,
                "judge_scores": judge_scores,
                "docs_used": [d.get("text", "")[:200] for d in top_docs_for_judge]
            }
            with open(live_file, "w", encoding="utf-8") as f:
                json.dump(payload_live, f, indent=2, ensure_ascii=False)

            print(f"Judge scores saved to {live_file}")
            print("Judge scores:", judge_scores)

            # Return judge_scores in state
            return {"judge_scores": judge_scores, **state}
        else:
            return state

    # --- Add nodes ---
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("quality_analysis", node_quality_analysis)
    graph.add_node("user_profile", node_user_profile)
    graph.add_node("explain", node_explain)
    graph.add_node("judge", node_judge)

    # --- Routing ---
    def router_after_retrieve(state: GraphState) -> str:
        return "quality_analysis"

    def router_after_quality(state: GraphState) -> str:
        reviewer_id = (state.get("reviewer_id") or "").strip()
        if reviewer_id and userprof.has_user(reviewer_id):
            return "user_profile"
        return "explain"

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges(
        "retrieve",
        router_after_retrieve,
        {"quality_analysis": "quality_analysis"}
    )
    graph.add_conditional_edges(
        "quality_analysis",
        router_after_quality,
        {"user_profile": "user_profile", "explain": "explain"}
    )

    graph.add_edge("user_profile", "explain")
    graph.add_edge("explain", "judge")
    graph.add_edge("judge", END)

    return graph.compile()
