# langgraph_graph.py
"""
- Multi-hop retrieval driven by PlannerAgent
- Only NEW documents are used for explanation
- Planner internal state is reset per run
"""

from typing import List, Optional, TypedDict, Any, Dict
from langgraph.graph import StateGraph, END

from agents.retriever_agent import RetrieverAgent, RetrievedDocument
from agents.planner_agent import PlannerAgent
from agents.explainer_agent import ExplainerAgent

# State definition

class RAGState(TypedDict, total=False):
    # user input
    user_query: str
    intent: Optional[str]

    # retrieval loop
    current_query: str
    previous_queries: List[str]
    hop: int
    stop_reason: Optional[str]

    # documents
    retrieved_docs: List[RetrievedDocument]
    new_docs: List[RetrievedDocument]

    # hop tracking (for debugging / tests)
    hops: List[Dict[str, Any]]

    # final output
    answer: Optional[str]
    sources: List[Dict[str, str]]

# Agent initialization

retriever = RetrieverAgent()
planner = PlannerAgent(max_hops=3, enable_sufficiency=True)
explainer = ExplainerAgent()


# LangGraph nodes

async def classify_intent_node(state: RAGState) -> RAGState:
    """
    Initialize state and classify intent once.
    """
    # Reset planner internal memory per run
    planner.seen_doc_ids.clear()
    planner.query_hashes.clear()

    intent = await planner.classify_intent(state["user_query"])

    state.update({
        "intent": intent,
        "current_query": state["user_query"],
        "previous_queries": [],
        "hop": 0,
        "retrieved_docs": [],
        "new_docs": [],
        "hops": [],
        "stop_reason": None,
        "answer": None,
        "sources": [],
    })

    return state


async def retrieve_node(state: RAGState) -> RAGState:
    """
    Retrieve documents for the current query.
    """
    docs = await retriever.retrieve(
        state["current_query"],
        top_k=5,
    )

    state["retrieved_docs"].extend(docs)

    return state


async def plan_node(state: RAGState) -> RAGState:
    """
    Planner decides:
    - next query (or stop)
    - which documents are NEW
    """
    next_query, stop_reason, new_docs = await planner.plan_next_query(
        original_query=state["user_query"],
        current_query=state["current_query"],
        retrieved_docs=state["retrieved_docs"],
        hop=state["hop"] + 1,
        previous_queries=state["previous_queries"],
        top_k=5,
    )

    state["hop"] += 1
    state["new_docs"].extend(new_docs)
    state["stop_reason"] = stop_reason

    # Track hop metadata
    state["hops"].append({
        "hop_number": state["hop"],
        "query_used": state["current_query"],
        "new_docs": new_docs,
        "next_query": next_query,
        "stop_reason": stop_reason,
    })

    if next_query:
        state["previous_queries"].append(state["current_query"])
        state["current_query"] = next_query

    return state


def should_continue(state: RAGState) -> str:
    """
    Decide whether to continue retrieval or move to explanation.
    """
    if state.get("stop_reason"):
        return "explain"
    return "retrieve"


async def explain_node(state: RAGState) -> RAGState:
    """
    Final explanation using ONLY deduplicated NEW documents.
    """
    deduped_docs = list({
        doc.id: doc for doc in state["new_docs"]
    }.values())

    state["answer"] = await explainer.explain(
        query=state["user_query"],
        docs=deduped_docs,
    )

    state["sources"] = [
        {
            "label": f"{doc.section_type} {doc.section_title}",
            "url": doc.url,
        }
        for doc in deduped_docs
    ]

    return state

# Graph builder

def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("plan", plan_node)
    graph.add_node("explain", explain_node)

    graph.set_entry_point("classify_intent")

    graph.add_edge("classify_intent", "retrieve")
    graph.add_edge("retrieve", "plan")

    graph.add_conditional_edges(
        "plan",
        should_continue,
        {
            "retrieve": "retrieve",
            "explain": "explain",
        },
    )

    graph.add_edge("explain", END)

    return graph.compile()
