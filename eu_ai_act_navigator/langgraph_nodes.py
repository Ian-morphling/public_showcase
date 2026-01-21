# langgraph_nodes.py
"""
LangGraph nodes for Agentic RAG with async PlannerAgent.

Flow:
PlannerAgent -> RetrieverAgent (multi-hop) -> ExplainerAgent

- Each hop uses only NEW documents for planning
- Final output is deduplicated
- Explanation is citation-aware and hallucination-resistant
- Explainer output is plain Markdown/text
"""

from agents.retriever_agent import RetrieverAgent, RetrievedDocument
from agents.planner_agent import PlannerAgent
from agents.explainer_agent import ExplainerAgent

# --- Initialize agents ---
retriever = RetrieverAgent()
planner = PlannerAgent(max_hops=3, enable_sufficiency=True)
explainer = ExplainerAgent()


async def run_agentic_rag(
    user_query: str,
    max_hops: int = 3,
    top_k: int = 5,
):
    all_retrieved_docs: list[RetrievedDocument] = []
    all_new_docs: list[RetrievedDocument] = []
    hops = []
    current_query = user_query
    previous_queries: list[str] = []

    # --- Infer intent once ---
    await planner.classify_intent(user_query)

    for hop_num in range(1, max_hops + 1):
        # --- Retrieval ---
        docs = await retriever.retrieve(current_query, top_k=top_k)
        all_retrieved_docs.extend(docs)

        # --- Planning ---
        next_query, stop_reason, new_docs = await planner.plan_next_query(
            original_query=user_query,
            current_query=current_query,
            retrieved_docs=all_retrieved_docs,
            hop=hop_num,
            previous_queries=previous_queries,
            top_k=top_k,
        )

        # --- Track only NEW docs ---
        all_new_docs.extend(new_docs)

        # --- Record hop ---
        hops.append({
            "hop_number": hop_num,
            "query_used": current_query,
            "retrieved_docs": docs,
            "new_docs": new_docs,
            "next_query": next_query,
            "stop_reason": stop_reason,
        })

        if not next_query:
            break

        previous_queries.append(current_query)
        current_query = next_query

    # --- Deduplicate final documents by ID ---
    deduped_docs = list({doc.id: doc for doc in all_new_docs}.values())

    # --- Final citation-aware explanation ---
    explanation = await explainer.explain(
        query=user_query,
        docs=deduped_docs,
    )

    return {
        "query": user_query,
        "intent": planner.intent,
        "hop_count": len(hops),
        "hops": hops,
        "retrieved_docs": deduped_docs,
        "answer": explanation,
        "sources": [f"{doc.section_type} {doc.section_title}" for doc in deduped_docs],
    }
