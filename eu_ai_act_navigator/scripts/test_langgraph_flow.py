import asyncio
from textwrap import shorten

from langgraph_graph import build_graph

def preview(text: str, max_chars: int = 220) -> str:
    if not text:
        return ""
    return shorten(text.replace("\n", " "), width=max_chars, placeholder="...")


def divider(title: str):
    print("\n" + "=" * 5 + f" {title} " + "=" * 5)

def print_agentic_trace(state: dict):
    divider("Agentic RAG Execution Trace")

    print(f"User query:\n{state['user_query']}")

    all_sources = {}

    for hop_data in state.get("hops", []):
        hop_num = hop_data["hop_number"]

        divider(f"Hop {hop_num}")

        print(f"Query used:\n{hop_data['query_used']}")

        new_docs = hop_data.get("new_docs", [])

        if not new_docs:
            print("\nNo new documents added in this hop.")
        else:
            print(f"\nNew documents added ({len(new_docs)}):")

            for i, doc in enumerate(new_docs, 1):
                section_label = f"{doc.section_type} {doc.section_title}"

                print(f"\nResult {i}")
                print(f"Section: {section_label}")
                print(f"Similarity: {doc.similarity:.4f}")
                print(f"URL: {doc.url}")
                print("Preview:")
                print(f"  {preview(doc.content)}")

                all_sources[doc.url] = section_label

        if hop_data.get("stop_reason"):
            print(f"\nStop reason: {hop_data['stop_reason']}")

    divider("Final Answer (Explainable)")

    final_answer = state.get("answer")

    if isinstance(final_answer, dict):
        print(final_answer.get("answer", "No answer text available."))
    else:
        print(final_answer or "No answer generated.")

    divider("Sources (Deduplicated)")

    for i, (url, label) in enumerate(all_sources.items(), 1):
        print(f"{i}. {label}")
        print(f"   {url}")

async def main():
    graph = build_graph()

    user_query = input("Enter your EU AI query: ").strip()

    initial_state = {
        "user_query": user_query,
    }

    final_state = await graph.ainvoke(initial_state)

    print_agentic_trace(final_state)


if __name__ == "__main__":
    asyncio.run(main())
