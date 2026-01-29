# tests/test_planner_agent.py
import asyncio
import pytest
from agents import planner_agent
from agents.retriever_agent import RetrievedDocument
from types import SimpleNamespace

# Fake Groq client response
class FakeGroqResponse:
    def __init__(self, content: str):
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=content)
            )
        ]


class FakeGroqClient:
    class chat:
        class completions:
            @staticmethod
            def create(model, messages, max_tokens):
                prompt = messages[0]["content"]
                # Simple logic: if "sufficiency" check, return YES
                if "sufficiently answered" in prompt:
                    return FakeGroqResponse("YES")
                # if planning next query, return a fake query unless STOP is requested
                return FakeGroqResponse("STOP")

# Fixtures
@pytest.fixture
def sample_docs():
    return [
        RetrievedDocument(
            id=f"doc{i}",
            content=f"Content of doc {i}",
            similarity=0.9,
            section_type="Article",
            section_title=f"Article {i}",
            url=f"http://example.com/doc{i}"
        )
        for i in range(3)
    ]


@pytest.fixture
def low_similarity_docs():
    return [
        RetrievedDocument(
            id=f"doc{i}",
            content=f"Content of doc {i}",
            similarity=0.4,  # low similarity
            section_type="Article",
            section_title=f"Article {i}",
            url=f"http://example.com/doc{i}"
        )
        for i in range(3)
    ]


# Tests
@pytest.mark.asyncio
async def test_check_answer_sufficiency(monkeypatch, sample_docs):
    monkeypatch.setattr(planner_agent, "groq_client", FakeGroqClient())
    planner = planner_agent.PlannerAgent()

    result = await planner.check_answer_sufficiency("Test query", sample_docs)
    assert result is True


@pytest.mark.asyncio
async def test_plan_next_query_stop_on_stop(monkeypatch, sample_docs):
    monkeypatch.setattr(planner_agent, "groq_client", FakeGroqClient())
    planner = planner_agent.PlannerAgent(max_hops=5)

    next_query, stop_reason, new_docs = await planner.plan_next_query(
        original_query="Test query",
        current_query="Test query",
        retrieved_docs=sample_docs,
        hop=1,
        previous_queries=[]
    )

    assert next_query is None
    assert "stop" in stop_reason.lower()
    assert len(new_docs) == len(sample_docs)
    assert all(doc.id in planner.seen_doc_ids for doc in new_docs)


@pytest.mark.asyncio
async def test_plan_next_query_filters_seen_docs(monkeypatch, sample_docs):
    monkeypatch.setattr(planner_agent, "groq_client", FakeGroqClient())
    planner = planner_agent.PlannerAgent(max_hops=5)

    planner.seen_doc_ids.add(sample_docs[0].id)

    next_query, stop_reason, new_docs = await planner.plan_next_query(
        original_query="Test query",
        current_query="Test query",
        retrieved_docs=sample_docs,
        hop=1,
        previous_queries=[]
    )

    # Ensure seen doc is filtered
    assert sample_docs[0] not in new_docs
    # Remaining docs are new
    assert len(new_docs) == len(sample_docs) - 1


@pytest.mark.asyncio
async def test_plan_next_query_stops_on_max_hops(monkeypatch, sample_docs):
    monkeypatch.setattr(planner_agent, "groq_client", FakeGroqClient())
    planner = planner_agent.PlannerAgent(max_hops=1)  # very low max_hops

    next_query, stop_reason, new_docs = await planner.plan_next_query(
        original_query="Test query",
        current_query="Test query",
        retrieved_docs=sample_docs,
        hop=1,  # hop already equals max_hops
        previous_queries=[]
    )

    assert next_query is None
    assert "max hops" in stop_reason.lower()


@pytest.mark.asyncio
async def test_plan_next_query_stops_on_low_similarity(monkeypatch, low_similarity_docs):
    monkeypatch.setattr(planner_agent, "groq_client", FakeGroqClient())
    planner = planner_agent.PlannerAgent(max_hops=5)

    next_query, stop_reason, new_docs = await planner.plan_next_query(
        original_query="Test query",
        current_query="Test query",
        retrieved_docs=low_similarity_docs,
        hop=1,
        previous_queries=[]
    )

    assert next_query is None
    assert "relevance" in stop_reason.lower()
