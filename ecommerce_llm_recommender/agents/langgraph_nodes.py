from typing import Optional, List, Dict
from .retriever_agent import RetrieverAgent
from .userprofile_agent import UserProfileAgent
from .explainer_agent import ExplainerAgent


class RetrieverNode:
    """Wrapper around RetrieverAgent for LangGraph-style node."""
    def __init__(self, index_path: str, mapping_path: str, docs_dir: str):
        self.agent = RetrieverAgent(index_path=index_path, mapping_path=mapping_path, docs_dir=docs_dir)

    def run(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top_k relevant reviews for the query."""
        return self.agent.retrieve(query, top_k=top_k)


class UserProfileNode:
    """Wrapper around UserProfileAgent for LangGraph-style node."""
    def __init__(self, user_profiles_dir: str):
        self.agent = UserProfileAgent(user_profiles_dir=user_profiles_dir)

    def run(self, reviewer_id: str) -> Optional[List[Dict]]:
        """Return user profile stats if reviewer exists."""
        if self.agent.has_user(reviewer_id):
            return self.agent.get_user_stats(reviewer_id)
        return None

    def has_user(self, reviewer_id: str) -> bool:
        return self.agent.has_user(reviewer_id)


class ExplainerNode:
    """Wrapper around ExplainerAgent for LangGraph-style node."""
    def __init__(self, groq_api_key: Optional[str] = None):
        self.agent = ExplainerAgent(groq_api_key)

    def run(self, query: str, retrieved_docs: List[Dict], user_profile: Optional[List[Dict]] = None) -> str:
        """Generate AI explanation / recommendation based on query, docs, and optional user profile."""
        return self.agent.generate_answer(query, retrieved_docs, user_profile=user_profile)