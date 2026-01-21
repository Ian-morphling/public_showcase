import os
from typing import List, Optional, Dict
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_HOST"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]


@dataclass
class RetrievedDocument:
    id: str
    section_type: str
    section_title: str
    url: str
    content: str
    similarity: float


class RetrieverAgent:
    """
    Semantic retriever for EU AI Act documents.
    Uses pgvector HNSW index via Supabase RPC.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)
        self.supabase: Client = create_client(
            SUPABASE_URL, SUPABASE_KEY
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[RetrievedDocument]:

        query_embedding = self.model.encode(query).tolist()

        rpc_args = {
            "query_embedding": query_embedding,
            "match_count": top_k,
        }

        response = self.supabase.rpc(
            "match_documents", rpc_args
        ).execute()

        if response.data is None:
            return []

        return [
            RetrievedDocument(
                id=row["id"],
                section_type=row["section_type"],
                section_title=row["section_title"],
                url=row["url"],
                content=row["content"],
                similarity=row["similarity"],
            )
            for row in response.data
        ]
