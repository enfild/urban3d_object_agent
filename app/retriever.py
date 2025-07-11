# app/retriever.py

from typing import List, Optional, Dict, Any

from .retrievers.base_retriever import BaseRetriever
from .retrievers.milvus_retriever import MilvusRetriever


def get_retrievers() -> List[BaseRetriever]:
    return [
        MilvusRetriever()
    ]


def find_existing(vector: List[float]) -> Optional[Dict[str, Any]]:
    for retriever in get_retrievers():
        result = retriever.retrieve(vector)
        if result:
            return result
    return None
