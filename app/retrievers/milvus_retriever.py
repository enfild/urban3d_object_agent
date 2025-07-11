from typing import List, Dict, Any, Optional

from ..milvus_client import search_vector
from .base_retriever import BaseRetriever


class MilvusRetriever(BaseRetriever):
    # Milvus

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def retrieve(self, vector: List[float]) -> Optional[Dict[str, Any]]:
        # find only the nearest
        results = search_vector(vector, top_k=1)
        if not results:
            return None

        top_hit = results[0]
        distance = top_hit["distance"]
        metadata = top_hit["metadata"]

        if distance < self.threshold:
            return {
                "vector": vector,
                "metadata": metadata,
                "score": distance
            }
        return None
