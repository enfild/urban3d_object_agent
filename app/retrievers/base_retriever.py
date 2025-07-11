from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(self, vector: List[float]) -> Optional[Dict[str, Any]]:
        """
        Take the embedding vector and try to find an existing object.
        Returns a dictionary with keys:
        - vector: original embedding (List[float])
        - metadata: metadata objects from Milvus (Dict[str, Any])
        - score: homogeneity/distance measure (float)
        or None if there is no match.
        ""
        ...


