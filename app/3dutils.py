import time
from typing import List

import requests

from .config import settings

def encode_pointcloud(points: List[List[float]]) -> List[float]:
    # Convert 3D-points to embedding-вектор with external API.
    url = settings.ENCODER_URL
    payload = {"points3d": points}
    headers = {"Content-Type": "application/json"}

    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("embedding")
            if embedding is None:
                raise ValueError("Response JSON does not contain 'embedding'")
            return embedding
        except (requests.RequestException, ValueError) as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise
