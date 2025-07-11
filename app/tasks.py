import logging
from typing import Any, Dict

import requests

from .config import settings
from .models import ObjectRequest

logger = logging.getLogger(__name__)


def notify_new_object(obj: ObjectRequest) -> None:
    """
    POST(notification) to settings.NEW_OBJ_URL
      {
        "id": ...,
        "type": ...,
        "timestamp": ...,
        "points3d": [...]
      }
    """
    payload: Dict[str, Any] = {
        "id": obj.id,
        "type": obj.type,
        "timestamp": obj.timestamp.isoformat(),
        "points3d": obj.pointcloud,
    }
    try:
        resp = requests.post(
            settings.NEW_OBJ_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        resp.raise_for_status()
        logger.info(f"Successfully notified new object {obj.id}")
    except requests.RequestException as e:
        logger.error(f"Failed to notify new object {obj.id}: {e}")
        raise
