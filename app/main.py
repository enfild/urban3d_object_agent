import logging
from typing import Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from .config import settings
from .models import (
    ObjectRequest,
    ObjectResponse,
    ExistingObject,
    ConditionRequest,
)
from .models import LLMFilterResponse
from .llm_utils import (
    normalize_type,
    decide_update,
    generate_filter_expression,
)
from ._3dutils import encode_pointcloud
from .milvus_client import (
    insert_vector,
    get_all_distinct_types,
    stream_ids_by_expression,
)
from .retriever import find_existing
from .tasks import notify_new_object

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="UrbanRAG3D Service",
    description="Service for tracking and managing 3D objects in city pointclouds",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/objects/new", response_model=ObjectResponse)
def process_object(request: ObjectRequest) -> Union[str, ExistingObject]:
    """
    1. Normalize the textual type via LLM.
    2. Generate 3D embedding.
    3. Search for an existing object in Milvus.
    4. If found, decide via LLM whether to update or keep.
    5. Insert/update the record and notify downstream.
    """
    # 1. Normalize type
    try:
        normalized_type = normalize_type(request.type)
        logger.info(f"Normalized type '{request.type}' → '{normalized_type}'")
    except Exception as e:
        logger.error(f"Error normalizing type for {request.id}: {e}")
        raise HTTPException(status_code=500, detail=f"Normalization error: {e}")

    # 2. Generate embedding
    try:
        vector = encode_pointcloud(request.pointcloud)
    except Exception as e:
        logger.error(f"Error encoding pointcloud for {request.id}: {e}")
        raise HTTPException(status_code=500, detail=f"Encoding error: {e}")

    # 3. Search for an existing object
    existing_hit = find_existing(vector)
    if existing_hit:
        meta = existing_hit["metadata"]
        score = existing_hit.get("score")
        decision, reason = decide_update(meta, request, {"score": score})
        logger.info(f"LLM decision for {request.id}: {decision} ({reason})")

        if decision == "keep":
            try:
                return ExistingObject(**meta)
            except ValidationError as e:
                logger.error(f"Validation error returning existing object: {e}")
                raise HTTPException(status_code=500, detail="Invalid existing object data")

        # update path
        updated_meta = {
            **meta,
            "timestamp": request.timestamp.isoformat(),
            "lat": request.lat,
            "lon": request.lon,
            "type": normalized_type,
            "bbox": request.bbox,
        }
        insert_vector(request.id, vector, updated_meta)
        try:
            notify_new_object(request)
        except Exception as e:
            logger.error(f"Error notifying update for {request.id}: {e}")
        return "object updated"

    # create-new path
    new_meta = {
        "id": request.id,
        "city": request.city,
        "timestamp": request.timestamp.isoformat(),
        "lat": request.lat,
        "lon": request.lon,
        "type": normalized_type,
        "bbox": request.bbox,
    }
    insert_vector(request.id, vector, new_meta)
    try:
        notify_new_object(request)
    except Exception as e:
        logger.error(f"Error notifying new object for {request.id}: {e}")
    return "new object created"


@app.post("/objects/filter_by_rule/included/stream", response_model=None)
def filter_by_rule_included_stream(req: ConditionRequest):
    """
    Stream IDs using an LLM-generated filter expression for inclusion.
    """
    expr = generate_filter_expression(req.condition)
    return StreamingResponse(
        stream_ids_by_expression(expr),
        media_type="text/plain"
    )


@app.post("/objects/filter_by_rule/excluded/stream", response_model=None)
def filter_by_rule_excluded_stream(req: ConditionRequest):
    """
    Stream IDs using an LLM-generated filter expression for exclusion.
    """
    expr = generate_filter_expression(req.condition)
    return StreamingResponse(
        stream_ids_by_expression(expr),
        media_type="text/plain"
    )
