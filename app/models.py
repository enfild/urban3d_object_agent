from typing import List, Literal, Union
from datetime import datetime
from pydantic import BaseModel, Field

class ObjectRequest(BaseModel):
    id: str
    city: str
    timestamp: datetime
    lat: float
    lon: float
    type: str = Field(..., description="source name of the obj")
    pointcloud: List[List[float]] = Field(
        ..., description="cloud [x, y, z]..."
    )
    bbox: List[float] = Field(
        ..., description="Bounding box [x, y, z, width, height, depth]"
    )

class ExistingObject(BaseModel):
    id: str
    city: str
    timestamp: datetime
    lat: float
    lon: float
    type: str = Field(..., description="Normilized object")
    bbox: List[float] = Field(
        ..., description="Bounding box [x, y, z, width, height, depth]"
    )

class ConditionRequest(BaseModel):
    # filter types
    condition: str

class LLMFilterResponse(BaseModel):
    # Response with included and excluded types
    included: List[str]
    excluded: List[str]

ObjectResponse = Union[str, ExistingObject]

class LLMNormalizeResponse(BaseModel):
    normalized_type: str

class LLMDecisionResponse(BaseModel):
    decision: Literal["update", "keep"]
    reason: str

