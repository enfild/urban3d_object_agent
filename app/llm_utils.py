import json
from datetime import datetime
from typing import Tuple, Dict

import openai
from pydantic import ValidationError

from .config import settings
from .models import ObjectRequest, LLMFilterResponse, LLMNormalizeResponse, LLMDecisionResponse

openai.api_key = settings.OPENAI_API_KEY


def normalize_type(raw: str) -> str:
    """
    Normalizes the value of the "type" field via the LLM, returning a canonical English term.
    """
    prompt = (
        'Convert the value of the "type" field to a canonical form:\n'
        f'Input: "{raw}"\n\n'
        'Expected output: a single English word (e.g., Car, Tree, Bench).'
    )

    resp = openai.ChatCompletion.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You help normalize textual labels of objects."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=10,
    )

    content = resp.choices[0].message.content.strip()
    # Attempt to parse into the model for validation
    try:
        parsed = LLMNormalizeResponse(normalized_type=content)
        return parsed.normalized_type
    except ValidationError:
        # If validation fails, return the raw content
        return content


def decide_update(
    existing: Dict,
    incoming: ObjectRequest,
    metadata: Dict
) -> Tuple[str, str]:
    """
    Determines whether to update an existing record or keep the old one.
    Returns a tuple: (decision, reason), where decision is "update" or "keep".
    """
    # Prepare description for the prompt
    existing_ts = existing.get("timestamp")
    new_ts = incoming.timestamp.isoformat()
    season = incoming.timestamp.strftime("%B")  # e.g., "July"
    hour = incoming.timestamp.hour
    time_of_day = (
        "night" if hour < 6 or hour >= 22 else
        "morning" if hour < 12 else
        "afternoon" if hour < 18 else
        "evening"
    )
    prompt = (
        "You have information about a previously recorded object and new data for the same object.\n\n"
        "Existing data:\n"
        f"- ID: {existing.get('id')}\n"
        f"- Type: {existing.get('type')}\n"
        f"- Timestamp: {existing_ts}\n"
        f"- BBox: {existing.get('bbox')}\n\n"
        "New data:\n"
        f"- ID: {incoming.id}\n"
        f"- Capture time: {new_ts} (season: {season}, time of day: {time_of_day})\n"
        f"- Type (normalized): {incoming.type}\n"
        f"- BBox: {incoming.bbox}\n"
        f"- Points (count): {len(incoming.pointcloud)}\n\n"
        "Additional metadata:\n"
        + "\n".join(f"- {k}: {v}" for k, v in metadata.items())
        + "\n\n"
        "Decide whether to UPDATE the record (update) or keep the existing one (keep). "
        "Return the response in JSON format with the following fields:\n"
        '```\n'
        '{\n'
        '  "decision": "update" or "keep",\n'
        '  "reason": "brief justification"\n'
        '}\n'
        '```'
    )

    resp = openai.ChatCompletion.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You assist with making a decision about updating a 3D object in the database."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=150,
    )

    content = resp.choices[0].message.content.strip()
    # Try to parse JSON from the model
    try:
        data = json.loads(content)
        parsed = LLMDecisionResponse(**data)
        return parsed.decision, parsed.reason
    except (json.JSONDecodeError, ValidationError):
        # If parsing fails, return "keep" with the raw model explanation
        return "keep", content


def filter_types(available_types: List[str], condition: str) -> LLMFilterResponse:
    """
    Uses the LLM to split available_types into included/excluded according to the textual condition.
    """
    prompt = (
        f"You have the following list of object TYPE labels:\n"
        f"{available_types}\n\n"
        f"Based on this condition, select which types SHOULD be included and which SHOULD be excluded.\n"
        f"Condition: \"{condition}\"\n\n"
        f"Return JSON with two arrays:\n"
        "```\n"
        "{\n"
        "  \"included\": [\"Type1\", \"Type2\", ...],\n"
        "  \"excluded\": [\"TypeA\", \"TypeB\", ...]\n"
        "}\n"
        "```"
    )
    resp = openai.ChatCompletion.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You split a list of labels into included and excluded based on a condition."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    content = resp.choices[0].message.content.strip()
    try:
        data = json.loads(content)
        return LLMFilterResponse(**data)
    except (json.JSONDecodeError, ValidationError):
        return LLMFilterResponse(
            included=[t for t in available_types if condition.lower() in t.lower()],
            excluded=[t for t in available_types if condition.lower() not in t.lower()]
        )

def generate_filter_expression(condition: str) -> str:
    """
    Ask the LLM to produce a Milvus query expression (e.g. `type in ("Car","Bus")`)
    based on the textual condition. Returns a valid Milvus boolean expression.
    """
    prompt = (
        f"You need to write a Milvus filter expression over the field `type` "
        f"that captures this condition:\n\n"
        f"Condition: \"{condition}\"\n\n"
        "Return ONLY the boolean expression, for example:\n"
        "`type in (\"Car\",\"Truck\") and type != \"Tree\"`"
    )
    resp = openai.ChatCompletion.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You produce a Milvus query filter."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=500,
    )
    expr = resp.choices[0].message.content.strip().strip("`")
    return expr