import json
from typing import List, Dict, Any

from pymilvus import (
    connections,
    FieldSchema, CollectionSchema,
    DataType, Collection, utility
)

from .config import settings

connections.connect(host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)

COLLECTION_NAME = "object_vectors"

def ensure_collection() -> None:
    """
    Create the collection
    fields :
      - id: primary key (VARCHAR)
      - embedding: FLOAT_VECTOR
      - metadata: VARCHAR (JSON-str)
    """
    if utility.has_collection(COLLECTION_NAME):
        return

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=64
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=settings.VECTOR_DIM
        ),
        FieldSchema(
            name="metadata",
            dtype=DataType.VARCHAR,
            max_length=8192
        )
    ]
    schema = CollectionSchema(fields, description="Collection of 3D object embeddings")
    Collection(name=COLLECTION_NAME, schema=schema, consistency_level="Strong")


def insert_vector(
    id: str,
    vector: List[float],
    metadata: Dict[str, Any]
) -> None:
    ensure_collection()
    collection = Collection(COLLECTION_NAME)
    metadata_str = json.dumps(metadata)
    ids = [id]
    vecs = [vector]
    metas = [metadata_str]
    collection.insert([ids, vecs, metas])
    collection.flush()


def search_vector(
    vector: List[float],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    ensure_collection()
    collection = Collection(COLLECTION_NAME)
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }
    results = collection.search(
        data=[vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["id", "metadata"]
    )

    processed: List[Dict[str, Any]] = []
    for hits in results:
        for hit in hits:
            entity = hit.entity
            obj_id = entity.id
            raw_meta = entity.metadata
            try:
                meta = json.loads(raw_meta)
            except (TypeError, json.JSONDecodeError):
                meta = {}
            processed.append({
                "id": obj_id,
                "distance": hit.distance,
                "metadata": meta
            })
    return processed


def get_all_distinct_types() -> List[str]:
    """
    Returns all distinct 'type' values currently stored in Milvus metadata.
    Warning: this does a full scan of metadata.
    """
    ensure_collection()
    collection = Collection(COLLECTION_NAME)
    results = collection.query(expr="", output_fields=["metadata"])
    types = set()
    for row in results:
        meta = json.loads(row["metadata"])
        types.add(meta.get("type"))
    return list(types)


def query_ids_by_types(types: List[str]) -> List[str]:
    """
    Returns all object IDs whose metadata.type is in the provided list.
    """
    ensure_collection()
    collection = Collection(COLLECTION_NAME)
    # build Milvus expression: e.g. 'metadata["type"] in ["Car","Tree"]'
    expr = " or ".join(f'type == "{t}"' for t in types)
    results = collection.query(expr=expr, output_fields=["id"])
    return [row["id"] for row in results]


def query_ids_excluding_types(types: List[str]) -> List[str]:
    """
    Returns all object IDs whose metadata.type is NOT in the provided list.
    """
    ensure_collection()
    collection = Collection(COLLECTION_NAME)
    expr = " and ".join(f'type != "{t}"' for t in types)
    results = collection.query(expr=expr, output_fields=["id"])
    return [row["id"] for row in results]

def stream_ids_by_expression(expr: str, batch_size: int = 1000) -> Iterator[str]:
    """
    Yield object IDs matching the Milvus expression in batches.
    """
    ensure_collection()
    collection = Collection(COLLECTION_NAME)
    # query in pages by using offset+limit
    offset = 0
    while True:
        results = collection.query(
            expr=expr,
            output_fields=["id"],
            limit=batch_size,
            offset=offset
        )
        if not results:
            break
        for row in results:
            yield row["id"]
        offset += batch_size