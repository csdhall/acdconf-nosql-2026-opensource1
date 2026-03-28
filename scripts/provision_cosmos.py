from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from azure.cosmos import PartitionKey
from azure.cosmos.aio import CosmosClient
from azure.cosmos.exceptions import CosmosHttpResponseError

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.config import get_settings


def _extract_vector_embedding_policy(container_props: dict) -> dict | None:
    return container_props.get("vectorEmbeddingPolicy") or container_props.get("vector_embedding_policy")


def _extract_full_text_policy(container_props: dict) -> dict | None:
    return container_props.get("fullTextPolicy") or container_props.get("full_text_policy")


def _validate_entity_container_config(
    *,
    container_props: dict,
    expect_vector: bool,
    expect_full_text: bool,
    expected_dimensions: int,
) -> None:
    indexing_policy = container_props.get("indexingPolicy") or {}
    if expect_vector:
        vector_indexes = indexing_policy.get("vectorIndexes") or []
        if not any(ix.get("path") == "/embedding" for ix in vector_indexes):
            raise RuntimeError(
                "Container exists but is missing vectorIndexes for /embedding. "
                "Delete and recreate the container or set COSMOS_ENABLE_VECTOR_SEARCH=false."
            )

        embedding_policy = _extract_vector_embedding_policy(container_props) or {}
        vector_embeddings = embedding_policy.get("vectorEmbeddings") or []
        embedding = next((e for e in vector_embeddings if e.get("path") == "/embedding"), None)
        if not embedding:
            raise RuntimeError(
                "Container exists but is missing vectorEmbeddingPolicy for /embedding. "
                "Delete and recreate the container or set COSMOS_ENABLE_VECTOR_SEARCH=false."
            )
        if int(embedding.get("dimensions", -1)) != int(expected_dimensions):
            raise RuntimeError(
                f"Container vectorEmbeddingPolicy dimensions mismatch: "
                f"expected {expected_dimensions}, got {embedding.get('dimensions')}. "
                "Delete and recreate the container with the correct dimensions."
            )

    if expect_full_text:
        full_text_indexes = indexing_policy.get("fullTextIndexes") or []
        if not any(ix.get("path") == "/searchText" for ix in full_text_indexes):
            raise RuntimeError(
                "Container exists but is missing fullTextIndexes for /searchText. "
                "Delete and recreate the container or set COSMOS_ENABLE_FULL_TEXT_SEARCH=false."
            )
        full_text_policy = _extract_full_text_policy(container_props) or {}
        full_text_paths = full_text_policy.get("fullTextPaths") or []
        if not any(p.get("path") == "/searchText" for p in full_text_paths):
            raise RuntimeError(
                "Container exists but is missing fullTextPolicy fullTextPaths for /searchText. "
                "Delete and recreate the container or set COSMOS_ENABLE_FULL_TEXT_SEARCH=false."
            )


async def _provision() -> None:
    s = get_settings()
    if not s.cosmos_endpoint or not s.cosmos_key:
        raise RuntimeError("COSMOS_ENDPOINT and COSMOS_KEY must be set")

    client = CosmosClient(
        s.cosmos_endpoint,
        credential=s.cosmos_key,
        connection_verify=s.cosmos_verify_ssl,
    )
    try:
        db = await client.create_database_if_not_exists(id=s.cosmos_database)

        await db.create_container_if_not_exists(
            id=s.cosmos_container_direct_llm,
            partition_key=PartitionKey(path="/session_id"),
            default_ttl=-1,
        )
        await db.create_container_if_not_exists(
            id=s.cosmos_container_sliding_window,
            partition_key=PartitionKey(path="/session_id"),
            default_ttl=-1,
        )
        await db.create_container_if_not_exists(
            id=s.cosmos_container_hierarchical,
            partition_key=PartitionKey(path="/session_id"),
            default_ttl=-1,
        )
        await db.create_container_if_not_exists(
            id=s.cosmos_container_entity_graph_sessions,
            partition_key=PartitionKey(path="/session_id"),
            default_ttl=-1,
        )

        entity_indexing_policy: dict | None = None
        entity_vector_embedding_policy: dict | None = None
        entity_full_text_policy: dict | None = None

        if s.cosmos_enable_vector_search or s.cosmos_enable_full_text_search:
            excluded_paths = [{"path": "/_etag/?"}]
            if s.cosmos_enable_vector_search:
                excluded_paths.append({"path": "/embedding/*"})

            entity_indexing_policy = {
                "indexingMode": "consistent",
                "automatic": True,
                "includedPaths": [{"path": "/*"}],
                "excludedPaths": excluded_paths,
            }
            if s.cosmos_enable_vector_search:
                entity_indexing_policy["vectorIndexes"] = [{"path": "/embedding", "type": "diskANN"}]
                entity_vector_embedding_policy = {
                    "vectorEmbeddings": [
                        {
                            "path": "/embedding",
                            "dataType": "float32",
                            "distanceFunction": "cosine",
                            "dimensions": s.openai_embedding_dimensions,
                        }
                    ]
                }
            if s.cosmos_enable_full_text_search:
                entity_indexing_policy["fullTextIndexes"] = [{"path": "/searchText"}]
                entity_full_text_policy = {
                    "defaultLanguage": "en-US",
                    "fullTextPaths": [{"path": "/searchText", "language": "en-US"}],
                }

        create_kwargs: dict = {}
        if entity_indexing_policy is not None:
            create_kwargs["indexing_policy"] = entity_indexing_policy
        if entity_vector_embedding_policy is not None:
            create_kwargs["vector_embedding_policy"] = entity_vector_embedding_policy
        if entity_full_text_policy is not None:
            create_kwargs["full_text_policy"] = entity_full_text_policy

        entities_container = await db.create_container_if_not_exists(
            id=s.cosmos_container_entity_graph_entities,
            partition_key=PartitionKey(path="/session_id"),
            default_ttl=-1,
            **create_kwargs,
        )

        try:
            props = await entities_container.read()
        except CosmosHttpResponseError as e:
            raise RuntimeError(f"Failed to read container properties: {e}") from e

        _validate_entity_container_config(
            container_props=props,
            expect_vector=s.cosmos_enable_vector_search,
            expect_full_text=s.cosmos_enable_full_text_search,
            expected_dimensions=s.openai_embedding_dimensions,
        )

        print("Cosmos provisioning complete.")
    finally:
        await client.close()


def main() -> None:
    asyncio.run(_provision())


if __name__ == "__main__":
    main()
