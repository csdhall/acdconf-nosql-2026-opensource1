from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, List

from azure.cosmos.aio import CosmosClient

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.config import get_settings
from backend.storage.cosmos_queries import delete_all_items_in_container


async def _delete_test_only(container: Any) -> int:
    query = "SELECT c.id, c.session_id FROM c WHERE STARTSWITH(c.session_id, 'test-')"
    rows = [row async for row in container.query_items(query=query)]
    for row in rows:
        await container.delete_item(item=row["id"], partition_key=row["session_id"])
    return len(rows)


async def _run(*, test_only: bool) -> None:
    s = get_settings()
    if not s.cosmos_endpoint or not s.cosmos_key:
        raise RuntimeError("COSMOS_ENDPOINT and COSMOS_KEY must be set")

    cosmos = CosmosClient(
        s.cosmos_endpoint,
        credential=s.cosmos_key,
        connection_verify=s.cosmos_verify_ssl,
    )
    try:
        db = cosmos.get_database_client(s.cosmos_database)
        containers: List[Any] = [
            db.get_container_client(s.cosmos_container_direct_llm),
            db.get_container_client(s.cosmos_container_sliding_window),
            db.get_container_client(s.cosmos_container_hierarchical),
            db.get_container_client(s.cosmos_container_entity_graph_sessions),
            db.get_container_client(s.cosmos_container_entity_graph_entities),
        ]

        if test_only:
            deleted = 0
            for c in containers:
                deleted += await _delete_test_only(c)
            print(f"Deleted {deleted} test items.")
            return

        for c in containers:
            await delete_all_items_in_container(c)
        print("Deleted ALL data from all containers.")
    finally:
        await cosmos.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete Cosmos test data or all data.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Delete only test-* sessions across all containers (default: delete everything).",
    )
    args = parser.parse_args()
    asyncio.run(_run(test_only=args.test))


if __name__ == "__main__":
    main()
