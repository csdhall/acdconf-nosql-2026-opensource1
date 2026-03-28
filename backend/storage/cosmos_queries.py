from __future__ import annotations

from typing import Any, Dict, List

from azure.cosmos.exceptions import CosmosResourceNotFoundError


async def read_or_default(container: Any, session_id: str, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        item = await container.read_item(item=session_id, partition_key=session_id)
        return dict(item)
    except CosmosResourceNotFoundError:
        return default


async def upsert_item(container: Any, doc: Dict[str, Any]) -> Dict[str, Any]:
    return await container.upsert_item(doc)


async def list_sessions(container: Any) -> List[Dict[str, Any]]:
    query = "SELECT TOP 50 c.session_id, c.turn_count FROM c WHERE c.id = c.session_id ORDER BY c.turn_count DESC"
    rows = [row async for row in container.query_items(query=query)]
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append({"session_id": row.get("session_id"), "turn_count": row.get("turn_count", 0)})
    return out


async def delete_all_items_for_session(container: Any, session_id: str) -> int:
    rows = [
        row
        async for row in container.query_items(
            query="SELECT c.id FROM c",
            partition_key=session_id,
        )
    ]
    for row in rows:
        await container.delete_item(item=row["id"], partition_key=session_id)
    return len(rows)


async def delete_entity_graph_entities_for_session(entities_container: Any, session_id: str) -> None:
    query = "SELECT c.id FROM c"
    rows = [
        row
        async for row in entities_container.query_items(
            query=query, partition_key=session_id
        )
    ]
    for row in rows:
        await entities_container.delete_item(item=row["id"], partition_key=session_id)


async def delete_all_items_in_container(container: Any) -> None:
    query = "SELECT c.id, c.session_id FROM c"
    rows = [row async for row in container.query_items(query=query)]
    for row in rows:
        await container.delete_item(item=row["id"], partition_key=row["session_id"])
