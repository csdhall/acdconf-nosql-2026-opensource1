from __future__ import annotations

from typing import Any, Dict, List, Optional


async def upsert_message(
    container: Any,
    *,
    session_id: str,
    seq: int,
    role: str,
    content: str,
    ts: int,
    ttl_s: Optional[int] = None,
) -> None:
    doc: Dict[str, Any] = {
        "id": f"msg:{seq}",
        "session_id": session_id,
        "doc_type": "msg",
        "seq": int(seq),
        "ts": int(ts),
        "role": role,
        "content": content,
    }
    if ttl_s is not None:
        doc["ttl"] = int(ttl_s)
    await container.upsert_item(doc)


async def read_recent_messages(
    container: Any,
    *,
    session_id: str,
    limit: int,
) -> List[Dict[str, str]]:
    query = (
        f"SELECT TOP {int(limit)} c.seq, c.role, c.content FROM c "
        "WHERE c.session_id = @sid AND c.doc_type = 'msg' "
        "ORDER BY c.seq DESC"
    )
    rows = [
        row
        async for row in container.query_items(
            query=query,
            parameters=[{"name": "@sid", "value": session_id}],
            partition_key=session_id,
        )
    ]
    rows.sort(key=lambda r: int(r.get("seq", 0)))
    out: List[Dict[str, str]] = []
    for row in rows:
        role = row.get("role")
        content = row.get("content")
        if isinstance(role, str) and isinstance(content, str):
            out.append({"role": role, "content": content})
    return out


async def read_messages_by_seq_range(
    container: Any,
    *,
    session_id: str,
    start_seq: int,
    end_seq: int,
) -> List[Dict[str, str]]:
    if end_seq <= start_seq:
        return []
    query = (
        "SELECT c.seq, c.role, c.content FROM c "
        "WHERE c.session_id = @sid AND c.doc_type = 'msg' "
        "AND c.seq >= @start AND c.seq < @end "
        "ORDER BY c.seq ASC"
    )
    rows = [
        row
        async for row in container.query_items(
            query=query,
            parameters=[
                {"name": "@sid", "value": session_id},
                {"name": "@start", "value": int(start_seq)},
                {"name": "@end", "value": int(end_seq)},
            ],
            partition_key=session_id,
        )
    ]
    out: List[Dict[str, str]] = []
    for row in rows:
        role = row.get("role")
        content = row.get("content")
        if isinstance(role, str) and isinstance(content, str):
            out.append({"role": role, "content": content})
    return out

