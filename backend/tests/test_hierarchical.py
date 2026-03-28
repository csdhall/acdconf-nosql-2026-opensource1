from __future__ import annotations

import re
import unittest
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

from azure.cosmos.exceptions import CosmosResourceNotFoundError


class _InMemoryContainer:
    def __init__(self) -> None:
        self.docs: Dict[tuple[str, str], Dict[str, Any]] = {}

    async def read_item(self, *, item: str, partition_key: str) -> Dict[str, Any]:
        doc = self.docs.get((partition_key, item))
        if doc is None:
            raise CosmosResourceNotFoundError()
        return dict(doc)

    async def upsert_item(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        session_id = str(doc.get("session_id") or "")
        doc_id = str(doc.get("id") or "")
        if not session_id or not doc_id:
            raise ValueError("Docs must have 'session_id' and 'id'")
        self.docs[(session_id, doc_id)] = dict(doc)
        return dict(doc)

    async def delete_item(self, *, item: str, partition_key: str) -> None:
        self.docs.pop((partition_key, item), None)

    async def query_items(
        self,
        *,
        query: str,
        parameters: List[Dict[str, Any]] | None = None,
        partition_key: str | None = None,
        **_: Any,
    ) -> Any:
        params = {p["name"]: p["value"] for p in (parameters or []) if "name" in p}
        sid = params.get("@sid") or partition_key

        candidates = [
            dict(doc)
            for (pk, _id), doc in self.docs.items()
            if partition_key is None or pk == partition_key
        ]

        if "doc_type = 'msg'" in query:
            candidates = [
                doc
                for doc in candidates
                if doc.get("session_id") == sid and doc.get("doc_type") == "msg"
            ]

            if "@start" in params and "@end" in params:
                start = int(params["@start"])
                end = int(params["@end"])
                in_range = [
                    doc
                    for doc in candidates
                    if start <= int(doc.get("seq", 0) or 0) < end
                ]
                in_range.sort(key=lambda d: int(d.get("seq", 0) or 0))
                for doc in in_range:
                    yield {"seq": doc.get("seq"), "role": doc.get("role"), "content": doc.get("content")}
                return

            m = re.search(r"SELECT TOP (\\d+)", query)
            limit = int(m.group(1)) if m else len(candidates)
            candidates.sort(key=lambda d: int(d.get("seq", 0) or 0), reverse=True)
            for doc in candidates[:limit]:
                yield {"seq": doc.get("seq"), "role": doc.get("role"), "content": doc.get("content")}
            return

        if "doc_type = 'tier2_summary'" in query:
            prefix = str(params.get("@prefix") or "/tier2/summary/")
            candidates = [
                doc
                for doc in candidates
                if doc.get("session_id") == sid and doc.get("doc_type") == "tier2_summary"
            ]
            candidates = [
                doc
                for doc in candidates
                if isinstance(doc.get("path"), str) and doc["path"].startswith(prefix)
            ]
            candidates.sort(key=lambda d: int(d.get("summary_index", 0) or 0))
            for doc in candidates:
                yield {"id": doc.get("id"), "summary_index": doc.get("summary_index"), "summary": doc.get("summary")}
            return

        if "SELECT c.id FROM c" in query:
            for doc in candidates:
                yield {"id": doc.get("id")}
            return


class HierarchicalFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_summaries_fallback_and_no_tier3_overwrite(self) -> None:
        from backend.strategies import hierarchical

        container = _InMemoryContainer()
        session_id = "s1"
        await container.upsert_item(
            {
                "id": "t3",
                "session_id": session_id,
                "doc_type": "tier3_facts",
                "path": "/tier3/facts",
                "facts": "existing facts",
            }
        )

        async def fake_create_chat_completion(
            _client: Any,
            *,
            model: str,
            messages: list[dict[str, str]],
            max_completion_tokens: int | None = None,
            response_format: dict[str, Any] | None = None,
        ) -> Tuple[str, Dict[str, int], float]:
            system_prompt = (messages[0].get("content") or "") if messages else ""
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            if system_prompt.startswith("Summarize this conversation block"):
                return "   ", usage, 0.0
            if system_prompt.startswith("Extract ALL important facts"):
                return "   ", usage, 0.0
            return "ok", usage, 0.0

        with (
            patch.object(hierarchical, "create_chat_completion", new=fake_create_chat_completion),
            patch.object(hierarchical, "TIER1_SIZE", 2),
            patch.object(hierarchical, "TIER2_BLOCK", 2),
            patch.object(hierarchical, "MAX_TIER2", 1),
        ):
            await hierarchical.chat(container, None, session_id, "m1", use_loading_models=True)
            await hierarchical.chat(container, None, session_id, "m2", use_loading_models=True)
            await hierarchical.chat(container, None, session_id, "m3", use_loading_models=True)

        mem = await hierarchical.get_memory(container, session_id)

        self.assertEqual(mem["tier3"], "existing facts")
        self.assertTrue(mem["tier2_summaries"])
        self.assertTrue(all(isinstance(s, str) and s.strip() for s in mem["tier2_summaries"]))
        self.assertIn("user: m2", mem["tier2_summaries"][0])
        self.assertIn("assistant: ok", mem["tier2_summaries"][0])


if __name__ == "__main__":
    unittest.main()
