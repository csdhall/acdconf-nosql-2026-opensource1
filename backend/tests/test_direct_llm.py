from __future__ import annotations

import unittest
from typing import Any, Dict, Tuple
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
        sid = str(doc.get("session_id") or "")
        doc_id = str(doc.get("id") or "")
        if not sid or not doc_id:
            raise ValueError("Docs must include session_id and id")
        self.docs[(sid, doc_id)] = dict(doc)
        return dict(doc)


class DirectLlmTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_updates_session_and_memory(self) -> None:
        from backend.strategies import direct_llm

        container = _InMemoryContainer()

        async def fake_create_chat_completion(
            _client: Any,
            *,
            model: str,
            messages: list[dict[str, str]],
            max_completion_tokens: int | None = None,
            response_format: dict[str, Any] | None = None,
        ) -> Tuple[str, Dict[str, int], float]:
            self.assertEqual(model, "gpt-test")
            self.assertEqual(len(messages), 2)
            usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
            return "baseline reply", usage, 12.5

        with (
            patch.object(direct_llm, "create_chat_completion", new=fake_create_chat_completion),
            patch.object(direct_llm, "get_chat_model", new=lambda **_: "gpt-test"),
        ):
            reply, metrics = await direct_llm.chat(
                container, None, "session-1", "Hello there", use_loading_models=True
            )

        self.assertEqual(reply, "baseline reply")
        self.assertEqual(metrics.context_turns_sent, 1)
        self.assertEqual(metrics.total_tokens, 18)

        mem = await direct_llm.get_memory(container, "session-1")
        self.assertEqual(mem.get("mode"), "direct_llm_no_memory")
        self.assertEqual(mem.get("turn_count"), 1)
        self.assertEqual(mem.get("recent_turns"), [])


if __name__ == "__main__":
    unittest.main()
