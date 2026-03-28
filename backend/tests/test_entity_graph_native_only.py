from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch


@dataclass(frozen=True)
class _SettingsStub:
    cosmos_enable_vector_search: bool
    cosmos_enable_full_text_search: bool
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536


class _DummyContainer:
    async def query_items(self, *args, **kwargs):  # pragma: no cover - should not be called in this test
        if False:
            yield {}


class EntityGraphNativeOnlyTests(unittest.IsolatedAsyncioTestCase):
    async def test_retrieve_entities_requires_native_cosmos_search(self) -> None:
        from backend.strategies import entity_graph

        with patch.object(
            entity_graph,
            "get_settings",
            new=lambda: _SettingsStub(
                cosmos_enable_vector_search=False,
                cosmos_enable_full_text_search=False,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "requires native Cosmos retrieval"):
                await entity_graph._retrieve_entities(
                    _DummyContainer(),
                    None,
                    session_id="s1",
                    query_text="What is our tech stack?",
                )


if __name__ == "__main__":
    unittest.main()
