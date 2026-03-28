from __future__ import annotations

from typing import Any, Optional, Tuple

from azure.cosmos.aio import CosmosClient
from openai import AsyncOpenAI

from backend.config import get_settings
from backend.llm import is_mock_openai
from backend.models import Metrics, Strategy
from backend.storage.cosmos_queries import (
    delete_all_items_for_session,
    delete_all_items_in_container,
    list_sessions,
)
from backend.strategies import direct_llm, entity_graph, hierarchical, sliding_window


class CosmosStore:
    def __init__(self) -> None:
        self._cosmos: Optional[CosmosClient] = None
        self._db = None
        self._containers: dict[str, Any] = {}
        self._openai: Optional[AsyncOpenAI] = None

    async def open(self) -> None:
        s = get_settings()
        if not s.cosmos_endpoint or not s.cosmos_key:
            raise RuntimeError("COSMOS_ENDPOINT and COSMOS_KEY must be set")
        self._cosmos = CosmosClient(
            s.cosmos_endpoint,
            credential=s.cosmos_key,
            connection_verify=s.cosmos_verify_ssl,
        )
        self._db = self._cosmos.get_database_client(s.cosmos_database)
        self._containers = {
            "direct_llm": self._db.get_container_client(s.cosmos_container_direct_llm),
            "sliding_window": self._db.get_container_client(s.cosmos_container_sliding_window),
            "hierarchical": self._db.get_container_client(s.cosmos_container_hierarchical),
            "entity_graph_sessions": self._db.get_container_client(
                s.cosmos_container_entity_graph_sessions
            ),
            "entity_graph_entities": self._db.get_container_client(
                s.cosmos_container_entity_graph_entities
            ),
        }

        if not is_mock_openai():
            if not s.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY must be set (or set MOCK_OPENAI=true)")
            self._openai = AsyncOpenAI(api_key=s.openai_api_key)

    async def close(self) -> None:
        if self._cosmos is not None:
            await self._cosmos.close()
            self._cosmos = None
        if self._openai is not None:
            await self._openai.close()
            self._openai = None

    def _container_for_strategy(self, strategy: Strategy) -> Any:
        if strategy == Strategy.direct_llm:
            return self._containers["direct_llm"]
        if strategy == Strategy.sliding_window:
            return self._containers["sliding_window"]
        if strategy == Strategy.hierarchical:
            return self._containers["hierarchical"]
        if strategy == Strategy.entity_graph:
            return self._containers["entity_graph_sessions"]
        raise ValueError(f"Unknown strategy: {strategy}")

    async def chat(self, strategy: Strategy, session_id: str, message: str) -> Tuple[str, Metrics]:
        if self._cosmos is None:
            raise RuntimeError("CosmosStore not initialized")
        openai_client: Any = self._openai
        if strategy == Strategy.direct_llm:
            return await direct_llm.chat(
                self._containers["direct_llm"],
                openai_client,
                session_id,
                message,
                use_loading_models=False,
            )
        if strategy == Strategy.sliding_window:
            return await sliding_window.chat(
                self._containers["sliding_window"],
                openai_client,
                session_id,
                message,
                use_loading_models=False,
            )
        if strategy == Strategy.hierarchical:
            return await hierarchical.chat(
                self._containers["hierarchical"],
                openai_client,
                session_id,
                message,
                use_loading_models=False,
            )
        if strategy == Strategy.entity_graph:
            return await entity_graph.chat(
                self._containers["entity_graph_sessions"],
                self._containers["entity_graph_entities"],
                openai_client,
                session_id,
                message,
                use_loading_models=False,
            )
        raise ValueError(f"Unknown strategy: {strategy}")

    async def get_memory(self, strategy: Strategy, session_id: str) -> Any:
        if strategy == Strategy.direct_llm:
            return await direct_llm.get_memory(self._containers["direct_llm"], session_id)
        if strategy == Strategy.sliding_window:
            return await sliding_window.get_memory(self._containers["sliding_window"], session_id)
        if strategy == Strategy.hierarchical:
            return await hierarchical.get_memory(self._containers["hierarchical"], session_id)
        if strategy == Strategy.entity_graph:
            return await entity_graph.get_memory(
                self._containers["entity_graph_sessions"],
                self._containers["entity_graph_entities"],
                session_id,
            )
        raise ValueError(f"Unknown strategy: {strategy}")

    async def list_sessions(self, strategy: Strategy) -> Any:
        container = self._container_for_strategy(strategy)
        return await list_sessions(container)

    async def delete_session(self, strategy: Strategy, session_id: str) -> None:
        container = self._container_for_strategy(strategy)
        await delete_all_items_for_session(container, session_id)
        if strategy == Strategy.entity_graph:
            await delete_all_items_for_session(self._containers["entity_graph_entities"], session_id)

    async def delete_all_data(self) -> None:
        for container in self._containers.values():
            await delete_all_items_in_container(container)
