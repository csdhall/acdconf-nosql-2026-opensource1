from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Containers:
    direct_llm: str
    sliding_window: str
    hierarchical: str
    entity_graph_sessions: str
    entity_graph_entities: str
