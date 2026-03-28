from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class Strategy(str, Enum):
    direct_llm = "direct_llm"
    sliding_window = "sliding_window"
    hierarchical = "hierarchical"
    entity_graph = "entity_graph"


class ChatRequest(BaseModel):
    session_id: str
    message: str
    strategy: Strategy


class Metrics(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    memory_turns_stored: int
    context_turns_sent: int


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    strategy: Strategy
    metrics: Metrics
