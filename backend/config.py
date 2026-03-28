from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    openai_summarizer_model: str
    openai_extractor_model: str

    loading_openai_model: Optional[str]
    loading_openai_summarizer_model: Optional[str]
    loading_openai_extractor_model: Optional[str]

    system_prompt: str

    openai_embedding_model: str
    openai_embedding_dimensions: int

    cosmos_endpoint: str
    cosmos_key: str
    cosmos_database: str

    cosmos_container_direct_llm: str
    cosmos_container_sliding_window: str
    cosmos_container_hierarchical: str
    cosmos_container_entity_graph_sessions: str
    cosmos_container_entity_graph_entities: str

    cosmos_enable_vector_search: bool
    cosmos_enable_full_text_search: bool
    cosmos_verify_ssl: bool


def get_settings() -> Settings:
    endpoint = os.getenv("COSMOS_ENDPOINT", "")
    default_verify = not ("localhost" in endpoint or "127.0.0.1" in endpoint)
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4"),
        openai_summarizer_model=os.getenv("OPENAI_SUMMARIZER_MODEL", "gpt-5.4"),
        openai_extractor_model=os.getenv("OPENAI_EXTRACTOR_MODEL", "gpt-5.4"),
        loading_openai_model=os.getenv("LOADING_OPENAI_MODEL") or None,
        loading_openai_summarizer_model=os.getenv("LOADING_OPENAI_SUMMARIZER_MODEL") or None,
        loading_openai_extractor_model=os.getenv("LOADING_OPENAI_EXTRACTOR_MODEL") or None,
        system_prompt=os.getenv(
            "SYSTEM_PROMPT",
            "You are a helpful assistant with persistent memory.\n"
            "Use the injected memory context to answer questions about prior turns.\n"
            "Only state details that are explicitly present in the conversation or injected memory; do not invent facts.\n"
            "If the needed information is missing, say you don't know and ask for the missing detail.",
        ),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_embedding_dimensions=_get_env_int("OPENAI_EMBEDDING_DIMENSIONS", 1536),
        cosmos_endpoint=endpoint,
        cosmos_key=os.getenv("COSMOS_KEY", ""),
        cosmos_database=os.getenv("COSMOS_DATABASE", "acdconf2"),
        cosmos_container_direct_llm=os.getenv(
            "COSMOS_CONTAINER_DIRECT_LLM", "direct_llm_sessions"
        ),
        cosmos_container_sliding_window=os.getenv(
            "COSMOS_CONTAINER_SLIDING_WINDOW", "sliding_window_sessions"
        ),
        cosmos_container_hierarchical=os.getenv(
            "COSMOS_CONTAINER_HIERARCHICAL", "hierarchical_sessions"
        ),
        cosmos_container_entity_graph_sessions=os.getenv(
            "COSMOS_CONTAINER_ENTITY_GRAPH_SESSIONS", "entity_graph_sessions"
        ),
        cosmos_container_entity_graph_entities=os.getenv(
            "COSMOS_CONTAINER_ENTITY_GRAPH_ENTITIES", "entity_graph_entities"
        ),
        cosmos_enable_vector_search=_get_env_bool("COSMOS_ENABLE_VECTOR_SEARCH", True),
        cosmos_enable_full_text_search=_get_env_bool("COSMOS_ENABLE_FULL_TEXT_SEARCH", False),
        cosmos_verify_ssl=_get_env_bool("COSMOS_VERIFY_SSL", default_verify),
    )


def get_chat_model(*, use_loading_models: bool) -> str:
    s = get_settings()
    if use_loading_models and s.loading_openai_model:
        return s.loading_openai_model
    return s.openai_model


def get_summarizer_model(*, use_loading_models: bool) -> str:
    s = get_settings()
    if use_loading_models and s.loading_openai_summarizer_model:
        return s.loading_openai_summarizer_model
    return s.openai_summarizer_model


def get_extractor_model(*, use_loading_models: bool) -> str:
    s = get_settings()
    if use_loading_models and s.loading_openai_extractor_model:
        return s.loading_openai_extractor_model
    return s.openai_extractor_model
