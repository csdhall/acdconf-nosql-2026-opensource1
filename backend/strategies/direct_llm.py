from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

from backend.config import get_chat_model, get_settings
from backend.llm import create_chat_completion
from backend.models import Metrics
from backend.storage.cosmos_queries import read_or_default, upsert_item


CHAT_MAX_COMPLETION_TOKENS = 600


async def chat(
    sessions_container: Any,
    openai_client: Any,
    session_id: str,
    user_message: str,
    *,
    use_loading_models: bool,
) -> Tuple[str, Metrics]:
    session = await read_or_default(
        sessions_container,
        session_id,
        default={
            "id": session_id,
            "session_id": session_id,
            "doc_type": "session",
            "turn_count": 0,
            "last_ts": 0,
        },
    )

    session["turn_count"] = int(session.get("turn_count", 0) or 0) + 1
    session["last_ts"] = int(time.time())

    settings = get_settings()
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": settings.system_prompt},
        {"role": "user", "content": user_message},
    ]

    reply, usage, latency_ms = await create_chat_completion(
        openai_client,
        model=get_chat_model(use_loading_models=use_loading_models),
        messages=messages,
        max_completion_tokens=CHAT_MAX_COMPLETION_TOKENS,
    )

    session["doc_type"] = "session"
    await upsert_item(sessions_container, session)

    metrics = Metrics(
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        latency_ms=latency_ms,
        memory_turns_stored=int(session["turn_count"]),
        context_turns_sent=1,
    )
    return reply, metrics


async def get_memory(sessions_container: Any, session_id: str) -> Dict[str, Any]:
    session = await read_or_default(
        sessions_container,
        session_id,
        default={
            "id": session_id,
            "session_id": session_id,
            "doc_type": "session",
            "turn_count": 0,
            "last_ts": 0,
        },
    )
    return {
        "mode": "direct_llm_no_memory",
        "recent_turns": [],
        "summary": None,
        "turn_count": int(session.get("turn_count", 0) or 0),
    }
