from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError

from backend.config import (
    get_chat_model,
    get_extractor_model,
    get_settings,
)
from backend.llm import create_chat_completion, create_embedding, parse_json_object
from backend.models import Metrics
from backend.storage.cosmos_queries import read_or_default, upsert_item
from backend.storage.cosmos_messages import read_recent_messages, upsert_message


RECENT_TURNS_KEPT = 3  # turn pairs
CHAT_MAX_COMPLETION_TOKENS = 600
MESSAGE_TTL_S = 60 * 60 * 6  # 6 hours

QUESTION_WORDS = {"what", "who", "where", "when", "which", "how", "tell", "remind", "remember", "recall"}
QUERY_STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "to",
    "for",
    "of",
    "and",
    "or",
    "in",
    "on",
    "at",
    "with",
    "our",
    "my",
    "your",
    "what",
    "who",
    "when",
    "where",
    "which",
    "how",
    "tell",
    "about",
    "including",
}
USER_IDENTITY_PROMPT = """Analyze this message. ONLY extract a name if the user is EXPLICITLY introducing THEMSELVES.
Return JSON: {"user_name": "Their Name"} or {"user_name": null}.

POSITIVE examples (user IS introducing themselves):
- "Hi, I'm Jordan Park" → {"user_name": "Jordan Park"}
- "My name is John Doe" → {"user_name": "John Doe"}
- "I am Jane Doe, a developer" → {"user_name": "Jane Doe"}
- "This is Jordan speaking" → {"user_name": "Jordan"}

NEGATIVE examples (user is NOT introducing themselves - return null):
- "What's the weather?" → {"user_name": null}
- "I work at FakeCompany" → {"user_name": null}
- "Tell me about Mary Doe" → {"user_name": null}
- "Mary Doe?" → {"user_name": null}
- "How is John doing?" → {"user_name": null}
- "What about Jack Doe?" → {"user_name": null}
- "Who is Jane Doe?" → {"user_name": null}
- "Mary Doe needs access" → {"user_name": null}
- "John said..." → {"user_name": null}

CRITICAL: Asking ABOUT someone is NOT the same as BEING that person.
Only return a name when the user says "I am X" or "My name is X" or "I'm X".
Return ONLY valid JSON.
"""

EXTRACTION_PROMPT = """Extract entities and facts from the USER message in this turn. Return JSON:
{
  "entities": [
    {
      "name": "entity name (person, place, organization, URL, project, concept, preference)",
      "type": "person|place|organization|preference|fact|project|date|url|concept",
      "facts": ["fact 1 about this entity", "fact 2"],
      "related_to": ["other entity names this is related to"]
    }
  ]
}
IMPORTANT:
- ONLY use information explicitly stated by the user. Ignore assistant text entirely.
- When the user says "I'm [name]" or "my name is [name]", extract that as a person entity with their actual name (NOT "User").
- When the user says "I work at [company]", extract the company AND add "works at [company]" to the person's facts.
- When the user mentions a repository/URL (example: "github.com/org/repo"), extract the URL EXACTLY as written:
  - Create a `url` entity named the URL with a fact like "project repo URL" (or what the URL is for).
  - If the related project name is present in the SAME turn, also add the URL as a fact on that `project` entity.
- Extract ALL clearly stated people, companies/organizations, projects, dates/deadlines, budgets/costs, SLAs, numeric requirements, technologies/tools/cloud services, integrations/platforms, preferences, locations, and URLs mentioned.
- For technology/tool/service names (React, FastAPI, PostgreSQL, AWS, EKS, Kafka, etc.), use `type: "concept"` (or `"fact"` if needed) and include a short role in `facts` (e.g., "frontend", "backend", "database", "cloud", "monitoring", "CI/CD", "auth", "streaming").
- When the user states multiple technologies with roles (especially frontend/backend/database), ALWAYS add a synthetic concept entity:
  - `name`: "Tech Stack"
  - `type`: "concept"
  - `facts`: role-tagged entries such as "frontend: React", "backend: FastAPI", "database: PostgreSQL"
  - `related_to`: the referenced technology names
- ALWAYS capture numeric values exactly as written (currency, percentages, counts, dates, and units like "/month"):
  - Example: "$3,500/month" (monthly AWS cost) → a `fact` entity with a fact like "estimated monthly AWS cost" and related_to ["Tom Doe", "AWS"].
  - Example: "87% accuracy" → add a fact on the relevant model/person/project entity (e.g., "churn prediction model achieved 87% accuracy").
- Only extract clearly stated information. If no entities found, return {"entities": []}.
Return ONLY valid JSON, no markdown.
"""


def _round_up_to_even(n: int) -> int:
    return n if n % 2 == 0 else n + 1


def _trim_pairs_to_last(messages: List[Dict[str, str]], max_messages: int) -> List[Dict[str, str]]:
    if len(messages) <= max_messages:
        return messages
    overflow = _round_up_to_even(len(messages) - max_messages)
    return messages[overflow:]


def _is_question(text: str) -> bool:
    if text.strip().endswith("?"):
        return True
    words = {w.lower() for w in re.findall(r"\b\w+\b", text.lower())}
    return bool(words & QUESTION_WORDS)


def _top_k_for_query(text: str) -> int:
    return 30 if _is_question(text) else 15


def _extract_query_keywords(query_text: str) -> List[str]:
    words = [w.lower() for w in re.findall(r"[a-zA-Z0-9$%]+", query_text or "")]
    out: List[str] = []
    seen = set()
    for w in words:
        if w in QUERY_STOP_WORDS:
            continue
        if len(w) < 3 and not any(ch.isdigit() for ch in w):
            continue
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= 6:
            break
    return out


def _merge_entity_rows(primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]], *, limit: int) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for source in (primary, secondary):
        for row in source:
            rid = str(row.get("id") or "")
            if not rid:
                rid = f'{row.get("name", "")}::{row.get("type", "")}'
            if rid in seen:
                continue
            seen.add(rid)
            merged.append(row)
            if len(merged) >= limit:
                return merged
    return merged


def _asks_for_self_identity(text: str) -> bool:
    q = (text or "").lower()
    if "my name" in q or "my role" in q:
        return True
    if "who am i" in q or "who i am" in q:
        return True
    return False


def _extract_hint_entities_from_user_message(user_message: str) -> List[Dict[str, Any]]:
    text = " ".join((user_message or "").split()).strip()
    if not text:
        return []

    hints: List[Dict[str, Any]] = []

    intro = re.search(
        r"(?:^|\b)(?:hi,\s*)?(?:i'?m|i am|my name is)\s+"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*(?:,|\band\b)?\s*"
        r"(?:(?:i'?m|i am)\s+)?(?:a|an)?\s*([^,.!?]+?)\s+at\s+([A-Z][A-Za-z0-9&._-]+)",
        text,
        re.IGNORECASE,
    )
    if intro:
        name = re.sub(r"\s+(?:and|&)$", "", intro.group(1).strip(), flags=re.IGNORECASE)
        role = re.sub(r"^(?:i'?m|i am)\s+", "", intro.group(2).strip(), flags=re.IGNORECASE)
        role = role.strip(" ,.")
        company = intro.group(3).strip().rstrip(".,;:")
        hints.append(
            {
                "name": name,
                "type": "person",
                "facts": [role, f"works at {company}"],
                "related_to": [company],
            }
        )
        hints.append(
            {
                "name": company,
                "type": "organization",
                "facts": ["organization"],
                "related_to": [name],
            }
        )

    pilot = re.search(r"pilot customers?\s+are\s*:\s*([^\n.]+)", text, re.IGNORECASE)
    if pilot:
        raw_names = pilot.group(1)
        names = [n.strip().strip(".") for n in re.split(r",|\band\b", raw_names, flags=re.IGNORECASE)]
        for name in names:
            if not name:
                continue
            hints.append(
                {
                    "name": name,
                    "type": "organization",
                    "facts": ["pilot customer", "soft launch pilot customer"],
                    "related_to": ["soft launch"],
                }
            )

    return hints


def _normalize_entity_key(name: str) -> str:
    s = " ".join((name or "").strip().split())
    s = s.strip(" \t\r\n\"'()[]{}<>.,;:")
    if not s:
        return ""
    s = re.sub(r"^https?://", "", s, flags=re.IGNORECASE)
    if re.search(r"[a-z0-9.-]+\.[a-z]{2,}", s, re.IGNORECASE):
        s = s.rstrip("/")
    s = s.lower()
    s = s.replace(" ", "_")
    s = re.sub(r"[\\\\/\\?#]", "|", s)
    return s


def _build_search_text(name: str, type_: str, facts: List[str], related_to: List[str]) -> str:
    parts: List[str] = []
    if name:
        parts.append(name)
    if type_:
        parts.append(type_)
    parts.extend([f for f in facts if f])
    parts.extend([r for r in related_to if r])
    text = " ".join(parts)
    text = " ".join(text.split()).strip().lower()
    return text


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clean_extracted_user_name(name: str) -> str:
    s = " ".join(name.strip().split())
    s = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", s).strip()
    s = re.sub(r"\b(and|&)$", "", s, flags=re.IGNORECASE).strip()
    return s


def _parse_json_object_loose(text: str) -> Dict[str, Any]:
    data = parse_json_object(text)
    if data:
        return data
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return parse_json_object(text[start : end + 1])
    return {}


async def _extract_json_with_fallback(
    openai_client: Any,
    *,
    primary_model: str,
    fallback_model: str,
    system_prompt: str,
    user_content: str,
    max_completion_tokens: int,
) -> Dict[str, Any]:
    async def _call(model: str, *, max_tokens: int) -> Dict[str, Any]:
        text, _, _ = await create_chat_completion(
            openai_client,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_completion_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return _parse_json_object_loose(text)

    data = await _call(primary_model, max_tokens=max_completion_tokens)
    if data:
        return data

    retry_tokens = min(max_completion_tokens * 2, 1200)
    data = await _call(primary_model, max_tokens=retry_tokens)
    if data:
        return data

    if fallback_model and fallback_model != primary_model:
        data = await _call(fallback_model, max_tokens=max_completion_tokens)
        if data:
            return data
        data = await _call(fallback_model, max_tokens=retry_tokens)
        if data:
            return data

    return {}


async def _read_entity(entities_container: Any, *, entity_id: str, session_id: str) -> Dict[str, Any] | None:
    try:
        item = await entities_container.read_item(item=entity_id, partition_key=session_id)
        return dict(item)
    except CosmosResourceNotFoundError:
        return None


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


async def _upsert_entity(
    entities_container: Any,
    openai_client: Any,
    *,
    session_id: str,
    entity: Dict[str, Any],
) -> None:
    settings = get_settings()

    name = (entity.get("name") or "").strip()
    type_ = (entity.get("type") or "").strip()
    facts = [str(f).strip() for f in (entity.get("facts") or []) if str(f).strip()]
    related_to = [str(r).strip() for r in (entity.get("related_to") or []) if str(r).strip()]

    entity_key = _normalize_entity_key(name)
    if not entity_key:
        return

    entity_id = f"{session_id}::{entity_key}"

    existing = await _read_entity(entities_container, entity_id=entity_id, session_id=session_id)
    if existing:
        merged_facts = _dedupe_preserve_order(list(existing.get("facts") or []) + facts)
        merged_related = _dedupe_preserve_order(list(existing.get("related_to") or []) + related_to)
        merged_type = type_ or existing.get("type") or "fact"
        merged_name = existing.get("name") or name or entity_key

        search_text = _build_search_text(merged_name, merged_type, merged_facts, merged_related)
        embedding = existing.get("embedding") or []
        if settings.cosmos_enable_vector_search and search_text != existing.get("searchText"):
            embedding = await create_embedding(
                openai_client,
                model=settings.openai_embedding_model,
                input_text=search_text,
                dimensions=settings.openai_embedding_dimensions,
            )

        doc = {
            **existing,
            "id": entity_id,
            "session_id": session_id,
            "name": merged_name,
            "type": merged_type,
            "facts": merged_facts,
            "related_to": merged_related,
            "searchText": search_text,
            "embedding": embedding if settings.cosmos_enable_vector_search else [],
            "updated_at": _utc_now_iso(),
        }
        await entities_container.upsert_item(doc)
        return

    search_text = _build_search_text(name, type_, facts, related_to)
    embedding: List[float] = []
    if settings.cosmos_enable_vector_search:
        embedding = await create_embedding(
            openai_client,
            model=settings.openai_embedding_model,
            input_text=search_text,
            dimensions=settings.openai_embedding_dimensions,
        )

    doc = {
        "id": entity_id,
        "session_id": session_id,
        "name": name,
        "type": type_ or "fact",
        "facts": _dedupe_preserve_order(facts),
        "related_to": _dedupe_preserve_order(related_to),
        "searchText": search_text,
        "embedding": embedding if settings.cosmos_enable_vector_search else [],
        "updated_at": _utc_now_iso(),
    }
    await entities_container.upsert_item(doc)


async def _retrieve_entities(
    entities_container: Any,
    openai_client: Any,
    *,
    session_id: str,
    query_text: str,
) -> List[Dict[str, Any]]:
    settings = get_settings()
    k = _top_k_for_query(query_text)

    vector_enabled = settings.cosmos_enable_vector_search
    full_text_enabled = settings.cosmos_enable_full_text_search
    if not vector_enabled and not full_text_enabled:
        raise RuntimeError(
            "Entity Graph requires native Cosmos retrieval. "
            "Set COSMOS_ENABLE_VECTOR_SEARCH=true (recommended) or COSMOS_ENABLE_FULL_TEXT_SEARCH=true."
        )

    try:
        if full_text_enabled and vector_enabled:
            query_vector = await create_embedding(
                openai_client,
                model=settings.openai_embedding_model,
                input_text=query_text,
                dimensions=settings.openai_embedding_dimensions,
            )
            query = (
                f"SELECT TOP {k} c.id, c.name, c.type, c.facts, c.related_to FROM c "
                "WHERE c.session_id = @sid "
                "ORDER BY RANK RRF(FULLTEXTSCORE(c.searchText, @searchString), VectorDistance(c.embedding, @queryVector))"
            )
            params = [
                {"name": "@sid", "value": session_id},
                {"name": "@searchString", "value": query_text},
                {"name": "@queryVector", "value": query_vector},
            ]
            rows = [
                row
                async for row in entities_container.query_items(
                    query=query,
                    parameters=params,
                    partition_key=session_id,
                )
            ]
        elif full_text_enabled:
            query = (
                f"SELECT TOP {k} c.id, c.name, c.type, c.facts, c.related_to FROM c "
                "WHERE c.session_id = @sid AND FULLTEXTCONTAINS(c.searchText, @phrase) "
                "ORDER BY RANK FULLTEXTSCORE(c.searchText, @phrase)"
            )
            params = [{"name": "@sid", "value": session_id}, {"name": "@phrase", "value": query_text}]
            rows = [
                row
                async for row in entities_container.query_items(
                    query=query,
                    parameters=params,
                    partition_key=session_id,
                )
            ]
        elif vector_enabled:
            query_vector = await create_embedding(
                openai_client,
                model=settings.openai_embedding_model,
                input_text=query_text,
                dimensions=settings.openai_embedding_dimensions,
            )
            query = (
                f"SELECT TOP {k} c.id, c.name, c.type, c.facts, c.related_to FROM c "
                "WHERE c.session_id = @sid "
                "ORDER BY VectorDistance(c.embedding, @queryVector)"
            )
            params = [{"name": "@sid", "value": session_id}, {"name": "@queryVector", "value": query_vector}]
            rows = [
                row
                async for row in entities_container.query_items(
                    query=query,
                    parameters=params,
                    partition_key=session_id,
                )
            ]
    except CosmosHttpResponseError as e:
        raise RuntimeError(
            "Cosmos native entity retrieval failed. "
            "Ensure vector/full-text capabilities are enabled and containers were provisioned with matching policies."
        ) from e

    keywords = _extract_query_keywords(query_text)
    if keywords:
        contains_terms = " OR ".join(
            [f"CONTAINS(c.searchText, @kw{i}, true)" for i in range(len(keywords))]
        )
        lexical_query = (
            f"SELECT TOP {k} c.id, c.name, c.type, c.facts, c.related_to FROM c "
            "WHERE c.session_id = @sid "
            f"AND ({contains_terms})"
        )
        lexical_params = [{"name": "@sid", "value": session_id}] + [
            {"name": f"@kw{i}", "value": kw} for i, kw in enumerate(keywords)
        ]
        lexical_rows = [
            row
            async for row in entities_container.query_items(
                query=lexical_query,
                parameters=lexical_params,
                partition_key=session_id,
            )
        ]
        rows = _merge_entity_rows(list(rows), lexical_rows, limit=k)

    return rows


def _entities_to_bullets(entities: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for e in entities:
        name = str(e.get("name", "")).strip()
        type_ = str(e.get("type", "")).strip()
        facts = [str(f).strip() for f in (e.get("facts") or []) if str(f).strip()]
        related = [str(r).strip() for r in (e.get("related_to") or []) if str(r).strip()]
        if not name:
            continue
        related_suffix = ""
        if related:
            related_suffix = f" (related: {', '.join(related[:6])})"
        if facts:
            lines.append(f"- {name} ({type_}): {'; '.join(facts)}{related_suffix}")
        else:
            lines.append(f"- {name} ({type_}){related_suffix}")
    return "\n".join(lines)
async def chat(
    sessions_container: Any,
    entities_container: Any,
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
            "next_seq": 0,
            "user_entity": None,
        },
    )

    user_entity = session.get("user_entity")
    next_seq = int(session.get("next_seq", 0) or 0)

    if not user_entity:
        primary = get_extractor_model(use_loading_models=use_loading_models)
        fallback = get_extractor_model(use_loading_models=False)
        identity = await _extract_json_with_fallback(
            openai_client,
            primary_model=primary,
            fallback_model=fallback,
            system_prompt=USER_IDENTITY_PROMPT,
            user_content=user_message,
            max_completion_tokens=50,
        )
        extracted = identity.get("user_name")
        if isinstance(extracted, str) and extracted.strip():
            user_entity = _clean_extracted_user_name(extracted)
            session["user_entity"] = user_entity

    session["turn_count"] = int(session.get("turn_count", 0) or 0) + 1
    await upsert_message(
        sessions_container,
        session_id=session_id,
        seq=next_seq,
        role="user",
        content=user_message,
        ts=int(time.time()),
        ttl_s=MESSAGE_TTL_S,
    )

    retrieved = await _retrieve_entities(
        entities_container, openai_client, session_id=session_id, query_text=user_message
    )
    if user_entity and _asks_for_self_identity(user_message):
        entity_id = f"{session_id}::{_normalize_entity_key(user_entity)}"
        pinned = await _read_entity(entities_container, entity_id=entity_id, session_id=session_id)
        if pinned:
            retrieved = _merge_entity_rows([dict(pinned)], list(retrieved), limit=_top_k_for_query(user_message))

    settings = get_settings()
    messages: List[Dict[str, str]] = [{"role": "system", "content": settings.system_prompt}]
    if user_entity:
        messages.append(
            {
                "role": "system",
                "content": (
                    f'The user is {user_entity}. When the user says "I", "me", or "my", '
                    f'it refers to {user_entity}.'
                ),
            }
        )
    if retrieved:
        bullets = _entities_to_bullets(retrieved)
        if bullets.strip():
            messages.append({"role": "system", "content": bullets})
    recent_turns = await read_recent_messages(
        sessions_container, session_id=session_id, limit=RECENT_TURNS_KEPT * 2
    )
    messages.extend(recent_turns)

    reply, usage, latency_ms = await create_chat_completion(
        openai_client,
        model=get_chat_model(use_loading_models=use_loading_models),
        messages=messages,
        max_completion_tokens=CHAT_MAX_COMPLETION_TOKENS,
    )

    await upsert_message(
        sessions_container,
        session_id=session_id,
        seq=next_seq + 1,
        role="assistant",
        content=reply,
        ts=int(time.time()),
        ttl_s=MESSAGE_TTL_S,
    )
    next_seq += 2

    primary = get_extractor_model(use_loading_models=use_loading_models)
    fallback = get_extractor_model(use_loading_models=False)
    extracted = await _extract_json_with_fallback(
        openai_client,
        primary_model=primary,
        fallback_model=fallback,
        system_prompt=EXTRACTION_PROMPT,
        user_content=f"User: {user_message}",
        max_completion_tokens=800,
    )
    entities = extracted.get("entities", [])
    hinted_entities = _extract_hint_entities_from_user_message(user_message)
    if hinted_entities:
        if not isinstance(entities, list):
            entities = []
        entities = list(entities) + hinted_entities
    if isinstance(entities, list):
        for ent in entities:
            if isinstance(ent, dict):
                await _upsert_entity(
                    entities_container, openai_client, session_id=session_id, entity=ent
                )

    session["doc_type"] = "session"
    session.pop("recent_turns", None)  # legacy field (pre message-doc refactor)
    session["next_seq"] = next_seq
    await upsert_item(sessions_container, session)

    metrics = Metrics(
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        latency_ms=latency_ms,
        memory_turns_stored=int(session["turn_count"]),
        context_turns_sent=max(len(messages) - 1, 0),
    )
    return reply, metrics


async def get_memory(
    sessions_container: Any, entities_container: Any, session_id: str
) -> Dict[str, Any]:
    session = await read_or_default(
        sessions_container,
        session_id,
        default={
            "id": session_id,
            "session_id": session_id,
            "doc_type": "session",
            "turn_count": 0,
            "next_seq": 0,
            "user_entity": None,
        },
    )
    entities = [
        dict(row)
        async for row in entities_container.query_items(
            query="SELECT TOP 500 * FROM c",
            partition_key=session_id,
        )
    ]
    cleaned: List[Dict[str, Any]] = []
    for e in entities:
        for k in list(e.keys()):
            if k.startswith("_"):
                e.pop(k, None)
        e.pop("embedding", None)
        cleaned.append(e)
    recent_turns = await read_recent_messages(
        sessions_container, session_id=session_id, limit=RECENT_TURNS_KEPT * 2
    )
    return {
        "recent_turns": recent_turns,
        "entities": cleaned,
        "turn_count": int(session.get("turn_count", 0) or 0),
        "user_entity": session.get("user_entity"),
    }
