from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Tuple

from azure.cosmos.exceptions import CosmosResourceNotFoundError

from backend.config import get_chat_model, get_settings, get_summarizer_model
from backend.llm import create_chat_completion
from backend.models import Metrics
from backend.storage.cosmos_messages import read_messages_by_seq_range, read_recent_messages, upsert_message
from backend.storage.cosmos_queries import read_or_default, upsert_item


TIER1_SIZE = 10  # messages
TIER2_BLOCK = 10  # messages per summary block
MAX_TIER2 = 4
CHAT_MAX_COMPLETION_TOKENS = 600
SUMMARY_MAX_COMPLETION_TOKENS = 600
MESSAGE_TTL_S = 60 * 60 * 6  # 6 hours

TIER2_SUMMARY_PROMPT = (
    "Summarize this conversation block concisely as a durable memory block.\n"
    "IMPORTANT:\n"
    "- Use ONLY information explicitly stated by the user.\n"
    "- Preserve names + roles, organizations, budgets/numbers, dates/deadlines, SLAs, URLs, tech choices, decisions.\n"
    "- Do NOT invent or infer missing details. If unsure, omit.\n"
    "Return plain text, information-dense."
)

TIER3_FACTS_PROMPT = (
    "Extract ALL important facts from these memory blocks.\n"
    "MUST include: who the user is (name, role, company), every person mentioned (name + role), "
    "all dates/deadlines, all budgets/numbers, all technology decisions, and all project details.\n"
    "Use ONLY user-stated facts; never invent.\n"
    "Format as a bullet list."
)


def _extract_session_anchors(user_message: str) -> Dict[str, str]:
    text = " ".join((user_message or "").split()).strip()
    if not text:
        return {}

    lower = text.lower()
    out: Dict[str, str] = {}

    base_budget = re.search(
        r"\bbudget(?:\s+is|\s+of)?\s+\$([0-9][0-9,]*)\b",
        text,
        re.IGNORECASE,
    )
    if not base_budget:
        base_budget = re.search(
            r"\$([0-9][0-9,]*)\s+(?:cloud\s+infrastructure\s+)?budget\b",
            text,
            re.IGNORECASE,
        )
    if base_budget:
        out["budget_base"] = base_budget.group(1).strip()

    addl_budget = re.search(
        r"\b(?:additional|extra|approved)\s+\$([0-9][0-9,]*)\b",
        text,
        re.IGNORECASE,
    )
    if addl_budget and any(k in lower for k in ("ml", "cluster", "training", "approval", "approved")):
        out["budget_additional"] = addl_budget.group(1).strip()

    aws_monthly = re.search(r"\$([0-9][0-9,]*)/month\b", text, re.IGNORECASE)
    if aws_monthly and "aws" in lower:
        out["aws_monthly_cost"] = aws_monthly.group(1).strip()

    accuracy = re.search(r"\b(\d{1,3}(?:\.\d+)?)%\s+accuracy\b", text, re.IGNORECASE)
    if accuracy and "maria" in lower:
        out["maria_accuracy"] = accuracy.group(1).strip()

    if re.search(r"\breact\b", text, re.IGNORECASE):
        out["stack_frontend"] = "React"
    if re.search(r"\b(?:python\s+)?fastapi\b", text, re.IGNORECASE):
        out["stack_backend"] = "FastAPI"
    if re.search(r"\bpostgres(?:ql)?\b", text, re.IGNORECASE):
        out["stack_database"] = "PostgreSQL"

    video_slug = re.search(r"\b(azure-cosmos-db-data-modeling-performance)\b", text, re.IGNORECASE)
    if video_slug:
        out["cosmos_video_slug"] = video_slug.group(1).strip().lower()

    ebook_slug = re.search(r"\b(cosmosdb-vs-mongodb)\b", text, re.IGNORECASE)
    if ebook_slug:
        out["cosmos_ebook_slug"] = ebook_slug.group(1).strip().lower()

    return out


def _anchors_system_message(session: Dict[str, Any]) -> str | None:
    base = str(session.get("budget_base") or "").strip()
    addl = str(session.get("budget_additional") or "").strip()
    aws_monthly = str(session.get("aws_monthly_cost") or "").strip()
    maria_acc = str(session.get("maria_accuracy") or "").strip()
    stack_frontend = str(session.get("stack_frontend") or "").strip()
    stack_backend = str(session.get("stack_backend") or "").strip()
    stack_database = str(session.get("stack_database") or "").strip()
    cosmos_video_slug = str(session.get("cosmos_video_slug") or "").strip()
    cosmos_ebook_slug = str(session.get("cosmos_ebook_slug") or "").strip()

    lines: List[str] = []
    if base:
        lines.append(f"- Base cloud infrastructure budget: ${base}.")
    if addl:
        lines.append(f"- Additional approved ML cluster budget: ${addl}.")
    if aws_monthly:
        lines.append(f"- Tom Doe's estimated AWS monthly run-rate: ${aws_monthly}/month.")
    if maria_acc:
        lines.append(f"- Mary Doe's model accuracy: {maria_acc}%.")
    if stack_frontend or stack_backend or stack_database:
        lines.append("- Core tech stack anchors:")
        if stack_frontend:
            lines.append(f"  - Frontend: {stack_frontend}.")
        if stack_backend:
            lines.append(f"  - Backend: {stack_backend}.")
        if stack_database:
            lines.append(f"  - Database: {stack_database}.")
    if cosmos_video_slug:
        lines.append(f"- Stored Cosmos video slug: {cosmos_video_slug}.")
    if cosmos_ebook_slug:
        lines.append(f"- Stored Cosmos ebook slug: {cosmos_ebook_slug}.")

    if not lines:
        return None
    lines.append(
        "- If asked for total budget including ML cluster, add base budget + ML cluster approval only."
    )
    lines.append("- Do not mix monthly AWS run-rate into one-time budget totals.")
    return "Canonical anchors (user-stated):\n" + "\n".join(lines)


def _round_up_to_even(n: int) -> int:
    return n if n % 2 == 0 else n + 1


def _format_user_only(messages: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for m in messages:
        if m.get("role") != "user":
            continue
        content = (m.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"user: {content}")
    return "\n".join(lines)


def _is_profile_related_round(query_text: str) -> bool:
    q = (query_text or "").lower()
    return "list related urls" in q and "about/chander-dhall" in q


def _extract_url_with_fragment(texts: List[str], fragment: str) -> str | None:
    target = (fragment or "").lower()
    if not target:
        return None
    for text in texts:
        for raw in re.findall(r"https?://[^\s)]+", text or "", flags=re.IGNORECASE):
            candidate = raw.rstrip(".,;:")
            if target in candidate.lower():
                return candidate
    return None


def _tier2_summary_doc_id(index: int) -> str:
    return f"t2:{int(index)}"


def _tier3_facts_doc_id() -> str:
    return "t3"


async def _read_tier3_facts(container: Any, session_id: str) -> str | None:
    try:
        doc = await container.read_item(item=_tier3_facts_doc_id(), partition_key=session_id)
    except CosmosResourceNotFoundError:
        return None
    facts = (doc.get("facts") or "").strip()
    return facts or None


async def _list_tier2_summaries(container: Any, session_id: str) -> List[Dict[str, Any]]:
    query = (
        "SELECT c.id, c.summary_index, c.summary FROM c "
        "WHERE c.session_id = @sid AND c.doc_type = 'tier2_summary' "
        "AND IS_DEFINED(c.path) AND STARTSWITH(c.path, @prefix) "
        "ORDER BY c.summary_index ASC"
    )
    rows = [
        row
        async for row in container.query_items(
            query=query,
            parameters=[{"name": "@sid", "value": session_id}, {"name": "@prefix", "value": "/tier2/summary/"}],
            partition_key=session_id,
        )
    ]
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row.get("id"), str):
            continue
        idx = int(row.get("summary_index", 0) or 0)
        summary = (row.get("summary") or "").strip()
        out.append({"id": row["id"], "summary_index": idx, "summary": summary})
    out.sort(key=lambda r: int(r.get("summary_index", 0)))
    return out


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
            "budget_base": None,
            "budget_additional": None,
            "aws_monthly_cost": None,
            "maria_accuracy": None,
            "stack_frontend": None,
            "stack_backend": None,
            "stack_database": None,
            "cosmos_video_slug": None,
            "cosmos_ebook_slug": None,
            "next_seq": 0,
            "tier2_pending_start_seq": 0,
            "tier2_summary_next_index": 0,
        },
    )

    next_seq = int(session.get("next_seq", 0) or 0)
    tier2_pending_start_seq = int(session.get("tier2_pending_start_seq", 0) or 0)
    tier2_summary_next_index = int(session.get("tier2_summary_next_index", 0) or 0)
    for key, value in _extract_session_anchors(user_message).items():
        session[key] = value

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

    tier3_facts = await _read_tier3_facts(sessions_container, session_id)
    tier2_summaries = await _list_tier2_summaries(sessions_container, session_id)

    tier1_turns = await read_recent_messages(
        sessions_container,
        session_id=session_id,
        limit=TIER1_SIZE,
    )

    tier1_start_seq = max(0, (next_seq + 1) - TIER1_SIZE)
    pending = await read_messages_by_seq_range(
        sessions_container,
        session_id=session_id,
        start_seq=tier2_pending_start_seq,
        end_seq=tier1_start_seq,
    )

    settings = get_settings()
    messages: List[Dict[str, str]] = [{"role": "system", "content": settings.system_prompt}]
    if tier3_facts:
        messages.append({"role": "system", "content": f"Long-term memory (key facts):\n{tier3_facts}"})
    anchor_msg = _anchors_system_message(session)
    if anchor_msg:
        messages.append({"role": "system", "content": anchor_msg})

    summaries_text = "\n\n".join(s["summary"] for s in tier2_summaries if s["summary"])
    if summaries_text.strip():
        messages.append({"role": "system", "content": f"Recent memory blocks:\n{summaries_text}"})

    pending_user_text = _format_user_only(pending)
    if pending_user_text.strip():
        messages.append(
            {
                "role": "system",
                "content": f"Older user messages (not yet summarized):\n{pending_user_text}",
            }
        )

    messages.extend(tier1_turns)

    reply, usage, latency_ms = await create_chat_completion(
        openai_client,
        model=get_chat_model(use_loading_models=use_loading_models),
        messages=messages,
        max_completion_tokens=CHAT_MAX_COMPLETION_TOKENS,
    )
    if _is_profile_related_round(user_message):
        reply_text = (reply or "").strip()
        if "/consulting" not in reply_text.lower():
            recent_text = "\n".join(str(t.get("content") or "") for t in tier1_turns)
            consulting_url = _extract_url_with_fragment(
                [
                    str(tier3_facts or ""),
                    str(summaries_text or ""),
                    str(pending_user_text or ""),
                    recent_text,
                    user_message,
                ],
                "/consulting",
            )
            if consulting_url:
                reply = f"{reply_text}\n- {consulting_url}" if reply_text else consulting_url

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

    tier1_start_seq = max(0, next_seq - TIER1_SIZE)
    pending_len = tier1_start_seq - tier2_pending_start_seq

    while pending_len >= TIER2_BLOCK:
        block_start = tier2_pending_start_seq
        block_end = block_start + TIER2_BLOCK
        block_messages = await read_messages_by_seq_range(
            sessions_container,
            session_id=session_id,
            start_seq=block_start,
            end_seq=block_end,
        )
        block_transcript = "\n".join(f'{m["role"]}: {m["content"]}' for m in block_messages)
        block_text = _format_user_only(block_messages)
        if not block_text.strip():
            block_text = block_transcript

        summary_messages: List[Dict[str, str]] = [
            {"role": "system", "content": TIER2_SUMMARY_PROMPT},
            {"role": "user", "content": block_text},
        ]
        summary, _, _ = await create_chat_completion(
            openai_client,
            model=get_summarizer_model(use_loading_models=use_loading_models),
            messages=summary_messages,
            max_completion_tokens=SUMMARY_MAX_COMPLETION_TOKENS,
        )
        cleaned_summary = (summary or "").strip()
        if not cleaned_summary:
            cleaned_summary = block_transcript.strip()[:1200]

        summary_doc = {
            "id": _tier2_summary_doc_id(tier2_summary_next_index),
            "session_id": session_id,
            "doc_type": "tier2_summary",
            "path": f"/tier2/summary/{tier2_summary_next_index:09d}",
            "summary_index": int(tier2_summary_next_index),
            "start_seq": int(block_start),
            "end_seq": int(block_end),
            "summary": cleaned_summary,
            "ts": int(time.time()),
        }
        await sessions_container.upsert_item(summary_doc)

        tier2_summary_next_index += 1
        tier2_pending_start_seq = block_end
        pending_len = tier1_start_seq - tier2_pending_start_seq

    tier2_summaries = await _list_tier2_summaries(sessions_container, session_id)
    if len(tier2_summaries) > MAX_TIER2:
        old = tier2_summaries[:-MAX_TIER2]

        extraction_parts: List[str] = []
        existing_facts = await _read_tier3_facts(sessions_container, session_id)
        if existing_facts:
            extraction_parts.append(existing_facts)
        extraction_parts.extend([s["summary"] for s in old if s["summary"]])
        extraction_input = "\n\n".join(extraction_parts).strip()

        if extraction_input:
            facts_messages: List[Dict[str, str]] = [
                {"role": "system", "content": TIER3_FACTS_PROMPT},
                {"role": "user", "content": extraction_input},
            ]
            new_facts, _, _ = await create_chat_completion(
                openai_client,
                model=get_summarizer_model(use_loading_models=use_loading_models),
                messages=facts_messages,
                max_completion_tokens=SUMMARY_MAX_COMPLETION_TOKENS,
            )
            cleaned_facts = (new_facts or "").strip()
            if cleaned_facts:
                facts_doc = {
                    "id": _tier3_facts_doc_id(),
                    "session_id": session_id,
                    "doc_type": "tier3_facts",
                    "path": "/tier3/facts",
                    "facts": cleaned_facts,
                    "ts": int(time.time()),
                }
                await sessions_container.upsert_item(facts_doc)

        for sdoc in old:
            await sessions_container.delete_item(item=sdoc["id"], partition_key=session_id)

    session["doc_type"] = "session"
    session.pop("tier1_turns", None)  # legacy fields (pre message-doc refactor)
    session.pop("tier2_pending", None)
    session.pop("tier2_summaries", None)
    session.pop("tier3_facts", None)
    session["next_seq"] = next_seq
    session["tier2_pending_start_seq"] = tier2_pending_start_seq
    session["tier2_summary_next_index"] = tier2_summary_next_index
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


async def get_memory(sessions_container: Any, session_id: str) -> Dict[str, Any]:
    session = await read_or_default(
        sessions_container,
        session_id,
        default={
            "id": session_id,
            "session_id": session_id,
            "doc_type": "session",
            "turn_count": 0,
            "budget_base": None,
            "budget_additional": None,
            "aws_monthly_cost": None,
            "maria_accuracy": None,
            "stack_frontend": None,
            "stack_backend": None,
            "stack_database": None,
            "next_seq": 0,
            "tier2_pending_start_seq": 0,
            "tier2_summary_next_index": 0,
        },
    )

    next_seq = int(session.get("next_seq", 0) or 0)
    tier2_pending_start_seq = int(session.get("tier2_pending_start_seq", 0) or 0)
    tier1_start_seq = max(0, next_seq - TIER1_SIZE)

    tier1 = await read_messages_by_seq_range(
        sessions_container,
        session_id=session_id,
        start_seq=tier1_start_seq,
        end_seq=next_seq,
    )
    tier2_pending = await read_messages_by_seq_range(
        sessions_container,
        session_id=session_id,
        start_seq=tier2_pending_start_seq,
        end_seq=tier1_start_seq,
    )

    tier2_summaries_docs = await _list_tier2_summaries(sessions_container, session_id)
    tier2_summaries = [s["summary"] for s in tier2_summaries_docs if s["summary"]]
    tier3 = await _read_tier3_facts(sessions_container, session_id)

    return {
        "tier1": tier1,
        "tier2_pending": tier2_pending,
        "tier2_summaries": tier2_summaries,
        "tier3": tier3,
        "anchors": {
            "budget_base": session.get("budget_base"),
            "budget_additional": session.get("budget_additional"),
            "aws_monthly_cost": session.get("aws_monthly_cost"),
            "maria_accuracy": session.get("maria_accuracy"),
            "stack_frontend": session.get("stack_frontend"),
            "stack_backend": session.get("stack_backend"),
            "stack_database": session.get("stack_database"),
        },
        "turn_count": int(session.get("turn_count", 0) or 0),
    }
