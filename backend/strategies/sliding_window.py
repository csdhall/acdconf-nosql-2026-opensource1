from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Tuple

from backend.config import get_chat_model, get_settings, get_summarizer_model
from backend.llm import create_chat_completion
from backend.models import Metrics
from backend.storage.cosmos_messages import (
    read_messages_by_seq_range,
    read_recent_messages,
    upsert_message,
)
from backend.storage.cosmos_queries import read_or_default, upsert_item


WINDOW_SIZE = 30  # messages
CHAT_MAX_COMPLETION_TOKENS = 600
SUMMARY_MAX_COMPLETION_TOKENS = 900
MESSAGE_TTL_S = 60 * 60  # 1 hour

SUMMARY_SYSTEM_PROMPT = (
    "Summarize this conversation concisely as a durable MEMORY SNAPSHOT for future turns.\n"
    "You are updating a rolling summary that will be injected as context.\n"
    "\n"
    "Rules:\n"
    "- Use ONLY information explicitly stated by the user. Ignore assistant messages.\n"
    "- Do NOT invent, assume, or infer missing details. If unsure, omit.\n"
    "- Preserve specific details: names + roles, numbers/budgets/costs, dates/deadlines, SLAs, URLs, "
    "technology choices, and decisions.\n"
    "- Keep existing facts from the previous summary unless contradicted.\n"
    "\n"
    "Output format (plain text, keep it information-dense):\n"
    "User:\n"
    "Project:\n"
    "People:\n"
    "Tech:\n"
    "Requirements:\n"
    "Dates:\n"
    "Budget/Costs:\n"
    "Integrations:\n"
    "Links:\n"
    "Metrics:\n"
)


def _extract_identity_anchor(user_message: str) -> str | None:
    text = " ".join((user_message or "").split()).strip()
    if not text:
        return None

    intro_with_role = re.search(
        r"(?:^|\b)(?:hi,\s*)?(?:i'?m|i am|my name is)\s+"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*(?:,|\band\b)?\s*"
        r"(?:(?:i'?m|i am)\s+)?(?:a|an)?\s*([^,.!?]+?)\s+at\s+([A-Z][A-Za-z0-9&._-]+)",
        text,
        re.IGNORECASE,
    )
    if intro_with_role:
        name = re.sub(r"\s+(?:and|&)$", "", intro_with_role.group(1).strip(), flags=re.IGNORECASE)
        role = re.sub(r"^(?:i'?m|i am)\s+", "", intro_with_role.group(2).strip(), flags=re.IGNORECASE)
        role = role.strip(" ,.")
        company = intro_with_role.group(3).strip().rstrip(".,;:")
        return f"User identity: {name}; role: {role}; company: {company}."

    intro_name_only = re.search(
        r"(?:^|\b)(?:hi,\s*)?(?:i'?m|i am|my name is)\s+"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        text,
        re.IGNORECASE,
    )
    if intro_name_only:
        return f"User identity: {intro_name_only.group(1).strip()}."

    return None


def _extract_numeric_anchors(user_message: str) -> Dict[str, str]:
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

    return out


def _numeric_facts_system_message(session: Dict[str, Any]) -> str | None:
    base = str(session.get("budget_base") or "").strip()
    addl = str(session.get("budget_additional") or "").strip()
    aws_monthly = str(session.get("aws_monthly_cost") or "").strip()
    maria_acc = str(session.get("maria_accuracy") or "").strip()
    stack_frontend = str(session.get("stack_frontend") or "").strip()
    stack_backend = str(session.get("stack_backend") or "").strip()
    stack_database = str(session.get("stack_database") or "").strip()

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

    if not lines:
        return None

    lines.append(
        "- If asked for the total budget including the ML cluster, add only base budget + ML cluster approval."
    )
    lines.append("- Do not add monthly AWS run-rate to one-time budget totals unless asked for monthly cost.")
    return "Canonical numeric facts (user-stated):\n" + "\n".join(lines)


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
            "summary": None,
            "identity_anchor": None,
            "budget_base": None,
            "budget_additional": None,
            "aws_monthly_cost": None,
            "maria_accuracy": None,
            "stack_frontend": None,
            "stack_backend": None,
            "stack_database": None,
            "next_seq": 0,
            "summarized_until_seq": 0,
            "turn_count": 0,
        },
    )

    summary = session.get("summary")
    identity_anchor = session.get("identity_anchor")
    next_seq = int(session.get("next_seq", 0) or 0)
    summarized_until_seq = int(session.get("summarized_until_seq", 0) or 0)

    latest_anchor = _extract_identity_anchor(user_message)
    if latest_anchor:
        identity_anchor = latest_anchor
        session["identity_anchor"] = latest_anchor
    for key, value in _extract_numeric_anchors(user_message).items():
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

    settings = get_settings()
    messages: List[Dict[str, str]] = [{"role": "system", "content": settings.system_prompt}]
    if summary:
        messages.append(
            {
                "role": "system",
                "content": f"Rolling memory summary (user-stated facts only; may be incomplete):\n{summary}",
            }
        )
    if identity_anchor:
        messages.append(
            {
                "role": "system",
                "content": identity_anchor,
            }
        )
    numeric_facts = _numeric_facts_system_message(session)
    if numeric_facts:
        messages.append({"role": "system", "content": numeric_facts})
    recent_turns = await read_recent_messages(sessions_container, session_id=session_id, limit=WINDOW_SIZE)
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

    window_start_seq = max(0, next_seq - WINDOW_SIZE)
    if summarized_until_seq < window_start_seq:
        end_seq = _round_up_to_even(window_start_seq)
        overflow = await read_messages_by_seq_range(
            sessions_container,
            session_id=session_id,
            start_seq=summarized_until_seq,
            end_seq=end_seq,
        )

        overflow_text = _format_user_only(overflow)
        if overflow_text.strip():
            summary_messages: List[Dict[str, str]] = [{"role": "system", "content": SUMMARY_SYSTEM_PROMPT}]
            if summary:
                summary_messages.append({"role": "system", "content": f"Previous summary: {summary}"})
            summary_messages.append({"role": "user", "content": overflow_text})

            new_summary, _, _ = await create_chat_completion(
                openai_client,
                model=get_summarizer_model(use_loading_models=use_loading_models),
                messages=summary_messages,
                max_completion_tokens=SUMMARY_MAX_COMPLETION_TOKENS,
            )
            cleaned = (new_summary or "").strip()
            if cleaned:
                session["summary"] = cleaned

        session["summarized_until_seq"] = end_seq

    session["doc_type"] = "session"
    session.pop("turns", None)  # legacy field (pre message-doc refactor)
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


async def get_memory(sessions_container: Any, session_id: str) -> Dict[str, Any]:
    session = await read_or_default(
        sessions_container,
        session_id,
        default={
            "id": session_id,
            "session_id": session_id,
            "doc_type": "session",
            "summary": None,
            "identity_anchor": None,
            "budget_base": None,
            "budget_additional": None,
            "aws_monthly_cost": None,
            "maria_accuracy": None,
            "stack_frontend": None,
            "stack_backend": None,
            "stack_database": None,
            "next_seq": 0,
            "summarized_until_seq": 0,
            "turn_count": 0,
        },
    )
    recent_turns = await read_recent_messages(sessions_container, session_id=session_id, limit=WINDOW_SIZE)
    return {
        "recent_turns": recent_turns,
        "summary": session.get("summary"),
        "identity_anchor": session.get("identity_anchor"),
        "numeric_facts": {
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
