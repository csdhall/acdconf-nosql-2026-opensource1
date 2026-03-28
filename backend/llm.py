from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple


def _get_env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_openai_temperature() -> float:
    raw = os.getenv("OPENAI_TEMPERATURE")
    if raw is None:
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 2.0:
        return 2.0
    return value


def is_mock_openai() -> bool:
    return _get_env_bool("MOCK_OPENAI", False)


def _message_text_blob(messages: List[Dict[str, str]]) -> str:
    return "\n".join(f'{m.get("role","")}: {m.get("content","")}' for m in messages if m.get("content"))


def _extract_facts_for_mock(text_blob: str) -> Dict[str, Any]:
    facts: Dict[str, Any] = {}

    # Name + company (common demo phrasing)
    m = re.search(
        r"\b(?:hi,\s*)?i'?m\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)[^.\n]*?\bat\s+([A-Z][A-Za-z0-9&_-]+)\b",
        text_blob,
        re.IGNORECASE,
    )
    if m:
        facts["name"] = m.group(1).strip()
        facts["company"] = m.group(2).strip()

    m = re.search(
        r"\b(?:i'?m|i am)\s+(?:a|an)\s+([^.,]+?)\s+at\s+([A-Z][A-Za-z0-9&_-]+)\b",
        text_blob,
        re.IGNORECASE,
    )
    if m:
        facts["role"] = m.group(1).strip()
        facts["company"] = facts.get("company") or m.group(2).strip()

    m = re.search(
        r"\bproject is called\s+['\"]?([A-Z][A-Za-z0-9_-]+)['\"]?\b",
        text_blob,
        re.IGNORECASE,
    )
    if m:
        facts["project_name"] = m.group(1).strip()

    m = re.search(r"\bcat named\s+([A-Z][a-z]+)\b", text_blob, re.IGNORECASE)
    if m:
        facts["cat_name"] = m.group(1).strip()
    m = re.search(r"\b(\d{1,2})\s+years\s+old\b", text_blob, re.IGNORECASE)
    if m:
        facts["cat_age"] = m.group(1).strip()

    m = re.search(
        r"\bteam lead is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text_blob, re.IGNORECASE
    )
    if m:
        facts["team_lead"] = m.group(1).strip()
    m = re.search(r"\b(?:she|he|they)\s+prefers\s+([A-Za-z#+]+)\b", text_blob, re.IGNORECASE)
    if m:
        facts["lead_language"] = m.group(1).strip()

    m = re.search(r"\bproject codename is\s+([A-Z][A-Za-z0-9_-]+)\b", text_blob, re.IGNORECASE)
    if m:
        facts["project_codename"] = m.group(1).strip()

    m = re.search(r"\bdeadline is\s+([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?)\b", text_blob)
    if m:
        facts["deadline"] = m.group(1).strip()

    m = re.search(r"\bbudget of\s+\$?([0-9][0-9,]*)\b", text_blob, re.IGNORECASE)
    if m:
        facts["budget"] = m.group(1).strip()
    else:
        m = re.search(r"\$([0-9][0-9,]*)\s+budget\b", text_blob, re.IGNORECASE)
        if m:
            facts["budget"] = m.group(1).strip()
        else:
            m = re.search(r"\bbudget\s+\$([0-9][0-9,]*)\b", text_blob, re.IGNORECASE)
            if m:
                facts["budget"] = m.group(1).strip()

    m = re.search(r"\badditional\s+\$([0-9][0-9,]*)\b", text_blob, re.IGNORECASE)
    if m:
        facts["additional_budget"] = m.group(1).strip()
    else:
        m = re.search(r"\badditional budget\s+\$([0-9][0-9,]*)\b", text_blob, re.IGNORECASE)
        if m:
            facts["additional_budget"] = m.group(1).strip()

    m = re.search(r"\baws[^\n]*?\$([0-9][0-9,]*)/month\b", text_blob, re.IGNORECASE)
    if m:
        facts["aws_monthly_cost"] = m.group(1).strip()

    m = re.search(r"\bmaria[^\n]*?\b(\d{1,3}(?:\.\d+)?)%\s+accuracy\b", text_blob, re.IGNORECASE)
    if m:
        facts["maria_accuracy"] = m.group(1).strip()

    m = re.search(r"\b(\d{1,3}(?:\.\d+)?)%\s+uptime\b", text_blob, re.IGNORECASE)
    if m:
        facts["sla_uptime"] = m.group(1).strip()

    m = re.search(
        r"\bfirst milestone is\s+([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?)\b",
        text_blob,
    )
    if m:
        facts["first_milestone"] = m.group(1).strip()

    team_roles: dict[str, str] = {}
    for nm, role in re.findall(r"\b([A-Z][a-z]+)\s*\(([^)]+)\)", text_blob):
        team_roles[nm.strip()] = role.strip()
    if team_roles:
        facts["team_roles"] = team_roles

    pilot_m = re.search(r"\bpilot customers\s+are:\s*([^\n]+)", text_blob, re.IGNORECASE)
    if pilot_m:
        s = pilot_m.group(1)
        names = [x.strip().strip(".") for x in re.split(r",|\band\b", s, flags=re.IGNORECASE)]
        names = [x for x in names if x]
        if names:
            facts["pilot_customers"] = names

    integ_m = re.search(r"\bintegrate with\s+([^\n]+)", text_blob, re.IGNORECASE)
    if integ_m:
        s = integ_m.group(1)
        s = s.split(" for ", 1)[0]
        names = [x.strip().strip(".") for x in re.split(r",|\band\b", s, flags=re.IGNORECASE)]
        names = [x for x in names if x]
        if names:
            facts["integrations"] = names

    m = re.search(r"\brepo is\s+([^\s]+)", text_blob, re.IGNORECASE)
    if m:
        facts["repo"] = m.group(1).strip().rstrip(".")

    return facts


def _mock_chat_reply(messages: List[Dict[str, str]]) -> str:
    user_msg = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    q = user_msg.lower()

    is_question = user_msg.strip().endswith("?") or bool(
        re.search(r"\b(what|who|where|when|which|how|tell|remind|remember|recall)\b", q)
    )
    if not is_question:
        return "Got it."

    system_contents = [m.get("content", "") for m in messages if m.get("role") == "system"]
    memory_system_contents = [c for c in system_contents[1:] if c and c.strip()]

    blob = _message_text_blob(messages)
    facts = _extract_facts_for_mock(blob)

    def as_int(v: Any) -> Optional[int]:
        if not isinstance(v, str):
            return None
        digits = v.replace(",", "").strip()
        if not digits.isdigit():
            return None
        return int(digits)

    if ("pilot" in q and "customer" in q) and isinstance(facts.get("pilot_customers"), list):
        names = [str(x) for x in (facts.get("pilot_customers") or []) if str(x).strip()]
        if names:
            return "Pilot customers: " + ", ".join(names) + "."

    if "integrat" in q and isinstance(facts.get("integrations"), list):
        names = [str(x) for x in (facts.get("integrations") or []) if str(x).strip()]
        if names:
            return "Integrations: " + ", ".join(names) + "."

    if "total budget" in q:
        base = as_int(facts.get("budget"))
        addl = as_int(facts.get("additional_budget"))
        if base is not None and addl is not None:
            total = base + addl
            return f"Total budget: ${total:,} (${base:,} base + ${addl:,} additional ML cluster)."

    if "aws" in q and "cost" in q and isinstance(facts.get("aws_monthly_cost"), str):
        amt = str(facts.get("aws_monthly_cost")).strip()
        if amt:
            return f"Tom Doe estimated AWS costs at about ${amt}/month."

    if "accuracy" in q and isinstance(facts.get("maria_accuracy"), str):
        pct = str(facts.get("maria_accuracy")).strip()
        if pct:
            return f"Mary Doe's churn prediction model achieved {pct}% accuracy."

    if ("sla" in q or "uptime" in q) and isinstance(facts.get("sla_uptime"), str):
        pct = str(facts.get("sla_uptime")).strip()
        if pct:
            return f"Uptime SLA: {pct}%."

    if "milestone" in q and isinstance(facts.get("first_milestone"), str):
        when = str(facts.get("first_milestone")).strip()
        if when:
            return f"First milestone deadline: {when}."

    if "team" in q and ("responsibil" in q or "role" in q) and isinstance(facts.get("team_roles"), dict):
        team: dict = facts.get("team_roles") or {}
        parts = [f"{k} ({v})" for k, v in team.items()]
        if parts:
            return "Team: " + "; ".join(parts) + "."

    parts: List[str] = []
    if facts.get("name") and facts.get("company"):
        parts.append(f'{facts["name"]} at {facts["company"]}')
    if facts.get("cat_name") and facts.get("cat_age"):
        parts.append(f'cat {facts["cat_name"]} ({facts["cat_age"]} years old)')
    if facts.get("project_codename"):
        parts.append(f'project {facts["project_codename"]}')
    if facts.get("deadline"):
        parts.append(f'deadline {facts["deadline"]}')
    if facts.get("budget"):
        parts.append(f'budget ${facts["budget"]}')
    if facts.get("team_lead"):
        lead = facts["team_lead"]
        lang = facts.get("lead_language")
        if lang:
            parts.append(f"team lead {lead} prefers {lang}")
        else:
            parts.append(f"team lead {lead}")
    if facts.get("repo"):
        parts.append(f'repo {facts["repo"]}')

    if not parts:
        if memory_system_contents:
            return "From memory:\n" + "\n\n".join(memory_system_contents).strip()
        return "I don't have any stored facts yet."
    answer = "From memory: " + "; ".join(parts) + "."
    if memory_system_contents:
        mem = "\n\n".join(memory_system_contents).strip()
        if mem:
            answer += "\n\n" + mem[:6000]
    return answer


def _mock_extract_user_name(text: str) -> Optional[str]:
    m = re.search(
        r"\b(?:hi,\s*)?(?:i'?m|i am|my name is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    return m.group(1).strip()


def _mock_extract_entities_from_turn(text: str) -> List[Dict[str, Any]]:
    # Expected input: "User: ...\nAssistant: ..."
    user_line = text.split("\n", 1)[0]
    user_msg = user_line.removeprefix("User:").strip()

    entities: List[Dict[str, Any]] = []

    def add_entity(name: str, type_: str, facts: List[str], related_to: List[str]) -> None:
        entities.append(
            {
                "name": name,
                "type": type_,
                "facts": [f for f in facts if f],
                "related_to": [r for r in related_to if r],
            }
        )

    lower = user_msg.lower()
    project: str | None = None

    # User identity + company
    name = _mock_extract_user_name(user_msg or text)
    company = None
    m = re.search(r"\bat\s+([A-Z][A-Za-z0-9&_-]+)\b", user_msg)
    if m:
        company = m.group(1).strip()
    role = None
    role_m = re.search(
        r"\b(?:i'?m|i am)\s+(?:a|an)\s+([^.,]+?)\s+at\s+([A-Z][A-Za-z0-9&_-]+)\b",
        user_msg,
        re.IGNORECASE,
    )
    if role_m:
        role = role_m.group(1).strip()
        company = company or role_m.group(2).strip()
    else:
        role_m2 = re.search(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*,\s*a\s+([^.,]+?)\s+at\s+([A-Z][A-Za-z0-9&_-]+)\b",
            user_msg,
            re.IGNORECASE,
        )
        if role_m2:
            role = role_m2.group(2).strip()
            company = company or role_m2.group(3).strip()
    if name:
        facts: List[str] = []
        related: List[str] = []
        if role:
            facts.append(role)
        if company:
            facts.append(f"works at {company}")
            related.append(company)
            add_entity(company, "organization", ["organization"], [name])
        add_entity(name, "person", facts, related)

    # Project name ("project is called ...")
    m = re.search(r"\bproject is called\s+['\"]?([A-Z][A-Za-z0-9_-]+)['\"]?\b", user_msg)
    if m:
        project = m.group(1).strip()
        proj_facts: List[str] = []
        if "analytics platform" in lower:
            proj_facts.append("customer analytics platform")
        if "e-commerce" in lower or "ecommerce" in lower:
            proj_facts.append("for e-commerce")
        add_entity(project, "project", proj_facts, [name] if name else [])

    # Cat
    m = re.search(r"\bcat named\s+([A-Z][a-z]+)\b", user_msg, re.IGNORECASE)
    if m:
        cat_name = m.group(1).strip()
        age_m = re.search(r"\b(\d{1,2})\s+years\s+old\b", user_msg, re.IGNORECASE)
        cat_facts = ["cat"]
        if age_m:
            cat_facts.append(f"is {age_m.group(1)} years old")
        add_entity(cat_name, "fact", cat_facts, [name] if name else [])

    # Team lead + preferred language
    m = re.search(r"\bteam lead is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", user_msg, re.IGNORECASE)
    if m:
        lead = m.group(1).strip()
        lead_facts = ["team lead"]
        lang_m = re.search(r"\b(?:she|he|they)\s+prefers\s+([A-Za-z#+]+)\b", user_msg, re.IGNORECASE)
        if lang_m:
            lead_facts.append(f"prefers {lang_m.group(1).strip()}")
        add_entity(lead, "person", lead_facts, [name] if name else [])

    # Project codename
    m = re.search(r"\bproject codename is\s+([A-Z][A-Za-z0-9_-]+)\b", user_msg, re.IGNORECASE)
    if m:
        project = project or m.group(1).strip()
        facts: List[str] = []
        if "recommendation engine" in user_msg.lower():
            facts.append("recommendation engine")
        add_entity(project, "project", facts, [name] if name else [])

    # Team members with roles like "Jane Doe (frontend)"
    for person_m in re.finditer(r"\b([A-Z][a-z]+)\s*\(([^)]+)\)", user_msg):
        person = person_m.group(1).strip()
        person_role = person_m.group(2).strip()
        rel = [project] if project else []
        add_entity(person, "person", ["team member", person_role], rel)

    # Pilot customers list
    if "pilot customer" in lower and ":" in user_msg:
        after = user_msg.split(":", 1)[1]
        raw_names = re.split(r",|\band\b", after, flags=re.IGNORECASE)
        for raw in raw_names:
            n = raw.strip().strip(".")
            if not n:
                continue
            add_entity(n, "organization", ["pilot customer"], [project] if project else [])

    # Integrations list
    if "integrate with" in lower:
        after = re.split(r"integrate with", user_msg, flags=re.IGNORECASE, maxsplit=1)[-1]
        raw_names = re.split(r",|\band\b", after, flags=re.IGNORECASE)
        for raw in raw_names:
            n = raw.strip().strip(".")
            if not n:
                continue
            add_entity(n, "organization", ["integration"], [project] if project else [])

    # Deadline + budget (often in same line)
    deadline_m = re.search(
        r"\bdeadline is\s+([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?)\b", user_msg
    )
    budget_m = re.search(r"\bbudget of\s+\$?([0-9][0-9,]*)\b", user_msg, re.IGNORECASE)
    if not budget_m:
        budget_m = re.search(r"\$([0-9][0-9,]*)\s+budget\b", user_msg, re.IGNORECASE)
    if deadline_m:
        deadline = deadline_m.group(1).strip()
        add_entity(deadline, "date", ["deadline"], [project] if project else [])
        if project:
            add_entity(project, "project", [f"deadline {deadline}"], [])
    if budget_m:
        amount = budget_m.group(1).strip()
        add_entity("Budget", "fact", [f"budget ${amount}"], [project] if project else [])
        if project:
            add_entity(project, "project", [f"budget ${amount}"], [])

    # Additional budget approval (e.g., dedicated ML cluster)
    addl_budget_m = re.search(r"\badditional\s+\$?([0-9][0-9,]*)\b", user_msg, re.IGNORECASE)
    if addl_budget_m and ("approved" in lower or "approval" in lower):
        amount = addl_budget_m.group(1).strip()
        add_entity("Budget", "fact", [f"additional budget ${amount}"], [project] if project else [])
        if project:
            add_entity(project, "project", [f"additional budget ${amount}"], [])

    # AWS monthly cost estimate
    aws_cost_m = re.search(r"\$([0-9][0-9,]*)/month", user_msg, re.IGNORECASE)
    if aws_cost_m and "aws" in lower and "cost" in lower:
        amount = aws_cost_m.group(1).strip()
        add_entity("AWS", "concept", [f"estimated monthly cost ${amount}/month"], [project] if project else [])
        if "tom" in lower:
            add_entity("Tom Doe", "person", [f"estimated AWS cost ${amount}/month"], [project] if project else [])

    # Model accuracy (e.g., "87% accuracy")
    acc_m = re.search(r"\b(\d{1,3}(?:\.\d+)?)%\s+accuracy\b", user_msg, re.IGNORECASE)
    if acc_m:
        pct = acc_m.group(1).strip()
        if "maria" in lower:
            add_entity("Mary Doe", "person", [f"model accuracy {pct}%"], [project] if project else [])
        add_entity("Model", "concept", [f"accuracy {pct}%"], [project] if project else [])

    # Uptime SLA (e.g., "99.9% uptime")
    sla_m = re.search(r"\b(\d{1,3}(?:\.\d+)?)%\s+uptime\b", user_msg, re.IGNORECASE)
    if sla_m:
        pct = sla_m.group(1).strip()
        add_entity("SLA", "fact", [f"uptime {pct}%"], [project] if project else [])
        if project:
            add_entity(project, "project", [f"SLA {pct}% uptime"], [])

    # First milestone date
    milestone_m = re.search(
        r"\bfirst milestone is\s+([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?)\b",
        user_msg,
    )
    if milestone_m:
        milestone = milestone_m.group(1).strip()
        add_entity(milestone, "date", ["first milestone"], [project] if project else [])
        if project:
            add_entity(project, "project", [f"first milestone {milestone}"], [])

    # Repo URL
    repo_m = re.search(r"\brepo is\s+([^\s]+)", user_msg, re.IGNORECASE)
    if repo_m:
        url = repo_m.group(1).strip().rstrip(".")
        add_entity(url, "url", ["project repo URL"], [project] if project else [])
        if project:
            add_entity(project, "project", [f"repo {url}"], [])

    # Staging URL
    staging_m = re.search(r"\bstaging\s+is\s+at\s+([^\s]+)", user_msg, re.IGNORECASE)
    if staging_m:
        url = staging_m.group(1).strip().rstrip(".")
        add_entity(url, "url", ["staging environment"], [project] if project else [])

    # Common tech stack concepts (minimal; used only in MOCK_OPENAI mode)
    tech_map: List[Tuple[str, str, List[str]]] = [
        (r"\breact\b", "React", ["frontend"]),
        (r"\bnext\.js\b|\bnextjs\b", "Next.js", ["frontend"]),
        (r"\btypescript\b", "TypeScript", ["frontend"]),
        (r"\bpython\s+fastapi\b|\bfastapi\b", "FastAPI", ["backend"]),
        (r"\bpostgresql\b", "PostgreSQL", ["database"]),
        (r"\bmongodb\b", "MongoDB", ["session storage"]),
        (r"\bredis\b", "Redis", ["cache"]),
        (r"\bgraphql\b", "GraphQL", ["api"]),
        (r"\brest\b", "REST", ["api"]),
        (r"\bapache\s+spark\b|\bspark\b", "Apache Spark", ["data processing"]),
        (r"\bsnowflake\b", "Snowflake", ["data warehouse"]),
        (r"\bkafka\b", "Kafka", ["event streaming"]),
        (r"\bdocker\b", "Docker", ["containers"]),
        (r"\bkubernetes\b", "Kubernetes", ["orchestration"]),
        (r"\beks\b", "EKS", ["kubernetes", "cloud"]),
        (r"\baws\b", "AWS", ["cloud"]),
        (r"\bauth0\b", "Auth0", ["auth"]),
        (r"\boauth2\b", "OAuth2", ["auth"]),
        (r"\bprometheus\b", "Prometheus", ["monitoring"]),
        (r"\bgrafana\b", "Grafana", ["monitoring"]),
        (r"\bgithub\s+actions\b", "GitHub Actions", ["ci/cd"]),
        (r"\bpytest\b", "pytest", ["testing"]),
        (r"\bjest\b", "Jest", ["testing"]),
        (r"\bwebsockets\b", "WebSockets", ["real-time"]),
        (r"\blaunchdarkly\b", "LaunchDarkly", ["feature flags"]),
        (r"\btailwind(?:\s+css)?\b", "Tailwind CSS", ["styling"]),
        (r"\bjupyter\b", "Jupyter", ["notebooks"]),
        (r"\bfigma\b", "Figma", ["design"]),
    ]
    for pattern, tech_name, tech_facts in tech_map:
        if re.search(pattern, user_msg, re.IGNORECASE):
            add_entity(tech_name, "concept", tech_facts, [project] if project else [])

    return entities


async def create_chat_completion(
    client: Any,
    *,
    model: str,
    messages: List[Dict[str, str]],
    max_completion_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, int], float]:
    if is_mock_openai():
        system_prompt = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
        user_text = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")

        if response_format and response_format.get("type") == "json_object":
            if "return json" in system_prompt.lower() and "user_name" in system_prompt:
                return (
                    json.dumps({"user_name": _mock_extract_user_name(user_text)}),
                    {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    0.0,
                )
            if "extract entities and facts" in system_prompt.lower():
                return (
                    json.dumps({"entities": _mock_extract_entities_from_turn(user_text)}),
                    {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    0.0,
                )
            return "{}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, 0.0

        if system_prompt.lower().startswith("summarize this conversation"):
            parts = []
            prev_raw = [m.get("content", "") for m in messages if m.get("role") == "system"][1:]
            prev: List[str] = []
            for c in prev_raw:
                c = (c or "").strip()
                if not c:
                    continue
                c = re.sub(r"^previous summary:\s*", "", c, flags=re.IGNORECASE).strip()
                c = re.sub(r"^existing summary:\s*", "", c, flags=re.IGNORECASE).strip()
                prev.append(c)
            if prev:
                parts.append("\n".join(prev))
            if user_text:
                parts.append(user_text)
            summary = "\n".join(p for p in parts if p).strip()
            max_chars = 8000
            if len(summary) <= max_chars:
                out = summary
            else:
                half = max_chars // 2
                out = summary[:half].rstrip() + "\n...\n" + summary[-half:].lstrip()
            return out, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, 0.0

        if system_prompt.lower().startswith("extract all important facts"):
            bullets = []
            for line in user_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                bullets.append(f"- {line}")
            return "\n".join(bullets)[:6000], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, 0.0

        reply = _mock_chat_reply(messages)
        return reply, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, 0.0

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": _get_openai_temperature(),
    }
    if max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = max_completion_tokens
    if response_format is not None:
        kwargs["response_format"] = response_format

    start = time.time()
    try:
        resp = await client.chat.completions.create(**kwargs)
    except TypeError:
        if "max_completion_tokens" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
            resp = await client.chat.completions.create(**kwargs)
        else:
            raise
    latency_ms = round((time.time() - start) * 1000, 1)

    content = resp.choices[0].message.content or ""
    usage_obj = getattr(resp, "usage", None)
    usage = {
        "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
    }
    return content, usage, latency_ms


async def create_embedding(client: Any, *, model: str, input_text: str, dimensions: int) -> List[float]:
    if is_mock_openai():
        # Deterministic lexical projection for mock mode so similar texts
        # land near each other in vector space.
        tokens = re.findall(r"[a-z0-9_./:-]+", input_text.lower())
        if not tokens:
            tokens = ["__empty__"]

        out: List[float] = [0.0] * dimensions
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            for i in range(8):
                idx = ((digest[i * 2] << 8) | digest[(i * 2) + 1]) % dimensions
                sign = 1.0 if (digest[16 + i] & 1) == 0 else -1.0
                out[idx] += sign

        norm = math.sqrt(sum(v * v for v in out))
        if norm > 0:
            out = [v / norm for v in out]
        return out

    resp = await client.embeddings.create(model=model, input=input_text)
    emb = resp.data[0].embedding
    return list(emb)


def parse_json_object(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        return data
    return {}
