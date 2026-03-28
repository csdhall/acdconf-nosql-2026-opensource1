from __future__ import annotations

import os
from typing import Dict, List, Tuple

import httpx


BASE_URL = os.getenv("BASE_URL", "http://localhost:9281")


SETUP_MESSAGES = [
    "Hi, I'm Jordan Park, a data scientist at NovaTech.",
    "My favorite programming language is Rust, but I also use Python daily.",
    "I have a cat named Whiskers who is 3 years old.",
    "My team lead is Sandra Chen. She prefers Go for backend services.",
    "Our project codename is Starlight. It's a recommendation engine.",
    "The project deadline is June 30th with a budget of $80,000.",
    "We're deploying on Google Cloud Platform using Kubernetes.",
    "Our project repo is github.com/novatech/starlight.",
    "Sandra approved using PyTorch for the ML models last Tuesday.",
    "Our main competitor is RecSys Pro. They claim 95% accuracy.",
    "I prefer vim over VS Code and always use dark mode.",
    "The team has 8 members total, including 2 interns from Stanford.",
]

RECALL_QUESTIONS: List[Tuple[str, List[str]]] = [
    ("What's my name and where do I work?", ["Jordan", "NovaTech"]),
    ("What's my cat's name and age?", ["Whiskers", "3"]),
    ("What's the project codename and deadline?", ["Starlight", "June"]),
    ("What's our budget?", ["80,000", "80000", "$80"]),
    ("Who is my team lead and what language does she prefer?", ["Sandra", "Go"]),
    ("What's our project repo URL?", ["github.com/novatech/starlight", "novatech/starlight"]),
]


def _contains_any(haystack: str, needles: List[str]) -> bool:
    h = haystack.lower()
    return any(n.lower() in h for n in needles)


def _print_memory_summary(strategy: str, mem: Dict) -> None:
    if strategy == "sliding_window":
        recent = mem.get("recent_turns") or []
        summary = mem.get("summary") or ""
        print(f"  memory: recent_turns={len(recent)} summary_len={len(summary)}")
    elif strategy == "hierarchical":
        t1 = mem.get("tier1") or []
        t2p = mem.get("tier2_pending") or []
        t2s = mem.get("tier2_summaries") or []
        t3 = mem.get("tier3") or ""
        print(
            f"  memory: tier1={len(t1)} tier2_pending={len(t2p)} tier2_summaries={len(t2s)} tier3_len={len(t3)}"
        )
    elif strategy == "entity_graph":
        recent = mem.get("recent_turns") or []
        entities = mem.get("entities") or []
        user_entity = mem.get("user_entity")
        print(f"  memory: recent_turns={len(recent)} entities={len(entities)} user_entity={user_entity!r}")


def main() -> None:
    strategies = ["sliding_window", "hierarchical", "entity_graph"]

    failures = 0
    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        # Quick connectivity check
        try:
            resp = client.get("/")
            resp.raise_for_status()
        except Exception as e:  # noqa: BLE001
            print(f"Server not reachable at {BASE_URL}: {e}")
            raise SystemExit(2)

        for strategy in strategies:
            session_id = f"integration-test-{strategy}"
            print(f"\n== {strategy} ==")

            client.delete(f"/api/sessions/{strategy}/{session_id}")

            for msg in SETUP_MESSAGES:
                r = client.post(
                    "/api/chat",
                    json={"session_id": session_id, "message": msg, "strategy": strategy},
                )
                r.raise_for_status()

            passed = 0
            for question, keywords in RECALL_QUESTIONS:
                r = client.post(
                    "/api/chat",
                    json={"session_id": session_id, "message": question, "strategy": strategy},
                )
                r.raise_for_status()
                reply = r.json().get("reply", "")
                ok = _contains_any(reply, keywords)
                passed += int(ok)
                if not ok:
                    failures += 1
                    print(f"  FAIL: {question}")
                    print(f"    expected one of: {keywords}")
                    print(f"    got: {reply[:300]!r}")

            mem = client.get(f"/api/memory/{strategy}/{session_id}")
            mem.raise_for_status()
            mem_json = mem.json()
            turn_count = mem_json.get("turn_count")
            print(f"  recall: {passed}/{len(RECALL_QUESTIONS)} passed, turn_count={turn_count}")
            _print_memory_summary(strategy, mem_json)

            client.delete(f"/api/sessions/{strategy}/{session_id}")

    if failures:
        print(f"\nFAILED: {failures} recall checks did not match expected keywords.")
        raise SystemExit(1)
    print("\nPASSED: all strategies passed recall checks.")


if __name__ == "__main__":
    main()
