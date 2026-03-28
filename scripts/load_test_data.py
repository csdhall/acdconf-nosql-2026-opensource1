from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, List

from azure.cosmos.aio import CosmosClient
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from openai import AsyncOpenAI

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.config import get_settings
from backend.llm import is_mock_openai
from backend.storage.cosmos_queries import delete_all_items_for_session
from backend.strategies import direct_llm, entity_graph, hierarchical, sliding_window


BASELINE_TEST_SESSIONS = {
    "direct_llm": "test-direct-llm",
    "sliding_window": "test-sliding-window",
    "hierarchical": "test-hierarchical",
    "entity_graph": "test-entity-graph",
}

DEFAULT_SEED_PATH = Path(__file__).resolve().parent / "default_seed_messages.json"


def _load_test_conversations() -> List[str]:
    if not DEFAULT_SEED_PATH.exists():
        raise RuntimeError("scripts/default_seed_messages.json not found")

    payload = json.loads(DEFAULT_SEED_PATH.read_text(encoding="utf-8"))
    messages = payload.get("messages")
    if not isinstance(messages, list) or not all(isinstance(item, str) and item.strip() for item in messages):
        raise RuntimeError("default_seed_messages.json must contain a non-empty messages list[str].")
    if len(messages) != 60:
        raise RuntimeError(f"Expected 60 seed messages; got {len(messages)}")
    return messages


async def _session_exists(container: Any, session_id: str) -> bool:
    try:
        rows = [
            row
            async for row in container.query_items(
                query="SELECT TOP 1 c.id FROM c",
                partition_key=session_id,
            )
        ]
        return bool(rows)
    except CosmosResourceNotFoundError:
        return False


async def _delete_session(container: Any, session_id: str) -> None:
    await delete_all_items_for_session(container, session_id)


async def _delete_entities_for_session(entities_container: Any, session_id: str) -> None:
    await delete_all_items_for_session(entities_container, session_id)


async def _load_strategy(
    *,
    name: str,
    run_one_message,
    messages: List[str],
) -> None:
    total = len(messages)
    for idx, message in enumerate(messages, start=1):
        if idx == 1 or idx % 5 == 0 or idx == total:
            print(f"[{name}] {idx}/{total}")

        retries = 5
        backoff_s = 0.75
        while True:
            try:
                await asyncio.wait_for(run_one_message(message), timeout=180.0)
                break
            except asyncio.TimeoutError as exc:
                if retries <= 0:
                    raise RuntimeError(f"{name} timed out on message: {message}") from exc
                retries -= 1
                await asyncio.sleep(backoff_s)
                backoff_s = min(backoff_s * 1.8, 15.0)
            except Exception as exc:  # noqa: BLE001
                if retries <= 0:
                    raise RuntimeError(f"{name} failed on message: {message}") from exc
                retries -= 1
                await asyncio.sleep(backoff_s)
                backoff_s = min(backoff_s * 1.8, 15.0)
        await asyncio.sleep(0.05)


async def _run(*, force: bool) -> None:
    settings = get_settings()
    if not settings.cosmos_endpoint or not settings.cosmos_key:
        raise RuntimeError("COSMOS_ENDPOINT and COSMOS_KEY must be set")
    if not is_mock_openai() and not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY must be set (or set MOCK_OPENAI=true)")

    cosmos = CosmosClient(
        settings.cosmos_endpoint,
        credential=settings.cosmos_key,
        connection_verify=settings.cosmos_verify_ssl,
    )
    openai_client: Any = None if is_mock_openai() else AsyncOpenAI(api_key=settings.openai_api_key)

    try:
        db = cosmos.get_database_client(settings.cosmos_database)
        direct_container = db.get_container_client(settings.cosmos_container_direct_llm)
        sliding_container = db.get_container_client(settings.cosmos_container_sliding_window)
        hierarchical_container = db.get_container_client(settings.cosmos_container_hierarchical)
        eg_sessions = db.get_container_client(settings.cosmos_container_entity_graph_sessions)
        eg_entities = db.get_container_client(settings.cosmos_container_entity_graph_entities)

        existing = []
        if await _session_exists(direct_container, BASELINE_TEST_SESSIONS["direct_llm"]):
            existing.append(BASELINE_TEST_SESSIONS["direct_llm"])
        if await _session_exists(sliding_container, BASELINE_TEST_SESSIONS["sliding_window"]):
            existing.append(BASELINE_TEST_SESSIONS["sliding_window"])
        if await _session_exists(hierarchical_container, BASELINE_TEST_SESSIONS["hierarchical"]):
            existing.append(BASELINE_TEST_SESSIONS["hierarchical"])
        if await _session_exists(eg_sessions, BASELINE_TEST_SESSIONS["entity_graph"]):
            existing.append(BASELINE_TEST_SESSIONS["entity_graph"])

        if existing and not force:
            joined = ", ".join(existing)
            raise RuntimeError(f"Test sessions already exist: {joined}. Re-run with --force to overwrite.")

        if force:
            await _delete_session(direct_container, BASELINE_TEST_SESSIONS["direct_llm"])
            await _delete_session(sliding_container, BASELINE_TEST_SESSIONS["sliding_window"])
            await _delete_session(hierarchical_container, BASELINE_TEST_SESSIONS["hierarchical"])
            await _delete_session(eg_sessions, BASELINE_TEST_SESSIONS["entity_graph"])
            await _delete_entities_for_session(eg_entities, BASELINE_TEST_SESSIONS["entity_graph"])

        conversations = _load_test_conversations()

        await asyncio.gather(
            _load_strategy(
                name="direct_llm",
                messages=conversations,
                run_one_message=lambda message: direct_llm.chat(
                    direct_container,
                    openai_client,
                    BASELINE_TEST_SESSIONS["direct_llm"],
                    message,
                    use_loading_models=True,
                ),
            ),
            _load_strategy(
                name="sliding_window",
                messages=conversations,
                run_one_message=lambda message: sliding_window.chat(
                    sliding_container,
                    openai_client,
                    BASELINE_TEST_SESSIONS["sliding_window"],
                    message,
                    use_loading_models=True,
                ),
            ),
            _load_strategy(
                name="hierarchical",
                messages=conversations,
                run_one_message=lambda message: hierarchical.chat(
                    hierarchical_container,
                    openai_client,
                    BASELINE_TEST_SESSIONS["hierarchical"],
                    message,
                    use_loading_models=True,
                ),
            ),
            _load_strategy(
                name="entity_graph",
                messages=conversations,
                run_one_message=lambda message: entity_graph.chat(
                    eg_sessions,
                    eg_entities,
                    openai_client,
                    BASELINE_TEST_SESSIONS["entity_graph"],
                    message,
                    use_loading_models=True,
                ),
            ),
        )

        print("Loaded baseline test sessions:")
        for strategy in ("direct_llm", "sliding_window", "hierarchical", "entity_graph"):
            print(f"- {strategy}: {BASELINE_TEST_SESSIONS[strategy]}")
    finally:
        await cosmos.close()
        if openai_client is not None:
            await openai_client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load the 60-turn baseline conversation into Cosmos."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing test sessions before loading.",
    )
    args = parser.parse_args()
    asyncio.run(_run(force=args.force))


if __name__ == "__main__":
    main()
