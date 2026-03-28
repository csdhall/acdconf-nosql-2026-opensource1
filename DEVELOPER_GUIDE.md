# Developer Guide

## What Ships in the Public Repo

The repo is intentionally reduced to the runnable app plus the minimum setup/test utilities:

- `backend/`: FastAPI app, strategy implementations, Cosmos/OpenAI integration
- `frontend/`: single-page comparison UI
- `scripts/provision_cosmos.py`: idempotent Cosmos provisioning
- `scripts/load_test_data.py`: seeds the shared 60-turn baseline
- `scripts/delete_test_data.py`: deletes `test-*` or all data
- `scripts/integration_test.py`: lightweight regression script against the running API
- `run.py`: local server entry point

## Request Flow

`POST /api/chat` is the only runtime entry point for a turn. Each request includes:

- `strategy`
- `session_id`
- `message`

The flow is:

1. `backend/main.py` receives the request.
2. `CosmosStore` in `backend/storage/cosmos_client.py` dispatches to the selected strategy.
3. The strategy reads the minimum Cosmos state needed for that turn.
4. The strategy calls OpenAI once for the assistant reply.
5. The strategy writes any updated memory back to Cosmos.
6. The API returns the reply plus token/latency metrics.

## Strategy Summary

| Strategy | Storage pattern | Tradeoff |
|---|---|---|
| `direct_llm` | session metadata only | clean no-memory baseline |
| `sliding_window` | recent messages plus one rolling summary | cheap, but can lose older detail |
| `hierarchical` | recent messages, tier-2 summaries, tier-3 facts | better long-horizon recall with more bookkeeping |
| `entity_graph` | recent messages plus structured entity docs | strongest fact recall when retrieval is configured correctly |

## Key Files

| Path | Purpose |
|---|---|
| `backend/main.py` | API routes, static hosting, baseline reset job |
| `backend/config.py` | `.env` loading and runtime settings |
| `backend/llm.py` | OpenAI wrapper and `MOCK_OPENAI` behavior |
| `backend/storage/cosmos_client.py` | Cosmos client lifecycle and strategy dispatch |
| `backend/storage/cosmos_queries.py` | shared list/delete/query helpers |
| `backend/strategies/*.py` | memory strategy implementations |
| `frontend/index.html` | comparison UI shell |
| `frontend/app.js` | UI logic, baseline warning behavior, metrics updates |
| `scripts/default_seed_messages.json` | canonical 60-turn seed data |

## Environment Variables

Minimum required settings:

- `COSMOS_ENDPOINT`
- `COSMOS_KEY`
- `OPENAI_API_KEY` unless `MOCK_OPENAI=true`

Notable optional settings:

- `OPENAI_MODEL`, `OPENAI_SUMMARIZER_MODEL`, `OPENAI_EXTRACTOR_MODEL`
- `LOADING_OPENAI_*` variants for faster or cheaper seeding
- `COSMOS_ENABLE_VECTOR_SEARCH`
- `COSMOS_ENABLE_FULL_TEXT_SEARCH`
- `COSMOS_VERIFY_SSL`

## Baseline Seeding

`scripts/load_test_data.py --force` seeds four canonical sessions from `scripts/default_seed_messages.json`.

Those session IDs are fixed:

- `test-direct-llm`
- `test-sliding-window`
- `test-hierarchical`
- `test-entity-graph`

The UI warns when a canonical session drifts above 60 turns, but it no longer blocks chat. That keeps the app usable while still telling the operator that the seeded comparison point has changed.

## Local Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python run.py --provision --open-browser
```

If you only want to validate app flow without live OpenAI calls:

```bash
MOCK_OPENAI=true python run.py --provision --open-browser
```

## Safe Extension Points

- Add a new strategy by extending `backend/models.py::Strategy`, updating `CosmosStore`, and adding a new module under `backend/strategies/`.
- Adjust retrieval behavior in `entity_graph.py` only if Cosmos vector/full-text configuration stays aligned with `scripts/provision_cosmos.py`.
- Keep seed data in dedicated files rather than hiding it inside docs or planning notes.
