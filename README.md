# Agent Memory Patterns Demo

This repo is a small FastAPI app for comparing four multi-turn memory strategies backed by Azure Cosmos DB for NoSQL:

> Demo note: this repository is intentionally optimized for clarity and repeatability in a conference-style demo. It is NOT A production reference implementation or a statement of how these patterns should be built in enterprise code. The app still uses Azure Cosmos DB for NoSQL for persistence, and the `entity_graph` strategy can use native Cosmos vector or full-text retrieval when those features are enabled, but the overall implementation is deliberately simplified.

- `direct_llm`: no-memory baseline
- `sliding_window`: recent turns plus a rolling summary
- `hierarchical`: tiered summaries plus long-term facts
- `entity_graph`: structured fact extraction plus retrieval

The public repo is intentionally trimmed to the runnable app, the provisioning/seed/test utilities, and the core architecture/developer/tester documentation.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python run.py --provision --open-browser
```

If port `6800` is already in use:

```bash
python run.py --provision --open-browser --kill-existing-port
```

The app runs at `http://127.0.0.1:6800/`.

## VS Code

Open the repo in VS Code and press `F5`.

The checked-in debug setup now:

- bootstraps `.venv` automatically if it does not exist
- installs `requirements.txt` when needed
- loads variables from `.env`
- starts `run.py` with `--provision --open-browser --kill-existing-port --no-reload`

## Seed the Shared 60-Turn Baseline

```bash
python scripts/load_test_data.py --force
```

This creates the canonical sessions used for side-by-side comparison:

- `test-direct-llm`
- `test-sliding-window`
- `test-hierarchical`
- `test-entity-graph`

You can also reseed the same baseline from the UI with `Reset to 60`.

## About the 60-Turn Warning

The seeded baseline is meant to be a common comparison point at 60 turns. If you keep chatting past 60, the app now warns but does not block the request. That matters most for reproducible comparisons, especially with `sliding_window`; it does not prevent normal use.

## Mock Mode

If you want to exercise the end-to-end flow without live OpenAI calls:

```bash
MOCK_OPENAI=true python run.py --provision --open-browser
```

This still requires Cosmos DB, but the model calls are replaced with deterministic mock behavior.

## Test and Cleanup

Run the lightweight API regression once the server is up:

```bash
python scripts/integration_test.py
```

Delete only the canonical test sessions:

```bash
python scripts/delete_test_data.py --test
```

Delete all data from all containers:

```bash
python scripts/delete_test_data.py
```

## Docs

- `DEVELOPER_GUIDE.md`
- `TESTER_GUIDE.md`
- `ARCHITECTURE.md`

## License

This repository is source-available, not OSI open source.

The code and documentation are licensed under the custom terms in `LICENSE`.
In practice, that means:

- non-commercial use, modification, and sharing are allowed
- attribution is required for redistribution and for public blog, tutorial, or training use
- commercial use is not allowed without separate written permission
- paid courses, paid workshops, paid training, paid certifications, consulting deliverables, and other monetized distributions are not allowed

See `LICENSE` and `NOTICE` for the exact terms and the attribution wording.
