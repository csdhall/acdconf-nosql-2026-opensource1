# Tester Guide

## Start the App

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python run.py --provision
```

The app runs at `http://localhost:9281`.

## Seed the Canonical Baseline

```bash
python scripts/load_test_data.py --force
```

This loads four shared 60-turn sessions:

- `test-direct-llm`
- `test-sliding-window`
- `test-hierarchical`
- `test-entity-graph`

The UI also exposes the same operation as `Reset to 60`.

## Manual Sanity Check

1. Open the app.
2. Switch between the four strategies.
3. Confirm the matching `test-*` session is available for each strategy.
4. Ask a few recall questions against the seeded baseline.
5. Compare answer quality, token totals, and `Memory State`.

Useful questions:

- `What's the tech stack?`
- `What's the total budget including the additional ML cluster approval?`
- `Who are the pilot customers?`
- `What's our uptime SLA requirement?`

## About the >60 Warning

If a seeded `test-*` session drifts above 60 turns, the UI shows a warning instead of blocking chat. That warning means the shared comparison point has changed. You can still continue testing; use `Reset to 60` when you want to return to the canonical baseline.

## Automated Check

With the server already running:

```bash
python scripts/integration_test.py
```

If the server is on a different host or port:

```bash
BASE_URL=http://localhost:9282 python scripts/integration_test.py
```

Optional mock-mode run:

```bash
MOCK_OPENAI=true python scripts/integration_test.py
```

## Cleanup

Delete only seeded test sessions:

```bash
python scripts/delete_test_data.py --test
```

Delete all data from every container:

```bash
python scripts/delete_test_data.py
```

## Troubleshooting

| Issue | What to check |
|---|---|
| Cosmos connection fails | Verify `COSMOS_ENDPOINT`, `COSMOS_KEY`, and `COSMOS_VERIFY_SSL` in `.env` |
| Server startup fails | Re-run `python run.py --provision` to ensure containers exist |
| OpenAI request errors | Use `MOCK_OPENAI=true` to isolate Cosmos and app behavior |
