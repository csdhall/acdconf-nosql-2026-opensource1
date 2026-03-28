from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.models import ChatRequest, ChatResponse, Strategy
from backend.storage.cosmos_client import CosmosStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = CosmosStore()
    await store.open()
    app.state.store = store
    try:
        yield
    finally:
        await store.close()


app = FastAPI(lifespan=lifespan)

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

RESET_JOBS: dict[str, dict[str, Any]] = {}
RESET_ACTIVE_JOB_ID: str | None = None


@app.middleware("http")
async def disable_frontend_asset_cache(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(
        str(FRONTEND_DIR / "index.html"),
        headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"},
    )


@app.get("/favicon.ico")
async def favicon() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "favicon.ico"))


@app.post("/api/baseline/reset")
async def reset_baseline() -> Any:
    global RESET_ACTIVE_JOB_ID

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "load_test_data.py"
    if not script_path.exists():
        raise HTTPException(status_code=500, detail="Missing scripts/load_test_data.py")

    cmd = [sys.executable, str(script_path), "--force"]

    if RESET_ACTIVE_JOB_ID:
        active_job = RESET_JOBS.get(RESET_ACTIVE_JOB_ID)
        if active_job and active_job.get("status") == "running":
            return {
                "status": "running",
                "job_id": RESET_ACTIVE_JOB_ID,
                "message": "A baseline reset is already running.",
            }

    job_id = uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    RESET_ACTIVE_JOB_ID = job_id
    RESET_JOBS[job_id] = {
        "job_id": job_id,
        "status": "running",
        "created_at": now,
        "started_at": now,
        "finished_at": None,
        "command": " ".join(cmd),
        "message": "Baseline reset started.",
        "stdout_tail": "",
        "stderr_tail": "",
    }

    async def _run_reset_job() -> None:
        global RESET_ACTIVE_JOB_ID

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(repo_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await proc.communicate()
            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")
            stdout_tail = "\n".join(stdout.splitlines()[-20:]).strip()
            stderr_tail = "\n".join(stderr.splitlines()[-20:]).strip()
            finished = datetime.now(timezone.utc).isoformat()

            job = RESET_JOBS.get(job_id)
            if job is None:
                return

            job["finished_at"] = finished
            job["stdout_tail"] = stdout_tail
            job["stderr_tail"] = stderr_tail
            if proc.returncode == 0:
                job["status"] = "completed"
                job["message"] = "Baseline reset completed."
            else:
                job["status"] = "failed"
                tail = "\n".join((stdout + "\n" + stderr).splitlines()[-40:]).strip()
                if not tail:
                    tail = f"load_test_data exited with code {proc.returncode}."
                job["message"] = f"Baseline reset failed. Last output:\n{tail}"
        except Exception as exc:  # noqa: BLE001
            job = RESET_JOBS.get(job_id)
            if job is not None:
                job["status"] = "failed"
                job["finished_at"] = datetime.now(timezone.utc).isoformat()
                job["message"] = f"Baseline reset failed: {exc}"
        finally:
            if RESET_ACTIVE_JOB_ID == job_id:
                RESET_ACTIVE_JOB_ID = None

    asyncio.create_task(_run_reset_job())

    return {
        "status": "started",
        "job_id": job_id,
        "message": "Baseline reset job started.",
    }


@app.get("/api/baseline/reset/{job_id}")
async def get_reset_baseline_job(job_id: str) -> Any:
    job = RESET_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Reset job not found: {job_id}")
    return job


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> Any:
    try:
        store: CosmosStore = app.state.store
        reply, metrics = await store.chat(req.strategy, req.session_id, req.message)
        return ChatResponse(
            reply=reply,
            session_id=req.session_id,
            strategy=req.strategy,
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/memory/{strategy}/{session_id}")
async def memory(strategy: Strategy, session_id: str) -> Any:
    try:
        store: CosmosStore = app.state.store
        state = await store.get_memory(strategy, session_id)
        return {"strategy": strategy.value, "session_id": session_id, **state}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/sessions/{strategy}")
async def sessions(strategy: Strategy) -> Any:
    try:
        store: CosmosStore = app.state.store
        return await store.list_sessions(strategy)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/api/sessions/{strategy}/{session_id}")
async def delete_session(strategy: Strategy, session_id: str) -> Any:
    try:
        store: CosmosStore = app.state.store
        await store.delete_session(strategy, session_id)
        return {"status": "deleted"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/api/all-data")
async def delete_all_data() -> Any:
    try:
        store: CosmosStore = app.state.store
        await store.delete_all_data()
        return {"status": "all data deleted"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
