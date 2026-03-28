from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import webbrowser

import uvicorn


def _provision_cosmos() -> None:
    res = subprocess.run(
        [sys.executable, "scripts/provision_cosmos.py"],
        check=False,
        text=True,
    )
    if res.returncode != 0:
        raise RuntimeError("Cosmos provisioning failed. See output above.")


def _listening_pids_for_port(port: int) -> list[int]:
    """Return PIDs listening on TCP port (macOS/Linux best-effort via lsof)."""
    try:
        res = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            check=False,
            text=True,
            capture_output=True,
        )
    except Exception:  # noqa: BLE001
        return []
    if res.returncode not in (0, 1):
        return []
    out = (res.stdout or "").strip().splitlines()
    pids: list[int] = []
    for line in out:
        try:
            pids.append(int(line.strip()))
        except ValueError:
            continue
    return pids


def _port_in_use(host: str, port: int) -> bool:
    """Fast preflight check so we can fail with a clear message before uvicorn stack traces."""
    check_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((check_host, port)) == 0


def _kill_listeners(port: int) -> list[int]:
    pids = [p for p in _listening_pids_for_port(port) if p != os.getpid()]
    killed: list[int] = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except OSError:
            continue
    if killed:
        time.sleep(0.6)
    return killed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the app server (Azure Cosmos DB for NoSQL + optional provisioning)."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=9281, type=int)
    parser.add_argument("--reload", dest="reload", action="store_true", default=True)
    parser.add_argument("--no-reload", dest="reload", action="store_false")
    parser.add_argument("--open-browser", action="store_true", help="Open the UI in your browser.")
    parser.add_argument(
        "--kill-existing-port",
        action="store_true",
        help="If the selected port is already in use, terminate the existing listener first.",
    )

    parser.add_argument(
        "--provision",
        action="store_true",
        default=False,
        help="Provision Cosmos database/containers before starting the server.",
    )
    parser.add_argument(
        "--mock-openai",
        action="store_true",
        help="Set MOCK_OPENAI=true (no OpenAI network calls).",
    )

    args = parser.parse_args()

    if args.mock_openai:
        os.environ["MOCK_OPENAI"] = "true"

    if args.provision:
        _provision_cosmos()

    if args.kill_existing_port:
        killed = _kill_listeners(args.port)
        if killed:
            print(f"Killed existing listener(s) on port {args.port}: {', '.join(map(str, killed))}")

    if _port_in_use(args.host, args.port):
        print(
            f"Port {args.port} is already in use.\n"
            f"- Re-run with --kill-existing-port to auto-stop the current listener.\n"
            f"- Or stop it manually: lsof -ti tcp:{args.port} | xargs kill",
            file=sys.stderr,
        )
        raise SystemExit(1)

    url = f"http://localhost:{args.port}"
    if args.open_browser:
        webbrowser.open(url)

    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
