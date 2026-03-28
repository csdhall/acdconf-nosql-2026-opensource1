from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
STAMP = VENV_DIR / ".requirements-stamp"


def _venv_python() -> Path:
    candidates = [
        VENV_DIR / "bin" / "python",
        VENV_DIR / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> None:
    venv_python = _venv_python()

    if not venv_python.exists():
        _run([sys.executable, "-m", "venv", str(VENV_DIR)])
        venv_python = _venv_python()

    if not STAMP.exists() or REQUIREMENTS.stat().st_mtime > STAMP.stat().st_mtime:
        _run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
        _run([str(venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
        STAMP.touch()
    else:
        print("Virtualenv already bootstrapped.")


if __name__ == "__main__":
    main()
