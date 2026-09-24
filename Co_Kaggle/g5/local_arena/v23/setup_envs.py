#!/usr/bin/env python3
"""Create isolated v23 agent and replay virtual environments."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def env_python(env_dir: Path) -> Path:
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def create_env(env_dir: Path, requirements: Path) -> Path:
    if not env_python(env_dir).is_file():
        venv.EnvBuilder(with_pip=True).create(env_dir)
    python = env_python(env_dir)
    subprocess.run(
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(
        [str(python), "-m", "pip", "install", "-r", str(requirements)],
        check=True,
        cwd=ROOT,
    )
    return python


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent-only", action="store_true")
    p.add_argument("--replay-only", action="store_true")
    args = p.parse_args()
    if args.agent_only and args.replay_only:
        raise SystemExit("choose at most one of --agent-only or --replay-only")

    if not args.replay_only:
        agent_python = create_env(ROOT / ".venv-agent", ROOT / "requirements-agent.txt")
        print(f"agent python:  {agent_python}")
    if not args.agent_only:
        replay_python = create_env(ROOT / ".venv-replay", ROOT / "requirements-replay.txt")
        print(f"replay python: {replay_python}")

    print("v23 isolated environments ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
