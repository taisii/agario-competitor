from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

DEFAULT_POOL = [
    "1:bots/entries/beam_rl_balanced.py",
    "1:bots/entries/beam_rl_survival.py",
    "1:bots/entries/beam_rl_farmer.py",
    "1:bots/entries/beam_rl_hunter.py",
    "1:bots/entries/beam_rl_opportunist.py",
    "3:bots/entries/random_opponent.py",
]


def run_episode(project_root: Path, workspace_root: Path, log_path: Path, episode: int, specs: list[str]) -> None:
    workspace = workspace_root / f"episode_{episode:04d}"
    env = os.environ.copy()
    env["BOT_LOG_PATH"] = str(log_path.resolve())
    env.setdefault("BOT_LOG_SAMPLE_RATE", "1")
    command = ["uv", "run", "simulation", *specs, "--headless", "--workspace", str(workspace)]
    print("RUN", " ".join(command))
    subprocess.run(command, cwd=project_root, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect JSONL decisions from local headless simulations.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="agario-competitor project root")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--workspace-root", type=Path, default=Path(".agario/rl_collect"))
    parser.add_argument("--log-path", type=Path, default=Path("rl/rollouts.jsonl"))
    parser.add_argument("--spec", action="append", default=None, help="count:path spec; repeat to override the default 8-player pool")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    workspace_root = (project_root / args.workspace_root).resolve()
    log_path = (project_root / args.log_path).resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    specs = args.spec if args.spec else DEFAULT_POOL
    if log_path.exists():
        print(f"Appending to existing log: {log_path}")
    for episode in range(args.episodes):
        run_episode(project_root, workspace_root, log_path, episode, specs)
    print(f"wrote {log_path}")


if __name__ == "__main__":
    main()
