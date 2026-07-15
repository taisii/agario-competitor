from __future__ import annotations

"""Run a headless match without the visualiser or its startup countdown.

This runner is only for high-throughput strategy experiments.  It uses the
same installed engine and submission processes as ``simulation``, but does not
record ``game.json`` by default. Pass ``--record`` when a deterministic
benchmark divergence needs replay-level diagnosis. Final candidates must still
be checked with the official ``simulation --headless`` process layout.
"""

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
from signal import SIGKILL
import subprocess
import sys

from agario_visualiser.launch_local_match import (
    parse_submission_specs,
    runtime_env,
    setup_match_environment,
)
from lib.config.arena import NUM_PLAYERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", nargs="+")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--record",
        action="store_true",
        help="Preserve game.json and visualiser events for replay diagnosis.",
    )
    return parser.parse_args()


def forward_engine_overrides(
    env: dict[str, str],
    source: Mapping[str, str] = os.environ,
) -> None:
    """Preserve explicit validation-only engine settings through runtime_env."""

    for key in (
        "AGARIO_LOCAL_CUMULATIVE_TIMEOUT_SECONDS",
        "AGARIO_LOCAL_TURN_TIMEOUT_SECONDS",
        "AGARIO_LOCAL_RELAXED_PLAYER_IDS",
        "AGARIO_STRICT_CUMULATIVE_TIMEOUT_SECONDS",
        "AGARIO_STRICT_TURN_TIMEOUT_SECONDS",
    ):
        if key in source:
            env[key] = source[key]


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    submissions = parse_submission_specs(args.submission, NUM_PLAYERS)
    seeded_engine = Path(__file__).with_name("run_seeded_engine.py").resolve()
    setup_match_environment(workspace)
    env = runtime_env(workspace)
    forward_engine_overrides(env)
    variant_env = json.loads(env.pop("BOT_BENCHMARK_VARIANT_ENV_JSON", "{}"))
    variant_slots = {
        int(value)
        for value in env.pop("BOT_BENCHMARK_VARIANT_SLOTS", "").split(",")
        if value
    }

    bots: list[subprocess.Popen[bytes]] = []
    opened_logs: list[object] = []
    try:
        for player_id, script in enumerate(submissions):
            io_dir = workspace / f"submission{player_id}" / "io"
            stdout_file = (io_dir / "submission.log").open("wb")
            stderr_file = (io_dir / "submission.err").open("wb")
            opened_logs.extend((stdout_file, stderr_file))
            bot_env = env.copy()
            bot_env["BOT_METRICS_ENABLED"] = "1" if player_id in variant_slots else "0"
            if player_id in variant_slots:
                bot_env.update(variant_env)
            bots.append(
                subprocess.Popen(
                    [sys.executable, str(script)],
                    cwd=workspace / f"submission{player_id}",
                    env=bot_env,
                    stdout=stdout_file,
                    stderr=stderr_file,
                )
            )

        engine_log_path = workspace / "output" / "engine.log"
        engine_err_path = workspace / "output" / "engine.err"
        with (
            engine_log_path.open("w") as engine_log,
            engine_err_path.open("w") as engine_err,
        ):
            engine_command = [sys.executable, str(seeded_engine)]
            if not args.record:
                engine_command.append("--no-recording")
            engine = subprocess.Popen(
                engine_command,
                cwd=workspace,
                env=env,
                stdout=subprocess.PIPE,
                stderr=engine_err,
                text=True,
                bufsize=1,
            )
            if engine.stdout is None:
                raise RuntimeError("engine stdout pipe was not created")
            for line in engine.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                engine_log.write(line)
            return_code = engine.wait()
            if return_code:
                raise SystemExit(return_code)
    finally:
        for bot in bots:
            if bot.poll() is None:
                try:
                    os.kill(bot.pid, SIGKILL)
                except ProcessLookupError:
                    pass
        for log_file in opened_logs:
            log_file.close()


if __name__ == "__main__":
    main()
