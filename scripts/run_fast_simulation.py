from __future__ import annotations

"""Run a headless match without the visualiser or its startup countdown.

This runner is only for high-throughput strategy experiments.  It uses the
same installed engine and submission processes as ``simulation``, but does not
record ``game.json`` or render a review video.  Final candidates must still be
checked with the official ``simulation --headless`` process layout.
"""

import argparse
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    submissions = parse_submission_specs(args.submission, NUM_PLAYERS)
    seeded_engine = Path(__file__).with_name("run_seeded_engine.py").resolve()
    setup_match_environment(workspace)
    env = runtime_env(workspace)
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
        with engine_log_path.open("w") as engine_log, engine_err_path.open(
            "w"
        ) as engine_err:
            engine = subprocess.Popen(
                [sys.executable, str(seeded_engine)],
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
