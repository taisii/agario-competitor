"""Run seeded matches and summarize the submitted bot's mean final mass."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
from statistics import fmean
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RANDOM_SEED = 20260712
DEFAULT_SUBMISSIONS = ("1:bots/my_bot.py", "7:bots/baseline_bot.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=ROOT
        / ".agario"
        / "benchmarks"
        / datetime.now().strftime("submission-%Y%m%d-%H%M%S"),
    )
    parser.add_argument(
        "--submission",
        nargs="+",
        default=list(DEFAULT_SUBMISSIONS),
        help="Bot specs whose counts sum to eight; tracked bot must occupy slot 0.",
    )
    args = parser.parse_args()
    if args.trials < 1 or args.jobs < 1:
        parser.error("--trials and --jobs must be positive")
    return args


def score_result(trial: int, report: dict[str, object]) -> dict[str, object]:
    successful = report.get("result_type") == "SUCCESS"
    final_masses = report.get("final_masses")
    ranking = report.get("ranking")
    mass = 0.0
    rank = None
    if successful and isinstance(final_masses, dict) and isinstance(ranking, list):
        mass = float(final_masses.get("0", final_masses.get(0, 0.0)))
        rank = ranking.index(0) + 1 if 0 in ranking else None
    return {
        "trial": trial,
        "result_type": report.get("result_type", "MISSING_OUTCOME"),
        "final_mass": mass,
        "rank": rank,
    }


def summarize(results: list[dict[str, object]]) -> dict[str, object]:
    masses = [float(result["final_mass"]) for result in results]
    ranks = [int(result["rank"]) for result in results if result["rank"] is not None]
    successful = sum(result["result_type"] == "SUCCESS" for result in results)
    return {
        "trials": len(results),
        "successful_trials": successful,
        "mean_final_mass": fmean(masses),
        "mean_rank": fmean(ranks) if ranks else None,
        "top_one_rate": sum(rank == 1 for rank in ranks) / len(results),
    }


async def run_trial(
    *,
    semaphore: asyncio.Semaphore,
    trial: int,
    workspace: Path,
    submissions: list[str],
    random_seed: int,
) -> dict[str, object]:
    async with semaphore:
        run_dir = workspace / f"run_{trial:03d}"
        match_dir = run_dir / "match"
        run_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(
            {
                "AGARIO_ENGINE_RANDOM_SEED": str(random_seed + trial),
                "AGARIO_LOCAL_RELAXED_PLAYER_IDS": "1,2,3,4,5,6,7",
                "AGARIO_LOCAL_TURN_TIMEOUT_SECONDS": "10",
                "AGARIO_LOCAL_CUMULATIVE_TIMEOUT_SECONDS": "60",
                "AGARIO_STRICT_TURN_TIMEOUT_SECONDS": "1",
                "AGARIO_STRICT_CUMULATIVE_TIMEOUT_SECONDS": "8",
                "BOT_BENCHMARK_VARIANT_SLOTS": "0",
                "BOT_METRICS_EVERY_N": "10",
                "PYTHONHASHSEED": "0",
            }
        )
        command = [
            sys.executable,
            "scripts/run_fast_simulation.py",
            *submissions,
            "--workspace",
            str(match_dir),
        ]
        started = time.monotonic()
        with (run_dir / "simulation.log").open("w") as log:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
            )
            return_code = await process.wait()

        report_path = match_dir / "output" / "results.json"
        report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        result = score_result(trial, report)
        result.update(
            {
                "return_code": return_code,
                "elapsed_seconds": time.monotonic() - started,
                "workspace": str(run_dir),
            }
        )
        (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            f"[trial {trial:03d}] {result['result_type']} "
            f"mass={result['final_mass']:.3f} rank={result['rank']}",
            flush=True,
        )
        return result


async def run(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.jobs)
    results = await asyncio.gather(
        *(
            run_trial(
                semaphore=semaphore,
                trial=trial,
                workspace=workspace,
                submissions=args.submission,
                random_seed=args.random_seed,
            )
            for trial in range(args.trials)
        )
    )
    results.sort(key=lambda result: int(result["trial"]))
    summary = summarize(results)
    (workspace / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (workspace / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return results, summary


def main() -> None:
    args = parse_args()
    _, summary = asyncio.run(run(args))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
