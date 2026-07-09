from __future__ import annotations

import argparse
import asyncio
import ast
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


BUILT_IN_VARIANTS = {
    "baseline": {
        "BOT_BEAM_TURN_PENALTY_WEIGHT": "0",
        "BOT_BEAM_KEEP_DIRECTION_CANDIDATE": "0",
    },
    "smooth": {
        "BOT_BEAM_TURN_PENALTY_WEIGHT": "1.4",
        "BOT_BEAM_KEEP_DIRECTION_CANDIDATE": "1",
    },
    "current": {},
}

OUTCOME_RE = re.compile(
    r"outcome was \{result_type='(?P<result_type>[^']+)' "
    r"ranking=(?P<ranking>\[[^\]]*\]) "
    r"final_masses=(?P<final_masses>\{[^}]*\})\}"
)
BANNED_OUTCOME_RE = re.compile(
    r"outcome was \{result_type='(?P<result_type>PLAYER_BANNED)' "
    r"ban_type='(?P<ban_type>[^']+)' player=(?P<player>\d+) "
    r"reason='(?P<reason>[^']+)'\}"
)


@dataclass(frozen=True)
class Variant:
    name: str
    env: dict[str, str]


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = resolve_workspace_root(repo_root, args.workspace_root)
    tracked_slots = parse_slots(args.tracked_slots)
    variants = [parse_variant(value) for value in args.variants]

    workspace_root.mkdir(parents=True, exist_ok=True)
    write_run_config(workspace_root, args, variants, tracked_slots)

    results = asyncio.run(
        run_all(
            repo_root=repo_root,
            workspace_root=workspace_root,
            variants=variants,
            trials=args.trials,
            jobs=args.jobs,
            submissions=args.submission,
            tracked_slots=tracked_slots,
            metrics_every_n=args.metrics_every_n,
            headless=not args.no_headless,
        )
    )
    write_outputs(workspace_root, results)
    print_summary(workspace_root, results, tracked_slots)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Agar.io simulations in parallel and aggregate outcomes."
    )
    parser.add_argument("--trials", type=int, default=4, help="Number of matches per variant.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=2,
        help="Parallel simulations. Each simulation launches the engine plus submissions.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "smooth"],
        help=(
            "Variant names. Built-ins: baseline, smooth, current. "
            "Custom form: name:KEY=VALUE,OTHER=VALUE"
        ),
    )
    parser.add_argument(
        "--submission",
        nargs="+",
        default=["4:bots/entries/food_greedy.py", "4:bots/entries/beam_survival.py"],
        help="Simulation submission specs. Counts must sum to 8.",
    )
    parser.add_argument(
        "--tracked-slots",
        default="4,5,6,7",
        help="Comma-separated player slots to score as the strategy under test.",
    )
    parser.add_argument(
        "--metrics-every-n",
        default="10",
        help="BOT_METRICS_EVERY_N value for submissions.",
    )
    parser.add_argument(
        "--workspace-root",
        help="Output root. Defaults to .agario/benchmarks/<timestamp>.",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Open the visualiser GUI. Not recommended for parallel benchmark runs.",
    )
    return parser.parse_args()


def resolve_workspace_root(repo_root: Path, workspace_root: str | None) -> Path:
    if workspace_root:
        return Path(workspace_root).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return repo_root / ".agario" / "benchmarks" / timestamp


def parse_slots(value: str) -> tuple[int, ...]:
    slots = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not slots:
        raise ValueError("--tracked-slots must contain at least one slot")
    return slots


def parse_variant(value: str) -> Variant:
    if value in BUILT_IN_VARIANTS:
        return Variant(name=value, env=dict(BUILT_IN_VARIANTS[value]))

    if ":" not in value:
        known = ", ".join(sorted(BUILT_IN_VARIANTS))
        raise ValueError(f"Unknown variant '{value}'. Use one of {known}, or name:KEY=VALUE.")

    name, raw_env = value.split(":", 1)
    env: dict[str, str] = {}
    for assignment in raw_env.split(","):
        if not assignment:
            continue
        key, separator, env_value = assignment.partition("=")
        if not separator:
            raise ValueError(f"Invalid variant assignment '{assignment}'")
        env[key] = env_value
    return Variant(name=name, env=env)


def write_run_config(
    workspace_root: Path,
    args: argparse.Namespace,
    variants: list[Variant],
    tracked_slots: tuple[int, ...],
) -> None:
    config = {
        "trials": args.trials,
        "jobs": args.jobs,
        "variants": [{"name": variant.name, "env": variant.env} for variant in variants],
        "submission": args.submission,
        "tracked_slots": tracked_slots,
        "metrics_every_n": args.metrics_every_n,
        "headless": not args.no_headless,
    }
    (workspace_root / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )


async def run_all(
    *,
    repo_root: Path,
    workspace_root: Path,
    variants: list[Variant],
    trials: int,
    jobs: int,
    submissions: list[str],
    tracked_slots: tuple[int, ...],
    metrics_every_n: str,
    headless: bool,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(jobs, 1))
    tasks = []
    for trial in range(trials):
        for variant in variants:
            tasks.append(
                asyncio.create_task(
                    run_match(
                        semaphore=semaphore,
                        repo_root=repo_root,
                        workspace_root=workspace_root,
                        variant=variant,
                        trial=trial,
                        submissions=submissions,
                        tracked_slots=tracked_slots,
                        metrics_every_n=metrics_every_n,
                        headless=headless,
                    )
                )
            )
    results: list[dict[str, Any]] = []
    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
        status = result["result_type"] if result["return_code"] == 0 else f"exit {result['return_code']}"
        print(
            f"[{result['variant']} #{result['trial']:03d}] {status} "
            f"elapsed={result['elapsed_seconds']:.1f}s workspace={result['workspace']}",
            flush=True,
        )
    return sorted(results, key=lambda item: (item["variant"], item["trial"]))


async def run_match(
    *,
    semaphore: asyncio.Semaphore,
    repo_root: Path,
    workspace_root: Path,
    variant: Variant,
    trial: int,
    submissions: list[str],
    tracked_slots: tuple[int, ...],
    metrics_every_n: str,
    headless: bool,
) -> dict[str, Any]:
    async with semaphore:
        run_dir = workspace_root / variant.name / f"run_{trial:03d}"
        simulation_workspace = run_dir / "match"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "simulation.log"
        env = os.environ.copy()
        env.update(variant.env)
        env["BOT_METRICS_EVERY_N"] = metrics_every_n
        env["PYTHONUNBUFFERED"] = "1"

        command = ["uv", "run", "simulation", *submissions]
        if headless:
            command.append("--headless")
        command.extend(["--workspace", str(simulation_workspace)])

        started = time.monotonic()
        with log_path.open("w") as log_file:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=repo_root,
                env=env,
                stdout=log_file,
                stderr=asyncio.subprocess.STDOUT,
            )
            return_code = await process.wait()

        elapsed_seconds = time.monotonic() - started
        outcome = parse_outcome(log_path)
        metrics = summarize_metrics(simulation_workspace, tracked_slots)
        result = {
            "variant": variant.name,
            "trial": trial,
            "workspace": str(run_dir),
            "simulation_workspace": str(simulation_workspace),
            "log_path": str(log_path),
            "return_code": return_code,
            "elapsed_seconds": elapsed_seconds,
            "command": command,
            "env": variant.env,
            **outcome,
            **score_outcome(outcome, tracked_slots),
            **metrics,
        }
        (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result


def parse_outcome(log_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "result_type": "MISSING_OUTCOME",
        "ranking": [],
        "final_masses": {},
    }
    for line in log_path.read_text(errors="replace").splitlines():
        match = OUTCOME_RE.search(line)
        if match:
            result["result_type"] = match.group("result_type")
            result["ranking"] = ast.literal_eval(match.group("ranking"))
            result["final_masses"] = {
                int(key): float(value)
                for key, value in ast.literal_eval(match.group("final_masses")).items()
            }
            continue

        banned_match = BANNED_OUTCOME_RE.search(line)
        if banned_match:
            result["result_type"] = banned_match.group("result_type")
            result["ban_type"] = banned_match.group("ban_type")
            result["banned_player"] = int(banned_match.group("player"))
            result["ban_reason"] = banned_match.group("reason")
    return result


def score_outcome(outcome: dict[str, Any], tracked_slots: tuple[int, ...]) -> dict[str, Any]:
    ranking = outcome.get("ranking") or []
    final_masses = outcome.get("final_masses") or {}
    rank_by_slot = {slot: index + 1 for index, slot in enumerate(ranking)}
    tracked_ranks = [rank_by_slot[slot] for slot in tracked_slots if slot in rank_by_slot]
    tracked_masses = [final_masses[slot] for slot in tracked_slots if slot in final_masses]
    return {
        "tracked_ranks": tracked_ranks,
        "tracked_best_rank": min(tracked_ranks) if tracked_ranks else None,
        "tracked_mean_rank": mean(tracked_ranks) if tracked_ranks else None,
        "tracked_top1": bool(ranking and ranking[0] in tracked_slots),
        "tracked_top4_count": sum(1 for slot in tracked_slots if rank_by_slot.get(slot, 99) <= 4),
        "tracked_mass_sum": sum(tracked_masses),
        "tracked_mass_mean": mean(tracked_masses) if tracked_masses else None,
        "tracked_mass_max": max(tracked_masses) if tracked_masses else None,
    }


def summarize_metrics(workspace: Path, tracked_slots: tuple[int, ...]) -> dict[str, Any]:
    elapsed_ms: list[float] = []
    dots_all: list[float] = []
    dots_predator_1: list[float] = []
    dots_predator_2: list[float] = []
    samples = 0

    for slot in tracked_slots:
        path = workspace / f"submission{slot}" / "bot_metrics.jsonl"
        rows = read_jsonl(path)
        samples += len(rows)
        elapsed_ms.extend(row.get("decision_elapsed_ms", 0.0) for row in rows)
        for previous, current in zip(rows, rows[1:]):
            dot = direction_dot(previous, current)
            if dot is None:
                continue
            dots_all.append(dot)
            predator_count = max(
                int(previous.get("predator_count") or 0),
                int(current.get("predator_count") or 0),
            )
            if predator_count >= 1:
                dots_predator_1.append(dot)
            if predator_count >= 2:
                dots_predator_2.append(dot)

    return {
        "metric_samples": samples,
        "decision_avg_ms": mean(elapsed_ms) if elapsed_ms else None,
        "decision_max_ms": max(elapsed_ms) if elapsed_ms else None,
        "direction_pairs": len(dots_all),
        "direction_avg_dot": mean(dots_all) if dots_all else None,
        "direction_reversals": sum(1 for dot in dots_all if dot < -0.25),
        "predator1_direction_pairs": len(dots_predator_1),
        "predator1_avg_dot": mean(dots_predator_1) if dots_predator_1 else None,
        "predator1_reversals": sum(1 for dot in dots_predator_1 if dot < -0.25),
        "predator2_direction_pairs": len(dots_predator_2),
        "predator2_avg_dot": mean(dots_predator_2) if dots_predator_2 else None,
        "predator2_reversals": sum(1 for dot in dots_predator_2 if dot < -0.25),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def direction_dot(previous: dict[str, Any], current: dict[str, Any]) -> float | None:
    previous_x = previous.get("decision_direction_x")
    previous_y = previous.get("decision_direction_y")
    current_x = current.get("decision_direction_x")
    current_y = current.get("decision_direction_y")
    if None in {previous_x, previous_y, current_x, current_y}:
        return None
    dot = previous_x * current_x + previous_y * current_y
    return max(-1.0, min(1.0, dot))


def write_outputs(workspace_root: Path, results: list[dict[str, Any]]) -> None:
    (workspace_root / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )

    fieldnames = [
        "variant",
        "trial",
        "return_code",
        "result_type",
        "tracked_best_rank",
        "tracked_mean_rank",
        "tracked_top1",
        "tracked_top4_count",
        "tracked_mass_sum",
        "tracked_mass_mean",
        "tracked_mass_max",
        "decision_avg_ms",
        "decision_max_ms",
        "direction_avg_dot",
        "direction_reversals",
        "predator1_avg_dot",
        "predator1_reversals",
        "predator2_avg_dot",
        "predator2_reversals",
        "elapsed_seconds",
        "workspace",
    ]
    with (workspace_root / "matches.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field) for field in fieldnames})


def print_summary(
    workspace_root: Path,
    results: list[dict[str, Any]],
    tracked_slots: tuple[int, ...],
) -> None:
    print(f"\nOutput: {workspace_root}")
    print(f"Tracked slots: {','.join(str(slot) for slot in tracked_slots)}")
    print(
        "variant  matches  success  top1_rate  avg_best_rank  avg_mean_rank  "
        "avg_mass_sum  avg_ms  max_ms  pred2_rev_rate"
    )
    for variant in sorted({result["variant"] for result in results}):
        rows = [result for result in results if result["variant"] == variant]
        successes = [row for row in rows if row["return_code"] == 0 and row["result_type"] == "SUCCESS"]
        pred2_pairs = sum(row.get("predator2_direction_pairs") or 0 for row in rows)
        pred2_reversals = sum(row.get("predator2_reversals") or 0 for row in rows)
        print(
            f"{variant:<8} "
            f"{len(rows):>7} "
            f"{len(successes):>7} "
            f"{avg_bool(rows, 'tracked_top1'):>9.3f} "
            f"{avg_number(rows, 'tracked_best_rank'):>13.3f} "
            f"{avg_number(rows, 'tracked_mean_rank'):>13.3f} "
            f"{avg_number(rows, 'tracked_mass_sum'):>12.3f} "
            f"{avg_number(rows, 'decision_avg_ms'):>7.3f} "
            f"{max_number(rows, 'decision_max_ms'):>6.3f} "
            f"{ratio(pred2_reversals, pred2_pairs):>14.3f}"
        )


def avg_bool(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return float("nan")
    return sum(1 for row in rows if row.get(field)) / len(rows)


def avg_number(rows: list[dict[str, Any]], field: str) -> float:
    values = [row[field] for row in rows if row.get(field) is not None]
    return mean(values) if values else float("nan")


def max_number(rows: list[dict[str, Any]], field: str) -> float:
    values = [row[field] for row in rows if row.get(field) is not None]
    return max(values) if values else float("nan")


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


if __name__ == "__main__":
    main()
