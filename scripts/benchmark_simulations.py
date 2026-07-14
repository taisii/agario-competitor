from __future__ import annotations

import argparse
import asyncio
import ast
import csv
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


BUILT_IN_VARIANTS = {"current": {}}
DEFAULT_RANDOM_SEED = 20260712

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
            random_seed=args.random_seed,
            headless=not args.no_headless,
            fast=args.fast,
        )
    )
    write_outputs(workspace_root, results)
    print_summary(workspace_root, results, tracked_slots)
    comparisons = paired_mass_comparisons(results, baseline=args.baseline_variant)
    if comparisons:
        (workspace_root / "mass_comparisons.json").write_text(
            json.dumps(comparisons, indent=2, sort_keys=True) + "\n"
        )
        print_mass_comparisons(comparisons)
    if args.factorial_cells:
        factorial = two_factor_mass_analysis(
            results,
            cells=parse_factorial_cells(args.factorial_cells),
        )
        (workspace_root / "factorial_mass_analysis.json").write_text(
            json.dumps(factorial, indent=2, sort_keys=True) + "\n"
        )
        print_factorial_mass_analysis(factorial)
    if args.require_mass_improvement:
        failed = [
            row["variant"]
            for row in comparisons
            if not row["passed"]
        ]
        if failed:
            raise SystemExit(
                "Mean-final-mass gate failed for: "
                + ", ".join(str(name) for name in failed)
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Agar.io simulations in parallel and aggregate outcomes."
    )
    parser.add_argument("--trials", type=int, default=4, help="Number of matches per variant.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Concurrent matches. Each match already launches the engine and eight "
            "bots; raise only after verifying the submissions are lightweight."
        ),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["current"],
        help=(
            "Variant names. Built-in: current. "
            "Custom form: name:KEY=VALUE,OTHER=VALUE"
        ),
    )
    parser.add_argument(
        "--submission",
        nargs="+",
        default=["4:bots/entries/food_greedy.py", "4:bots/entries/survival_greedy.py"],
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
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=(
            "Base seed for random opponents. The same trial and player slot "
            "select the same opponent in every variant."
        ),
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
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Skip the visualiser, startup countdown, game recording, and review "
            "video. Intended for development screens, not final verification."
        ),
    )
    parser.add_argument(
        "--baseline-variant",
        default="current",
        help="Variant used for paired final-mass differences.",
    )
    parser.add_argument(
        "--require-mass-improvement",
        action="store_true",
        help=(
            "Fail unless every non-baseline match succeeds, its paired mean "
            "final-mass difference is positive, and the one-sided 95%% "
            "cluster-bootstrap lower bound is above zero."
        ),
    )
    parser.add_argument(
        "--factorial-cells",
        help=(
            "Report paired two-factor effects using four variant names in "
            "BASE,A,B,AB order. This separates each feature's conditional "
            "effect from their interaction instead of rejecting a feature "
            "from one combined result."
        ),
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


def parse_factorial_cells(value: str) -> dict[str, str]:
    names = tuple(part.strip() for part in value.split(","))
    if len(names) != 4 or any(not name for name in names):
        raise ValueError(
            "--factorial-cells must contain four variant names in BASE,A,B,AB order"
        )
    if len(set(names)) != 4:
        raise ValueError("--factorial-cells variant names must be distinct")
    return dict(zip(("base", "a", "b", "ab"), names, strict=True))


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
        "random_seed": args.random_seed,
        "headless": not args.no_headless,
        "fast": args.fast,
        "factorial_cells": args.factorial_cells,
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
    random_seed: int,
    headless: bool,
    fast: bool,
    semaphore: asyncio.Semaphore | None = None,
) -> list[dict[str, Any]]:
    if semaphore is None:
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
                        random_seed=random_seed,
                        headless=headless,
                        fast=fast,
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
    random_seed: int,
    headless: bool,
    fast: bool = False,
) -> dict[str, Any]:
    async with semaphore:
        run_dir = workspace_root / variant.name / f"run_{trial:03d}"
        simulation_workspace = run_dir / "match"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "simulation.log"
        env = os.environ.copy()
        env["BOT_BENCHMARK_TRIAL"] = str(trial)
        env["BOT_RANDOM_SEED"] = str(random_seed)
        env["AGARIO_ENGINE_RANDOM_SEED"] = str(random_seed + trial)
        env["BOT_METRICS_ENABLED"] = "1"
        env["BOT_METRICS_EVERY_N"] = metrics_every_n
        env["PYTHONUNBUFFERED"] = "1"

        if fast:
            env["BOT_BENCHMARK_VARIANT_ENV_JSON"] = json.dumps(variant.env)
            env["BOT_BENCHMARK_VARIANT_SLOTS"] = ",".join(
                str(slot) for slot in tracked_slots
            )
            command = [
                "uv",
                "run",
                "python",
                "scripts/run_fast_simulation.py",
                *submissions,
            ]
        else:
            # The official launcher provides one environment to every bot.
            # It remains a final-format smoke test; paired strategy comparisons
            # use the fast runner, which scopes variant values to tracked slots.
            env.update(variant.env)
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
            "random_seed": random_seed,
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
    competition_remaining_ms: list[float] = []
    competition_remaining_last_ms: list[float] = []
    dots_all: list[float] = []
    dots_predator_1: list[float] = []
    dots_predator_2: list[float] = []
    samples = 0
    proxy_only_turns = 0
    candidate_total = 0
    refined_total = 0
    exact_transitions = 0
    prey_turns = 0
    split_turns = 0
    observed_round_gaps = 0
    response_timings_path = workspace / "output" / "response_timings.json"
    response_timings = (
        json.loads(response_timings_path.read_text())
        if response_timings_path.exists()
        else {}
    )
    tracked_response_seconds = [
        float(response_timings[str(slot)])
        for slot in tracked_slots
        if str(slot) in response_timings
    ]

    for slot in tracked_slots:
        path = workspace / f"submission{slot}" / "bot_metrics.jsonl"
        rows = read_jsonl(path)
        samples += len(rows)
        elapsed_ms.extend(row.get("decision_elapsed_ms", 0.0) for row in rows)
        slot_remaining: list[float] = []
        for row in rows:
            diagnostics = row.get("decision_diagnostics", {})
            if not isinstance(diagnostics, dict):
                continue
            remaining = diagnostics.get("competition_compute_remaining_ms")
            if isinstance(remaining, (int, float)):
                slot_remaining.append(float(remaining))
            proxy_only_turns += bool(diagnostics.get("proxy_only"))
            candidate_total += int(
                diagnostics.get("fallback_candidates")
                or diagnostics.get("root_actions_generated")
                or 0
            )
            refined_total += int(diagnostics.get("proxy_candidates_refined") or 0)
            exact_transitions += int(diagnostics.get("transitions_evaluated") or 0)
            prey_turns += "prey" in str(row.get("decision_reason") or "")
            split_turns += bool(row.get("decision_split"))
        competition_remaining_ms.extend(slot_remaining)
        if slot_remaining:
            competition_remaining_last_ms.append(slot_remaining[-1])
        for previous, current in zip(rows, rows[1:]):
            previous_round = previous.get("round")
            current_round = current.get("round")
            if isinstance(previous_round, int) and isinstance(current_round, int):
                observed_round_gaps += max(0, current_round - previous_round - 1)
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
        "response_cumulative_sum_seconds": sum(tracked_response_seconds),
        "response_cumulative_mean_seconds": (
            mean(tracked_response_seconds) if tracked_response_seconds else None
        ),
        "response_cumulative_max_seconds": (
            max(tracked_response_seconds) if tracked_response_seconds else None
        ),
        "metric_samples": samples,
        "decision_total_ms": sum(elapsed_ms),
        "decision_avg_ms": mean(elapsed_ms) if elapsed_ms else None,
        "decision_p95_ms": percentile(elapsed_ms, 0.95),
        "decision_p99_ms": percentile(elapsed_ms, 0.99),
        "decision_max_ms": max(elapsed_ms) if elapsed_ms else None,
        "competition_remaining_min_ms": (
            min(competition_remaining_ms) if competition_remaining_ms else None
        ),
        "competition_remaining_last_ms": (
            min(competition_remaining_last_ms)
            if competition_remaining_last_ms
            else None
        ),
        "competition_exhausted_samples": sum(
            remaining <= 0.0 for remaining in competition_remaining_ms
        ),
        "proxy_only_turns": proxy_only_turns,
        "candidate_total": candidate_total,
        "candidate_avg": candidate_total / samples if samples else None,
        "refined_total": refined_total,
        "refined_avg": refined_total / samples if samples else None,
        "exact_transitions": exact_transitions,
        "prey_turns": prey_turns,
        "split_turns": split_turns,
        "observed_round_gaps": observed_round_gaps,
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


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


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
        "response_cumulative_sum_seconds",
        "response_cumulative_mean_seconds",
        "response_cumulative_max_seconds",
        "decision_total_ms",
        "decision_avg_ms",
        "decision_p95_ms",
        "decision_p99_ms",
        "decision_max_ms",
        "competition_remaining_min_ms",
        "competition_remaining_last_ms",
        "competition_exhausted_samples",
        "proxy_only_turns",
        "candidate_total",
        "candidate_avg",
        "refined_total",
        "refined_avg",
        "exact_transitions",
        "prey_turns",
        "split_turns",
        "observed_round_gaps",
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


def paired_mass_comparisons(
    results: list[dict[str, Any]],
    *,
    baseline: str = "current",
    bootstrap_samples: int = 20_000,
) -> list[dict[str, Any]]:
    """Compare variants on the leaderboard objective using paired trials.

    A failed or banned match contributes zero final mass instead of silently
    disappearing from the mean.  The confidence interval resamples whole
    trials, preserving the shared seed/opponent layout within each pair.
    """

    by_variant_trial = {
        (str(row["variant"]), int(row["trial"])): row for row in results
    }
    baseline_trials = sorted(
        trial
        for variant, trial in by_variant_trial
        if variant == baseline
    )
    if not baseline_trials:
        return []

    comparisons: list[dict[str, Any]] = []
    variants = sorted({str(row["variant"]) for row in results} - {baseline})
    for variant in variants:
        paired_trials = [
            trial
            for trial in baseline_trials
            if (variant, trial) in by_variant_trial
        ]
        differences: list[float] = []
        baseline_masses: list[float] = []
        variant_masses: list[float] = []
        valid_pairs = 0
        for trial in paired_trials:
            baseline_row = by_variant_trial[(baseline, trial)]
            variant_row = by_variant_trial[(variant, trial)]
            baseline_valid = _successful_mass_row(baseline_row)
            variant_valid = _successful_mass_row(variant_row)
            baseline_mass = (
                float(baseline_row["tracked_mass_sum"])
                if baseline_valid
                else 0.0
            )
            variant_mass = (
                float(variant_row["tracked_mass_sum"])
                if variant_valid
                else 0.0
            )
            valid_pairs += baseline_valid and variant_valid
            baseline_masses.append(baseline_mass)
            variant_masses.append(variant_mass)
            differences.append(variant_mass - baseline_mass)

        lower_bound = _bootstrap_mean_lower_bound(
            differences,
            samples=bootstrap_samples,
            seed=f"{baseline}:{variant}:{len(differences)}",
        )
        mean_difference = mean(differences) if differences else None
        all_pairs_valid = (
            bool(paired_trials)
            and len(paired_trials) == len(baseline_trials)
            and valid_pairs == len(paired_trials)
        )
        comparisons.append(
            {
                "baseline": baseline,
                "variant": variant,
                "pairs": len(paired_trials),
                "valid_pairs": valid_pairs,
                "all_pairs_valid": all_pairs_valid,
                "baseline_mass_mean": (
                    mean(baseline_masses) if baseline_masses else None
                ),
                "variant_mass_mean": mean(variant_masses) if variant_masses else None,
                "paired_mean_difference": mean_difference,
                "paired_differences": differences,
                "one_sided_95_lower_bound": lower_bound,
                "passed": (
                    all_pairs_valid
                    and mean_difference is not None
                    and mean_difference > 0.0
                    and lower_bound is not None
                    and lower_bound > 0.0
                ),
            }
        )
    return comparisons


def _successful_mass_row(row: dict[str, Any]) -> bool:
    return (
        row.get("return_code") == 0
        and row.get("result_type") == "SUCCESS"
        and row.get("tracked_mass_sum") is not None
    )


def two_factor_mass_analysis(
    results: list[dict[str, Any]],
    *,
    cells: dict[str, str],
) -> dict[str, Any]:
    """Measure conditional and interaction effects on paired match layouts.

    ``cells`` names the four variants for neither feature (``base``), only A,
    only B, and both (``ab``). Failed matches count as zero mass, matching the
    leaderboard-oriented comparison. Only trials present in all four cells are
    comparable; validity is reported separately so missing or banned runs are
    never mistaken for evidence about a feature.
    """

    required_cells = {"base", "a", "b", "ab"}
    if set(cells) != required_cells:
        raise ValueError("cells must define exactly base, a, b, and ab")
    if len(set(cells.values())) != 4:
        raise ValueError("factorial cell variants must be distinct")

    by_variant_trial = {
        (str(row["variant"]), int(row["trial"])): row for row in results
    }
    trials_by_cell = {
        cell: {
            trial
            for variant, trial in by_variant_trial
            if variant == variant_name
        }
        for cell, variant_name in cells.items()
    }
    paired_trials = sorted(set.intersection(*trials_by_cell.values()))

    trial_effects: list[dict[str, Any]] = []
    for trial in paired_trials:
        cell_rows = {
            cell: by_variant_trial[(variant_name, trial)]
            for cell, variant_name in cells.items()
        }
        masses = {
            cell: (
                float(row["tracked_mass_sum"])
                if _successful_mass_row(row)
                else 0.0
            )
            for cell, row in cell_rows.items()
        }
        trial_effects.append(
            {
                "trial": trial,
                "all_cells_valid": all(
                    _successful_mass_row(row) for row in cell_rows.values()
                ),
                "masses": masses,
                "a_effect_when_b_off": masses["a"] - masses["base"],
                "a_effect_when_b_on": masses["ab"] - masses["b"],
                "b_effect_when_a_off": masses["b"] - masses["base"],
                "b_effect_when_a_on": masses["ab"] - masses["a"],
                "interaction": (
                    masses["ab"] - masses["a"] - masses["b"] + masses["base"]
                ),
            }
        )

    def average(field: str) -> float | None:
        values = [float(row[field]) for row in trial_effects]
        return mean(values) if values else None

    cell_means = {
        cell: (
            mean(float(row["masses"][cell]) for row in trial_effects)
            if trial_effects
            else None
        )
        for cell in ("base", "a", "b", "ab")
    }
    conditional_effects = {
        "a_when_b_off": average("a_effect_when_b_off"),
        "a_when_b_on": average("a_effect_when_b_on"),
        "b_when_a_off": average("b_effect_when_a_off"),
        "b_when_a_on": average("b_effect_when_a_on"),
        "interaction": average("interaction"),
    }
    conditional_effects["a_marginal"] = _mean_optional(
        conditional_effects["a_when_b_off"],
        conditional_effects["a_when_b_on"],
    )
    conditional_effects["b_marginal"] = _mean_optional(
        conditional_effects["b_when_a_off"],
        conditional_effects["b_when_a_on"],
    )
    return {
        "cells": cells,
        "paired_trials": len(paired_trials),
        "valid_paired_trials": sum(
            bool(row["all_cells_valid"]) for row in trial_effects
        ),
        "cell_mass_means": cell_means,
        "mean_effects": conditional_effects,
        "trial_effects": trial_effects,
    }


def _mean_optional(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    return (first + second) / 2.0


def _bootstrap_mean_lower_bound(
    values: list[float],
    *,
    samples: int,
    seed: str,
) -> float | None:
    if not values:
        return None
    rng = random.Random(seed)
    count = len(values)
    bootstrapped = sorted(
        mean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(max(1, samples))
    )
    return bootstrapped[int(0.05 * (len(bootstrapped) - 1))]


def print_mass_comparisons(comparisons: list[dict[str, Any]]) -> None:
    print("\nPaired mean-final-mass comparisons (failures count as zero)")
    print("variant  pairs  valid  baseline  candidate  mean_diff  lower95  passed")
    for row in comparisons:
        print(
            f"{row['variant']:<8} "
            f"{row['pairs']:>5} "
            f"{row['valid_pairs']:>6} "
            f"{float(row['baseline_mass_mean']):>9.3f} "
            f"{float(row['variant_mass_mean']):>10.3f} "
            f"{float(row['paired_mean_difference']):>9.3f} "
            f"{float(row['one_sided_95_lower_bound']):>8.3f} "
            f"{row['passed']}"
        )


def print_factorial_mass_analysis(analysis: dict[str, Any]) -> None:
    cells = analysis["cells"]
    means = analysis["cell_mass_means"]
    effects = analysis["mean_effects"]
    print("\nPaired two-factor final-mass analysis (failures count as zero)")
    print(
        "cells: "
        f"base={cells['base']}  A={cells['a']}  "
        f"B={cells['b']}  AB={cells['ab']}"
    )
    print(
        f"paired trials: {analysis['paired_trials']}  "
        f"all valid: {analysis['valid_paired_trials']}"
    )
    print("cell  variant  mean_mass")
    for cell in ("base", "a", "b", "ab"):
        value = means[cell]
        rendered = "n/a" if value is None else f"{float(value):.3f}"
        print(f"{cell:<4}  {cells[cell]:<24}  {rendered:>9}")
    print("effect                  mean_mass_delta")
    for field in (
        "a_when_b_off",
        "a_when_b_on",
        "b_when_a_off",
        "b_when_a_on",
        "a_marginal",
        "b_marginal",
        "interaction",
    ):
        value = effects[field]
        rendered = "n/a" if value is None else f"{float(value):.3f}"
        print(f"{field:<23} {rendered:>15}")


def print_summary(
    workspace_root: Path,
    results: list[dict[str, Any]],
    tracked_slots: tuple[int, ...],
) -> None:
    print(f"\nOutput: {workspace_root}")
    print(f"Tracked slots: {','.join(str(slot) for slot in tracked_slots)}")
    print(
        "variant  matches  success  top1_rate  avg_best_rank  avg_mean_rank  "
        "avg_mass_sum  response_s  total_ms  p99_ms  min_remaining_ms  "
        "pred2_rev_rate"
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
            f"{avg_number(rows, 'response_cumulative_mean_seconds'):>10.3f} "
            f"{avg_number(rows, 'decision_total_ms'):>8.1f} "
            f"{max_number(rows, 'decision_p99_ms'):>7.3f} "
            f"{min_number(rows, 'competition_remaining_min_ms'):>16.1f} "
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


def min_number(rows: list[dict[str, Any]], field: str) -> float:
    values = [row[field] for row in rows if row.get(field) is not None]
    return min(values) if values else float("nan")


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


if __name__ == "__main__":
    main()
