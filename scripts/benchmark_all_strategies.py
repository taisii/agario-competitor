"""Benchmark one strategy against seven copies of every saved strategy.

Each opponent is tested with the candidate in the first and last player slot.
This is deliberately stricter than a mixed-field screen: a win means the
candidate finished ahead of seven instances of the named opponent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

if __package__:
    from .benchmark_simulations import (
        DEFAULT_RANDOM_SEED,
        Variant,
        run_all,
        write_outputs,
    )
else:
    from benchmark_simulations import (
        DEFAULT_RANDOM_SEED,
        Variant,
        run_all,
        write_outputs,
    )


ROOT = Path(__file__).resolve().parents[1]
ENTRIES = ROOT / "bots" / "entries"
DEFAULT_OUTPUT = ROOT / ".agario" / "benchmarks" / "all-strategy-matrix"
IGNORED_ENTRIES = {
    "random_opponent",
    "random_replay_opponent",
}


def saved_strategy_names(candidate: str) -> tuple[str, ...]:
    names = {
        path.stem
        for path in ENTRIES.glob("*.py")
        if path.stem not in IGNORED_ENTRIES and path.stem != candidate
    }
    return tuple(sorted(names))


def summarize_opponent_rows(
    opponent: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """Summarize a matrix cell without treating failed matches as abstentions."""

    successful = [
        row
        for row in rows
        if row.get("return_code") == 0 and row.get("result_type") == "SUCCESS"
    ]
    ranked = [row for row in successful if row.get("tracked_ranks")]
    ranks = [int(row["tracked_ranks"][0]) for row in ranked]  # type: ignore[index]
    wins = sum(rank == 1 for rank in ranks)
    all_matches_valid = bool(rows) and len(ranked) == len(rows)
    return {
        "opponent": opponent,
        "matches": len(rows),
        "successful_matches": len(successful),
        "wins": wins,
        "average_rank": sum(ranks) / len(ranks) if ranks else None,
        "ranks": ranks,
        "passed": all_matches_valid and wins > len(rows) / 2,
    }


async def run_cell(
    *,
    candidate: str,
    opponent: str,
    candidate_slot: int,
    trials: int,
    output_root: Path,
    semaphore: asyncio.Semaphore,
    fast: bool,
) -> list[dict[str, object]]:
    if candidate_slot == 0:
        submissions = [
            f"1:bots/entries/{candidate}.py",
            f"7:bots/entries/{opponent}.py",
        ]
    elif candidate_slot == 7:
        submissions = [
            f"7:bots/entries/{opponent}.py",
            f"1:bots/entries/{candidate}.py",
        ]
    else:
        raise ValueError("candidate_slot must be 0 or 7")

    workspace = output_root / opponent / f"slot_{candidate_slot}"
    rows = await run_all(
        repo_root=ROOT,
        workspace_root=workspace,
        variants=[Variant(name="current", env={})],
        trials=trials,
        jobs=1,
        submissions=submissions,
        tracked_slots=(candidate_slot,),
        metrics_every_n="20",
        random_seed=DEFAULT_RANDOM_SEED,
        headless=True,
        fast=fast,
        semaphore=semaphore,
    )
    write_outputs(workspace, rows)
    return rows


async def run_matrix(args: argparse.Namespace) -> list[dict[str, object]]:
    opponents = tuple(args.opponents or saved_strategy_names(args.candidate))
    layouts = (0,) if args.single_layout else (0, 7)
    semaphore = asyncio.Semaphore(max(1, args.jobs))
    args.output_root.mkdir(parents=True, exist_ok=True)

    async def run_opponent(opponent: str) -> tuple[str, list[dict[str, object]]]:
        cells = await asyncio.gather(
            *(
                run_cell(
                    candidate=args.candidate,
                    opponent=opponent,
                    candidate_slot=candidate_slot,
                    trials=args.trials,
                    output_root=args.output_root,
                    semaphore=semaphore,
                    fast=not args.official,
                )
                for candidate_slot in layouts
            )
        )
        return opponent, [row for cell in cells for row in cell]

    completed = await asyncio.gather(*(run_opponent(name) for name in opponents))
    summary: list[dict[str, object]] = []
    for opponent, rows in sorted(completed):
        item = summarize_opponent_rows(opponent, rows)
        summary.append(item)
        print(
            f"{opponent:<36} wins={item['wins']}/{len(rows)} "
            f"avg_rank={item['average_rank']} passed={item['passed']}",
            flush=True,
        )
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="replay_dominance")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opponents", nargs="*")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--official",
        action="store_true",
        help=(
            "Use the recording-capable official launcher. The default matrix "
            "uses the engine-equivalent fast runner for high-throughput screening."
        ),
    )
    parser.add_argument(
        "--single-layout",
        action="store_true",
        help="Only test candidate slot 0; useful for an initial weakness screen.",
    )
    args = parser.parse_args()

    summary = asyncio.run(run_matrix(args))

    failed = [item["opponent"] for item in summary if not item["passed"]]
    if failed:
        raise SystemExit(f"Candidate did not beat: {', '.join(str(x) for x in failed)}")


if __name__ == "__main__":
    main()
