from __future__ import annotations

"""Benchmark one strategy against seven copies of every saved strategy.

Each opponent is tested with the candidate in the first and last player slot.
This is deliberately stricter than a mixed-field screen: a win means the
candidate finished ahead of seven instances of the named opponent.
"""

import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ENTRIES = ROOT / "bots" / "entries"
DEFAULT_OUTPUT = ROOT / ".agario" / "benchmarks" / "all-strategy-matrix"
IGNORED_ENTRIES = {
    "random_opponent",
    "random_replay_opponent",
    # Parameterised harness rather than one concrete strategy.  Its concrete
    # replay policies have dedicated replay_team_* entries and are included.
    "replay_profile",
}


def saved_strategy_names(candidate: str) -> tuple[str, ...]:
    names = {
        path.stem
        for path in ENTRIES.glob("*.py")
        if path.stem not in IGNORED_ENTRIES and path.stem != candidate
    }
    return tuple(sorted(names))


async def run_cell(
    *,
    candidate: str,
    opponent: str,
    candidate_slot: int,
    trials: int,
    output_root: Path,
    semaphore: asyncio.Semaphore,
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
    command = [
        sys.executable,
        str(ROOT / "scripts" / "benchmark_simulations.py"),
        "--trials",
        str(trials),
        "--jobs",
        "1",
        "--variants",
        "current",
        "--submission",
        *submissions,
        "--tracked-slots",
        str(candidate_slot),
        "--metrics-every-n",
        "20",
        "--workspace-root",
        str(workspace),
    ]
    async with semaphore:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=ROOT,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.STDOUT,
        )
        return_code = await process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return json.loads((workspace / "results.json").read_text())


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
                )
                for candidate_slot in layouts
            )
        )
        return opponent, [row for cell in cells for row in cell]

    completed = await asyncio.gather(*(run_opponent(name) for name in opponents))
    summary: list[dict[str, object]] = []
    for opponent, rows in sorted(completed):
        successful = [row for row in rows if row.get("result_type") == "SUCCESS"]
        ranks = [
            int(row["tracked_ranks"][0])
            for row in successful
            if row.get("tracked_ranks")
        ]
        item = {
            "opponent": opponent,
            "matches": len(rows),
            "successful_matches": len(successful),
            "wins": sum(rank == 1 for rank in ranks),
            "average_rank": sum(ranks) / len(ranks) if ranks else None,
            "ranks": ranks,
            "passed": bool(ranks) and sum(rank == 1 for rank in ranks) > len(ranks) / 2,
        }
        summary.append(item)
        print(
            f"{opponent:<36} wins={item['wins']}/{len(ranks)} "
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
