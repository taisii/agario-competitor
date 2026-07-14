from __future__ import annotations

"""Smoke-test every replay-derived opponent in parallel mixed matches."""

import argparse
import asyncio
import json
from pathlib import Path
import sys

if __package__:
    from .benchmark_simulations import parse_outcome
else:
    from benchmark_simulations import parse_outcome


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from strategies.replay_opponents import REPLAY_TEAM_IDS, replay_opponent_name  # noqa: E402


def discover_entries() -> dict[int, Path]:
    entries = {
        team_id: ROOT / "bots" / "entries" / f"{replay_opponent_name(team_id)}.py"
        for team_id in REPLAY_TEAM_IDS
    }
    missing = [team_id for team_id, path in entries.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing replay opponent entries for teams: {missing}")
    return entries


def batches(values: list[int], size: int = 8) -> list[list[int]]:
    result = [values[index : index + size] for index in range(0, len(values), size)]
    if result and len(result[-1]) < size:
        missing = size - len(result[-1])
        result[-1].extend(values[index % len(values)] for index in range(missing))
    return result


async def run_batch(
    *,
    index: int,
    team_ids: list[int],
    entries: dict[int, Path],
    workspace_root: Path,
    semaphore: asyncio.Semaphore,
    fast: bool,
) -> dict[str, object]:
    workspace = workspace_root / f"batch-{index:02d}"
    log_path = workspace_root / f"batch-{index:02d}.log"
    command = (
        ["uv", "run", "python", "scripts/run_fast_simulation.py"]
        if fast
        else ["uv", "run", "simulation"]
    )
    command.extend(
        f"1:{entries[team_id].relative_to(ROOT)}" for team_id in team_ids
    )
    if not fast:
        command.append("--headless")
    command.extend(["--workspace", str(workspace)])
    async with semaphore:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await process.communicate()
    workspace_root.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(output)
    result_path = workspace / "output/results.json"
    result = (
        parse_outcome(log_path)
        if fast
        else (json.loads(result_path.read_text()) if result_path.exists() else None)
    )
    success = (
        process.returncode == 0
        and result is not None
        and result.get("result_type") == "SUCCESS"
    )
    return {
        "batch": index,
        "team_ids": team_ids,
        "command": command,
        "workspace": str(workspace),
        "returncode": process.returncode,
        "result": result,
        "success": success,
    }


async def async_main(args: argparse.Namespace) -> int:
    entries = discover_entries()
    selected = sorted(entries)
    if args.teams:
        requested = {int(value) for value in args.teams.split(",") if value.strip()}
        missing = sorted(requested - entries.keys())
        if missing:
            raise SystemExit(f"Missing replay entries for teams: {missing}")
        selected = sorted(requested)
    if not selected:
        raise SystemExit("No replay team entries found")

    args.workspace.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.jobs)
    results = await asyncio.gather(
        *(
            run_batch(
                index=index,
                team_ids=team_ids,
                entries=entries,
                workspace_root=args.workspace,
                semaphore=semaphore,
                fast=not args.official,
            )
            for index, team_ids in enumerate(batches(selected))
        )
    )
    report = {
        "entry_count": len(selected),
        "teams": selected,
        "jobs": args.jobs,
        "batches": results,
        "success": all(result["success"] for result in results),
    }
    report_path = args.workspace / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for result in results:
        status = "PASS" if result["success"] else "FAIL"
        print(f"batch {result['batch']:02d} {status}: teams={result['team_ids']}")
    print(f"report: {report_path}")
    return 0 if report["success"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teams", help="comma-separated team IDs; default is every entry")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--official",
        action="store_true",
        help="Use the recording-capable official runner instead of the fast smoke runner",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=ROOT / ".agario/replay-imitation/simulations",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least one")
    return args


def main() -> None:
    raise SystemExit(asyncio.run(async_main(parse_args())))


if __name__ == "__main__":
    main()
