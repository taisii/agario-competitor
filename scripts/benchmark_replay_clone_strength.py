"""Measure replay-clone liveness and strength in a deterministic local league."""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import fmean
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bots"))

from scripts.benchmark_simulations import (  # noqa: E402
    DEFAULT_RANDOM_SEED,
    Variant,
    run_match,
)
from strategies.replay_opponents import (  # noqa: E402
    REPLAY_OPPONENT_SPECS,
    REPLAY_STRENGTH_CANDIDATE_TEAM_IDS,
)


ALL_SLOTS = tuple(range(8))


@dataclass(frozen=True)
class RuntimeStrengthGate:
    minimum_appearances: int = 8
    minimum_completion_rate: float = 1.0
    maximum_mean_rank: float = 4.5
    minimum_top_three_rate: float = 0.25


@dataclass(frozen=True)
class RuntimeStrength:
    team_id: int
    appearances: int
    completion_rate: float
    mean_rank: float | None
    top_three_rate: float

    def qualifies(self, gate: RuntimeStrengthGate) -> bool:
        return (
            self.appearances >= gate.minimum_appearances
            and self.completion_rate >= gate.minimum_completion_rate
            and self.mean_rank is not None
            and self.mean_rank <= gate.maximum_mean_rank
            and self.top_three_rate >= gate.minimum_top_three_rate
        )


def league_layouts(team_ids: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Use cyclic 8-player tables so every candidate appears exactly eight times."""

    if len(team_ids) < 8:
        raise ValueError("The balanced league requires at least eight live candidates")
    return tuple(
        tuple(team_ids[(start + offset) % len(team_ids)] for offset in range(8))
        for start in range(len(team_ids))
    )


def evaluation_entry(team_id: int) -> str:
    """Return the active or archived adapter used only for strength evaluation."""

    if team_id in REPLAY_OPPONENT_SPECS:
        return f"bots/entries/replay_team_{team_id}.py"
    return f"bots/replay_candidates/replay_team_{team_id}.py"


def summarise_runtime_strength(
    team_ids: tuple[int, ...],
    league_results: list[tuple[tuple[int, ...], dict[str, object]]],
) -> tuple[RuntimeStrength, ...]:
    appearances: dict[int, list[int | None]] = defaultdict(list)
    for layout, result in league_results:
        ranking = result.get("ranking")
        successful = result.get("result_type") == "SUCCESS" and isinstance(ranking, list)
        rank_by_slot = {int(slot): index + 1 for index, slot in enumerate(ranking or [])}
        for slot, team_id in enumerate(layout):
            appearances[team_id].append(rank_by_slot.get(slot) if successful else None)

    return tuple(
        RuntimeStrength(
            team_id=team_id,
            appearances=len(rows := appearances[team_id]),
            completion_rate=sum(rank is not None for rank in rows) / len(rows),
            mean_rank=(
                fmean(float(rank) for rank in rows if rank is not None)
                if all(rank is not None for rank in rows)
                else None
            ),
            top_three_rate=sum(rank is not None and rank <= 3 for rank in rows) / len(rows),
        )
        for team_id in team_ids
    )


async def evaluate(args: argparse.Namespace) -> dict[str, object]:
    semaphore = asyncio.Semaphore(1)
    candidate_team_ids = tuple(args.teams)
    liveness_results: dict[int, dict[str, object]] = {}
    for index, team_id in enumerate(candidate_team_ids):
        result = await run_match(
            semaphore=semaphore,
            repo_root=ROOT,
            workspace_root=args.workspace / "liveness",
            variant=Variant(
                name=f"team_{team_id}",
                env={"BOT_REPLAY_TEAM_ID": str(team_id)},
            ),
            trial=index,
            submissions=["8:bots/entries/replay_candidate.py"],
            tracked_slots=ALL_SLOTS,
            metrics_every_n="20",
            random_seed=args.random_seed,
            headless=True,
            fast=True,
        )
        liveness_results[team_id] = result

    live_team_ids = tuple(
        team_id
        for team_id in candidate_team_ids
        if liveness_results[team_id].get("result_type") == "SUCCESS"
    )
    layouts = league_layouts(live_team_ids)
    league_results: list[tuple[tuple[int, ...], dict[str, object]]] = []
    for match_index, layout in enumerate(layouts):
        result = await run_match(
            semaphore=semaphore,
            repo_root=ROOT,
            workspace_root=args.workspace / "league",
            variant=Variant(name=f"match_{match_index:02d}", env={}),
            trial=match_index,
            submissions=[f"1:{evaluation_entry(team_id)}" for team_id in layout],
            tracked_slots=ALL_SLOTS,
            metrics_every_n="20",
            random_seed=args.random_seed,
            headless=True,
            fast=True,
        )
        league_results.append((layout, result))

    strengths = summarise_runtime_strength(live_team_ids, league_results)
    gate = RuntimeStrengthGate()
    return {
        "candidate_team_ids": candidate_team_ids,
        "liveness": {
            str(team_id): {
                "result_type": result.get("result_type"),
                "ban_type": result.get("ban_type"),
                "banned_player": result.get("banned_player"),
            }
            for team_id, result in liveness_results.items()
        },
        "league": [
            {
                "team_ids": layout,
                "result_type": result.get("result_type"),
                "ranking": result.get("ranking"),
            }
            for layout, result in league_results
        ],
        "runtime_gate": asdict(gate),
        "strengths": [asdict(strength) for strength in strengths],
        "selected_team_ids": tuple(
            strength.team_id for strength in strengths if strength.qualifies(gate)
        ),
    }


def liveness_run_path(workspace: Path, team_id: int, trial: int) -> Path:
    return workspace / "liveness" / f"team_{team_id}" / f"run_{trial:03d}" / "result.json"


def load_liveness_results(
    workspace: Path,
    candidate_team_ids: tuple[int, ...],
) -> dict[int, dict[str, object]]:
    results: dict[int, dict[str, object]] = {}
    for trial, team_id in enumerate(candidate_team_ids):
        path = liveness_run_path(workspace, team_id, trial)
        if not path.is_file():
            raise SystemExit(f"Missing liveness result for team {team_id}: {path}")
        results[team_id] = json.loads(path.read_text())
    return results


async def run_one_liveness(args: argparse.Namespace, team_id: int) -> None:
    candidate_team_ids = tuple(args.teams)
    trial = candidate_team_ids.index(team_id)
    await run_match(
        semaphore=asyncio.Semaphore(1),
        repo_root=ROOT,
        workspace_root=args.workspace / "liveness",
        variant=Variant(name=f"team_{team_id}", env={"BOT_REPLAY_TEAM_ID": str(team_id)}),
        trial=trial,
        submissions=["8:bots/entries/replay_candidate.py"],
        tracked_slots=ALL_SLOTS,
        metrics_every_n="20",
        random_seed=args.random_seed,
        headless=True,
        fast=True,
    )


async def run_one_league_match(args: argparse.Namespace, match_index: int) -> None:
    liveness = load_liveness_results(args.workspace, tuple(args.teams))
    live_team_ids = tuple(
        team_id
        for team_id in args.teams
        if liveness[team_id].get("result_type") == "SUCCESS"
    )
    layouts = league_layouts(live_team_ids)
    try:
        layout = layouts[match_index]
    except IndexError as exc:
        raise SystemExit(f"League match index must be below {len(layouts)}") from exc
    await run_match(
        semaphore=asyncio.Semaphore(1),
        repo_root=ROOT,
        workspace_root=args.workspace / "league",
        variant=Variant(name=f"match_{match_index:02d}", env={}),
        trial=match_index,
        submissions=[f"1:{evaluation_entry(team_id)}" for team_id in layout],
        tracked_slots=ALL_SLOTS,
        metrics_every_n="20",
        random_seed=args.random_seed,
        headless=True,
        fast=True,
    )


def write_completed_report(args: argparse.Namespace) -> None:
    candidate_team_ids = tuple(args.teams)
    liveness_results = load_liveness_results(args.workspace, candidate_team_ids)
    live_team_ids = tuple(
        team_id
        for team_id in candidate_team_ids
        if liveness_results[team_id].get("result_type") == "SUCCESS"
    )
    layouts = league_layouts(live_team_ids)
    league_results: list[tuple[tuple[int, ...], dict[str, object]]] = []
    for match_index, layout in enumerate(layouts):
        path = args.workspace / "league" / f"match_{match_index:02d}" / f"run_{match_index:03d}" / "result.json"
        if not path.is_file():
            raise SystemExit(f"Missing league result {match_index}: {path}")
        league_results.append((layout, json.loads(path.read_text())))
    strengths = summarise_runtime_strength(live_team_ids, league_results)
    gate = RuntimeStrengthGate()
    report = {
        "candidate_team_ids": candidate_team_ids,
        "liveness": {
            str(team_id): {
                "result_type": result.get("result_type"),
                "ban_type": result.get("ban_type"),
                "banned_player": result.get("banned_player"),
            }
            for team_id, result in liveness_results.items()
        },
        "league": [
            {
                "team_ids": layout,
                "result_type": result.get("result_type"),
                "ranking": result.get("ranking"),
            }
            for layout, result in league_results
        ],
        "runtime_gate": asdict(gate),
        "strengths": [asdict(strength) for strength in strengths],
        "selected_team_ids": tuple(
            strength.team_id for strength in strengths if strength.qualifies(gate)
        ),
    }
    args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"selected replay clones: {', '.join(map(str, report['selected_team_ids']))}")
    print(f"report: {args.report_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teams",
        type=int,
        nargs="+",
        default=REPLAY_STRENGTH_CANDIDATE_TEAM_IDS,
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=ROOT / ".agario" / "replay-imitation" / "clone-strength",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=ROOT / "docs" / "replay-opponent-runtime-strength.json",
    )
    parser.add_argument("--liveness-team", type=int)
    parser.add_argument("--league-match", type=int)
    parser.add_argument("--write-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_stages = sum(
        value is not None
        for value in (args.liveness_team, args.league_match)
    ) + int(args.write_report)
    if selected_stages > 1:
        raise SystemExit("Choose only one staged evaluation action")
    if args.liveness_team is not None:
        if args.liveness_team not in args.teams:
            raise SystemExit("--liveness-team must be included in --teams")
        asyncio.run(run_one_liveness(args, args.liveness_team))
        return
    if args.league_match is not None:
        asyncio.run(run_one_league_match(args, args.league_match))
        return
    if args.write_report:
        write_completed_report(args)
        return
    report = asyncio.run(evaluate(args))
    args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"selected replay clones: {', '.join(map(str, report['selected_team_ids']))}")
    print(f"report: {args.report_out}")


if __name__ == "__main__":
    main()
