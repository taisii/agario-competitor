"""Rank replay opponents from official results and select a challenging panel."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bots"))

from scripts.analyze_official_match import analyze  # noqa: E402
from strategies.replay_opponents import (  # noqa: E402
    OBSERVED_REPLAY_TEAM_IDS,
    REPLAY_STRENGTH_CANDIDATE_TEAM_IDS,
    REPLAY_TEAM_IDS,
)


@dataclass(frozen=True)
class StrengthGate:
    minimum_matches: int = 3
    maximum_mean_rank: float = 4.0
    minimum_top_three_rate: float = 1.0 / 3.0
    minimum_mean_kills: float = 4.0
    minimum_mean_splits: float = 1.0


@dataclass(frozen=True)
class TeamStrength:
    team_id: int
    matches: int
    mean_rank: float
    top_three_rate: float
    mean_kills: float
    mean_splits: float

    def qualifies(self, gate: StrengthGate) -> bool:
        return (
            self.matches >= gate.minimum_matches
            and self.mean_rank <= gate.maximum_mean_rank
            and self.top_three_rate >= gate.minimum_top_three_rate
            and self.mean_kills >= gate.minimum_mean_kills
            and self.mean_splits >= gate.minimum_mean_splits
        )


def team_strengths(match_results: list[dict[str, object]]) -> tuple[TeamStrength, ...]:
    """Aggregate final placement and tactical activity for each official team."""

    rows_by_team: dict[int, list[dict[str, object]]] = defaultdict(list)
    for match in match_results:
        for player in match["players"]:  # type: ignore[index]
            assert isinstance(player, dict)
            team_id = int(player["team_id"])
            if team_id != 73:
                rows_by_team[team_id].append(player)
    return tuple(
        sorted(
            (
                TeamStrength(
                    team_id=team_id,
                    matches=len(rows),
                    mean_rank=statistics.fmean(float(row["final_rank"]) for row in rows),
                    top_three_rate=(
                        sum(float(row["final_rank"]) <= 3 for row in rows) / len(rows)
                    ),
                    mean_kills=statistics.fmean(float(row["kills"]) for row in rows),
                    mean_splits=statistics.fmean(float(row["splits"]) for row in rows),
                )
                for team_id, rows in rows_by_team.items()
            ),
            key=lambda strength: (strength.mean_rank, -strength.top_three_rate),
        )
    )


def selected_team_ids(
    strengths: tuple[TeamStrength, ...],
    *,
    gate: StrengthGate = StrengthGate(),
    runtime_selected_team_ids: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """Return teams that clear official and clone-runtime strength gates."""

    observed = set(OBSERVED_REPLAY_TEAM_IDS)
    official_selected = tuple(
        sorted(
            strength.team_id
            for strength in strengths
            if (
                strength.team_id in observed
                and strength.qualifies(gate)
            )
        )
    )
    if runtime_selected_team_ids is None:
        return official_selected
    runtime_selected = set(runtime_selected_team_ids)
    return tuple(team_id for team_id in official_selected if team_id in runtime_selected)


def load_runtime_selected_team_ids(path: Path) -> tuple[int, ...]:
    """Load the reproducible clone-league verdict used by the public catalog."""

    report = json.loads(path.read_text())
    candidates = tuple(int(team_id) for team_id in report["candidate_team_ids"])
    if candidates != REPLAY_STRENGTH_CANDIDATE_TEAM_IDS:
        raise ValueError(
            "Runtime strength report candidates do not match the official-results "
            "candidate set"
        )
    gate = report["runtime_gate"]
    measured_selection = tuple(
        int(strength["team_id"])
        for strength in report["strengths"]
        if (
            int(strength["appearances"]) >= int(gate["minimum_appearances"])
            and float(strength["completion_rate"])
            >= float(gate["minimum_completion_rate"])
            and strength["mean_rank"] is not None
            and float(strength["mean_rank"]) <= float(gate["maximum_mean_rank"])
            and float(strength["top_three_rate"])
            >= float(gate["minimum_top_three_rate"])
        )
    )
    reported_selection = tuple(int(team_id) for team_id in report["selected_team_ids"])
    if reported_selection != measured_selection:
        raise ValueError(
            "Runtime strength report selection does not match its measured gate"
        )
    return measured_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay-dir",
        action="append",
        type=Path,
        required=True,
        help="Official replay directory; repeat to combine cohorts",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=ROOT / ".agario" / "replay-imitation" / "strength-report.json",
    )
    parser.add_argument(
        "--runtime-report",
        type=Path,
        default=ROOT / "docs" / "replay-opponent-runtime-strength.json",
        help="Recorded balanced-league clone strength report",
    )
    parser.add_argument(
        "--verify-catalog",
        action="store_true",
        help="Fail unless the evaluated selection exactly matches the active catalog",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay_paths = sorted(
        replay_path
        for replay_dir in args.replay_dir
        for replay_path in replay_dir.glob("match-*-replay.json")
    )
    if not replay_paths:
        raise SystemExit("No replay files found")

    completed: list[dict[str, object]] = []
    incomplete: list[str] = []
    for replay_path in replay_paths:
        try:
            completed.append(analyze(replay_path))
        except StopIteration:
            incomplete.append(replay_path.name)

    strengths = team_strengths(completed)
    runtime_selected = load_runtime_selected_team_ids(args.runtime_report)
    selected = selected_team_ids(
        strengths,
        runtime_selected_team_ids=runtime_selected,
    )
    if args.verify_catalog and selected != REPLAY_TEAM_IDS:
        raise SystemExit(
            "Replay opponent catalog is stale: "
            f"expected {selected}, found {REPLAY_TEAM_IDS}"
        )
    report = {
        "replay_count": len(replay_paths),
        "completed_match_count": len(completed),
        "incomplete_replays": incomplete,
        "gate": asdict(StrengthGate()),
        "runtime_report": str(args.runtime_report),
        "runtime_selected_team_ids": runtime_selected,
        "selected_team_ids": selected,
        "teams": [asdict(strength) for strength in strengths],
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"selected replay opponents: {', '.join(map(str, selected))}")
    print(f"report: {args.report_out}")


if __name__ == "__main__":
    main()
