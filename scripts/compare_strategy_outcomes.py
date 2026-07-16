from __future__ import annotations

"""Compare paired strategy replays using outcome metrics, not action imitation."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics


CHECKPOINT_ROUNDS = (350, 700, 1050)


@dataclass(frozen=True, slots=True)
class OutcomeMetrics:
    final_mass: float
    final_alive: bool
    final_blob_count: int
    max_mass: float
    early_mean_mass: float
    middle_mean_mass: float
    late_mean_mass: float
    mass_at_350: float
    mass_at_700: float
    mass_at_1050: float
    deaths: int
    death_round: int | None
    captures: int
    captured_mass: float
    first_capture_round: int | None
    blobs_lost: int
    lost_mass: float
    viruses: int
    food: int

    @property
    def net_player_mass(self) -> float:
        return self.captured_mass - self.lost_mass


def analyze_replay(path: Path, *, player_id: int) -> OutcomeMetrics:
    events = json.loads(path.read_text(encoding="utf-8"))
    started = events[0]
    initial = next(
        player
        for player in started["players"]
        if int(player["player_id"]) == player_id
    )
    current_mass = sum(
        float(blob["radius"]) ** 2 for blob in initial.get("blobs", ())
    )
    current_alive = bool(initial.get("alive", True))
    current_blob_count = len(initial.get("blobs", ()))
    masses_by_round: dict[int, float] = {-1: current_mass}
    round_number = -1
    in_move_batch = False
    deaths = 0
    death_round: int | None = None
    captures = 0
    captured_mass = 0.0
    first_capture_round: int | None = None
    blobs_lost = 0
    lost_mass = 0.0
    viruses = 0
    food = 0

    for event in events[1:]:
        event_type = event.get("event_type")
        if event_type == "move_player":
            if not in_move_batch:
                round_number += 1
                in_move_batch = True
            continue
        in_move_batch = False
        if (
            event_type == "event_player_moved"
            and int(event["player_id"]) == player_id
        ):
            prior_alive = current_alive
            current_alive = bool(event.get("alive", True))
            blobs = event.get("blobs", ())
            current_blob_count = len(blobs)
            current_mass = sum(
                float(blob["radius"]) ** 2 for blob in blobs
            )
            masses_by_round[round_number] = current_mass
            if prior_alive and not current_alive:
                deaths += 1
                death_round = round_number
        elif event_type == "event_player_eaten":
            mass = float(event["eaten_radius"]) ** 2
            if int(event["eater_player_id"]) == player_id:
                captures += 1
                captured_mass += mass
                if first_capture_round is None:
                    first_capture_round = round_number
            if int(event["eaten_player_id"]) == player_id:
                blobs_lost += 1
                lost_mass += mass
        elif (
            event_type == "event_virus_consumed"
            and int(event["player_id"]) == player_id
        ):
            viruses += 1
        elif (
            event_type == "event_food_eaten"
            and int(event["player_id"]) == player_id
        ):
            food += len(event.get("food_ids", ()))

    max_rounds = int(started.get("max_rounds", 1400))
    timeline = _filled_timeline(
        masses_by_round,
        max_rounds=max_rounds,
        initial_mass=masses_by_round[-1],
    )
    third = max_rounds // 3
    return OutcomeMetrics(
        final_mass=current_mass,
        final_alive=current_alive,
        final_blob_count=current_blob_count,
        max_mass=max(timeline),
        early_mean_mass=statistics.fmean(timeline[:third]),
        middle_mean_mass=statistics.fmean(timeline[third : third * 2]),
        late_mean_mass=statistics.fmean(timeline[third * 2 :]),
        mass_at_350=_checkpoint(timeline, CHECKPOINT_ROUNDS[0]),
        mass_at_700=_checkpoint(timeline, CHECKPOINT_ROUNDS[1]),
        mass_at_1050=_checkpoint(timeline, CHECKPOINT_ROUNDS[2]),
        deaths=deaths,
        death_round=death_round,
        captures=captures,
        captured_mass=captured_mass,
        first_capture_round=first_capture_round,
        blobs_lost=blobs_lost,
        lost_mass=lost_mass,
        viruses=viruses,
        food=food,
    )


def _filled_timeline(
    masses_by_round: dict[int, float],
    *,
    max_rounds: int,
    initial_mass: float,
) -> list[float]:
    timeline: list[float] = []
    current = initial_mass
    for round_number in range(max_rounds):
        current = masses_by_round.get(round_number, current)
        timeline.append(current)
    return timeline


def _checkpoint(timeline: list[float], round_number: int) -> float:
    return timeline[min(max(round_number, 0), len(timeline) - 1)]


def compare(root: Path, *, player_id: int) -> dict[str, object]:
    rows: dict[str, dict[int, OutcomeMetrics]] = {
        "semantic": {},
        "replay": {},
    }
    for strategy in rows:
        for workspace in sorted(root.glob(f"{strategy}_*")):
            trial = int(workspace.name.rsplit("_", 1)[1])
            replay = workspace / "output" / "game.json"
            if replay.exists():
                rows[strategy][trial] = analyze_replay(
                    replay,
                    player_id=player_id,
                )

    paired_trials = sorted(set(rows["semantic"]) & set(rows["replay"]))
    aggregates = {
        strategy: _aggregate([rows[strategy][trial] for trial in paired_trials])
        for strategy in rows
    }
    paired = {
        metric: statistics.fmean(
            getattr(rows["replay"][trial], metric)
            - getattr(rows["semantic"][trial], metric)
            for trial in paired_trials
        )
        if paired_trials
        else 0.0
        for metric in (
            "final_mass",
            "middle_mean_mass",
            "mass_at_700",
            "captures",
            "captured_mass",
            "blobs_lost",
            "lost_mass",
            "net_player_mass",
        )
    }
    return {
        "root": str(root.resolve()),
        "player_id": player_id,
        "paired_trials": paired_trials,
        "strategies": aggregates,
        "paired_replay_minus_semantic": paired,
        "teacher_recommendation": _teacher_recommendation(aggregates),
        "matches": {
            strategy: {
                str(trial): asdict(rows[strategy][trial])
                | {
                    "net_player_mass": rows[strategy][trial].net_player_mass,
                }
                for trial in paired_trials
            }
            for strategy in rows
        },
    }


def _aggregate(rows: list[OutcomeMetrics]) -> dict[str, float | int]:
    if not rows:
        return {"matches": 0}
    numeric_fields = (
        "final_mass",
        "max_mass",
        "early_mean_mass",
        "middle_mean_mass",
        "late_mean_mass",
        "mass_at_350",
        "mass_at_700",
        "mass_at_1050",
        "deaths",
        "captures",
        "captured_mass",
        "blobs_lost",
        "lost_mass",
        "net_player_mass",
        "viruses",
        "food",
    )
    return {
        "matches": len(rows),
        "survival_rate": sum(row.final_alive for row in rows) / len(rows),
        **{
            f"mean_{field}": statistics.fmean(
                getattr(row, field) for row in rows
            )
            for field in numeric_fields
        },
    }


def _teacher_recommendation(
    aggregates: dict[str, dict[str, float | int]],
) -> dict[str, str]:
    def higher(metric: str) -> str:
        return max(aggregates, key=lambda name: float(aggregates[name][metric]))

    def lower(metric: str) -> str:
        return min(aggregates, key=lambda name: float(aggregates[name][metric]))

    return {
        "survival": higher("survival_rate"),
        "early_growth": higher("mean_early_mean_mass"),
        "middle_game": higher("mean_middle_mean_mass"),
        "capture": higher("mean_net_player_mass"),
        "loss_avoidance": lower("mean_lost_mass"),
        "endgame": higher("mean_final_mass"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--player", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = compare(args.root, player_id=args.player)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
