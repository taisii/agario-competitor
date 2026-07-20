from __future__ import annotations

"""Classify the state immediately before one player loses a fragment."""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Iterable


# Engine 2026.1.15 constants. Keep the replay classifier aligned with the
# official simulator; stale values materially change nearby-predator labels.
EAT_SIZE_RATIO = 1.2
SPLIT_MIN_MASS = 2.0
SPLIT_EJECT_SPEED = 1.6
SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True, slots=True)
class LossContext:
    path: str
    round_number: int
    eaten_blob_id: int
    eater_player_id: int
    lost_mass: float
    prior_total_mass: float
    prior_fragment_count: int
    lost_mass_share: float
    eater_split_this_round: bool
    target_split_age: int | None
    target_virus_age: int | None
    nearby_predator_players: int
    wall_clearance: float
    command_toward_eater: float | None


def _player_speed(radius: float) -> float:
    return 2.0 / max(radius, 1.0e-9)


def _attack_reach(predator_radius: float, prey_radius: float) -> float:
    reach = predator_radius + _player_speed(predator_radius)
    child_radius = predator_radius / SQRT2
    if (
        predator_radius * predator_radius >= SPLIT_MIN_MASS
        and child_radius * child_radius >= prey_radius * prey_radius * EAT_SIZE_RATIO
    ):
        reach = max(
            reach,
            3.0 * child_radius + SPLIT_EJECT_SPEED + _player_speed(child_radius),
        )
    return reach


def _target_player_id(
    started: dict[str, object],
    *,
    team_id: int | None,
    player_id: int | None,
) -> int:
    if (team_id is None) == (player_id is None):
        raise ValueError("provide exactly one of team_id and player_id")
    players = started["players"]
    if player_id is not None:
        return player_id
    matches = [
        int(player["player_id"])
        for player in players
        if int(player["team_id"]) == team_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one player for team {team_id}, found {len(matches)}")
    return matches[0]


def _blob_map(blobs: Iterable[dict[str, object]]) -> dict[int, dict[str, object]]:
    return {int(blob["blob_id"]): dict(blob) for blob in blobs}


def analyze_match(
    path: Path,
    *,
    team_id: int | None = None,
    player_id: int | None = None,
) -> tuple[LossContext, ...]:
    events = json.loads(path.read_text(encoding="utf-8"))
    if not events or events[0].get("event_type") != "event_game_started":
        raise ValueError(f"Replay has no initial game event: {path}")
    started = events[0]
    target = _target_player_id(started, team_id=team_id, player_id=player_id)
    arena_size = float(started.get("arena_size", 60.0))
    snapshots = {
        int(player["player_id"]): _blob_map(player.get("blobs", ()))
        for player in started["players"]
    }
    round_number = -1
    in_move_batch = False
    commands: dict[int, dict[str, object]] = {}
    last_target_split: int | None = None
    last_target_virus: int | None = None
    losses: list[LossContext] = []

    for event in events[1:]:
        event_type = event.get("event_type")
        if event_type == "move_player":
            if not in_move_batch:
                round_number += 1
                commands = {}
                in_move_batch = True
            actor = int(event["player_id"])
            commands[actor] = event
            if actor == target and bool(event.get("split")):
                last_target_split = round_number
            continue
        in_move_batch = False

        if event_type == "event_virus_consumed" and int(event["player_id"]) == target:
            last_target_virus = round_number
            continue
        if event_type == "event_player_moved":
            snapshots[int(event["player_id"])] = _blob_map(event.get("blobs", ()))
            continue
        if event_type != "event_player_eaten":
            continue

        eaten_player = int(event["eaten_player_id"])
        eater_player = int(event["eater_player_id"])
        eaten_blob_id = int(event["eaten_blob_id"])
        eaten_radius = float(event["eaten_radius"])
        eaten_pos = tuple(float(value) for value in event["eaten_pos"])
        if eaten_player == target:
            prior_blobs = snapshots.get(target, {})
            prior_total_mass = sum(
                float(blob["radius"]) ** 2 for blob in prior_blobs.values()
            )
            nearby_predators = {
                other_player
                for other_player, blobs in snapshots.items()
                if other_player != target
                and any(
                    float(blob["radius"]) ** 2
                    >= eaten_radius * eaten_radius * EAT_SIZE_RATIO
                    and math.dist(
                        tuple(float(value) for value in blob["pos"]),
                        eaten_pos,
                    )
                    <= _attack_reach(float(blob["radius"]), eaten_radius)
                    + _player_speed(eaten_radius) * 4.0
                    for blob in blobs.values()
                )
            }
            command = commands.get(target)
            command_toward_eater = None
            if command is not None:
                direction = command["direction"]
                dx = float(event["eater_pos"][0]) - eaten_pos[0]
                dy = float(event["eater_pos"][1]) - eaten_pos[1]
                distance = math.hypot(dx, dy)
                command_norm = math.hypot(float(direction["x"]), float(direction["y"]))
                if distance > 1.0e-9 and command_norm > 1.0e-9:
                    command_toward_eater = (
                        float(direction["x"]) * dx + float(direction["y"]) * dy
                    ) / (command_norm * distance)
            wall_clearance = min(
                eaten_pos[0] - eaten_radius,
                eaten_pos[1] - eaten_radius,
                arena_size - eaten_radius - eaten_pos[0],
                arena_size - eaten_radius - eaten_pos[1],
            )
            lost_mass = eaten_radius * eaten_radius
            losses.append(
                LossContext(
                    path=str(path.resolve()),
                    round_number=round_number,
                    eaten_blob_id=eaten_blob_id,
                    eater_player_id=eater_player,
                    lost_mass=lost_mass,
                    prior_total_mass=prior_total_mass,
                    prior_fragment_count=len(prior_blobs),
                    lost_mass_share=(
                        lost_mass / prior_total_mass if prior_total_mass else 1.0
                    ),
                    eater_split_this_round=bool(
                        commands.get(eater_player, {}).get("split")
                    ),
                    target_split_age=(
                        None
                        if last_target_split is None
                        else round_number - last_target_split
                    ),
                    target_virus_age=(
                        None
                        if last_target_virus is None
                        else round_number - last_target_virus
                    ),
                    nearby_predator_players=len(nearby_predators),
                    wall_clearance=wall_clearance,
                    command_toward_eater=command_toward_eater,
                )
            )

        snapshots.get(eaten_player, {}).pop(eaten_blob_id, None)
        eater_blob = snapshots.get(eater_player, {}).get(int(event["eater_blob_id"]))
        if eater_blob is not None:
            eater_blob["radius"] = float(event["eater_radius"])
            eater_blob["pos"] = list(event["eater_pos"])

    return tuple(losses)


def summarize(losses: Iterable[LossContext]) -> dict[str, object]:
    rows = tuple(losses)
    total_mass = sum(row.lost_mass for row in rows)

    def category(predicate) -> dict[str, float | int]:
        selected = tuple(row for row in rows if predicate(row))
        mass = sum(row.lost_mass for row in selected)
        return {
            "events": len(selected),
            "event_share": len(selected) / len(rows) if rows else 0.0,
            "lost_mass": mass,
            "lost_mass_share": mass / total_mass if total_mass else 0.0,
        }

    categories = {
        "eater_split_this_round": category(lambda row: row.eater_split_this_round),
        "target_split_within_40": category(
            lambda row: row.target_split_age is not None and row.target_split_age <= 40
        ),
        "target_virus_within_40": category(
            lambda row: row.target_virus_age is not None and row.target_virus_age <= 40
        ),
        "fragmented": category(lambda row: row.prior_fragment_count >= 2),
        "large_fragment_mass_ge_1": category(lambda row: row.lost_mass >= 1.0),
        "lost_share_ge_25pct": category(lambda row: row.lost_mass_share >= 0.25),
        "multiple_predator_players": category(
            lambda row: row.nearby_predator_players >= 2
        ),
        "near_wall_clearance_lt_3": category(lambda row: row.wall_clearance < 3.0),
        "moving_toward_eater": category(
            lambda row: row.command_toward_eater is not None
            and row.command_toward_eater > 0.25
        ),
    }
    return {
        "events": len(rows),
        "lost_mass": total_mass,
        "lost_mass_per_event": total_mass / len(rows) if rows else 0.0,
        "fragment_count_distribution": dict(
            sorted(Counter(row.prior_fragment_count for row in rows).items())
        ),
        "lost_mass_share_median": (
            statistics.median(row.lost_mass_share for row in rows) if rows else 0.0
        ),
        "categories": categories,
        "details": [asdict(row) for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_dir", type=Path)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--team-id", type=int)
    target.add_argument("--player-id", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = tuple(sorted(args.replay_dir.glob("match-*-replay.json")))
    if not paths:
        raise SystemExit(f"No replay files found under {args.replay_dir}")
    losses = tuple(
        loss
        for path in paths
        for loss in analyze_match(path, team_id=args.team_id, player_id=args.player_id)
    )
    report = summarize(losses)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
