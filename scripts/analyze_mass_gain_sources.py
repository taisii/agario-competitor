from __future__ import annotations

"""Attribute gross mass gains and test semantic imitation of replay actions."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
BOTS = ROOT / "bots"
sys.path.insert(0, str(BOTS))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_strategy_decisions import _angle_degrees, _context  # noqa: E402
from replay_frames import extract_frames  # noqa: E402
from lib.config.player import FOOD_RADIUS  # noqa: E402
from strategies.receding_horizon import ReplayDominanceStrategy  # noqa: E402
from strategies.semantic_potential import SemanticPotentialStrategy  # noqa: E402


RESOURCE_KINDS = ("enemy", "virus", "food")


@dataclass(frozen=True, slots=True)
class Gain:
    count: int = 0
    mass: float = 0.0


def _extract_gain_timeline(
    events: list[dict[str, object]],
) -> tuple[dict[int, dict[int, dict[str, Gain]]], dict[int, float]]:
    gains: dict[int, dict[int, dict[str, Gain]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    peak_mass: dict[int, float] = defaultdict(float)
    virus_radii: dict[int, float] = {}
    round_number = -1
    in_move_batch = False

    def add(player_id: int, kind: str, *, count: int, mass: float) -> None:
        prior = gains[round_number][player_id].get(kind, Gain())
        gains[round_number][player_id][kind] = Gain(
            count=prior.count + count,
            mass=prior.mass + mass,
        )

    for event in events:
        event_type = event.get("event_type")
        if event_type == "event_game_started":
            for player in event.get("players", []):
                player_id = int(player["player_id"])
                peak_mass[player_id] = max(
                    peak_mass[player_id],
                    sum(float(blob["radius"]) ** 2 for blob in player.get("blobs", [])),
                )
            continue
        if event_type == "move_player":
            if not in_move_batch:
                round_number += 1
                in_move_batch = True
            continue

        in_move_batch = False
        if event_type == "event_food_eaten":
            food_count = len(event.get("food_ids", []))
            add(
                int(event["player_id"]),
                "food",
                count=food_count,
                mass=food_count * FOOD_RADIUS * FOOD_RADIUS,
            )
        elif event_type == "event_virus_spawned":
            for virus in event.get("viruses", []):
                virus_radii[int(virus["virus_id"])] = float(virus["radius"])
        elif event_type == "event_virus_consumed":
            virus_id = int(event["virus_id"])
            radius = virus_radii.pop(virus_id)
            add(
                int(event["player_id"]),
                "virus",
                count=1,
                mass=radius * radius,
            )
        elif event_type == "event_player_eaten":
            radius = float(event["eaten_radius"])
            add(
                int(event["eater_player_id"]),
                "enemy",
                count=1,
                mass=radius * radius,
            )
        elif event_type == "event_player_moved":
            player_id = int(event["player_id"])
            mass = sum(float(blob["radius"]) ** 2 for blob in event.get("blobs", []))
            peak_mass[player_id] = max(peak_mass[player_id], mass)
    return gains, peak_mass


def analyze(path: Path, *, player_id: int) -> dict[str, object]:
    events = json.loads(path.read_text(encoding="utf-8"))
    started = events[0]
    max_rounds = int(started.get("max_rounds", 1400))
    gains, peak_masses = _extract_gain_timeline(events)
    semantic = SemanticPotentialStrategy()
    replay_oracle = ReplayDominanceStrategy()

    total_rounds = 0
    oracle_direction_matches = 0
    oracle_split_matches = 0
    resource_rows = {
        kind: {
            "count": 0,
            "gross_mass": 0.0,
            "actual_split_mass": 0.0,
            "semantic_direction_match_mass": 0.0,
            "semantic_split_match_mass": 0.0,
            "semantic_full_match_mass": 0.0,
            "semantic_reason_mass": Counter(),
            "oracle_reason_mass": Counter(),
        }
        for kind in RESOURCE_KINDS
    }

    for frame in extract_frames(path):
        context = _context(frame, player_id=player_id, max_rounds=max_rounds)
        actual = frame.commands.get(player_id)
        if context is None or actual is None:
            continue
        semantic_decision = semantic.choose(context)
        oracle_decision = replay_oracle.choose(context)
        total_rounds += 1
        oracle_direction_matches += (
            _angle_degrees(actual.direction, oracle_decision.direction) <= 30.0
        )
        oracle_split_matches += actual.split == oracle_decision.split

        round_gains = gains.get(frame.round_number, {}).get(player_id, {})
        for kind, gain in round_gains.items():
            row = resource_rows[kind]
            direction_match = (
                _angle_degrees(actual.direction, semantic_decision.direction) <= 30.0
            )
            split_match = actual.split == semantic_decision.split
            row["count"] += gain.count
            row["gross_mass"] += gain.mass
            row["actual_split_mass"] += gain.mass * actual.split
            row["semantic_direction_match_mass"] += gain.mass * direction_match
            row["semantic_split_match_mass"] += gain.mass * split_match
            row["semantic_full_match_mass"] += gain.mass * (
                direction_match and split_match
            )
            row["semantic_reason_mass"][semantic_decision.reason] += gain.mass
            row["oracle_reason_mass"][oracle_decision.reason] += gain.mass

    total_gross_mass = sum(row["gross_mass"] for row in resource_rows.values())
    rendered_resources = {}
    for kind, row in resource_rows.items():
        gross_mass = float(row["gross_mass"])
        rendered_resources[kind] = {
            "count": row["count"],
            "gross_mass": gross_mass,
            "gross_mass_share": (
                gross_mass / total_gross_mass if total_gross_mass else 0.0
            ),
            "mean_mass_per_event": (gross_mass / row["count"] if row["count"] else 0.0),
            "actual_split_mass_share": (
                row["actual_split_mass"] / gross_mass if gross_mass else 0.0
            ),
            "semantic_direction_imitation_mass_rate": (
                row["semantic_direction_match_mass"] / gross_mass if gross_mass else 0.0
            ),
            "semantic_split_imitation_mass_rate": (
                row["semantic_split_match_mass"] / gross_mass if gross_mass else 0.0
            ),
            "semantic_full_imitation_mass_rate": (
                row["semantic_full_match_mass"] / gross_mass if gross_mass else 0.0
            ),
            "semantic_reason_mass": dict(row["semantic_reason_mass"].most_common()),
            "oracle_reason_mass": dict(row["oracle_reason_mass"].most_common()),
        }
    return {
        "player_id": player_id,
        "peak_mass": peak_masses.get(player_id, 0.0),
        "total_gross_resource_mass": total_gross_mass,
        "resources": rendered_resources,
        "replay_oracle_validation": {
            "rounds": total_rounds,
            "direction_within_30_degrees_rate": (
                oracle_direction_matches / total_rounds if total_rounds else 0.0
            ),
            "split_agreement_rate": (
                oracle_split_matches / total_rounds if total_rounds else 0.0
            ),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--player", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = analyze(args.replay.resolve(), player_id=args.player)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
