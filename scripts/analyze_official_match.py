from __future__ import annotations

"""Compare player behaviour in an official replay against the match winner."""

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
BOTS = ROOT / "bots"
sys.path.insert(0, str(BOTS))

from lib.config.player import EAT_SIZE_RATIO  # noqa: E402


VISION_REFERENCE_SUM_OF_RADII = 12.0


@dataclass(frozen=True)
class Blob:
    blob_id: int
    x: float
    y: float
    radius: float

    @property
    def mass(self) -> float:
        return self.radius * self.radius


@dataclass
class Player:
    player_id: int
    team_id: int
    alive: bool
    blobs: dict[int, Blob]

    @property
    def mass(self) -> float:
        return sum(blob.mass for blob in self.blobs.values())


@dataclass
class Metrics:
    player_id: int
    team_id: int
    final_mass: float = 0.0
    max_mass: float = 0.0
    mean_mass: float = 0.0
    endgame_mean_mass: float = 0.0
    endgame_mass_delta: float = 0.0
    splits: int = 0
    reversals: int = 0
    food: int = 0
    viruses: int = 0
    kills: int = 0
    blobs_lost: int = 0
    deaths: int = 0
    death_rounds: list[int] = field(default_factory=list)
    major_losses: list[dict[str, float | int]] = field(default_factory=list)
    eaten_events: list[dict[str, float | int | bool]] = field(default_factory=list)
    prey_opportunities: int = 0
    prey_approaches: int = 0
    prey_retreats: int = 0
    prey_alignment_mean: float = 0.0
    prey_split_attempts: int = 0
    edge_prey_opportunities: int = 0
    edge_prey_pursuits: int = 0
    edge_prey_split_attempts: int = 0
    late_prey_opportunities: int = 0
    late_prey_approaches: int = 0
    late_prey_split_attempts: int = 0
    rounds_alive: int = 0


def _unit(x: float, y: float) -> tuple[float, float]:
    length = math.hypot(x, y)
    if length <= 1e-9:
        return (0.0, 0.0)
    return (x / length, y / length)


def _player(payload: dict[str, object], team_id: int) -> Player:
    player_id = int(payload["player_id"])
    blobs = {
        int(blob["blob_id"]): Blob(
            blob_id=int(blob["blob_id"]),
            x=float(blob["pos"][0]),
            y=float(blob["pos"][1]),
            radius=float(blob["radius"]),
        )
        for blob in payload.get("blobs", [])
    }
    return Player(
        player_id=player_id,
        team_id=team_id,
        alive=bool(payload.get("alive", True)),
        blobs=blobs,
    )


def _center(player: Player) -> tuple[float, float]:
    mass = player.mass
    if mass <= 1e-9:
        return (30.0, 30.0)
    return (
        sum(blob.x * blob.mass for blob in player.blobs.values()) / mass,
        sum(blob.y * blob.mass for blob in player.blobs.values()) / mass,
    )


def _vision_size(player: Player, base_vision: float) -> float:
    sum_radii = sum(blob.radius for blob in player.blobs.values())
    if sum_radii <= 0.0:
        return base_vision
    return max(sum_radii / VISION_REFERENCE_SUM_OF_RADII, 1.0) ** 0.4 * base_vision


def _view_center(
    player: Player, arena_size: float, vision_size: float
) -> tuple[float, float]:
    x, y = _center(player)
    half = min(vision_size, arena_size) / 2.0
    return (
        min(max(x, half), arena_size - half),
        min(max(y, half), arena_size - half),
    )


def _visible(
    center: tuple[float, float], vision_size: float, blob: Blob
) -> bool:
    half = vision_size / 2.0
    dx = max(abs(blob.x - center[0]) - half, 0.0)
    dy = max(abs(blob.y - center[1]) - half, 0.0)
    return dx * dx + dy * dy <= blob.radius * blob.radius


def _can_eat(eater: Blob, target: Blob) -> bool:
    return eater.mass >= target.mass * EAT_SIZE_RATIO


def _prey_target(
    player: Player,
    players: dict[int, Player],
    arena_size: float,
    base_vision: float,
) -> Blob | None:
    vision_size = _vision_size(player, base_vision)
    view_center = _view_center(player, arena_size, vision_size)
    edible = [
        enemy_blob
        for enemy_id, enemy in players.items()
        if enemy_id != player.player_id and enemy.alive
        for enemy_blob in enemy.blobs.values()
        if _visible(view_center, vision_size, enemy_blob)
        and any(_can_eat(own_blob, enemy_blob) for own_blob in player.blobs.values())
    ]
    if not edible:
        return None
    center = _center(player)
    return max(
        edible,
        key=lambda blob: (
            blob.mass / (1.0 + math.hypot(blob.x - center[0], blob.y - center[1])),
            blob.mass,
        ),
    )


def analyze(path: Path) -> dict[str, object]:
    events = json.loads(path.read_text())
    started = events[0]
    arena_size = float(started["arena_size"])
    base_vision = float(started["vision_size"])
    max_rounds = int(started["max_rounds"])
    team_by_player = {
        int(payload["player_id"]): int(payload["team_id"])
        for payload in started["players"]
    }
    players = {
        int(payload["player_id"]): _player(
            payload, team_by_player[int(payload["player_id"])]
        )
        for payload in started["players"]
    }
    metrics = {
        player_id: Metrics(player_id=player_id, team_id=team_id)
        for player_id, team_id in team_by_player.items()
    }
    mass_history: dict[int, list[float]] = defaultdict(list)
    prey_alignments: dict[int, list[float]] = defaultdict(list)
    previous_direction: dict[int, tuple[float, float]] = {}
    round_number = -1
    in_move_batch = False

    for event in events[1:]:
        event_type = event["event_type"]
        if event_type == "move_player":
            if not in_move_batch:
                round_number += 1
                in_move_batch = True
            player_id = int(event["player_id"])
            player = players[player_id]
            direction = _unit(
                float(event["direction"]["x"]), float(event["direction"]["y"])
            )
            previous = previous_direction.get(player_id)
            if previous is not None and previous[0] * direction[0] + previous[1] * direction[1] < -0.25:
                metrics[player_id].reversals += 1
            previous_direction[player_id] = direction
            if bool(event["split"]):
                metrics[player_id].splits += 1

            prey = _prey_target(player, players, arena_size, base_vision)
            if prey is not None:
                center = _center(player)
                toward = _unit(prey.x - center[0], prey.y - center[1])
                alignment = direction[0] * toward[0] + direction[1] * toward[1]
                item = metrics[player_id]
                item.prey_opportunities += 1
                prey_alignments[player_id].append(alignment)
                if alignment >= 0.35:
                    item.prey_approaches += 1
                elif alignment <= -0.35:
                    item.prey_retreats += 1
                if bool(event["split"]):
                    item.prey_split_attempts += 1
                edge_clearance = min(
                    prey.x - prey.radius,
                    prey.y - prey.radius,
                    arena_size - prey.x - prey.radius,
                    arena_size - prey.y - prey.radius,
                )
                if edge_clearance <= 0.35:
                    item.edge_prey_opportunities += 1
                    if alignment >= 0.35:
                        item.edge_prey_pursuits += 1
                    if bool(event["split"]):
                        item.edge_prey_split_attempts += 1
                if round_number >= max_rounds - 400:
                    item.late_prey_opportunities += 1
                    if alignment >= 0.35:
                        item.late_prey_approaches += 1
                    if bool(event["split"]):
                        item.late_prey_split_attempts += 1
            continue

        in_move_batch = False
        if event_type == "event_player_moved":
            player_id = int(event["player_id"])
            prior_alive = players[player_id].alive
            players[player_id] = _player(event, team_by_player[player_id])
            updated = players[player_id]
            history = mass_history[player_id]
            if history and updated.mass < history[-1] * 0.72:
                metrics[player_id].major_losses.append(
                    {
                        "round": round_number,
                        "before": history[-1],
                        "after": updated.mass,
                    }
                )
            history.append(updated.mass)
            if updated.alive:
                metrics[player_id].rounds_alive += 1
            elif prior_alive:
                metrics[player_id].deaths += 1
                metrics[player_id].death_rounds.append(round_number)
        elif event_type == "event_food_eaten":
            metrics[int(event["player_id"])].food += len(event["food_ids"])
        elif event_type == "event_virus_consumed":
            metrics[int(event["player_id"])].viruses += 1
        elif event_type == "event_player_eaten":
            metrics[int(event["eater_player_id"])].kills += 1
            eaten_player_id = int(event["eaten_player_id"])
            metrics[eaten_player_id].blobs_lost += 1
            metrics[eaten_player_id].eaten_events.append(
                {
                    "round": round_number,
                    "eater_player_id": int(event["eater_player_id"]),
                    "eater_team_id": team_by_player[int(event["eater_player_id"])],
                    "eater_radius": float(event["eater_radius"]),
                    "eaten_blob_id": int(event["eaten_blob_id"]),
                    "survived": bool(event["eaten_player_alive"]),
                }
            )

    winner_id = next(
        int(event["player_id"])
        for event in reversed(events)
        if event["event_type"] == "event_player_won"
    )
    for player_id, item in metrics.items():
        history = mass_history[player_id]
        if history:
            item.final_mass = history[-1]
            item.max_mass = max(history)
            item.mean_mass = statistics.fmean(history)
            endgame = history[-400:]
            item.endgame_mean_mass = statistics.fmean(endgame)
            item.endgame_mass_delta = endgame[-1] - endgame[0]
        alignments = prey_alignments[player_id]
        if alignments:
            item.prey_alignment_mean = statistics.fmean(alignments)

    ranked = sorted(metrics.values(), key=lambda item: item.final_mass, reverse=True)
    return {
        "match_id": int(path.name.split("-")[1]),
        "winner_player_id": winner_id,
        "winner_team_id": team_by_player[winner_id],
        "players": [asdict(item) | {"final_rank": rank} for rank, item in enumerate(ranked, 1)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("replays", nargs="+", type=Path)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    results = [analyze(path) for path in args.replays]
    print(json.dumps(results, indent=2 if args.pretty else None, sort_keys=True))


if __name__ == "__main__":
    main()
