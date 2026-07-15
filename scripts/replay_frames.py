from __future__ import annotations

"""Read engine event recordings into immutable per-round snapshots.

Replay parsing is analysis infrastructure.  It deliberately contains no
team-specific policy or response calibration logic.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
BOTS = ROOT / "bots"
sys.path.insert(0, str(BOTS))

from strategies.features import normalise  # noqa: E402
from strategies.world_transition import PlayerCommand  # noqa: E402


VISION_REFERENCE_SUM_OF_RADII = 12.0


@dataclass(frozen=True, slots=True)
class ReplayBlob:
    x: float
    y: float
    radius: float
    merge_cooldown: int = 0

    @property
    def mass(self) -> float:
        return self.radius * self.radius

    @property
    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True, slots=True)
class ReplayPlayer:
    player_id: int
    team_id: int
    alive: bool
    blobs: tuple[ReplayBlob, ...]


@dataclass(frozen=True, slots=True)
class ReplayPoint:
    x: float
    y: float
    radius: float = 0.0
    source_id: int = -1

    @property
    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(slots=True)
class ReplayFrame:
    match_id: int
    round_number: int
    engine_version: str
    arena_size: float
    base_vision_size: float
    players: dict[int, ReplayPlayer]
    foods: tuple[ReplayPoint, ...]
    viruses: tuple[ReplayPoint, ...]
    previous_commands: dict[int, PlayerCommand]
    commands: dict[int, PlayerCommand]
    gain_player_ids: set[int]


def _match_id(path: Path) -> int:
    parts = path.name.split("-")
    return int(parts[1]) if len(parts) > 2 and parts[1].isdigit() else 0


def _blob(payload: dict[str, object]) -> ReplayBlob:
    position = payload["pos"]
    return ReplayBlob(
        x=float(position[0]),
        y=float(position[1]),
        radius=float(payload["radius"]),
        merge_cooldown=int(payload.get("merge_cooldown", 0)),
    )


def _player(payload: dict[str, object], team_id: int) -> ReplayPlayer:
    return ReplayPlayer(
        player_id=int(payload["player_id"]),
        team_id=team_id,
        alive=bool(payload.get("alive", True)),
        blobs=tuple(
            sorted(
                (_blob(blob) for blob in payload.get("blobs", ())),
                key=lambda blob: (blob.x, blob.y, blob.radius),
            )
        ),
    )


def extract_frames(path: Path) -> list[ReplayFrame]:
    events = json.loads(path.read_text(encoding="utf-8"))
    started = events[0]
    team_by_player = {
        int(player["player_id"]): int(player["team_id"])
        for player in started["players"]
    }
    players = {
        int(player["player_id"]): _player(
            player,
            team_by_player[int(player["player_id"])],
        )
        for player in started["players"]
    }
    foods: dict[int, ReplayPoint] = {}
    viruses: dict[int, ReplayPoint] = {}
    previous_commands: dict[int, PlayerCommand] = {}
    frames: list[ReplayFrame] = []
    current: ReplayFrame | None = None
    round_number = -1
    in_move_batch = False

    for event in events[1:]:
        event_type = event["event_type"]
        if event_type == "move_player":
            if not in_move_batch:
                round_number += 1
                in_move_batch = True
                current = ReplayFrame(
                    match_id=_match_id(path),
                    round_number=round_number,
                    engine_version=str(started.get("engine_version", "unknown")),
                    arena_size=float(started["arena_size"]),
                    base_vision_size=float(started["vision_size"]),
                    players=dict(players),
                    foods=tuple(foods.values()),
                    viruses=tuple(viruses.values()),
                    previous_commands=dict(previous_commands),
                    commands={},
                    gain_player_ids=set(),
                )
                frames.append(current)
            assert current is not None
            player_id = int(event["player_id"])
            direction = event["direction"]
            command = PlayerCommand(
                normalise((float(direction["x"]), float(direction["y"]))),
                split=bool(event["split"]),
            )
            current.commands[player_id] = command
            previous_commands[player_id] = command
            continue

        in_move_batch = False
        if event_type == "event_player_moved":
            player_id = int(event["player_id"])
            players[player_id] = _player(event, team_by_player[player_id])
        elif event_type == "event_food_spawned":
            for food in event["foods"]:
                position = food["pos"]
                foods[int(food["food_id"])] = ReplayPoint(
                    float(position[0]),
                    float(position[1]),
                    source_id=int(food["food_id"]),
                )
        elif event_type == "event_food_eaten":
            for food_id in event["food_ids"]:
                foods.pop(int(food_id), None)
        elif event_type == "event_virus_spawned":
            for virus in event["viruses"]:
                position = virus["pos"]
                viruses[int(virus["virus_id"])] = ReplayPoint(
                    float(position[0]),
                    float(position[1]),
                    float(virus["radius"]),
                    source_id=int(virus["virus_id"]),
                )
        elif event_type == "event_virus_consumed":
            viruses.pop(int(event["virus_id"]), None)
            if current is not None:
                current.gain_player_ids.add(int(event["player_id"]))
        elif event_type == "event_player_eaten" and current is not None:
            current.gain_player_ids.add(int(event["eater_player_id"]))
    return frames


def _mass_center(blobs: tuple[ReplayBlob, ...]) -> tuple[float, float]:
    mass = sum(blob.mass for blob in blobs)
    return (
        sum(blob.x * blob.mass for blob in blobs) / mass,
        sum(blob.y * blob.mass for blob in blobs) / mass,
    )


def _vision_size(player: ReplayPlayer, base: float) -> float:
    sum_radii = sum(blob.radius for blob in player.blobs)
    return max(sum_radii / VISION_REFERENCE_SUM_OF_RADII, 1.0) ** 0.4 * base


def _view_center(
    player: ReplayPlayer,
    arena_size: float,
    vision_size: float,
) -> tuple[float, float]:
    x, y = _mass_center(player.blobs)
    half = min(vision_size, arena_size) / 2.0
    return (
        min(max(x, half), arena_size - half),
        min(max(y, half), arena_size - half),
    )


def _circle_visible(
    center: tuple[float, float],
    vision_size: float,
    point: ReplayPoint | ReplayBlob,
) -> bool:
    half = vision_size / 2.0
    dx = max(abs(point.x - center[0]) - half, 0.0)
    dy = max(abs(point.y - center[1]) - half, 0.0)
    return dx * dx + dy * dy <= point.radius * point.radius


def _point_visible(
    center: tuple[float, float],
    vision_size: float,
    point: ReplayPoint,
) -> bool:
    half = vision_size / 2.0
    return (
        abs(point.x - center[0]) <= half
        and abs(point.y - center[1]) <= half
    )
