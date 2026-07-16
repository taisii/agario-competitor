from __future__ import annotations

"""Measure resource contribution and empirical pursuit value in replay data."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
BOTS = ROOT / "bots"
sys.path.insert(0, str(BOTS))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_mass_gain_sources import Gain, _extract_gain_timeline  # noqa: E402
from calibrate_expected_responses import (
    ReplayBlob,
    ReplayFrame,
    ReplayPlayer,
    extract_frames,
)  # noqa: E402
from lib.config.player import EAT_SIZE_RATIO  # noqa: E402
from strategies.features import player_speed  # noqa: E402


RESOURCE_KINDS = ("enemy", "virus", "food")
VISION_REFERENCE_SUM_OF_RADII = 12.0
PURSUIT_ALIGNMENT_THRESHOLD = 0.35
PURSUIT_RESTART_COOLDOWN = 10


@dataclass(frozen=True, slots=True)
class EatenEvent:
    round_number: int
    eaten_player_id: int
    mass: float


@dataclass(frozen=True, slots=True)
class PreyTarget:
    player_id: int
    blob: ReplayBlob
    eta_if_stationary: float
    edge_clearance: float
    direction: tuple[float, float]


@dataclass(frozen=True, slots=True)
class PursuitEpisode:
    eta_if_stationary: float
    edge_clearance: float
    split: bool
    target_mass: float
    target_mass_by_horizon: dict[int, float]
    any_enemy_mass_by_horizon: dict[int, float]


@dataclass(frozen=True, slots=True)
class PlayerMilestone:
    player_id: int
    won: bool
    first_enemy_capture_round: int | None
    mass_after_first_enemy_capture: float | None
    first_captured_enemy_mass: float | None
    final_mass: float


def _mass(player: ReplayPlayer) -> float:
    return sum(blob.radius * blob.radius for blob in player.blobs)


def _center(player: ReplayPlayer) -> tuple[float, float]:
    mass = _mass(player)
    if mass <= 1.0e-9:
        return (30.0, 30.0)
    return (
        sum(blob.x * blob.radius * blob.radius for blob in player.blobs) / mass,
        sum(blob.y * blob.radius * blob.radius for blob in player.blobs) / mass,
    )


def _vision_size(player: ReplayPlayer, base_vision_size: float) -> float:
    sum_radii = sum(blob.radius for blob in player.blobs)
    if sum_radii <= 0.0:
        return base_vision_size
    return max(sum_radii / VISION_REFERENCE_SUM_OF_RADII, 1.0) ** 0.4 * base_vision_size


def _view_center(
    player: ReplayPlayer,
    *,
    arena_size: float,
    vision_size: float,
) -> tuple[float, float]:
    center = _center(player)
    half = min(vision_size, arena_size) / 2.0
    return (
        min(max(center[0], half), arena_size - half),
        min(max(center[1], half), arena_size - half),
    )


def _visible(
    center: tuple[float, float],
    vision_size: float,
    blob: ReplayBlob,
) -> bool:
    half = vision_size / 2.0
    dx = max(abs(blob.x - center[0]) - half, 0.0)
    dy = max(abs(blob.y - center[1]) - half, 0.0)
    return dx * dx + dy * dy <= blob.radius * blob.radius


def _normalise(vector: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(*vector)
    if length <= 1.0e-9:
        return (1.0, 0.0)
    return (vector[0] / length, vector[1] / length)


def _prey_target(frame: ReplayFrame, own: ReplayPlayer) -> PreyTarget | None:
    own_center = _center(own)
    vision_size = _vision_size(own, frame.base_vision_size)
    view_center = _view_center(
        own,
        arena_size=frame.arena_size,
        vision_size=vision_size,
    )
    candidates: list[tuple[float, PreyTarget]] = []
    for enemy in frame.players.values():
        if enemy.player_id == own.player_id or not enemy.alive:
            continue
        for enemy_blob in enemy.blobs:
            if not _visible(view_center, vision_size, enemy_blob):
                continue
            capable = tuple(
                blob
                for blob in own.blobs
                if blob.radius * blob.radius
                >= enemy_blob.radius * enemy_blob.radius * EAT_SIZE_RATIO
            )
            if not capable:
                continue
            hunter = min(
                capable,
                key=lambda blob: math.dist(
                    (blob.x, blob.y),
                    (enemy_blob.x, enemy_blob.y),
                ),
            )
            center_distance = math.dist(
                own_center,
                (enemy_blob.x, enemy_blob.y),
            )
            hunter_distance = math.dist(
                (hunter.x, hunter.y),
                (enemy_blob.x, enemy_blob.y),
            )
            clearance = max(0.0, hunter_distance - hunter.radius)
            edge_clearance = min(
                enemy_blob.x - enemy_blob.radius,
                enemy_blob.y - enemy_blob.radius,
                frame.arena_size - enemy_blob.x - enemy_blob.radius,
                frame.arena_size - enemy_blob.y - enemy_blob.radius,
            )
            candidates.append(
                (
                    enemy_blob.radius * enemy_blob.radius / (1.0 + center_distance),
                    PreyTarget(
                        player_id=enemy.player_id,
                        blob=enemy_blob,
                        eta_if_stationary=clearance
                        / max(player_speed(hunter.radius), 1.0e-9),
                        edge_clearance=edge_clearance,
                        direction=_normalise(
                            (
                                enemy_blob.x - own_center[0],
                                enemy_blob.y - own_center[1],
                            )
                        ),
                    ),
                )
            )
    if not candidates:
        return None
    return max(candidates, key=lambda row: row[0])[1]


def _eaten_events(
    events: list[dict[str, object]],
    *,
    eater_player_id: int,
) -> tuple[EatenEvent, ...]:
    rows: list[EatenEvent] = []
    round_number = -1
    in_move_batch = False
    for event in events:
        event_type = event.get("event_type")
        if event_type == "move_player":
            if not in_move_batch:
                round_number += 1
                in_move_batch = True
            continue
        in_move_batch = False
        if (
            event_type == "event_player_eaten"
            and int(event["eater_player_id"]) == eater_player_id
        ):
            rows.append(
                EatenEvent(
                    round_number=round_number,
                    eaten_player_id=int(event["eaten_player_id"]),
                    mass=float(event["eaten_radius"]) ** 2,
                )
            )
    return tuple(rows)


def _player_milestones(
    events: list[dict[str, object]],
) -> tuple[dict[int, PlayerMilestone], int]:
    started = events[0]
    current_mass = {
        int(player["player_id"]): sum(
            float(blob["radius"]) ** 2 for blob in player.get("blobs", ())
        )
        for player in started["players"]
    }
    winner_player_id = next(
        int(event["player_id"])
        for event in reversed(events)
        if event.get("event_type") == "event_player_won"
    )
    first_capture_round: dict[int, int] = {}
    first_captured_mass: dict[int, float] = {}
    mass_after_first_capture: dict[int, float] = {}
    awaiting_post_capture_mass: set[int] = set()
    round_number = -1
    in_move_batch = False

    for event in events[1:]:
        event_type = event.get("event_type")
        if event_type == "move_player":
            if not in_move_batch:
                round_number += 1
                in_move_batch = True
            continue
        in_move_batch = False
        if event_type == "event_player_eaten":
            eater_player_id = int(event["eater_player_id"])
            if eater_player_id not in first_capture_round:
                first_capture_round[eater_player_id] = round_number
                first_captured_mass[eater_player_id] = float(event["eaten_radius"]) ** 2
                awaiting_post_capture_mass.add(eater_player_id)
        elif event_type == "event_player_moved":
            player_id = int(event["player_id"])
            current_mass[player_id] = sum(
                float(blob["radius"]) ** 2 for blob in event.get("blobs", ())
            )
            if player_id in awaiting_post_capture_mass:
                mass_after_first_capture[player_id] = current_mass[player_id]
                awaiting_post_capture_mass.remove(player_id)

    return (
        {
            player_id: PlayerMilestone(
                player_id=player_id,
                won=player_id == winner_player_id,
                first_enemy_capture_round=first_capture_round.get(player_id),
                mass_after_first_enemy_capture=mass_after_first_capture.get(player_id),
                first_captured_enemy_mass=first_captured_mass.get(player_id),
                final_mass=final_mass,
            )
            for player_id, final_mass in current_mass.items()
        },
        round_number,
    )


def _pursuit_episodes(
    path: Path,
    *,
    team_id: int,
    horizons: tuple[int, ...],
) -> tuple[PursuitEpisode, ...]:
    events = json.loads(path.read_text(encoding="utf-8"))
    team_by_player = {
        int(player["player_id"]): int(player["team_id"])
        for player in events[0]["players"]
    }
    player_id = next(
        (
            player_id
            for player_id, replay_team_id in team_by_player.items()
            if replay_team_id == team_id
        ),
        None,
    )
    if player_id is None:
        return ()

    eaten_events = _eaten_events(events, eater_player_id=player_id)
    episodes: list[PursuitEpisode] = []
    previous_target: int | None = None
    last_start_by_target: dict[int, int] = {}
    for frame in extract_frames(path):
        own = frame.players[player_id]
        command = frame.commands.get(player_id)
        if not own.alive or not own.blobs or command is None:
            previous_target = None
            continue
        target = _prey_target(frame, own)
        if target is None:
            previous_target = None
            continue
        alignment = (
            command.direction[0] * target.direction[0]
            + command.direction[1] * target.direction[1]
        )
        pursuing = alignment >= PURSUIT_ALIGNMENT_THRESHOLD
        can_start = (
            pursuing
            and target.player_id != previous_target
            and frame.round_number - last_start_by_target.get(target.player_id, -10_000)
            > PURSUIT_RESTART_COOLDOWN
        )
        if can_start:
            target_mass_by_horizon: dict[int, float] = {}
            any_enemy_mass_by_horizon: dict[int, float] = {}
            for horizon in horizons:
                within_horizon = tuple(
                    event
                    for event in eaten_events
                    if frame.round_number
                    < event.round_number
                    <= frame.round_number + horizon
                )
                target_mass_by_horizon[horizon] = sum(
                    event.mass
                    for event in within_horizon
                    if event.eaten_player_id == target.player_id
                )
                any_enemy_mass_by_horizon[horizon] = sum(
                    event.mass for event in within_horizon
                )
            episodes.append(
                PursuitEpisode(
                    eta_if_stationary=target.eta_if_stationary,
                    edge_clearance=target.edge_clearance,
                    split=command.split,
                    target_mass=target.blob.radius * target.blob.radius,
                    target_mass_by_horizon=target_mass_by_horizon,
                    any_enemy_mass_by_horizon=any_enemy_mass_by_horizon,
                )
            )
            last_start_by_target[target.player_id] = frame.round_number
        previous_target = target.player_id if pursuing else None
    return tuple(episodes)


def _empty_resource_totals() -> dict[str, dict[str, float | int]]:
    return {kind: {"count": 0, "gross_mass": 0.0} for kind in RESOURCE_KINDS}


def _add_gains(
    totals: dict[str, dict[str, float | int]],
    gains: Iterable[dict[str, Gain]],
) -> None:
    for round_gains in gains:
        for kind, gain in round_gains.items():
            totals[kind]["count"] += gain.count
            totals[kind]["gross_mass"] += gain.mass


def _render_resource_totals(
    totals: dict[str, dict[str, float | int]],
) -> dict[str, object]:
    gross_total = sum(float(row["gross_mass"]) for row in totals.values())
    return {
        "gross_mass_total": gross_total,
        "resources": {
            kind: {
                "count": int(row["count"]),
                "gross_mass": float(row["gross_mass"]),
                "gross_mass_share": (
                    float(row["gross_mass"]) / gross_total if gross_total else 0.0
                ),
                "mean_mass_per_event": (
                    float(row["gross_mass"]) / int(row["count"])
                    if row["count"]
                    else 0.0
                ),
            }
            for kind, row in totals.items()
        },
    }


def _episode_summary(
    episodes: tuple[PursuitEpisode, ...],
    *,
    horizon: int,
) -> dict[str, float | int]:
    target_masses = tuple(
        episode.target_mass_by_horizon[horizon] for episode in episodes
    )
    any_enemy_masses = tuple(
        episode.any_enemy_mass_by_horizon[horizon] for episode in episodes
    )
    successful = tuple(mass for mass in target_masses if mass > 0.0)
    return {
        "episodes": len(episodes),
        "target_capture_rate": len(successful) / len(episodes) if episodes else 0.0,
        "expected_target_mass_per_start": (
            statistics.fmean(target_masses) if target_masses else 0.0
        ),
        "mean_target_mass_given_capture": (
            statistics.fmean(successful) if successful else 0.0
        ),
        "expected_any_enemy_mass_per_start": (
            statistics.fmean(any_enemy_masses) if any_enemy_masses else 0.0
        ),
        "mean_offered_target_mass": (
            statistics.fmean(episode.target_mass for episode in episodes)
            if episodes
            else 0.0
        ),
    }


def _breakdown(
    episodes: tuple[PursuitEpisode, ...],
    *,
    horizon: int,
    groups: tuple[
        tuple[str, Callable[[PursuitEpisode], bool]],
        ...,
    ],
) -> dict[str, object]:
    return {
        label: _episode_summary(
            tuple(episode for episode in episodes if predicate(episode)),
            horizon=horizon,
        )
        for label, predicate in groups
    }


def _percentile(values: tuple[float, ...], quantile: float) -> float:
    ordered = tuple(sorted(values))
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: Iterable[float | int]) -> dict[str, float | int | None]:
    rendered = tuple(float(value) for value in values)
    if not rendered:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(rendered),
        "mean": statistics.fmean(rendered),
        "median": _percentile(rendered, 0.5),
        "p25": _percentile(rendered, 0.25),
        "p75": _percentile(rendered, 0.75),
        "min": min(rendered),
        "max": max(rendered),
    }


def _milestone_summary(
    milestones: Iterable[PlayerMilestone],
) -> dict[str, object]:
    rows = tuple(milestones)
    captured = tuple(row for row in rows if row.first_enemy_capture_round is not None)
    return {
        "matches": len(rows),
        "matches_with_enemy_capture": len(captured),
        "enemy_capture_match_rate": len(captured) / len(rows) if rows else 0.0,
        "first_enemy_capture_round": _distribution(
            row.first_enemy_capture_round
            for row in captured
            if row.first_enemy_capture_round is not None
        ),
        "mass_after_first_enemy_capture": _distribution(
            row.mass_after_first_enemy_capture
            for row in captured
            if row.mass_after_first_enemy_capture is not None
        ),
        "first_captured_enemy_mass": _distribution(
            row.first_captured_enemy_mass
            for row in captured
            if row.first_captured_enemy_mass is not None
        ),
        "final_mass": _distribution(row.final_mass for row in rows),
    }


def analyze(
    replay_paths: Iterable[Path],
    *,
    team_id: int,
    horizons: tuple[int, ...],
) -> dict[str, object]:
    paths = tuple(sorted(path.resolve() for path in replay_paths))
    resource_totals = {
        scope: _empty_resource_totals()
        for scope in ("all_players", "winners", f"team_{team_id}")
    }
    team_appearances = 0
    team_wins = 0
    team_final_masses: list[float] = []
    pursuit_episodes: list[PursuitEpisode] = []
    winner_milestones: list[PlayerMilestone] = []
    team_milestones: list[PlayerMilestone] = []
    terminal_rounds: list[int] = []
    loss_first_capture_lags: list[int] = []

    for path in paths:
        events = json.loads(path.read_text(encoding="utf-8"))
        started = events[0]
        team_by_player = {
            int(player["player_id"]): int(player["team_id"])
            for player in started["players"]
        }
        winner_player_id = next(
            (
                int(event["player_id"])
                for event in reversed(events)
                if event.get("event_type") == "event_player_won"
            ),
            None,
        )
        milestones, terminal_round = _player_milestones(events)
        terminal_rounds.append(terminal_round)
        if winner_player_id is None:
            raise ValueError(f"Replay has no winner: {path}")
        winner_milestone = milestones[winner_player_id]
        winner_milestones.append(winner_milestone)

        gains, _ = _extract_gain_timeline(events)
        for player_id, replay_team_id in team_by_player.items():
            player_gains = tuple(
                round_rows.get(player_id, {}) for round_rows in gains.values()
            )
            _add_gains(resource_totals["all_players"], player_gains)
            if player_id == winner_player_id:
                _add_gains(resource_totals["winners"], player_gains)
            if replay_team_id == team_id:
                team_appearances += 1
                team_wins += player_id == winner_player_id
                team_milestone = milestones[player_id]
                team_milestones.append(team_milestone)
                team_final_masses.append(team_milestone.final_mass)
                _add_gains(resource_totals[f"team_{team_id}"], player_gains)
                if (
                    not team_milestone.won
                    and team_milestone.first_enemy_capture_round is not None
                    and winner_milestone.first_enemy_capture_round is not None
                ):
                    loss_first_capture_lags.append(
                        team_milestone.first_enemy_capture_round
                        - winner_milestone.first_enemy_capture_round
                    )

        pursuit_episodes.extend(
            _pursuit_episodes(
                path,
                team_id=team_id,
                horizons=horizons,
            )
        )

    episodes = tuple(pursuit_episodes)
    rendered_team_milestones = tuple(team_milestones)
    reference_horizon = min(horizons, key=lambda value: abs(value - 40))
    early_capture_cutoffs = (100, 200, 400, 600)
    return {
        "dataset": {
            "replay_count": len(paths),
            "team_id": team_id,
            "team_appearances": team_appearances,
            "team_wins": team_wins,
            "team_mean_final_mass": (
                statistics.fmean(team_final_masses) if team_final_masses else 0.0
            ),
            "terminal_round_zero_based": _distribution(terminal_rounds),
        },
        "match_progression": {
            "definition": {
                "first_enemy_capture_round": (
                    "zero-based round of the first event_player_eaten credited "
                    "to the player"
                ),
                "mass_after_first_enemy_capture": (
                    "player total mass in the event_player_moved snapshot at "
                    "the end of that capture round"
                ),
                "terminal_round_zero_based": (
                    "1399 means the match completed all 1400 configured rounds"
                ),
            },
            "winners": _milestone_summary(winner_milestones),
            f"team_{team_id}_all": _milestone_summary(rendered_team_milestones),
            f"team_{team_id}_wins": _milestone_summary(
                row for row in rendered_team_milestones if row.won
            ),
            f"team_{team_id}_losses": _milestone_summary(
                row for row in rendered_team_milestones if not row.won
            ),
            "team_first_capture_cutoffs": {
                str(cutoff): {
                    "matches": len(
                        selected := tuple(
                            row
                            for row in rendered_team_milestones
                            if row.first_enemy_capture_round is not None
                            and row.first_enemy_capture_round <= cutoff
                        )
                    ),
                    "wins": sum(row.won for row in selected),
                    "win_rate": (
                        sum(row.won for row in selected) / len(selected)
                        if selected
                        else 0.0
                    ),
                }
                for cutoff in early_capture_cutoffs
            },
            "team_late_or_no_capture_after_round_400": {
                "matches": len(
                    late_or_none := tuple(
                        row
                        for row in rendered_team_milestones
                        if row.first_enemy_capture_round is None
                        or row.first_enemy_capture_round > 400
                    )
                ),
                "wins": sum(row.won for row in late_or_none),
                "win_rate": (
                    sum(row.won for row in late_or_none) / len(late_or_none)
                    if late_or_none
                    else 0.0
                ),
            },
            "team_loss_first_capture_lag_vs_winner": {
                "rounds": _distribution(loss_first_capture_lags),
                "team_captured_earlier_count": sum(
                    lag < 0 for lag in loss_first_capture_lags
                ),
                "paired_losses": len(loss_first_capture_lags),
            },
        },
        "resource_contribution": {
            scope: _render_resource_totals(totals)
            for scope, totals in resource_totals.items()
        },
        "pursuit": {
            "definition": {
                "alignment_threshold": PURSUIT_ALIGNMENT_THRESHOLD,
                "same_target_restart_cooldown_rounds": PURSUIT_RESTART_COOLDOWN,
                "target_selection": "highest edible visible mass / (1 + center distance)",
                "eta": "stationary target center-clearance / hunter speed",
            },
            "by_horizon": {
                str(horizon): _episode_summary(episodes, horizon=horizon)
                for horizon in horizons
            },
            "reference_horizon": reference_horizon,
            "eta_breakdown": _breakdown(
                episodes,
                horizon=reference_horizon,
                groups=(
                    ("eta_le_5", lambda episode: episode.eta_if_stationary <= 5.0),
                    (
                        "eta_5_to_10",
                        lambda episode: 5.0 < episode.eta_if_stationary <= 10.0,
                    ),
                    (
                        "eta_10_to_20",
                        lambda episode: 10.0 < episode.eta_if_stationary <= 20.0,
                    ),
                    ("eta_gt_20", lambda episode: episode.eta_if_stationary > 20.0),
                ),
            ),
            "edge_breakdown": _breakdown(
                episodes,
                horizon=reference_horizon,
                groups=(
                    (
                        "edge_le_0_35",
                        lambda episode: episode.edge_clearance <= 0.35,
                    ),
                    (
                        "edge_0_35_to_2",
                        lambda episode: 0.35 < episode.edge_clearance <= 2.0,
                    ),
                    (
                        "interior_gt_2",
                        lambda episode: episode.edge_clearance > 2.0,
                    ),
                ),
            ),
            "command_breakdown": _breakdown(
                episodes,
                horizon=reference_horizon,
                groups=(
                    ("split", lambda episode: episode.split),
                    ("normal", lambda episode: not episode.split),
                ),
            ),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_dir", type=Path)
    parser.add_argument("--team-id", type=int, default=73)
    parser.add_argument("--horizons", default="20,40,80")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    horizons = tuple(
        sorted({int(raw) for item in args.horizons.split(",") if (raw := item.strip())})
    )
    if not horizons or min(horizons) <= 0:
        raise SystemExit("--horizons must contain positive integers")
    paths = tuple(args.replay_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"No replay JSON files found in {args.replay_dir}")
    report = analyze(paths, team_id=args.team_id, horizons=horizons)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
