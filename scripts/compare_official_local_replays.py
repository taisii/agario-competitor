from __future__ import annotations

"""Compare one submitted player across official and local replay cohorts."""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
BOTS = ROOT / "bots"
sys.path.insert(0, str(BOTS))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_mass_gain_sources import RESOURCE_KINDS, _extract_gain_timeline  # noqa: E402


DEFAULT_TERMINAL_HORIZON = 100
DEFAULT_STALLED_DISTANCE = 1.0
MIN_TERMINAL_ALIVE_SHARE = 0.8


@dataclass(frozen=True, slots=True)
class MatchMetrics:
    path: str
    player_id: int
    won: bool
    final_alive: bool
    final_mass: float
    peak_mass: float
    low_final_mass: bool
    captured_fragments: int
    captured_mass: float
    eliminations: int
    lost_fragments: int
    lost_mass: float
    full_deaths: int
    resources: dict[str, dict[str, float | int]]
    terminal_horizon: int
    terminal_alive_rounds: int
    terminal_command_rounds: int
    terminal_zero_commands: int
    terminal_center_distance: float
    terminal_stalled: bool
    terminal_dead: bool
    terminal_lost_fragments: int
    terminal_lost_mass: float


def _mass_and_center(
    blobs: Iterable[dict[str, object]],
) -> tuple[float, tuple[float, float] | None]:
    rows = tuple(blobs)
    mass = sum(float(blob["radius"]) ** 2 for blob in rows)
    if mass <= 1.0e-12:
        return (0.0, None)
    return (
        mass,
        (
            sum(
                float(blob["pos"][0]) * float(blob["radius"]) ** 2
                for blob in rows
            )
            / mass,
            sum(
                float(blob["pos"][1]) * float(blob["radius"]) ** 2
                for blob in rows
            )
            / mass,
        ),
    )


def _target_player_id(
    started: dict[str, object],
    *,
    team_id: int | None,
    player_id: int | None,
) -> int:
    if (team_id is None) == (player_id is None):
        raise ValueError("provide exactly one of team_id and player_id")
    players = started.get("players", ())
    if player_id is not None:
        if not any(int(player["player_id"]) == player_id for player in players):
            raise ValueError(f"player {player_id} is absent from replay")
        return player_id
    matches = tuple(
        int(player["player_id"])
        for player in players
        if int(player["team_id"]) == team_id
    )
    if len(matches) != 1:
        raise ValueError(f"expected one player for team {team_id}, found {len(matches)}")
    return matches[0]


def analyze_match(
    path: Path,
    *,
    team_id: int | None = None,
    player_id: int | None = None,
    terminal_horizon: int = DEFAULT_TERMINAL_HORIZON,
    stalled_distance: float = DEFAULT_STALLED_DISTANCE,
) -> MatchMetrics:
    events = json.loads(path.read_text(encoding="utf-8"))
    if not events or events[0].get("event_type") != "event_game_started":
        raise ValueError(f"Replay has no initial game event: {path}")
    started = events[0]
    target = _target_player_id(started, team_id=team_id, player_id=player_id)
    max_rounds = int(started.get("max_rounds", 1400))
    terminal_start = max(0, max_rounds - terminal_horizon)

    initial = next(
        player for player in started["players"] if int(player["player_id"]) == target
    )
    final_mass, _ = _mass_and_center(initial.get("blobs", ()))
    final_alive = bool(initial.get("alive", True))
    peak_mass = final_mass
    winner_player_id: int | None = None
    captured_fragments = 0
    captured_mass = 0.0
    eliminations = 0
    lost_fragments = 0
    lost_mass = 0.0
    full_deaths = 0
    terminal_lost_fragments = 0
    terminal_lost_mass = 0.0
    snapshots: dict[int, tuple[bool, tuple[float, float] | None]] = {}
    commands: dict[int, tuple[float, float]] = {}
    round_number = -1
    in_move_batch = False

    for event in events[1:]:
        event_type = event.get("event_type")
        if event_type == "move_player":
            if not in_move_batch:
                round_number += 1
                in_move_batch = True
            if int(event["player_id"]) == target:
                direction = event["direction"]
                commands[round_number] = (
                    float(direction["x"]),
                    float(direction["y"]),
                )
            continue
        in_move_batch = False

        if event_type == "event_player_moved" and int(event["player_id"]) == target:
            final_mass, center = _mass_and_center(event.get("blobs", ()))
            final_alive = bool(event.get("alive", True))
            peak_mass = max(peak_mass, final_mass)
            snapshots[round_number] = (final_alive, center)
        elif event_type == "event_player_eaten":
            event_mass = float(event["eaten_radius"]) ** 2
            if int(event["eater_player_id"]) == target:
                captured_fragments += 1
                captured_mass += event_mass
                eliminations += not bool(event.get("eaten_player_alive", True))
            if int(event["eaten_player_id"]) == target:
                lost_fragments += 1
                lost_mass += event_mass
                died = not bool(event.get("eaten_player_alive", True))
                full_deaths += died
                if round_number >= terminal_start:
                    terminal_lost_fragments += 1
                    terminal_lost_mass += event_mass
        elif event_type == "event_player_won":
            winner_player_id = int(event["player_id"])

    gains, extracted_peaks = _extract_gain_timeline(events)
    peak_mass = max(peak_mass, extracted_peaks.get(target, 0.0))
    resources = {
        kind: {"count": 0, "gross_mass": 0.0}
        for kind in RESOURCE_KINDS
    }
    for round_gains in gains.values():
        for kind, gain in round_gains.get(target, {}).items():
            resources[kind]["count"] += gain.count
            resources[kind]["gross_mass"] += gain.mass

    terminal_snapshots = tuple(
        (round_id, alive, center)
        for round_id, (alive, center) in sorted(snapshots.items())
        if round_id >= terminal_start
    )
    terminal_alive_rounds = sum(alive for _, alive, _ in terminal_snapshots)
    terminal_commands = tuple(
        direction
        for command_round, direction in sorted(commands.items())
        if command_round >= terminal_start
    )
    terminal_zero_commands = sum(
        math.hypot(*direction) <= 1.0e-12 for direction in terminal_commands
    )
    terminal_center_distance = sum(
        math.dist(previous[2], current[2])
        for previous, current in zip(terminal_snapshots, terminal_snapshots[1:])
        if previous[0] + 1 == current[0]
        and previous[1]
        and current[1]
        and previous[2] is not None
        and current[2] is not None
    )
    enough_alive_rounds = terminal_alive_rounds >= math.ceil(
        terminal_horizon * MIN_TERMINAL_ALIVE_SHARE
    )
    terminal_stalled = (
        final_alive
        and enough_alive_rounds
        and terminal_center_distance < stalled_distance
    )

    return MatchMetrics(
        path=str(path.resolve()),
        player_id=target,
        won=winner_player_id == target,
        final_alive=final_alive,
        final_mass=final_mass,
        peak_mass=peak_mass,
        low_final_mass=final_mass < 2.0,
        captured_fragments=captured_fragments,
        captured_mass=captured_mass,
        eliminations=eliminations,
        lost_fragments=lost_fragments,
        lost_mass=lost_mass,
        full_deaths=full_deaths,
        resources=resources,
        terminal_horizon=terminal_horizon,
        terminal_alive_rounds=terminal_alive_rounds,
        terminal_command_rounds=len(terminal_commands),
        terminal_zero_commands=terminal_zero_commands,
        terminal_center_distance=terminal_center_distance,
        terminal_stalled=terminal_stalled,
        terminal_dead=not final_alive,
        terminal_lost_fragments=terminal_lost_fragments,
        terminal_lost_mass=terminal_lost_mass,
    )


def _percentile(values: tuple[float, ...], quantile: float) -> float:
    ordered = tuple(sorted(values))
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: Iterable[float | int]) -> dict[str, float | int | None]:
    rows = tuple(float(value) for value in values)
    if not rows:
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None}
    return {
        "count": len(rows),
        "mean": statistics.fmean(rows),
        "median": _percentile(rows, 0.5),
        "p25": _percentile(rows, 0.25),
        "p75": _percentile(rows, 0.75),
    }


def summarize(rows: Iterable[MatchMetrics]) -> dict[str, object]:
    matches = tuple(rows)
    if not matches:
        raise ValueError("at least one match is required")
    count = len(matches)
    terminal_horizons = {row.terminal_horizon for row in matches}
    if len(terminal_horizons) != 1:
        raise ValueError("all matches must use the same terminal horizon")
    terminal_horizon = terminal_horizons.pop()
    resource_counts = {
        kind: sum(int(row.resources[kind]["count"]) for row in matches)
        for kind in RESOURCE_KINDS
    }
    resource_masses = {
        kind: sum(float(row.resources[kind]["gross_mass"]) for row in matches)
        for kind in RESOURCE_KINDS
    }
    total_resource_count = sum(resource_counts.values())
    total_resource_mass = sum(resource_masses.values())
    lost_mass = sum(row.lost_mass for row in matches)
    terminal_lost_mass = sum(row.terminal_lost_mass for row in matches)
    terminal_commands = sum(row.terminal_command_rounds for row in matches)
    terminal_zero_commands = sum(row.terminal_zero_commands for row in matches)

    def mean(attribute: str) -> float:
        return statistics.fmean(float(getattr(row, attribute)) for row in matches)

    return {
        "matches": count,
        "win_rate": sum(row.won for row in matches) / count,
        "survival_rate": sum(row.final_alive for row in matches) / count,
        "low_final_mass_rate": sum(row.low_final_mass for row in matches) / count,
        "final_mass": _distribution(row.final_mass for row in matches),
        "peak_mass": _distribution(row.peak_mass for row in matches),
        "predation": {
            "captured_fragments_total": sum(row.captured_fragments for row in matches),
            "captured_fragments_per_match": mean("captured_fragments"),
            "captured_mass_total": sum(row.captured_mass for row in matches),
            "captured_mass_per_match": mean("captured_mass"),
            "eliminations_total": sum(row.eliminations for row in matches),
            "eliminations_per_match": mean("eliminations"),
            "lost_fragments_total": sum(row.lost_fragments for row in matches),
            "lost_fragments_per_match": mean("lost_fragments"),
            "lost_mass_total": lost_mass,
            "lost_mass_per_match": mean("lost_mass"),
            "full_deaths_total": sum(row.full_deaths for row in matches),
            "full_deaths_per_match": mean("full_deaths"),
        },
        "resources": {
            kind: {
                "count": resource_counts[kind],
                "count_per_match": resource_counts[kind] / count,
                "count_share": (
                    resource_counts[kind] / total_resource_count
                    if total_resource_count
                    else 0.0
                ),
                "gross_mass": resource_masses[kind],
                "gross_mass_per_match": resource_masses[kind] / count,
                "gross_mass_share": (
                    resource_masses[kind] / total_resource_mass
                    if total_resource_mass
                    else 0.0
                ),
            }
            for kind in RESOURCE_KINDS
        },
        "terminal": {
            "horizon_rounds": terminal_horizon,
            "stalled_matches": sum(row.terminal_stalled for row in matches),
            "stalled_match_rate": sum(row.terminal_stalled for row in matches) / count,
            "dead_at_end_matches": sum(row.terminal_dead for row in matches),
            "dead_at_end_rate": sum(row.terminal_dead for row in matches) / count,
            "matches_with_loss": sum(
                row.terminal_lost_fragments > 0 for row in matches
            ),
            "matches_with_loss_rate": sum(
                row.terminal_lost_fragments > 0 for row in matches
            )
            / count,
            "lost_fragments_total": sum(
                row.terminal_lost_fragments for row in matches
            ),
            "lost_mass_total": terminal_lost_mass,
            "lost_mass_share_of_all_enemy_loss": (
                terminal_lost_mass / lost_mass if lost_mass else 0.0
            ),
            "zero_command_rate": (
                terminal_zero_commands / terminal_commands
                if terminal_commands
                else 0.0
            ),
            "center_distance": _distribution(
                row.terminal_center_distance for row in matches if row.final_alive
            ),
        },
        "match_details": [asdict(row) for row in matches],
    }


def compare(
    official_paths: Iterable[Path],
    local_paths: Iterable[Path],
    *,
    official_team_id: int,
    local_player_id: int,
) -> dict[str, object]:
    official = tuple(
        analyze_match(path, team_id=official_team_id)
        for path in sorted(official_paths)
    )
    local = tuple(
        analyze_match(path, player_id=local_player_id)
        for path in sorted(local_paths)
    )
    return {
        "definitions": {
            "mass_unit": "radius_squared",
            "capture_unit": "player_fragment_event",
            "full_death": "target lost its final fragment; respawn follows after 30 rounds",
            "terminal_horizon_rounds": DEFAULT_TERMINAL_HORIZON,
            "terminal_stalled": (
                "alive for at least 80% of the terminal horizon and mass-center "
                f"path length below {DEFAULT_STALLED_DISTANCE}"
            ),
            "resource_share": "share of gross acquired mass before decay or later loss",
        },
        "official": summarize(official),
        "local": summarize(local),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official_dir", type=Path)
    parser.add_argument("local_root", type=Path)
    parser.add_argument("--official-team-id", type=int, default=73)
    parser.add_argument("--local-player-id", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    official_paths = tuple(args.official_dir.glob("match-*-replay.json"))
    local_paths = tuple(args.local_root.glob("**/output/game.json"))
    if not official_paths:
        raise SystemExit(f"No official replays found under {args.official_dir}")
    if not local_paths:
        raise SystemExit(f"No local replays found under {args.local_root}")
    report = compare(
        official_paths,
        local_paths,
        official_team_id=args.official_team_id,
        local_player_id=args.local_player_id,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
