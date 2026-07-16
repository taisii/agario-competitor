from __future__ import annotations

"""Compare the submitted semantic lookahead with the bounded offensive beam.

Both policies receive the exact observation reconstructed from each official
replay frame.  The replay's actual state remains authoritative, so this is an
action comparison rather than a long counterfactual simulation.
"""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
BOTS = ROOT / "bots"
sys.path.insert(0, str(BOTS))
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_expected_responses import extract_frames  # noqa: E402
from compare_strategy_decisions import _angle_degrees, _context  # noqa: E402
from strategies.features import can_eat_player_blob, normalise  # noqa: E402
from strategies.semantic_offensive_beam import (  # noqa: E402
    SemanticOffensiveBeamStrategy,
)
from strategies.semantic_potential import SemanticLookaheadStrategy  # noqa: E402


ANGLE_CHANGE_DEGREES = 30.0
APPROACH_ALIGNMENT = 0.35
RETREAT_ALIGNMENT = -0.35
CORNER_CLEARANCE = 5.0


@dataclass(frozen=True, slots=True)
class ChangedScene:
    match_id: int
    round_number: int
    mass: float
    blob_count: int
    angle_degrees: float
    current_reason: str
    offensive_reason: str
    current_split: bool
    offensive_split: bool
    current_prey_alignment: float | None
    offensive_prey_alignment: float | None
    prey_player_id: int | None
    prey_mass: float
    prey_corner_clearance: float | None
    predator_visible: bool
    current_safety_margin: float | None
    offensive_safety_margin: float | None
    offensive_value: float
    offensive_score: float


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _mass_center(blobs) -> tuple[float, float]:
    total = sum(blob.radius * blob.radius for blob in blobs)
    if total <= 1.0e-9:
        return (0.0, 0.0)
    return (
        sum(blob.pos[0] * blob.radius * blob.radius for blob in blobs) / total,
        sum(blob.pos[1] * blob.radius * blob.radius for blob in blobs) / total,
    )


def _prey(context):
    state = context.game.state
    own = tuple(state.me.blobs.values())
    edible = tuple(
        enemy
        for enemy in state.visible_blobs
        if any(
            can_eat_player_blob(blob.radius, enemy.radius, radius_margin=1.03)
            for blob in own
        )
    )
    if not edible:
        return None
    center = _mass_center(own)
    return max(
        edible,
        key=lambda enemy: (
            enemy.radius
            * enemy.radius
            / (1.0 + math.dist(center, enemy.pos)),
            enemy.radius,
        ),
    )


def _alignment(direction, center, prey) -> float:
    toward = normalise((prey.pos[0] - center[0], prey.pos[1] - center[1]))
    return direction[0] * toward[0] + direction[1] * toward[1]


def _player_id(path: Path, team_id: int) -> tuple[int, int]:
    started = json.loads(path.read_text(encoding="utf-8"))[0]
    matches = [
        int(player["player_id"])
        for player in started["players"]
        if int(player["team_id"]) == team_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{path}: expected one player for team {team_id}, found {matches}"
        )
    return matches[0], int(started.get("max_rounds", 1400))


def compare(paths: list[Path], *, team_id: int) -> dict[str, object]:
    scenes: list[ChangedScene] = []
    current_times: list[float] = []
    offensive_times: list[float] = []
    reason_changes: Counter[tuple[str, str]] = Counter()
    offensive_reasons: Counter[str] = Counter()
    total_frames = 0
    angle_changes = 0
    split_changes = 0
    safe_prey_frames = 0
    current_approaches = 0
    offensive_approaches = 0
    current_retreats = 0
    offensive_retreats = 0
    corner_prey_frames = 0
    corner_cutoff_selections = 0

    for path in paths:
        player_id, max_rounds = _player_id(path, team_id)
        current = SemanticLookaheadStrategy()
        offensive = SemanticOffensiveBeamStrategy()
        match_id = int(path.name.split("-")[1])

        for frame in extract_frames(path):
            context = _context(
                frame,
                player_id=player_id,
                max_rounds=max_rounds,
            )
            if context is None:
                continue
            total_frames += 1
            own = tuple(context.game.state.me.blobs.values())
            enemies = tuple(context.game.state.visible_blobs)
            mass = sum(blob.radius * blob.radius for blob in own)
            center = _mass_center(own)
            prey = _prey(context)
            predator_visible = any(
                can_eat_player_blob(enemy.radius, blob.radius)
                for blob in own
                for enemy in enemies
            )

            started_at = perf_counter()
            current_decision = current.choose(context)
            current_times.append((perf_counter() - started_at) * 1000.0)
            started_at = perf_counter()
            offensive_decision = offensive.choose(context)
            offensive_times.append((perf_counter() - started_at) * 1000.0)

            angle = _angle_degrees(
                current_decision.direction,
                offensive_decision.direction,
            )
            changed = angle > ANGLE_CHANGE_DEGREES
            split_changed = current_decision.split != offensive_decision.split
            angle_changes += changed
            split_changes += split_changed
            offensive_reasons[offensive_decision.reason] += 1
            if current_decision.reason != offensive_decision.reason:
                reason_changes[
                    (current_decision.reason, offensive_decision.reason)
                ] += 1

            current_alignment = None
            offensive_alignment = None
            prey_clearance = None
            if prey is not None:
                current_alignment = _alignment(
                    current_decision.direction,
                    center,
                    prey,
                )
                offensive_alignment = _alignment(
                    offensive_decision.direction,
                    center,
                    prey,
                )
                prey_clearance = min(
                    prey.pos[0] - prey.radius,
                    prey.pos[1] - prey.radius,
                    frame.arena_size - prey.pos[0] - prey.radius,
                    frame.arena_size - prey.pos[1] - prey.radius,
                )
                if not predator_visible:
                    safe_prey_frames += 1
                    current_approaches += current_alignment >= APPROACH_ALIGNMENT
                    offensive_approaches += (
                        offensive_alignment >= APPROACH_ALIGNMENT
                    )
                    current_retreats += current_alignment <= RETREAT_ALIGNMENT
                    offensive_retreats += (
                        offensive_alignment <= RETREAT_ALIGNMENT
                    )
                if prey_clearance <= CORNER_CLEARANCE:
                    corner_prey_frames += 1
                    corner_cutoff_selections += (
                        offensive_decision.reason == "corner_cutoff_enemy"
                    )

            offensive_diagnostics = offensive_decision.diagnostics.get(
                "offensive_beam",
                {},
            )
            offensive_value = float(
                offensive_diagnostics.get("selected_value", 0.0)
            )
            if changed or split_changed:
                scenes.append(
                    ChangedScene(
                        match_id=match_id,
                        round_number=frame.round_number,
                        mass=mass,
                        blob_count=len(own),
                        angle_degrees=angle,
                        current_reason=current_decision.reason,
                        offensive_reason=offensive_decision.reason,
                        current_split=current_decision.split,
                        offensive_split=offensive_decision.split,
                        current_prey_alignment=current_alignment,
                        offensive_prey_alignment=offensive_alignment,
                        prey_player_id=(
                            None if prey is None else int(prey.player_id)
                        ),
                        prey_mass=(
                            0.0 if prey is None else prey.radius * prey.radius
                        ),
                        prey_corner_clearance=prey_clearance,
                        predator_visible=predator_visible,
                        current_safety_margin=current_decision.diagnostics.get(
                            "selected_safety_margin"
                        ),
                        offensive_safety_margin=(
                            offensive_decision.diagnostics.get(
                                "selected_safety_margin"
                            )
                        ),
                        offensive_value=offensive_value,
                        offensive_score=offensive_decision.score,
                    )
                )

    def rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    ranked_scenes = sorted(
        scenes,
        key=lambda scene: (
            -(scene.offensive_prey_alignment or -2.0)
            + (scene.current_prey_alignment or -2.0),
            -scene.offensive_value,
            -scene.angle_degrees,
        ),
    )
    return {
        "team_id": team_id,
        "replays": len(paths),
        "frames": total_frames,
        "action_changes_over_30_degrees": angle_changes,
        "action_change_rate": rate(angle_changes, total_frames),
        "split_changes": split_changes,
        "split_change_rate": rate(split_changes, total_frames),
        "safe_prey": {
            "frames": safe_prey_frames,
            "current_approach_rate": rate(
                current_approaches,
                safe_prey_frames,
            ),
            "offensive_approach_rate": rate(
                offensive_approaches,
                safe_prey_frames,
            ),
            "current_retreat_rate": rate(
                current_retreats,
                safe_prey_frames,
            ),
            "offensive_retreat_rate": rate(
                offensive_retreats,
                safe_prey_frames,
            ),
        },
        "corner_prey": {
            "frames": corner_prey_frames,
            "corner_cutoff_selections": corner_cutoff_selections,
            "corner_cutoff_rate": rate(
                corner_cutoff_selections,
                corner_prey_frames,
            ),
        },
        "runtime_ms": {
            "current": {
                "median": statistics.median(current_times),
                "p95": _percentile(current_times, 0.95),
                "p99": _percentile(current_times, 0.99),
                "max": max(current_times, default=0.0),
            },
            "offensive": {
                "median": statistics.median(offensive_times),
                "p95": _percentile(offensive_times, 0.95),
                "p99": _percentile(offensive_times, 0.99),
                "max": max(offensive_times, default=0.0),
            },
        },
        "offensive_reason_counts": dict(offensive_reasons.most_common()),
        "reason_changes": [
            {
                "current": current_reason,
                "offensive": offensive_reason,
                "count": count,
            }
            for (current_reason, offensive_reason), count in reason_changes.most_common()
        ],
        "changed_scenes": [asdict(scene) for scene in ranked_scenes[:100]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replays", nargs="+", type=Path)
    parser.add_argument("--team", type=int, default=73)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(
        [path.resolve() for path in args.replays],
        team_id=args.team,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
