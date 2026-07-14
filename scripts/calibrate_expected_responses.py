from __future__ import annotations

"""Calibrate legal joint-response scenario weights from official replays.

This stage classifies only observed opponent commands.  It does not fit an
override threshold or enable ExpectedEvidence in production.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
BOTS = ROOT / "bots"
sys.path.insert(0, str(BOTS))

from lib.models.food_model import FoodModel  # noqa: E402
from lib.models.virus_model import VirusModel  # noqa: E402
from strategies.features import normalise  # noqa: E402
from strategies.potential_tactical_hybrid import (  # noqa: E402
    PotentialTacticalHybridStrategy,
)
from strategies.receding_horizon import (  # noqa: E402
    Action,
    EnemyBlob,
    OwnBlob,
    SearchNode,
)
from strategies.world_transition import PlayerCommand  # noqa: E402


VISION_REFERENCE_SUM_OF_RADII = 12.0
SCENARIO_LABELS = (
    "observed",
    "intercept_chase",
    "flee",
    "lateral_positive",
    "lateral_negative",
    "adaptive",
)


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


@dataclass(frozen=True, slots=True)
class ClassificationSample:
    match_id: int
    round_number: int
    opponent_player_id: int
    scenario_id: int
    angle_error_degrees: float
    split_match: bool


@dataclass(frozen=True, slots=True)
class TwoStepSample:
    match_id: int
    round_number: int
    predicted_final_mass: float
    actual_final_mass: float
    absolute_error: float
    predicted_gain_probability: float
    actual_gain: bool
    initial_mass: float = 0.0
    own_fragment_count: int = 0
    visible_enemy_count: int = 0
    relative_error: float = 0.0
    first_action_split: bool = False
    second_action_split: bool = False

    @property
    def any_action_split(self) -> bool:
        return self.first_action_split or self.second_action_split


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
        # Geometry order is deterministic within a snapshot. Replay blob IDs
        # are deliberately not carried across rounds (engine 1.13 reuses the
        # public IDs differently from the current 1.14 observation contract).
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


def _team_player(frame: ReplayFrame, team_id: int) -> ReplayPlayer | None:
    return next(
        (
            player
            for player in frame.players.values()
            if player.team_id == team_id and player.alive and player.blobs
        ),
        None,
    )


def _classification_world(
    frame: ReplayFrame,
    team_id: int,
) -> tuple[
    SearchNode,
    dict[int, list[EnemyBlob]],
    tuple[FoodModel, ...],
    tuple[VirusModel, ...],
] | None:
    own_player = _team_player(frame, team_id)
    if own_player is None:
        return None
    vision_size = _vision_size(own_player, frame.base_vision_size)
    center = _view_center(own_player, frame.arena_size, vision_size)
    own_blobs = tuple(
        OwnBlob(index, blob.x, blob.y, blob.radius, blob.merge_cooldown)
        for index, blob in enumerate(own_player.blobs)
    )
    by_player: dict[int, list[EnemyBlob]] = {}
    for player in frame.players.values():
        if player.player_id == own_player.player_id or not player.alive:
            continue
        visible = tuple(
            blob
            for blob in player.blobs
            if _circle_visible(center, vision_size, blob)
        )
        if not visible:
            continue
        previous = frame.previous_commands.get(
            player.player_id,
            PlayerCommand((0.0, 0.0)),
        ).unit
        by_player[player.player_id] = [
            EnemyBlob(
                player_id=player.player_id,
                blob_id=index,
                x=blob.x,
                y=blob.y,
                radius=blob.radius,
                direction=previous,
                merge_cooldown=blob.merge_cooldown,
            )
            for index, blob in enumerate(visible)
        ]
    previous_own = frame.previous_commands.get(
        own_player.player_id,
        PlayerCommand((1.0, 0.0)),
    ).unit
    node = SearchNode(
        own_blobs=own_blobs,
        enemies=tuple(
            enemy
            for player_id in sorted(by_player)
            for enemy in by_player[player_id]
        ),
        score=0.0,
        first_direction=previous_own,
        first_split=False,
        first_reason="replay",
        last_direction=previous_own,
    )
    foods = tuple(
        FoodModel(food_id=food.source_id, pos=food.pos)
        for food in frame.foods
        if _point_visible(center, vision_size, food)
    )
    viruses = tuple(
        VirusModel(virus_id=virus.source_id, pos=virus.pos, radius=virus.radius)
        for virus in frame.viruses
        if _circle_visible(center, vision_size, virus)
    )
    return node, by_player, foods, viruses


def _angle_error(
    actual: tuple[float, float],
    predicted: tuple[float, float],
) -> float:
    actual = normalise(actual)
    predicted = normalise(predicted)
    if actual == (0.0, 0.0) or predicted == (0.0, 0.0):
        return 180.0
    dot = max(-1.0, min(1.0, actual[0] * predicted[0] + actual[1] * predicted[1]))
    return math.degrees(math.acos(dot))


def classify_frame(
    frame: ReplayFrame,
    *,
    team_id: int = 73,
) -> list[ClassificationSample]:
    world = _classification_world(frame, team_id)
    own_player = _team_player(frame, team_id)
    if world is None or own_player is None:
        return []
    node, by_player, foods, viruses = world
    samples = []
    for player_id, group in by_player.items():
        actual = frame.commands.get(player_id)
        if actual is None:
            continue
        candidates = tuple(
            PotentialTacticalHybridStrategy._expected_enemy_command(
                group,
                node=node,
                response_type=scenario_id,
                foods=foods,
                viruses=viruses,
            )
            for scenario_id in range(len(SCENARIO_LABELS))
        )
        scenario_id, candidate = min(
            enumerate(candidates),
            key=lambda item: (
                item[1].split != actual.split,
                _angle_error(actual.unit, item[1].unit),
                item[0],
            ),
        )
        samples.append(
            ClassificationSample(
                match_id=frame.match_id,
                round_number=frame.round_number,
                opponent_player_id=player_id,
                scenario_id=scenario_id,
                angle_error_degrees=_angle_error(actual.unit, candidate.unit),
                split_match=candidate.split == actual.split,
            )
        )
    return samples


def simulate_match_two_step(
    frames: list[ReplayFrame],
    *,
    scenario_weights: tuple[float, ...],
    team_id: int = 73,
    joint_sample_count: int = 8,
) -> list[TwoStepSample]:
    """Replay actual consecutive own actions without policy optimisation."""

    if len(scenario_weights) != len(SCENARIO_LABELS):
        raise ValueError("one weight is required for each response scenario")
    weight_sum = sum(scenario_weights)
    if weight_sum <= 0.0:
        raise ValueError("scenario weights must have positive total mass")
    response_weights = tuple(weight / weight_sum for weight in scenario_weights)
    strategy = PotentialTacticalHybridStrategy()
    samples = []
    for index in range(len(frames) - 2):
        first_frame = frames[index]
        second_frame = frames[index + 1]
        actual_frame = frames[index + 2]
        if not (
            second_frame.round_number == first_frame.round_number + 1
            and actual_frame.round_number == second_frame.round_number + 1
        ):
            continue
        initial = _classification_world(first_frame, team_id)
        own_player = _team_player(first_frame, team_id)
        second_own_player = _team_player(second_frame, team_id)
        actual_player = _team_player(actual_frame, team_id)
        if initial is None or own_player is None or second_own_player is None:
            continue
        first_command = first_frame.commands.get(own_player.player_id)
        second_command = second_frame.commands.get(own_player.player_id)
        if first_command is None or second_command is None:
            continue
        node, _, first_foods, first_viruses = initial
        strategy._tactical._own_player_id = own_player.player_id
        response_table_type = type(
            "ReplayWeightedResponseTable",
            (PotentialTacticalHybridStrategy,),
            {"_EXPECTED_RESPONSE_WEIGHTS": response_weights},
        )
        response_table = response_table_type._expected_response_table(
            {enemy.player_id for enemy in node.enemies},
            sample_count=joint_sample_count,
        )
        first_action = Action(
            first_command.direction,
            split=first_command.split,
            reason="replay_actual_first",
        )
        second_action = Action(
            second_command.direction,
            split=second_command.split,
            reason="replay_actual_second",
        )
        branch_masses = []
        branch_gains = []
        for scenario_id in range(joint_sample_count):
            enemy_commands = strategy._expected_enemy_commands(
                node=node,
                response_types=response_table[scenario_id],
                foods=first_foods,
                viruses=first_viruses,
            )
            first_joint = strategy._expected_joint_command(
                node=node,
                own_action=first_action,
                enemy_commands=enemy_commands,
            )
            first = strategy._tactical._joint_physical_step(
                node=node,
                action=first_action,
                foods=first_foods,
                viruses=first_viruses,
                arena_size=first_frame.arena_size,
                first_step=True,
                joint_command=first_joint,
            )
            if first.dead:
                branch_masses.append(0.0)
                branch_gains.append(
                    first.state.projected_captures > node.projected_captures
                    or first.state.projected_viruses > node.projected_viruses
                )
                continue
            second_enemy_commands = strategy._expected_enemy_commands(
                node=first.state,
                response_types=response_table[scenario_id],
                foods=first_foods,
                viruses=first_viruses,
            )
            second_joint = strategy._expected_joint_command(
                node=first.state,
                own_action=second_action,
                enemy_commands=second_enemy_commands,
            )
            second = strategy._tactical._joint_physical_step(
                node=first.state,
                action=second_action,
                # Runtime has no future observation during its two-step
                # rollout. Keep the initial visible resource set; the carried
                # eaten/consumed IDs suppress already-resolved resources.
                foods=first_foods,
                viruses=first_viruses,
                arena_size=second_frame.arena_size,
                first_step=False,
                joint_command=second_joint,
            )
            branch_masses.append(second.final_own_mass)
            branch_gains.append(
                second.state.projected_captures > node.projected_captures
                or second.state.projected_viruses > node.projected_viruses
            )
        predicted_mass = sum(branch_masses) / joint_sample_count
        predicted_gain = sum(branch_gains) / joint_sample_count
        actual_mass = (
            sum(blob.mass for blob in actual_player.blobs)
            if actual_player is not None
            else 0.0
        )
        actual_gain = (
            own_player.player_id in first_frame.gain_player_ids
            or own_player.player_id in second_frame.gain_player_ids
        )
        initial_mass = node.total_mass
        absolute_error = abs(predicted_mass - actual_mass)
        samples.append(
            TwoStepSample(
                match_id=first_frame.match_id,
                round_number=first_frame.round_number,
                predicted_final_mass=predicted_mass,
                actual_final_mass=actual_mass,
                absolute_error=absolute_error,
                predicted_gain_probability=predicted_gain,
                actual_gain=actual_gain,
                initial_mass=initial_mass,
                own_fragment_count=len(node.own_blobs),
                visible_enemy_count=len(node.enemies),
                relative_error=absolute_error / max(initial_mass, 1e-9),
                first_action_split=first_action.split,
                second_action_split=second_action.split,
            )
        )
    return samples


def _wilson_95_lower(successes: int, total: int) -> float | None:
    if total <= 0:
        return None
    precision = successes / total
    z = 1.959963984540054
    denominator = 1.0 + z * z / total
    centre = precision + z * z / (2.0 * total)
    radius = z * math.sqrt(
        precision * (1.0 - precision) / total
        + z * z / (4.0 * total * total)
    )
    return (centre - radius) / denominator


def _precision_curve(samples: list[TwoStepSample]) -> list[dict[str, object]]:
    actual_positives = sum(sample.actual_gain for sample in samples)
    thresholds = sorted(
        {
            0.0,
            1.0,
            *(sample.predicted_gain_probability for sample in samples),
        }
    )
    curve = []
    for threshold in thresholds:
        predicted = [
            sample
            for sample in samples
            if sample.predicted_gain_probability >= threshold
        ]
        true_positives = sum(sample.actual_gain for sample in predicted)
        predicted_count = len(predicted)
        precision = true_positives / predicted_count if predicted_count else None
        curve.append(
            {
                "threshold": threshold,
                "predicted_positive_count": predicted_count,
                "true_positive_count": true_positives,
                "precision": precision,
                "precision_wilson_95_lower": _wilson_95_lower(
                    true_positives,
                    predicted_count,
                ),
                "recall": (
                    true_positives / actual_positives if actual_positives else None
                ),
            }
        )
    return curve


def _two_step_summary(samples: list[TwoStepSample]) -> dict[str, object]:
    errors = [sample.absolute_error for sample in samples]
    relative_errors = [sample.relative_error for sample in samples]
    curve = _precision_curve(samples)
    qualifying = [
        row
        for row in curve
        if row["precision"] is not None and row["precision"] >= 0.8
    ]
    frame_level_qualifying = [
        row
        for row in curve
        if row["predicted_positive_count"] >= 20
        and row["precision_wilson_95_lower"] is not None
        and row["precision_wilson_95_lower"] >= 0.8
    ]
    mass_bins = (
        ("mass_lt_4", 0.0, 4.0),
        ("mass_4_to_16", 4.0, 16.0),
        ("mass_16_to_64", 16.0, 64.0),
        ("mass_ge_64", 64.0, math.inf),
    )

    def error_distribution(rows: list[TwoStepSample]) -> dict[str, object]:
        absolute = [row.absolute_error for row in rows]
        relative = [row.relative_error for row in rows]
        return {
            "sample_count": len(rows),
            "absolute": {
                "mean": statistics.fmean(absolute) if absolute else None,
                "median": statistics.median(absolute) if absolute else None,
                "p95": _percentile(absolute, 0.95),
                "finite_sample_q97_5": _conformal_quantile(absolute, 0.975),
                "max": max(absolute) if absolute else None,
            },
            "relative": {
                "mean": statistics.fmean(relative) if relative else None,
                "median": statistics.median(relative) if relative else None,
                "p95": _percentile(relative, 0.95),
                "finite_sample_q97_5": _conformal_quantile(relative, 0.975),
                "max": max(relative) if relative else None,
            },
        }

    absolute_q = _conformal_quantile(errors, 0.975)
    relative_q = _conformal_quantile(relative_errors, 0.975)
    return {
        "sample_count": len(samples),
        "absolute_final_mass_error": {
            "mean": statistics.fmean(errors) if errors else None,
            "median": statistics.median(errors) if errors else None,
            "p95": _percentile(errors, 0.95),
            "finite_sample_q97_5": absolute_q,
            "max": max(errors) if errors else None,
        },
        "relative_final_mass_error": {
            "mean": statistics.fmean(relative_errors) if relative_errors else None,
            "median": statistics.median(relative_errors) if relative_errors else None,
            "p95": _percentile(relative_errors, 0.95),
            "finite_sample_q97_5": relative_q,
            "max": max(relative_errors) if relative_errors else None,
        },
        "mass_bins": {
            label: error_distribution(
                [sample for sample in samples if lower <= sample.initial_mass < upper]
            )
            for label, lower, upper in mass_bins
        },
        "context": {
            "own_fragment_count_max": max(
                (sample.own_fragment_count for sample in samples), default=0
            ),
            "visible_enemy_count_max": max(
                (sample.visible_enemy_count for sample in samples), default=0
            ),
        },
        "counterfactual_global_bonferroni_bound": {
            "absolute_2x_q97_5": 2.0 * absolute_q if absolute_q is not None else None,
            "relative_2x_q97_5": 2.0 * relative_q if relative_q is not None else None,
            "formula": "B_delta = q97.5(error_tactical) + q97.5(error_base)",
        },
        "actual_gain_count": sum(sample.actual_gain for sample in samples),
        "gain_positive_precision_curve": curve,
        "minimum_probability_for_80pct_precision": (
            qualifying[0]["threshold"] if qualifying else None
        ),
        "frame_level_minimum_probability_for_80pct_precision": (
            frame_level_qualifying[0]["threshold"]
            if frame_level_qualifying
            else None
        ),
        "frame_level_precision_threshold_rule": (
            "minimum threshold with predicted_positive_count >= 20 and "
            "Wilson 95% lower confidence bound >= 0.8; otherwise null"
        ),
    }


_MATCH_BLOCK_MIN_SAMPLES = 20
_MASS_BINS = (
    ("mass_lt_4", 0.0, 4.0),
    ("mass_4_to_16", 4.0, 16.0),
    ("mass_16_to_64", 16.0, 64.0),
    ("mass_ge_64", 64.0, math.inf),
)


def _block_error_distribution(
    by_match: dict[int, list[TwoStepSample]],
) -> dict[str, object]:
    blocks = {}
    eligible_absolute = []
    eligible_relative = []
    for match_id, rows in sorted(by_match.items()):
        absolute_q = _conformal_quantile(
            [row.absolute_error for row in rows],
            0.975,
        )
        relative_q = _conformal_quantile(
            [row.relative_error for row in rows],
            0.975,
        )
        eligible = len(rows) >= _MATCH_BLOCK_MIN_SAMPLES
        blocks[str(match_id)] = {
            "sample_count": len(rows),
            "eligible": eligible,
            "within_match_absolute_q97_5": absolute_q,
            "within_match_relative_q97_5": relative_q,
        }
        if eligible:
            assert absolute_q is not None and relative_q is not None
            eligible_absolute.append(absolute_q)
            eligible_relative.append(relative_q)

    def distribution(values: list[float]) -> dict[str, float | None]:
        return {
            "median": statistics.median(values) if values else None,
            "p90": _percentile(values, 0.9),
            "finite_sample_q97_5": _conformal_quantile(values, 0.975),
            "max": max(values) if values else None,
        }

    eligible_count = len(eligible_absolute)
    return {
        "minimum_samples_per_match_block": _MATCH_BLOCK_MIN_SAMPLES,
        "exclusion_rule": (
            "exclude a match block from the across-match distribution when "
            f"it contains fewer than {_MATCH_BLOCK_MIN_SAMPLES} frames"
        ),
        "observed_match_block_count": len(blocks),
        "eligible_match_block_count": eligible_count,
        "excluded_match_block_count": len(blocks) - eligible_count,
        "absolute_within_match_q97_5_distribution": distribution(
            eligible_absolute
        ),
        "relative_within_match_q97_5_distribution": distribution(
            eligible_relative
        ),
        "blocks": blocks,
    }


def _match_block_error_summary(
    by_match: dict[int, list[TwoStepSample]],
) -> dict[str, object]:
    return {
        "unit": "official replay match",
        "dependence_note": (
            "Frames within one replay are treated as correlated. Each match "
            "contributes at most one within-match q97.5 score per regime."
        ),
        "overall": _block_error_distribution(by_match),
        "mass_bins": {
            label: _block_error_distribution(
                {
                    match_id: [
                        row
                        for row in rows
                        if lower <= row.initial_mass < upper
                    ]
                    for match_id, rows in by_match.items()
                }
            )
            for label, lower, upper in _MASS_BINS
        },
    }


def _lomo_gain_precision_summary(
    by_match: dict[int, list[TwoStepSample]],
) -> dict[str, object]:
    all_samples = [row for rows in by_match.values() for row in rows]
    thresholds = sorted(
        {
            0.0,
            1.0,
            *(row.predicted_gain_probability for row in all_samples),
        }
    )
    rows = []
    match_ids = sorted(by_match)
    for threshold in thresholds:
        per_match_counts = {
            match_id: (
                sum(
                    sample.predicted_gain_probability >= threshold
                    for sample in samples
                ),
                sum(
                    sample.predicted_gain_probability >= threshold
                    and sample.actual_gain
                    for sample in samples
                ),
            )
            for match_id, samples in by_match.items()
        }
        total_count = sum(count for count, _ in per_match_counts.values())
        total_true = sum(true for _, true in per_match_counts.values())
        folds = []
        for heldout_match_id in match_ids:
            heldout_count, heldout_true = per_match_counts[heldout_match_id]
            training_count = total_count - heldout_count
            training_true = total_true - heldout_true
            folds.append(
                {
                    "heldout_match_id": heldout_match_id,
                    "predicted_positive_count": training_count,
                    "true_positive_count": training_true,
                    "precision": (
                        training_true / training_count if training_count else None
                    ),
                    "precision_wilson_95_lower": _wilson_95_lower(
                        training_true,
                        training_count,
                    ),
                }
            )
        eligible = [
            fold
            for fold in folds
            if fold["predicted_positive_count"] >= _MATCH_BLOCK_MIN_SAMPLES
            and fold["precision_wilson_95_lower"] is not None
        ]
        lowers = [fold["precision_wilson_95_lower"] for fold in eligible]
        rows.append(
            {
                "threshold": threshold,
                "fold_count": len(folds),
                "eligible_fold_count": len(eligible),
                "minimum_predicted_positive_count": min(
                    (fold["predicted_positive_count"] for fold in folds),
                    default=0,
                ),
                "minimum_wilson_95_lower": min(lowers) if lowers else None,
                "median_wilson_95_lower": (
                    statistics.median(lowers) if lowers else None
                ),
                "folds": folds,
            }
        )
    qualifying = [
        row
        for row in rows
        if row["eligible_fold_count"] == row["fold_count"]
        and row["minimum_wilson_95_lower"] is not None
        and row["minimum_wilson_95_lower"] >= 0.8
    ]
    return {
        "method": "leave-one-match-out pooled Wilson 95% lower bound",
        "minimum_predicted_positives_per_fold": _MATCH_BLOCK_MIN_SAMPLES,
        "curve": rows,
        "minimum_probability_for_all_folds_80pct_lower_bound": (
            qualifying[0]["threshold"] if qualifying else None
        ),
    }


def _action_regime_summary(
    by_match: dict[int, list[TwoStepSample]],
) -> dict[str, object]:
    all_samples = [sample for samples in by_match.values() for sample in samples]
    regimes = {
        "split_any": lambda sample: sample.any_action_split,
        "no_split": lambda sample: not sample.any_action_split,
    }
    result = {
        "first_action_split_count": sum(
            sample.first_action_split for sample in all_samples
        ),
        "second_action_split_count": sum(
            sample.second_action_split for sample in all_samples
        ),
        "any_action_split_count": sum(
            sample.any_action_split for sample in all_samples
        ),
    }
    for label, predicate in regimes.items():
        regime_by_match = {
            match_id: [sample for sample in samples if predicate(sample)]
            for match_id, samples in by_match.items()
        }
        regime_samples = [
            sample for samples in regime_by_match.values() for sample in samples
        ]
        frame_summary = _two_step_summary(regime_samples)
        result[label] = {
            "frame_level": frame_summary,
            "match_block_errors": _block_error_distribution(regime_by_match),
            "gain_precision_match_robust": _lomo_gain_precision_summary(
                regime_by_match
            ),
        }
    return result


def build_stage_b_report(
    replay_paths: list[Path],
    *,
    stage_a_report: dict[str, object],
    team_id: int = 73,
) -> dict[str, object]:
    """Evaluate each match with scenario weights trained on all other matches."""

    lomo = stage_a_report["leave_one_match_out"]
    by_match: dict[int, list[TwoStepSample]] = {}
    for path in replay_paths:
        match_id = _match_id(path)
        training_weights = lomo[str(match_id)]["training"]["scenario_weights"]
        weights = tuple(
            float(training_weights[label]) for label in SCENARIO_LABELS
        )
        by_match[match_id] = simulate_match_two_step(
            extract_frames(path),
            scenario_weights=weights,
            team_id=team_id,
        )
    all_samples = [sample for samples in by_match.values() for sample in samples]
    overall = _two_step_summary(all_samples)
    lomo_gain = _lomo_gain_precision_summary(by_match)
    replay_candidate = lomo_gain[
        "minimum_probability_for_all_folds_80pct_lower_bound"
    ]
    overall["replay_lomo_minimum_probability_for_80pct_precision"] = (
        replay_candidate
    )
    overall["runtime_minimum_probability_for_80pct_precision"] = None
    overall["runtime_precision_threshold_status"] = (
        "fail_closed_pending_local_engine_heldout_seeds"
    )
    return {
        "stage": "two_step_holdout_calibration",
        "weight_source": "leave_one_match_out_stage_a_training_weights",
        "joint_assignment": "player_stratified_latin_hypercube",
        "joint_sample_count": 8,
        "branch_weights": "equal_1_over_joint_sample_count",
        "own_action_source": "actual_consecutive_team_actions",
        "continuation_optimisation": False,
        "calibration_caveat": (
            "Errors are measured on actual consecutive team73 actions. Tactical "
            "counterfactual roots may be out of distribution; use the split-"
            "conformal Bonferroni bound and fall back to the global regime when "
            "a Mondrian mass bin has insufficient samples."
        ),
        "overall": overall,
        "match_block_robust": _match_block_error_summary(by_match),
        "gain_precision_match_robust": lomo_gain,
        "action_regimes": _action_regime_summary(by_match),
        "matches": {
            str(match_id): _two_step_summary(by_match[match_id])
            for match_id in sorted(by_match)
        },
    }


def _coverage_summary(
    samples: list[TwoStepSample],
    *,
    absolute_bound: float | None,
    relative_bound: float | None,
) -> dict[str, object]:
    def coverage(values: list[float], bound: float | None) -> float | None:
        if not values or bound is None:
            return None
        return sum(value <= bound for value in values) / len(values)

    return {
        "sample_count": len(samples),
        "absolute_bound": absolute_bound,
        "absolute_coverage": coverage(
            [sample.absolute_error for sample in samples],
            absolute_bound,
        ),
        "relative_bound": relative_bound,
        "relative_coverage": coverage(
            [sample.relative_error for sample in samples],
            relative_bound,
        ),
    }


def _gain_precision_at_threshold(
    samples: list[TwoStepSample],
    threshold: float | None,
) -> dict[str, object]:
    if threshold is None:
        return {
            "threshold": None,
            "predicted_positive_count": 0,
            "true_positive_count": 0,
            "precision": None,
            "precision_wilson_95_lower": None,
        }
    predicted = [
        sample
        for sample in samples
        if sample.predicted_gain_probability >= threshold
    ]
    true_positives = sum(sample.actual_gain for sample in predicted)
    return {
        "threshold": threshold,
        "predicted_positive_count": len(predicted),
        "true_positive_count": true_positives,
        "precision": true_positives / len(predicted) if predicted else None,
        "precision_wilson_95_lower": _wilson_95_lower(
            true_positives,
            len(predicted),
        ),
    }


def build_engine_validation_report(
    recording_paths: list[Path],
    *,
    stage_a_report: dict[str, object],
    stage_b_report: dict[str, object],
    team_id: int,
) -> dict[str, object]:
    """Validate on separate engine recordings without fitting any parameter."""

    overall_weights = stage_a_report["overall"]["scenario_weights"]
    weights = tuple(float(overall_weights[label]) for label in SCENARIO_LABELS)
    by_recording = {}
    engine_versions = set()
    for index, path in enumerate(recording_paths):
        frames = extract_frames(path)
        engine_versions.update(frame.engine_version for frame in frames)
        by_recording[index] = simulate_match_two_step(
            frames,
            scenario_weights=weights,
            team_id=team_id,
        )
    samples = [sample for rows in by_recording.values() for sample in rows]
    official = stage_b_report["overall"]
    absolute_q = official["absolute_final_mass_error"]["finite_sample_q97_5"]
    relative_q = official["relative_final_mass_error"]["finite_sample_q97_5"]
    absolute_bonferroni = official["counterfactual_global_bonferroni_bound"][
        "absolute_2x_q97_5"
    ]
    relative_bonferroni = official["counterfactual_global_bonferroni_bound"][
        "relative_2x_q97_5"
    ]
    replay_threshold = official[
        "replay_lomo_minimum_probability_for_80pct_precision"
    ]
    official_block = stage_b_report["match_block_robust"]["overall"]
    block_absolute_q = official_block[
        "absolute_within_match_q97_5_distribution"
    ]["finite_sample_q97_5"]
    block_relative_q = official_block[
        "relative_within_match_q97_5_distribution"
    ]["finite_sample_q97_5"]

    def action_regime(label: str, split: bool) -> dict[str, object]:
        rows = [sample for sample in samples if sample.any_action_split is split]
        official_regime = stage_b_report["action_regimes"][label]["frame_level"]
        official_regime_block = stage_b_report["action_regimes"][label][
            "match_block_errors"
        ]
        return {
            "error_summary": _two_step_summary(rows),
            "coverage_against_official_regime_q97_5": _coverage_summary(
                rows,
                absolute_bound=official_regime["absolute_final_mass_error"][
                    "finite_sample_q97_5"
                ],
                relative_bound=official_regime["relative_final_mass_error"][
                    "finite_sample_q97_5"
                ],
            ),
            "coverage_against_official_match_block_q97_5": _coverage_summary(
                rows,
                absolute_bound=official_regime_block[
                    "absolute_within_match_q97_5_distribution"
                ]["finite_sample_q97_5"],
                relative_bound=official_regime_block[
                    "relative_within_match_q97_5_distribution"
                ]["finite_sample_q97_5"],
            ),
            "gain_precision_at_replay_lomo_threshold": (
                _gain_precision_at_threshold(rows, replay_threshold)
            ),
        }

    return {
        "source": "separate local engine recordings; excluded from calibration",
        "recording_count": len(recording_paths),
        "recordings": [str(path) for path in recording_paths],
        "team_id": team_id,
        "engine_versions": sorted(engine_versions),
        "weight_source": "stage_a_overall_fixed",
        "joint_sample_count": 8,
        "sample_count": len(samples),
        "error_summary": _two_step_summary(samples),
        "coverage_against_official_frame_q97_5": _coverage_summary(
            samples,
            absolute_bound=absolute_q,
            relative_bound=relative_q,
        ),
        "coverage_against_official_match_block_q97_5": _coverage_summary(
            samples,
            absolute_bound=block_absolute_q,
            relative_bound=block_relative_q,
        ),
        "coverage_against_official_counterfactual_bonferroni_bound": (
            _coverage_summary(
                samples,
                absolute_bound=absolute_bonferroni,
                relative_bound=relative_bonferroni,
            )
        ),
        "gain_precision_at_replay_lomo_threshold": _gain_precision_at_threshold(
            samples,
            replay_threshold,
        ),
        "action_regimes": {
            "split_any": action_regime("split_any", True),
            "no_split": action_regime("no_split", False),
        },
        "per_recording": {
            str(recording_paths[index]): _two_step_summary(rows)
            for index, rows in by_recording.items()
        },
        "runtime_threshold_eligible": False,
        "runtime_threshold_status": (
            "fail_closed: fewer than 20 independent local engine seed recordings"
        ),
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _conformal_quantile(values: list[float], coverage: float) -> float | None:
    """Finite-sample split-conformal quantile, clipped to the observed maximum."""

    if not values:
        return None
    ordered = sorted(values)
    rank = min(len(ordered), math.ceil((len(ordered) + 1) * coverage))
    return ordered[rank - 1]


def _summary(samples: list[ClassificationSample]) -> dict[str, object]:
    counts = Counter(sample.scenario_id for sample in samples)
    sample_count = len(samples)
    errors = [sample.angle_error_degrees for sample in samples]
    return {
        "sample_count": sample_count,
        "scenario_counts": {
            SCENARIO_LABELS[index]: counts[index]
            for index in range(len(SCENARIO_LABELS))
        },
        "scenario_weights": {
            SCENARIO_LABELS[index]: (
                counts[index] / sample_count if sample_count else 0.0
            )
            for index in range(len(SCENARIO_LABELS))
        },
        "angle_error_degrees": {
            "mean": statistics.fmean(errors) if errors else None,
            "median": statistics.median(errors) if errors else None,
            "p75": _percentile(errors, 0.75),
            "p95": _percentile(errors, 0.95),
        },
        "split_accuracy": (
            sum(sample.split_match for sample in samples) / sample_count
            if sample_count
            else None
        ),
    }


def build_report(
    replay_paths: list[Path],
    *,
    replay_dirs: list[Path],
    team_id: int = 73,
) -> dict[str, object]:
    by_match: dict[int, list[ClassificationSample]] = {}
    engine_versions: set[str] = set()
    for path in replay_paths:
        frames = extract_frames(path)
        engine_versions.update(frame.engine_version for frame in frames)
        samples = [
            sample
            for frame in frames
            for sample in classify_frame(frame, team_id=team_id)
        ]
        by_match[_match_id(path)] = samples
    all_samples = [sample for samples in by_match.values() for sample in samples]
    match_ids = sorted(by_match)
    return {
        "schema_version": 1,
        "stage": "scenario_action_classification",
        "config": {
            "team_id": team_id,
            "replay_dirs": [str(path.resolve()) for path in replay_dirs],
            "match_count": len(replay_paths),
            "engine_versions": sorted(engine_versions),
            "classification_order": "split_match_then_minimum_angle_error",
            "scenario_labels": list(SCENARIO_LABELS),
            "identity_caveat": (
                "2026.1.13 replay blob IDs are not used for round-to-round "
                "tracking; classification uses snapshot geometry and one "
                "command per player."
            ),
        },
        "overall": _summary(all_samples),
        "matches": {
            str(match_id): _summary(by_match[match_id]) for match_id in match_ids
        },
        "leave_one_match_out": {
            str(held_out): {
                "training": _summary(
                    [
                        sample
                        for match_id in match_ids
                        if match_id != held_out
                        for sample in by_match[match_id]
                    ]
                ),
                "holdout": _summary(by_match[held_out]),
            }
            for held_out in match_ids
        },
    }


def _replay_files(directories: list[Path]) -> list[Path]:
    paths = {
        path.resolve()
        for directory in directories
        for path in directory.glob("*-replay.json")
    }
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify visible opponent commands into ExpectedEvidence scenarios."
    )
    parser.add_argument(
        "--replay-dir",
        type=Path,
        action="append",
        required=True,
        help="Absolute directory containing official *-replay.json files (repeatable).",
    )
    parser.add_argument("--team-id", type=int, default=73)
    parser.add_argument(
        "--engine-validation",
        type=Path,
        action="append",
        default=[],
        help=(
            "Separate local-engine game.json recording (repeatable, maximum 3); "
            "reported as validation and never included in fitting."
        ),
    )
    parser.add_argument("--engine-validation-team-id", type=int, default=0)
    parser.add_argument(
        "--include-stage-b",
        action="store_true",
        help=(
            "Also run two-step physical calibration. Use only after the runtime "
            "joint-scenario assignment and physics kernel are final."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "expected-response-calibration-stage-a.json",
    )
    args = parser.parse_args()
    replay_dirs = [path.resolve() for path in args.replay_dir]
    if any(not path.is_absolute() for path in args.replay_dir):
        parser.error("--replay-dir values must be absolute paths")
    replay_paths = _replay_files(replay_dirs)
    if not replay_paths:
        parser.error("no *-replay.json files found")
    engine_validation_paths = [path.resolve() for path in args.engine_validation]
    if len(engine_validation_paths) > 3:
        parser.error("at most three --engine-validation recordings are supported")
    if any(not path.is_file() for path in engine_validation_paths):
        parser.error("every --engine-validation value must be an existing file")
    if engine_validation_paths and not args.include_stage_b:
        parser.error("--engine-validation requires --include-stage-b")
    report = build_report(
        replay_paths,
        replay_dirs=replay_dirs,
        team_id=args.team_id,
    )
    if args.include_stage_b:
        report["stage_b"] = build_stage_b_report(
            replay_paths,
            stage_a_report=report,
            team_id=args.team_id,
        )
        if engine_validation_paths:
            report["stage_b"]["engine_1_14_validation"] = (
                build_engine_validation_report(
                    engine_validation_paths,
                    stage_a_report=report,
                    stage_b_report=report["stage_b"],
                    team_id=args.engine_validation_team_id,
                )
            )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote {output} "
        f"({report['overall']['sample_count']} classified commands)"
    )


if __name__ == "__main__":
    main()
