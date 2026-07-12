from __future__ import annotations

"""Fit and evaluate local opponent policies from official replay JSON files."""

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
BOTS = ROOT / "bots"
sys.path.insert(0, str(BOTS))

from strategies.replay_imitation import (  # noqa: E402
    FEATURE_NAMES,
    SPLIT_FEATURE_NAMES,
    ImitationBlob,
    ImitationObservation,
    ImitationPoint,
    ReplayProfile,
    direction_feature_vectors,
    split_feature_values,
    stable_unit_interval,
)


USER_TEAM_ID = 73
VISION_REFERENCE_SUM_OF_RADII = 12.0


@dataclass
class PlayerState:
    player_id: int
    team_id: int
    alive: bool
    blobs: dict[int, ImitationBlob]


@dataclass(frozen=True)
class ReplaySample:
    match_id: int
    team_id: int
    player_id: int
    round_number: int
    direction_features: tuple[tuple[float, float], ...]
    split_features: tuple[float, ...]
    target_direction: tuple[float, float]
    target_split: bool


def _unit(x: float, y: float) -> tuple[float, float]:
    magnitude = math.hypot(x, y)
    if magnitude <= 1e-9 or not math.isfinite(magnitude):
        return (0.0, 0.0)
    return (x / magnitude, y / magnitude)


def _player_from_event(payload: dict[str, object], team_id: int) -> PlayerState:
    player_id = int(payload["player_id"])
    blobs = {
        int(blob["blob_id"]): ImitationBlob(
            x=float(blob["pos"][0]),
            y=float(blob["pos"][1]),
            radius=float(blob["radius"]),
            player_id=player_id,
            team_id=team_id,
            blob_id=int(blob["blob_id"]),
            merge_cooldown=int(blob.get("merge_cooldown", 0)),
        )
        for blob in payload.get("blobs", [])
    }
    return PlayerState(
        player_id=player_id,
        team_id=team_id,
        alive=bool(payload.get("alive", True)),
        blobs=blobs,
    )


def _mass_center(player: PlayerState) -> tuple[float, float]:
    mass = sum(blob.radius * blob.radius for blob in player.blobs.values())
    if mass <= 1e-9:
        return (30.0, 30.0)
    return (
        sum(blob.x * blob.radius * blob.radius for blob in player.blobs.values()) / mass,
        sum(blob.y * blob.radius * blob.radius for blob in player.blobs.values()) / mass,
    )


def _vision_size(player: PlayerState, base_vision: float) -> float:
    if not player.alive:
        return base_vision
    sum_of_radii = sum(blob.radius for blob in player.blobs.values())
    if sum_of_radii <= 0.0:
        return base_vision
    scale = max(sum_of_radii / VISION_REFERENCE_SUM_OF_RADII, 1.0) ** 0.4
    return scale * base_vision


def _view_center(player: PlayerState, arena_size: float, vision_size: float) -> tuple[float, float]:
    x, y = _mass_center(player)
    half = min(vision_size, arena_size) / 2.0
    return (
        min(max(x, half), arena_size - half),
        min(max(y, half), arena_size - half),
    )


def _point_visible(center: tuple[float, float], vision_size: float, x: float, y: float) -> bool:
    half = vision_size / 2.0
    return abs(x - center[0]) <= half and abs(y - center[1]) <= half


def _circle_visible(
    center: tuple[float, float], vision_size: float, x: float, y: float, radius: float
) -> bool:
    half = vision_size / 2.0
    dx = max(abs(x - center[0]) - half, 0.0)
    dy = max(abs(y - center[1]) - half, 0.0)
    return dx * dx + dy * dy <= radius * radius


def _observation(
    *,
    player_id: int,
    round_number: int,
    max_rounds: int,
    arena_size: float,
    base_vision: float,
    players: dict[int, PlayerState],
    foods: dict[int, ImitationPoint],
    viruses: dict[int, ImitationPoint],
) -> ImitationObservation:
    player = players[player_id]
    vision_size = _vision_size(player, base_vision)
    center = _view_center(player, arena_size, vision_size)
    visible_blobs = tuple(
        blob
        for other_id, other in players.items()
        if other_id != player_id and other.alive
        for blob in sorted(other.blobs.values(), key=lambda item: item.blob_id)
        if _circle_visible(center, vision_size, blob.x, blob.y, blob.radius)
    )
    visible_food = tuple(
        food
        for food in foods.values()
        if _point_visible(center, vision_size, food.x, food.y)
    )
    visible_viruses = tuple(
        virus
        for virus in viruses.values()
        if _circle_visible(center, vision_size, virus.x, virus.y, virus.radius)
    )
    return ImitationObservation(
        round_number=round_number,
        max_rounds=max_rounds,
        arena_size=arena_size,
        own_blobs=tuple(player.blobs.values()),
        visible_blobs=visible_blobs,
        visible_food=visible_food,
        visible_viruses=visible_viruses,
    )


def extract_samples(path: Path) -> list[ReplaySample]:
    match_id = int(path.name.split("-")[1])
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
        int(payload["player_id"]): _player_from_event(payload, int(payload["team_id"]))
        for payload in started["players"]
    }
    foods: dict[int, ImitationPoint] = {}
    viruses: dict[int, ImitationPoint] = {}
    previous_directions: dict[int, tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))
    samples: list[ReplaySample] = []
    round_number = -1
    in_move_batch = False

    for event in events[1:]:
        event_type = event["event_type"]
        if event_type == "move_player":
            if not in_move_batch:
                round_number += 1
                in_move_batch = True
            player_id = int(event["player_id"])
            direction = _unit(float(event["direction"]["x"]), float(event["direction"]["y"]))
            observation = _observation(
                player_id=player_id,
                round_number=round_number,
                max_rounds=max_rounds,
                arena_size=arena_size,
                base_vision=base_vision,
                players=players,
                foods=foods,
                viruses=viruses,
            )
            previous = previous_directions[player_id]
            provisional = direction_feature_vectors(observation, previous)
            samples.append(
                ReplaySample(
                    match_id=match_id,
                    team_id=team_by_player[player_id],
                    player_id=player_id,
                    round_number=round_number,
                    direction_features=provisional,
                    split_features=split_feature_values(observation, previous, direction),
                    target_direction=direction,
                    target_split=bool(event["split"]),
                )
            )
            previous_directions[player_id] = direction
            continue

        in_move_batch = False
        if event_type == "event_food_spawned":
            for food in event["foods"]:
                food_id = int(food["food_id"])
                foods[food_id] = ImitationPoint(
                    float(food["pos"][0]), float(food["pos"][1]), entity_id=food_id
                )
        elif event_type == "event_food_eaten":
            for food_id in event["food_ids"]:
                foods.pop(int(food_id), None)
        elif event_type == "event_virus_spawned":
            for virus in event["viruses"]:
                virus_id = int(virus["virus_id"])
                viruses[virus_id] = ImitationPoint(
                    float(virus["pos"][0]),
                    float(virus["pos"][1]),
                    float(virus["radius"]),
                    virus_id,
                )
        elif event_type == "event_virus_consumed":
            viruses.pop(int(event["virus_id"]), None)
        elif event_type == "event_player_moved":
            player_id = int(event["player_id"])
            players[player_id] = _player_from_event(event, team_by_player[player_id])

    return samples


def _solve(matrix: list[list[float]], target: list[float]) -> tuple[float, ...]:
    n = len(target)
    augmented = [matrix[i][:] + [target[i]] for i in range(n)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        if abs(divisor) <= 1e-12:
            continue
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return tuple(augmented[row][-1] for row in range(n))


def fit_direction(
    samples: Sequence[ReplaySample],
    ridge: float = 0.25,
    sample_weights: Sequence[float] | None = None,
) -> tuple[float, ...]:
    size = len(FEATURE_NAMES)
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    target = [0.0 for _ in range(size)]
    if sample_weights is not None and len(sample_weights) != len(samples):
        raise ValueError("sample_weights must align with samples")
    for sample_index, sample in enumerate(samples):
        sample_weight = sample_weights[sample_index] if sample_weights is not None else 1.0
        tx, ty = sample.target_direction
        if (tx, ty) == (0.0, 0.0):
            continue
        for i, first in enumerate(sample.direction_features):
            target[i] += sample_weight * (first[0] * tx + first[1] * ty)
            for j, second in enumerate(sample.direction_features):
                matrix[i][j] += sample_weight * (
                    first[0] * second[0] + first[1] * second[1]
                )
    for index in range(size):
        matrix[index][index] += ridge
    return _solve(matrix, target)


def _sample_regime(sample: ReplaySample) -> int:
    return (
        (1 if sample.split_features[9] > 0.5 else 0)
        | (2 if sample.split_features[6] > 0.5 else 0)
        | (4 if sample.split_features[12] > 0.5 else 0)
    )


def fit_regime_directions(
    samples: Sequence[ReplaySample],
    global_weights: tuple[float, ...],
    ridge: float = 0.25,
) -> tuple[tuple[float, ...], ...]:
    result: list[tuple[float, ...]] = []
    for regime in range(8):
        subset = [sample for sample in samples if _sample_regime(sample) == regime]
        result.append(fit_direction(subset, ridge) if len(subset) >= 40 else global_weights)
    return tuple(result)


def fit_autonomous_directions(
    samples: Sequence[ReplaySample],
    ridge: float = 0.25,
    iterations: int = 1,
    distinguish_fragmented: bool = False,
    robust_delta_degrees: float | None = None,
) -> tuple[
    tuple[float, ...],
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...],
]:
    """Fit against the direction history the clone will actually produce.

    Replay extraction records the teacher's preceding command in the stateful
    ``previous`` features.  At runtime those features contain the clone's own
    preceding prediction instead, so a plain regression is trained on state it
    will not observe after its first mistake.  Iteratively replaying each trace
    with the current model removes that teacher-forcing mismatch.
    """
    previous_index = FEATURE_NAMES.index("previous")
    previous_left_index = FEATURE_NAMES.index("previous_left")
    direction_weights = fit_direction(samples, ridge)
    regime_count = 16 if distinguish_fragmented else 8

    def regime(sample: ReplaySample) -> int:
        base = _sample_regime(sample)
        return base | (8 if distinguish_fragmented and sample.split_features[3] > 0.0625 else 0)

    def fit_regimes(training: Sequence[ReplaySample]) -> tuple[tuple[float, ...], ...]:
        fitted: list[tuple[float, ...]] = []
        for index in range(regime_count):
            subset = [sample for sample in training if regime(sample) == index]
            fitted.append(
                fit_direction(subset, ridge) if len(subset) >= 40 else direction_weights
            )
        return tuple(fitted)

    regime_weights = fit_regimes(samples)

    fitting_samples: Sequence[ReplaySample] = samples
    for _ in range(iterations):
        previous_by_trace: dict[tuple[int, int], tuple[float, float]] = defaultdict(
            lambda: (0.0, 0.0)
        )
        autonomous_samples: list[ReplaySample] = []
        for sample in sorted(
            samples,
            key=lambda item: (item.match_id, item.player_id, item.round_number),
        ):
            key = (sample.match_id, sample.player_id)
            previous = previous_by_trace[key]
            features = list(sample.direction_features)
            features[previous_index] = previous
            features[previous_left_index] = (-previous[1], previous[0])
            autonomous_sample = replace(sample, direction_features=tuple(features))
            autonomous_samples.append(autonomous_sample)

            weights = regime_weights[regime(sample)]
            x = sum(weight * vector[0] for weight, vector in zip(weights, features))
            y = sum(weight * vector[1] for weight, vector in zip(weights, features))
            previous_by_trace[key] = _unit(x, y)

        direction_weights = fit_direction(autonomous_samples, ridge)
        fitting_samples = autonomous_samples
        regime_weights = fit_regimes(autonomous_samples)

    if robust_delta_degrees is not None:
        for _ in range(3):
            robust_weights: list[tuple[float, ...]] = []
            for regime_index in range(regime_count):
                subset = [
                    sample
                    for sample in fitting_samples
                    if regime(sample) == regime_index
                ]
                if len(subset) < 40:
                    robust_weights.append(direction_weights)
                    continue
                current = regime_weights[regime_index]
                weights: list[float] = []
                for sample in subset:
                    x = sum(
                        weight * vector[0]
                        for weight, vector in zip(current, sample.direction_features)
                    )
                    y = sum(
                        weight * vector[1]
                        for weight, vector in zip(current, sample.direction_features)
                    )
                    predicted = _unit(x, y)
                    dot = max(
                        -1.0,
                        min(
                            1.0,
                            predicted[0] * sample.target_direction[0]
                            + predicted[1] * sample.target_direction[1],
                        ),
                    )
                    error = math.degrees(math.acos(dot))
                    weights.append(min(1.0, robust_delta_degrees / max(error, 1e-6)))
                robust_weights.append(fit_direction(subset, ridge, weights))
            regime_weights = tuple(robust_weights)

    return (
        direction_weights,
        regime_weights[:8],
        regime_weights[8:] if distinguish_fragmented else (),
    )


def _split_scores(weights: Sequence[float], samples: Sequence[ReplaySample]) -> list[float]:
    return [
        sum(weight * value for weight, value in zip(weights, sample.split_features))
        for sample in samples
    ]


def _f1(labels: Sequence[bool], predictions: Sequence[bool]) -> tuple[float, float, float]:
    true_positive = sum(label and prediction for label, prediction in zip(labels, predictions))
    false_positive = sum(not label and prediction for label, prediction in zip(labels, predictions))
    false_negative = sum(label and not prediction for label, prediction in zip(labels, predictions))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _tolerant_split_f1(
    samples: Sequence[ReplaySample], predictions: Sequence[bool], tolerance: int = 2
) -> tuple[float, float, float]:
    true_by_trace: dict[tuple[int, int], list[int]] = defaultdict(list)
    predicted_by_trace: dict[tuple[int, int], list[int]] = defaultdict(list)
    for sample, prediction in zip(samples, predictions):
        key = (sample.match_id, sample.player_id)
        if sample.target_split:
            true_by_trace[key].append(sample.round_number)
        if prediction:
            predicted_by_trace[key].append(sample.round_number)
    matched = 0
    true_count = sum(len(rounds) for rounds in true_by_trace.values())
    predicted_count = sum(len(rounds) for rounds in predicted_by_trace.values())
    for key in set(true_by_trace) | set(predicted_by_trace):
        remaining = set(predicted_by_trace[key])
        for true_round in true_by_trace[key]:
            candidates = [round_number for round_number in remaining if abs(round_number - true_round) <= tolerance]
            if not candidates:
                continue
            chosen = min(candidates, key=lambda round_number: abs(round_number - true_round))
            remaining.remove(chosen)
            matched += 1
    precision = matched / predicted_count if predicted_count else (1.0 if true_count == 0 else 0.0)
    recall = matched / true_count if true_count else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def fit_split(samples: Sequence[ReplaySample], ridge: float = 0.5) -> tuple[tuple[float, ...], float]:
    size = len(SPLIT_FEATURE_NAMES)
    positives = sum(sample.target_split for sample in samples)
    if positives == 0:
        return (tuple(0.0 for _ in range(size)), math.inf)
    positive_weight = min((len(samples) - positives) / max(positives, 1), 100.0)
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    target = [0.0 for _ in range(size)]
    for sample in samples:
        weight = positive_weight if sample.target_split else 1.0
        label = 1.0 if sample.target_split else 0.0
        for i, first in enumerate(sample.split_features):
            target[i] += weight * first * label
            for j, second in enumerate(sample.split_features):
                matrix[i][j] += weight * first * second
    for index in range(size):
        matrix[index][index] += ridge
    weights = _solve(matrix, target)
    scores = _split_scores(weights, samples)
    ordered = sorted(scores)
    candidates = {
        ordered[min(int(len(ordered) * q / 30), len(ordered) - 1)]
        for q in range(1, 30)
    }
    threshold = max(
        candidates,
        key=lambda value: _tolerant_split_f1(
            samples, [score >= value for score in scores]
        )[2],
    )
    return weights, threshold


def detect_angle_grid(samples: Sequence[ReplaySample]) -> tuple[int, float]:
    angles = [
        math.atan2(sample.target_direction[1], sample.target_direction[0])
        for sample in samples
        if sample.target_direction != (0.0, 0.0)
    ]
    if not angles:
        return 0, 0.0
    for bins in (8, 12, 16, 24, 32):
        phases = [angle * bins for angle in angles]
        scaled_phase = math.atan2(
            sum(math.sin(value) for value in phases),
            sum(math.cos(value) for value in phases),
        )
        # A bot can aim an occasional split directly at prey while quantizing
        # every ordinary move. Refit the phase from circular inliers so those
        # sparse tactical angles do not destroy detection of the movement grid.
        for tolerance in (0.1, 0.01):
            inliers = [
                value
                for value in phases
                if abs((value - scaled_phase + math.pi) % math.tau - math.pi)
                <= tolerance
            ]
            if inliers:
                scaled_phase = math.atan2(
                    sum(math.sin(value) for value in inliers),
                    sum(math.cos(value) for value in inliers),
                )
        phase = scaled_phase / bins
        step = math.tau / bins
        errors = [abs((angle - phase + step / 2.0) % step - step / 2.0) for angle in angles]
        if sorted(errors)[int(0.95 * (len(errors) - 1))] <= 1e-5:
            return bins, phase
    return 0, 0.0


def fit_angle_grid_rates(
    samples: Sequence[ReplaySample], bins: int
) -> tuple[float, ...]:
    step = math.tau / bins
    def aligned(sample: ReplaySample) -> bool:
        x, y = sample.target_direction
        return abs((math.atan2(y, x) + step / 2.0) % step - step / 2.0) <= 1e-5

    global_rate = sum(aligned(sample) for sample in samples) / max(len(samples), 1)
    rates: list[float] = []
    for regime in range(8):
        subset = [sample for sample in samples if _sample_regime(sample) == regime]
        if len(subset) < 40:
            rates.append(global_rate)
            continue
        rates.append(
            sum(aligned(sample) for sample in subset) / len(subset)
        )
    return tuple(rates)


def _profile(team_id: int, samples: Sequence[ReplaySample]) -> ReplayProfile:
    direction_ridge = 0.005 if team_id in {35, 49} else 0.25
    autonomous_teams = {3, 9, 14, 15, 35, 49, 58, 59, 63}
    direction_weights, regime_direction_weights, fragmented_direction_weights = fit_autonomous_directions(
        samples,
        direction_ridge,
        iterations=1 if team_id in autonomous_teams else 0,
        distinguish_fragmented=team_id == 9,
        robust_delta_degrees=45.0 if team_id == 59 else None,
    )
    split_weights, split_threshold = fit_split(samples)
    angle_bins, angle_offset = detect_angle_grid(samples)
    probabilistic_angle_bins = 16 if team_id == 58 else 0
    angle_grid_rates = (
        fit_angle_grid_rates(samples, probabilistic_angle_bins)
        if probabilistic_angle_bins
        else ()
    )
    # Team 49's split policy is edge-triggered. Static classification fires on
    # every frame while the same prey remains visible; the observed policy
    # waits roughly one engine split interval before it can fire again.
    split_rules = {
        3: ((1.00, 2.0, 0.25, 0.125, 0.0, 0.0), 18),
        9: ((0.65, 2.5, 0.14, 0.0625, 0.0, 0.0), 90),
        12: ((0.60, 1.5, 0.14, 0.0625, 0.0, 0.0), 15),
        14: ((0.65, 2.0, 0.25, 0.125, 0.0, 0.0), 15),
        15: ((1.00, 2.0, 0.20, 0.25, 0.0, 0.0), 10),
        35: ((2.00, 0.0, 0.20, 0.0625, 1.0, 0.0), 0),
        49: ((0.65, 1.5, 0.15, 0.125, 0.0, 0.0), 18),
        # The original two traces made this look like a narrow, single-blob
        # prey split.  Five later Submission #4 traces show the real policy is
        # wider, permits fragmented states and predators, and rearms after
        # roughly thirty rounds.
        58: ((0.55, 1.0, 0.30, 0.50, 0.0, 1.0), 30),
        59: ((0.80, 2.0, 0.15, 0.125, 0.0, 0.0), 17),
        63: ((0.45, 2.0, 0.20, 0.25, 0.0, 0.0), 5),
    }
    split_rule, split_cooldown_rounds = split_rules.get(team_id, ((), 0))
    direction_override_weights: tuple[tuple[float, ...], ...] = ()
    if team_id == 35:
        overrides = [[0.0] * len(FEATURE_NAMES) for _ in range(8)]
        overrides[1][FEATURE_NAMES.index("previous")] = 0.25
        overrides[1][FEATURE_NAMES.index("predator_field")] = 2.0
        overrides[3][FEATURE_NAMES.index("previous")] = 2.0
        overrides[3][FEATURE_NAMES.index("predator_field")] = 0.5
        overrides[3][FEATURE_NAMES.index("prey_field")] = 1.0
        direction_override_weights = tuple(tuple(weights) for weights in overrides)
    return ReplayProfile(
        team_id=team_id,
        direction_weights=direction_weights,
        regime_direction_weights=regime_direction_weights,
        fragmented_direction_weights=fragmented_direction_weights,
        direction_override_weights=direction_override_weights,
        split_weights=split_weights,
        split_threshold=split_threshold,
        split_rule=split_rule,
        split_cooldown_rounds=split_cooldown_rounds,
        angle_bins=angle_bins,
        angle_offset=angle_offset,
        probabilistic_angle_bins=probabilistic_angle_bins,
        angle_grid_rates=angle_grid_rates,
        source_matches=tuple(sorted({sample.match_id for sample in samples})),
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)


def evaluate_profile(profile: ReplayProfile, samples: Sequence[ReplaySample]) -> dict[str, object]:
    angles: list[float] = []
    labels: list[bool] = []
    predictions: list[bool] = []
    previous_by_trace: dict[tuple[int, int], tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))
    last_split_by_trace: dict[tuple[int, int], int] = defaultdict(lambda: -10_000)
    for sample in sorted(samples, key=lambda item: (item.match_id, item.round_number)):
        key = (sample.match_id, sample.player_id)
        previous = previous_by_trace[key]
        # Training legitimately observes the teacher's previous command, but a
        # deployed clone only has its own previous prediction.  Replace the two
        # stateful direction features here to avoid teacher-forcing leakage.
        feature_vectors = list(sample.direction_features)
        feature_vectors[FEATURE_NAMES.index("previous")] = previous
        feature_vectors[FEATURE_NAMES.index("previous_left")] = (-previous[1], previous[0])
        regime = _sample_regime(sample)
        override_weights = (
            profile.direction_override_weights[regime]
            if profile.direction_override_weights
            else ()
        )
        fragmented_weights = (
            profile.fragmented_direction_weights[regime]
            if sample.split_features[3] > 0.0625
            and profile.fragmented_direction_weights
            else ()
        )
        weights = override_weights if override_weights and any(override_weights) else (
            fragmented_weights if fragmented_weights else (
                profile.regime_direction_weights[regime]
                if profile.regime_direction_weights
                else profile.direction_weights
            )
        )
        x = sum(
            weight * vector[0]
            for weight, vector in zip(weights, feature_vectors)
        )
        y = sum(
            weight * vector[1]
            for weight, vector in zip(weights, feature_vectors)
        )
        predicted_direction = _unit(x, y)
        if profile.angle_bins > 0 and predicted_direction != (0.0, 0.0):
            step = math.tau / profile.angle_bins
            angle = math.atan2(predicted_direction[1], predicted_direction[0])
            angle = round((angle - profile.angle_offset) / step) * step + profile.angle_offset
            predicted_direction = (math.cos(angle), math.sin(angle))
        elif (
            profile.probabilistic_angle_bins > 0
            and profile.angle_grid_rates
            and stable_unit_interval(
                profile.team_id,
                sample.player_id,
                sample.round_number,
            )
            < profile.angle_grid_rates[regime]
        ):
            step = math.tau / profile.probabilistic_angle_bins
            angle = math.atan2(predicted_direction[1], predicted_direction[0])
            angle = round(angle / step) * step
            predicted_direction = (math.cos(angle), math.sin(angle))
        target = sample.target_direction
        if target != (0.0, 0.0) and predicted_direction != (0.0, 0.0):
            dot = max(-1.0, min(1.0, predicted_direction[0] * target[0] + predicted_direction[1] * target[1]))
            angles.append(math.degrees(math.acos(dot)))
        split_values = list(sample.split_features)
        split_values[-1] = 1.0 - max(
            -1.0,
            min(
                1.0,
                previous[0] * predicted_direction[0] + previous[1] * predicted_direction[1],
            ),
        )
        score = sum(weight * value for weight, value in zip(profile.split_weights, split_values))
        labels.append(sample.target_split)
        if profile.split_rule:
            (
                prey_distance_max,
                prey_radius_ratio_min,
                largest_radius_min,
                blob_count_max,
                merge_ready_fraction_min,
                predator_visible_max,
            ) = profile.split_rule
            predicted_split = (
                split_values[6] > 0.5
                and split_values[7] <= prey_distance_max
                and split_values[8] >= prey_radius_ratio_min
                and split_values[4] >= largest_radius_min
                and split_values[3] <= blob_count_max
                and split_values[5] >= merge_ready_fraction_min
                and split_values[9] <= predator_visible_max
                and sample.round_number - last_split_by_trace[key]
                >= profile.split_cooldown_rounds
            )
        else:
            predicted_split = score >= profile.split_threshold
        predictions.append(predicted_split)
        if predicted_split:
            last_split_by_trace[key] = sample.round_number
        previous_by_trace[key] = predicted_direction
    precision, recall, split_f1 = _f1(labels, predictions)
    tolerant_precision, tolerant_recall, tolerant_split_f1 = _tolerant_split_f1(
        samples, predictions
    )
    median = _percentile(angles, 0.5)
    p75 = _percentile(angles, 0.75)
    within_30 = sum(angle <= 30.0 for angle in angles) / max(len(angles), 1)
    over_90 = sum(angle > 90.0 for angle in angles) / max(len(angles), 1)
    positive_count = sum(labels)
    false_split_rate = (
        sum(predictions) / max(len(predictions), 1) if positive_count == 0 else None
    )
    direction_pass = median <= 15.0 and p75 <= 30.0 and within_30 >= 0.70 and over_90 <= 0.10
    if positive_count >= 5:
        split_pass = tolerant_precision >= 0.70 and tolerant_recall >= 0.70
    elif positive_count == 0:
        split_pass = false_split_rate is not None and false_split_rate <= 0.002
    else:
        split_pass = precision >= 0.75 and recall >= 0.75
    return {
        "samples": len(samples),
        "direction_median_error_degrees": median,
        "direction_p75_error_degrees": p75,
        "direction_within_30_rate": within_30,
        "direction_over_90_rate": over_90,
        "split_positive_count": positive_count,
        "split_precision": precision,
        "split_recall": recall,
        "split_f1": split_f1,
        "split_tolerant_precision": tolerant_precision,
        "split_tolerant_recall": tolerant_recall,
        "split_tolerant_f1": tolerant_split_f1,
        "false_split_rate": false_split_rate,
        "direction_pass": direction_pass,
        "split_pass": split_pass,
        "passed": direction_pass and split_pass,
    }


def cross_validate(team_id: int, samples: Sequence[ReplaySample]) -> dict[str, object]:
    matches = sorted({sample.match_id for sample in samples})
    fold_metrics: list[dict[str, object]] = []
    if len(matches) > 1:
        folds = [
            (
                [sample for sample in samples if sample.match_id != held_out],
                [sample for sample in samples if sample.match_id == held_out],
                str(held_out),
            )
            for held_out in matches
        ]
    else:
        ordered = sorted(samples, key=lambda sample: sample.round_number)
        boundary = max(1, int(len(ordered) * 0.8))
        folds = [(ordered[:boundary], ordered[boundary:], "last_20_percent")]
    for training, validation, held_out in folds:
        profile = _profile(team_id, training)
        metrics = evaluate_profile(profile, validation)
        metrics["held_out"] = held_out
        fold_metrics.append(metrics)
    direction_median = statistics.median(
        float(metrics["direction_median_error_degrees"]) for metrics in fold_metrics
    )
    direction_within_30 = statistics.mean(
        float(metrics["direction_within_30_rate"]) for metrics in fold_metrics
    )
    split_f1 = statistics.mean(float(metrics["split_tolerant_f1"]) for metrics in fold_metrics)
    passed = all(bool(metrics["passed"]) for metrics in fold_metrics)
    return {
        "method": "leave_one_match_out" if len(matches) > 1 else "last_20_percent",
        "generalization_proven": len(matches) > 1,
        "folds": fold_metrics,
        "direction_median_error_degrees": direction_median,
        "direction_within_30_rate": direction_within_30,
        "split_f1": split_f1,
        "passed": passed,
    }


def _profile_with_validation(
    profile: ReplayProfile, validation: dict[str, object]
) -> ReplayProfile:
    return ReplayProfile(
        team_id=profile.team_id,
        direction_weights=profile.direction_weights,
        regime_direction_weights=profile.regime_direction_weights,
        direction_override_weights=profile.direction_override_weights,
        fragmented_direction_weights=profile.fragmented_direction_weights,
        split_weights=profile.split_weights,
        split_threshold=profile.split_threshold,
        split_rule=profile.split_rule,
        split_cooldown_rounds=profile.split_cooldown_rounds,
        angle_bins=profile.angle_bins,
        angle_offset=profile.angle_offset,
        probabilistic_angle_bins=profile.probabilistic_angle_bins,
        angle_grid_rates=profile.angle_grid_rates,
        source_matches=profile.source_matches,
        direction_median_error=float(validation["direction_median_error_degrees"]),
        direction_within_30_rate=float(validation["direction_within_30_rate"]),
        split_f1=float(validation["split_f1"]),
        validation_passed=bool(validation["passed"]),
    )


def write_profiles(path: Path, profiles: Sequence[ReplayProfile]) -> None:
    lines = [
        "from __future__ import annotations",
        "",
        '"""Generated replay-imitation profiles. Run scripts/replay_imitation.py to refresh."""',
        "",
        "from strategies.replay_imitation import ReplayProfile",
        "",
        "PROFILES = {",
    ]
    for profile in profiles:
        split_threshold = (
            'float("inf")' if math.isinf(profile.split_threshold) else repr(profile.split_threshold)
        )
        lines.extend(
            [
                f"    {profile.team_id}: ReplayProfile(",
                f"        team_id={profile.team_id},",
                f"        direction_weights={profile.direction_weights!r},",
                f"        regime_direction_weights={profile.regime_direction_weights!r},",
                f"        fragmented_direction_weights={profile.fragmented_direction_weights!r},",
                f"        direction_override_weights={profile.direction_override_weights!r},",
                f"        split_weights={profile.split_weights!r},",
                f"        split_threshold={split_threshold},",
                f"        split_rule={profile.split_rule!r},",
                f"        split_cooldown_rounds={profile.split_cooldown_rounds!r},",
                f"        angle_bins={profile.angle_bins!r},",
                f"        angle_offset={profile.angle_offset!r},",
                f"        probabilistic_angle_bins={profile.probabilistic_angle_bins!r},",
                f"        angle_grid_rates={profile.angle_grid_rates!r},",
                f"        source_matches={profile.source_matches!r},",
                f"        direction_median_error={profile.direction_median_error!r},",
                f"        direction_within_30_rate={profile.direction_within_30_rate!r},",
                f"        split_f1={profile.split_f1!r},",
                f"        validation_passed={profile.validation_passed!r},",
                "    ),",
            ]
        )
    lines.append("}")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay-dir",
        type=Path,
        action="append",
        dest="replay_dirs",
        help="Replay directory; repeat to combine isolated cohorts",
    )
    parser.add_argument(
        "--profiles-out",
        type=Path,
        default=ROOT / "bots/strategies/replay_profiles.py",
    )
    parser.add_argument(
        "--team-replay-dir",
        action="append",
        default=[],
        metavar="TEAM_ID=DIR",
        help="Replace one team's training set with replays from DIR; repeat DIRs to combine",
    )
    parser.add_argument(
        "--existing-strategies-only",
        action="store_true",
        help="Generate profiles only for replay_team_<id>.py strategies already in the repository",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=ROOT / ".agario/replay-imitation/report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples_by_team: dict[int, list[ReplaySample]] = defaultdict(list)
    samples_by_path: dict[Path, list[ReplaySample]] = {}
    replay_dirs = args.replay_dirs or [ROOT / ".agario/replays/official/latest-20"]
    replay_paths = sorted(
        path
        for replay_dir in replay_dirs
        for path in replay_dir.glob("match-*-replay.json")
    )
    if not replay_paths:
        raise SystemExit(f"No replays found in {', '.join(map(str, replay_dirs))}")
    for path in replay_paths:
        path_samples = extract_samples(path)
        samples_by_path[path.resolve()] = path_samples
        for sample in path_samples:
            if sample.team_id != USER_TEAM_ID:
                samples_by_team[sample.team_id].append(sample)

    team_replay_dirs: dict[int, list[Path]] = defaultdict(list)
    for specification in args.team_replay_dir:
        try:
            raw_team_id, raw_path = specification.split("=", 1)
            team_id = int(raw_team_id)
        except (ValueError, TypeError) as exc:
            raise SystemExit(
                f"Invalid --team-replay-dir {specification!r}; expected TEAM_ID=DIR"
            ) from exc
        team_replay_dirs[team_id].append(Path(raw_path))
    for team_id, directories in team_replay_dirs.items():
        selected = [
            sample
            for directory in directories
            for path in sorted(directory.glob("match-*-replay.json"))
            for sample in samples_by_path.get(path.resolve()) or extract_samples(path)
            if sample.team_id == team_id
        ]
        if not selected:
            raise SystemExit(f"No team {team_id} samples found in {directories}")
        samples_by_team[team_id] = selected

    if args.existing_strategies_only:
        existing_team_ids = {
            int(path.stem.removeprefix("replay_team_"))
            for path in (ROOT / "bots/strategies").glob("replay_team_*.py")
            if path.stem.removeprefix("replay_team_").isdigit()
        }
        samples_by_team = defaultdict(
            list,
            {
                team_id: samples
                for team_id, samples in samples_by_team.items()
                if team_id in existing_team_ids
            },
        )

    profiles: list[ReplayProfile] = []
    report: dict[str, object] = {
        "replay_count": len(replay_paths),
        "opponent_count": len(samples_by_team),
        "user_team_id": USER_TEAM_ID,
        "replay_dirs": [str(path) for path in replay_dirs],
        "team_replay_dirs": {
            str(team_id): [str(path) for path in paths]
            for team_id, paths in sorted(team_replay_dirs.items())
        },
        "existing_strategies_only": args.existing_strategies_only,
        "gates": {
            "direction_median_error_degrees_max": 15.0,
            "direction_p75_error_degrees_max": 30.0,
            "direction_within_30_rate_min": 0.70,
            "direction_over_90_rate_max": 0.10,
            "split_precision_recall_min": 0.70,
        },
        "teams": {},
    }
    for team_id, samples in sorted(samples_by_team.items()):
        validation = cross_validate(team_id, samples)
        final_profile = _profile_with_validation(_profile(team_id, samples), validation)
        profiles.append(final_profile)
        report["teams"][str(team_id)] = {
            "matches": list(final_profile.source_matches),
            "sample_count": len(samples),
            "angle_bins": final_profile.angle_bins,
            "validation": validation,
        }
        print(
            f"team {team_id:>2}: {'PASS' if final_profile.validation_passed else 'FAIL'} "
            f"median={final_profile.direction_median_error:.1f}deg "
            f"within30={final_profile.direction_within_30_rate:.1%} "
            f"splitF1={final_profile.split_f1:.2f}"
        )

    args.profiles_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    write_profiles(args.profiles_out, profiles)
    args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    passed = sum(profile.validation_passed for profile in profiles)
    print(f"wrote {len(profiles)} profiles ({passed} passed) to {args.profiles_out}")
    print(f"wrote validation report to {args.report_out}")


if __name__ == "__main__":
    main()
