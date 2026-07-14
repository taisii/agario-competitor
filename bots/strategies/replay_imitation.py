from __future__ import annotations

"""Shared policy used by strategies inferred from official match replays.

The official recordings contain the private world state and every submitted
move.  Profiles are fitted offline against bot-visible observations rebuilt
with the 2026.1.14 engine visibility and public-ID rules. Runtime code
deliberately stays small and standard-library-only so every inferred opponent
can be used in the normal local simulator.
"""

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from lib.config.player import EAT_SIZE_RATIO
from strategies.base import StrategyContext, StrategyDecision
from strategies.randomness import MASK_64, mix64


EPS = 1e-9
VISION_REFERENCE_SUM_OF_RADII = 12.0

FEATURE_NAMES = (
    "east",
    "north",
    "previous",
    "previous_left",
    "center",
    "wall",
    "nearest_food",
    "food_field",
    "nearest_prey",
    "prey_field",
    "nearest_predator_escape",
    "predator_field",
    "nearest_neutral_escape",
    "edible_virus",
    "virus_escape",
)

SPLIT_FEATURE_NAMES = (
    "bias",
    "round_fraction",
    "log_mass",
    "blob_count",
    "largest_radius",
    "merge_ready_fraction",
    "prey_visible",
    "prey_distance",
    "prey_radius_ratio",
    "predator_visible",
    "predator_distance",
    "predator_radius_ratio",
    "edible_virus_visible",
    "edible_virus_distance",
    "wall_distance",
    "turn_amount",
)


@dataclass(frozen=True)
class ImitationBlob:
    x: float
    y: float
    radius: float
    player_id: int = -1
    blob_id: int = -1
    merge_cooldown: int = 0


@dataclass(frozen=True)
class ImitationPoint:
    x: float
    y: float
    radius: float = 0.0


@dataclass(frozen=True)
class ImitationObservation:
    round_number: int
    max_rounds: int
    arena_size: float
    own_blobs: tuple[ImitationBlob, ...]
    visible_blobs: tuple[ImitationBlob, ...]
    visible_food: tuple[ImitationPoint, ...]
    visible_viruses: tuple[ImitationPoint, ...]


@dataclass(frozen=True, slots=True)
class ObservationAnalysis:
    center: tuple[float, float]
    total_mass: float
    largest_radius: float
    merge_ready_fraction: float
    predators: tuple[ImitationBlob, ...]
    prey: tuple[ImitationBlob, ...]
    neutral: tuple[ImitationBlob, ...]
    edible_viruses: tuple[ImitationPoint, ...]
    dangerous_viruses: tuple[ImitationPoint, ...]


@dataclass(frozen=True, slots=True)
class EntitySummary:
    nearest: ImitationBlob | ImitationPoint | None
    nearest_distance: float
    nearest_vector: tuple[float, float]
    field_vector: tuple[float, float]


@dataclass(frozen=True, slots=True)
class PredictionFeatures:
    analysis: ObservationAnalysis
    food: EntitySummary
    prey: EntitySummary
    predators: EntitySummary
    neutral: EntitySummary
    edible_viruses: EntitySummary
    dangerous_viruses: EntitySummary


@dataclass(frozen=True)
class ReplayProfile:
    team_id: int
    direction_weights: tuple[float, ...]
    split_weights: tuple[float, ...]
    split_threshold: float
    split_rule: tuple[float, ...] = ()
    split_cooldown_rounds: int = 0
    regime_direction_weights: tuple[tuple[float, ...], ...] = ()
    fragmented_direction_weights: tuple[tuple[float, ...], ...] = ()
    direction_override_weights: tuple[tuple[float, ...], ...] = ()
    angle_bins: int = 0
    angle_offset: float = 0.0
    probabilistic_angle_bins: int = 0
    angle_grid_rates: tuple[float, ...] = ()
    source_matches: tuple[int, ...] = ()
    direction_median_error: float | None = None
    direction_within_30_rate: float | None = None
    split_f1: float | None = None
    validation_passed: bool = False

    def __post_init__(self) -> None:
        if len(self.direction_weights) != len(FEATURE_NAMES):
            raise ValueError("direction_weights do not match FEATURE_NAMES")
        if len(self.split_weights) != len(SPLIT_FEATURE_NAMES):
            raise ValueError("split_weights do not match SPLIT_FEATURE_NAMES")
        if self.split_rule and len(self.split_rule) != 6:
            raise ValueError("split_rule must contain six thresholds")
        for field_name, conditional_weights in (
            ("regime_direction_weights", self.regime_direction_weights),
            ("fragmented_direction_weights", self.fragmented_direction_weights),
            ("direction_override_weights", self.direction_override_weights),
        ):
            if conditional_weights and (
                len(conditional_weights) != 8
                or any(
                    len(weights) != len(FEATURE_NAMES)
                    for weights in conditional_weights
                )
            ):
                raise ValueError(f"{field_name} must contain eight feature vectors")
        if self.angle_grid_rates and len(self.angle_grid_rates) != 8:
            raise ValueError("angle_grid_rates must contain eight regime rates")


_LAST_ANALYSIS: tuple[ImitationObservation, ObservationAnalysis] | None = None
_LAST_PREDICTION: tuple[ImitationObservation, PredictionFeatures] | None = None


def analyze_observation(observation: ImitationObservation) -> ObservationAnalysis:
    """Derive all relation features once for one immutable observation."""

    global _LAST_ANALYSIS
    if _LAST_ANALYSIS is not None and _LAST_ANALYSIS[0] is observation:
        return _LAST_ANALYSIS[1]

    own_masses = tuple(blob.radius * blob.radius for blob in observation.own_blobs)
    total_mass = sum(own_masses)
    smallest_own_mass = min(own_masses, default=math.inf)
    largest_own_mass = max(own_masses, default=0.0)
    predators: list[ImitationBlob] = []
    prey: list[ImitationBlob] = []
    neutral: list[ImitationBlob] = []
    for other in observation.visible_blobs:
        other_mass = other.radius * other.radius
        if other_mass >= smallest_own_mass * EAT_SIZE_RATIO:
            predators.append(other)
        elif largest_own_mass >= other_mass * EAT_SIZE_RATIO:
            prey.append(other)
        else:
            neutral.append(other)

    edible_viruses: list[ImitationPoint] = []
    dangerous_viruses: list[ImitationPoint] = []
    for virus in observation.visible_viruses:
        target = (
            edible_viruses
            if largest_own_mass > virus.radius * virus.radius * EAT_SIZE_RATIO
            else dangerous_viruses
        )
        target.append(virus)
    analysis = ObservationAnalysis(
        center=(
            (
                sum(blob.x * mass for blob, mass in zip(observation.own_blobs, own_masses))
                / total_mass,
                sum(blob.y * mass for blob, mass in zip(observation.own_blobs, own_masses))
                / total_mass,
            )
            if total_mass > EPS
            else (30.0, 30.0)
        ),
        total_mass=total_mass,
        largest_radius=max((blob.radius for blob in observation.own_blobs), default=0.0),
        merge_ready_fraction=(
            sum(blob.merge_cooldown <= 0 for blob in observation.own_blobs)
            / len(observation.own_blobs)
            if observation.own_blobs
            else 0.0
        ),
        predators=tuple(predators),
        prey=tuple(prey),
        neutral=tuple(neutral),
        edible_viruses=tuple(edible_viruses),
        dangerous_viruses=tuple(dangerous_viruses),
    )
    _LAST_ANALYSIS = (observation, analysis)
    return analysis


def observation_regime(observation: ImitationObservation) -> int:
    analysis = analyze_observation(observation)
    return (
        (1 if analysis.predators else 0)
        | (2 if analysis.prey else 0)
        | (4 if analysis.edible_viruses else 0)
    )


def stable_unit_interval(team_id: int, player_id: int, round_number: int) -> float:
    """Return a reproducible pseudo-random value without hidden process state."""
    value = (
        team_id * 0x9E3779B97F4A7C15
        + player_id * 0x94D049BB133111EB
        + round_number * 0xBF58476D1CE4E5B9
    ) & MASK_64
    value = mix64(value)
    return (value & ((1 << 53) - 1)) / float(1 << 53)


def _unit(vector: tuple[float, float]) -> tuple[float, float]:
    magnitude = math.hypot(*vector)
    if magnitude <= EPS or not math.isfinite(magnitude):
        return (0.0, 0.0)
    return (vector[0] / magnitude, vector[1] / magnitude)


def _mass_center(blobs: Sequence[ImitationBlob]) -> tuple[float, float]:
    mass = sum(blob.radius * blob.radius for blob in blobs)
    if mass <= EPS:
        return (30.0, 30.0)
    return (
        sum(blob.x * blob.radius * blob.radius for blob in blobs) / mass,
        sum(blob.y * blob.radius * blob.radius for blob in blobs) / mass,
    )


def _nearest_vector(
    origin: tuple[float, float],
    entities: Sequence[ImitationBlob | ImitationPoint],
    *,
    away: bool = False,
) -> tuple[float, float]:
    return _summarize_entities(origin, entities, away=away).nearest_vector


def _field_vector(
    origin: tuple[float, float],
    entities: Iterable[ImitationBlob | ImitationPoint],
    *,
    away: bool = False,
) -> tuple[float, float]:
    return _summarize_entities(origin, entities, away=away).field_vector


def _summarize_entities(
    origin: tuple[float, float],
    entities: Iterable[ImitationBlob | ImitationPoint],
    *,
    away: bool = False,
) -> EntitySummary:
    nearest: ImitationBlob | ImitationPoint | None = None
    nearest_distance_squared = math.inf
    x = 0.0
    y = 0.0
    sign = -1.0 if away else 1.0
    for entity in entities:
        dx = entity.x - origin[0]
        dy = entity.y - origin[1]
        distance_squared = dx * dx + dy * dy
        if distance_squared < nearest_distance_squared:
            nearest = entity
            nearest_distance_squared = distance_squared
        if distance_squared <= EPS:
            continue
        scale = sign / (distance_squared + 0.25)
        x += dx * scale
        y += dy * scale
    nearest_vector = (
        _unit(
            (
                sign * (nearest.x - origin[0]),
                sign * (nearest.y - origin[1]),
            )
        )
        if nearest is not None
        else (0.0, 0.0)
    )
    return EntitySummary(
        nearest=nearest,
        nearest_distance=math.sqrt(nearest_distance_squared),
        nearest_vector=nearest_vector,
        field_vector=_unit((x, y)),
    )


def _prediction_features(observation: ImitationObservation) -> PredictionFeatures:
    global _LAST_PREDICTION
    if _LAST_PREDICTION is not None and _LAST_PREDICTION[0] is observation:
        return _LAST_PREDICTION[1]

    analysis = analyze_observation(observation)
    center = analysis.center
    features = PredictionFeatures(
        analysis=analysis,
        food=_summarize_entities(center, observation.visible_food),
        prey=_summarize_entities(center, analysis.prey),
        predators=_summarize_entities(center, analysis.predators, away=True),
        neutral=_summarize_entities(center, analysis.neutral, away=True),
        edible_viruses=_summarize_entities(center, analysis.edible_viruses),
        dangerous_viruses=_summarize_entities(
            center, analysis.dangerous_viruses, away=True
        ),
    )
    _LAST_PREDICTION = (observation, features)
    return features


def _relations(
    observation: ImitationObservation,
) -> tuple[
    tuple[ImitationBlob, ...], tuple[ImitationBlob, ...], tuple[ImitationBlob, ...]
]:
    analysis = analyze_observation(observation)
    return analysis.predators, analysis.prey, analysis.neutral


def direction_feature_vectors(
    observation: ImitationObservation,
    previous_direction: tuple[float, float] = (0.0, 0.0),
) -> tuple[tuple[float, float], ...]:
    features = _prediction_features(observation)
    analysis = features.analysis
    center = analysis.center

    left = max(center[0], 0.15)
    right = max(observation.arena_size - center[0], 0.15)
    bottom = max(center[1], 0.15)
    top = max(observation.arena_size - center[1], 0.15)
    wall = _unit((1.0 / left - 1.0 / right, 1.0 / bottom - 1.0 / top))
    previous = _unit(previous_direction)

    vectors = (
        (1.0, 0.0),
        (0.0, 1.0),
        previous,
        (-previous[1], previous[0]),
        _unit(
            (
                observation.arena_size / 2.0 - center[0],
                observation.arena_size / 2.0 - center[1],
            )
        ),
        wall,
        features.food.nearest_vector,
        features.food.field_vector,
        features.prey.nearest_vector,
        features.prey.field_vector,
        features.predators.nearest_vector,
        features.predators.field_vector,
        features.neutral.nearest_vector,
        features.edible_viruses.nearest_vector,
        features.dangerous_viruses.nearest_vector,
    )
    assert len(vectors) == len(FEATURE_NAMES)
    return vectors


def split_feature_values(
    observation: ImitationObservation,
    previous_direction: tuple[float, float] = (0.0, 0.0),
    proposed_direction: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, ...]:
    features = _prediction_features(observation)
    analysis = features.analysis
    center = analysis.center
    nearest_prey = features.prey.nearest
    nearest_predator = features.predators.nearest
    largest_radius = analysis.largest_radius
    total_mass = analysis.total_mass
    nearest_virus = features.edible_viruses.nearest
    wall_distance = min(
        center[0],
        center[1],
        observation.arena_size - center[0],
        observation.arena_size - center[1],
    ) / max(observation.arena_size, EPS)
    previous = _unit(previous_direction)
    proposed = _unit(proposed_direction)
    turn_amount = 1.0 - max(
        -1.0, min(1.0, previous[0] * proposed[0] + previous[1] * proposed[1])
    )

    values = (
        1.0,
        observation.round_number / max(observation.max_rounds, 1),
        math.log1p(total_mass),
        len(observation.own_blobs) / 16.0,
        largest_radius / 10.0,
        analysis.merge_ready_fraction,
        1.0 if nearest_prey is not None else 0.0,
        (
            features.prey.nearest_distance / 20.0
            if nearest_prey is not None
            else 2.0
        ),
        (
            largest_radius / max(nearest_prey.radius, EPS)
            if nearest_prey is not None
            else 0.0
        ),
        1.0 if nearest_predator is not None else 0.0,
        (
            features.predators.nearest_distance / 20.0
            if nearest_predator is not None
            else 2.0
        ),
        (
            nearest_predator.radius / max(largest_radius, EPS)
            if nearest_predator is not None
            else 0.0
        ),
        1.0 if nearest_virus is not None else 0.0,
        (
            features.edible_viruses.nearest_distance / 20.0
            if nearest_virus is not None
            else 2.0
        ),
        wall_distance,
        turn_amount,
    )
    assert len(values) == len(SPLIT_FEATURE_NAMES)
    return values


def predict_direction(
    profile: ReplayProfile,
    observation: ImitationObservation,
    previous_direction: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    feature_vectors = direction_feature_vectors(observation, previous_direction)
    conditional_weights = profile.regime_direction_weights
    regime = observation_regime(observation)
    override_weights = (
        profile.direction_override_weights[regime]
        if profile.direction_override_weights
        else ()
    )
    fragmented_weights = (
        profile.fragmented_direction_weights[regime]
        if len(observation.own_blobs) > 1 and profile.fragmented_direction_weights
        else ()
    )
    weights = (
        override_weights
        if override_weights and any(override_weights)
        else (
            fragmented_weights
            if fragmented_weights
            else (
                conditional_weights[regime]
                if conditional_weights
                else profile.direction_weights
            )
        )
    )
    x = sum(weight * vector[0] for weight, vector in zip(weights, feature_vectors))
    y = sum(weight * vector[1] for weight, vector in zip(weights, feature_vectors))
    direction = _unit((x, y))
    if direction == (0.0, 0.0):
        food_direction = feature_vectors[FEATURE_NAMES.index("nearest_food")]
        direction = food_direction if food_direction != (0.0, 0.0) else (1.0, 0.0)
    if profile.angle_bins > 0:
        step = math.tau / profile.angle_bins
        angle = math.atan2(direction[1], direction[0])
        angle = (
            round((angle - profile.angle_offset) / step) * step + profile.angle_offset
        )
        direction = (math.cos(angle), math.sin(angle))
    elif (
        profile.probabilistic_angle_bins > 0
        and profile.angle_grid_rates
        and stable_unit_interval(
            profile.team_id,
            observation.own_blobs[0].player_id if observation.own_blobs else -1,
            observation.round_number,
        )
        < profile.angle_grid_rates[regime]
    ):
        step = math.tau / profile.probabilistic_angle_bins
        angle = math.atan2(direction[1], direction[0])
        angle = round(angle / step) * step
        direction = (math.cos(angle), math.sin(angle))
    return direction


def predict_split(
    profile: ReplayProfile,
    observation: ImitationObservation,
    previous_direction: tuple[float, float],
    proposed_direction: tuple[float, float],
    last_split_round: int = -10_000,
) -> tuple[bool, float]:
    values = split_feature_values(observation, previous_direction, proposed_direction)
    score = sum(weight * value for weight, value in zip(profile.split_weights, values))
    can_split = any(blob.radius * blob.radius >= 2.0 for blob in observation.own_blobs)
    if profile.split_rule:
        (
            prey_distance_max,
            prey_radius_ratio_min,
            largest_radius_min,
            blob_count_max,
            merge_ready_fraction_min,
            predator_visible_max,
        ) = profile.split_rule
        rule_matches = (
            values[6] > 0.5
            and values[7] <= prey_distance_max
            and values[8] >= prey_radius_ratio_min
            and values[4] >= largest_radius_min
            and values[3] <= blob_count_max
            and values[5] >= merge_ready_fraction_min
            and values[9] <= predator_visible_max
            and observation.round_number - last_split_round
            >= profile.split_cooldown_rounds
        )
        return can_split and rule_matches, score
    return can_split and score >= profile.split_threshold, score


def observation_from_context(context: StrategyContext) -> ImitationObservation:
    game = context.game
    own_blobs = tuple(
        ImitationBlob(
            x=blob.pos[0],
            y=blob.pos[1],
            radius=blob.radius,
            player_id=game.state.me.player_id,
            blob_id=blob.blob_id,
            merge_cooldown=blob.merge_cooldown,
        )
        for blob in game.state.me.blobs.values()
    )
    visible_blobs = tuple(
        ImitationBlob(
            x=blob.pos[0],
            y=blob.pos[1],
            radius=blob.radius,
            player_id=blob.player_id,
            blob_id=blob.blob_id,
            merge_cooldown=blob.merge_cooldown,
        )
        for blob in game.state.visible_blobs
    )
    visible_food = tuple(
        ImitationPoint(food.pos[0], food.pos[1])
        for food in game.state.visible_food
    )
    visible_viruses = tuple(
        ImitationPoint(virus.pos[0], virus.pos[1], virus.radius)
        for virus in game.state.visible_viruses
    )
    return ImitationObservation(
        round_number=game.state.round,
        max_rounds=game.state.max_rounds,
        arena_size=game.state.map.size,
        own_blobs=own_blobs,
        visible_blobs=visible_blobs,
        visible_food=visible_food,
        visible_viruses=visible_viruses,
    )


class ReplayImitationStrategy:
    """A stateful strategy driven by one fitted opponent profile."""

    def __init__(self, profile: ReplayProfile) -> None:
        self.profile = profile
        self.name = f"replay_team_{profile.team_id}"
        self._previous_direction = (0.0, 0.0)
        self._last_split_round = -10_000

    def choose(self, context: StrategyContext) -> StrategyDecision:
        observation = observation_from_context(context)
        direction = predict_direction(
            self.profile, observation, self._previous_direction
        )
        split, split_score = predict_split(
            self.profile,
            observation,
            self._previous_direction,
            direction,
            self._last_split_round,
        )
        if split:
            self._last_split_round = observation.round_number
        self._previous_direction = direction
        return StrategyDecision(
            direction=direction,
            split=split,
            target_kind="replay_imitation",
            target_id=str(self.profile.team_id),
            reason=f"fitted from official team {self.profile.team_id} replays",
            diagnostics={
                "source_matches": self.profile.source_matches,
                "split_score": split_score,
                "validation_passed": self.profile.validation_passed,
            },
        )


def create_profiled_replay_strategy(team_id: int) -> ReplayImitationStrategy:
    """Construct a fitted replay strategy without a team-specific wrapper."""

    from strategies.replay_profiles import PROFILES

    return ReplayImitationStrategy(PROFILES[team_id])
