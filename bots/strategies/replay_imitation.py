from __future__ import annotations

"""Shared policy used by strategies inferred from official match replays.

The official recordings contain the private world state and every submitted
move.  Profiles are fitted offline against bot-visible observations rebuilt
with the 2026.1.13 engine visibility rules.  Runtime code deliberately stays
small and standard-library-only so every inferred opponent can be used in the
normal local simulator.
"""

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from strategies.base import StrategyContext, StrategyDecision


EPS = 1e-9
EAT_SIZE_RATIO = 1.2
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
    team_id: int = -1
    blob_id: int = -1
    merge_cooldown: int = 0


@dataclass(frozen=True)
class ImitationPoint:
    x: float
    y: float
    radius: float = 0.0
    entity_id: int = -1


@dataclass(frozen=True)
class ImitationObservation:
    round_number: int
    max_rounds: int
    arena_size: float
    own_blobs: tuple[ImitationBlob, ...]
    visible_blobs: tuple[ImitationBlob, ...]
    visible_food: tuple[ImitationPoint, ...]
    visible_viruses: tuple[ImitationPoint, ...]


@dataclass(frozen=True)
class ReplayProfile:
    team_id: int
    direction_weights: tuple[float, ...]
    split_weights: tuple[float, ...]
    split_threshold: float
    regime_direction_weights: tuple[tuple[float, ...], ...] = ()
    angle_bins: int = 0
    angle_offset: float = 0.0
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
        if self.regime_direction_weights and (
            len(self.regime_direction_weights) != 8
            or any(len(weights) != len(FEATURE_NAMES) for weights in self.regime_direction_weights)
        ):
            raise ValueError("regime_direction_weights must contain eight feature vectors")


def observation_regime(observation: ImitationObservation) -> int:
    predators, prey, _ = _relations(observation)
    edible_virus = any(
        own.radius * own.radius > virus.radius * virus.radius * EAT_SIZE_RATIO
        for own in observation.own_blobs
        for virus in observation.visible_viruses
    )
    return (1 if predators else 0) | (2 if prey else 0) | (4 if edible_virus else 0)


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
    if not entities:
        return (0.0, 0.0)
    entity = min(entities, key=lambda item: math.hypot(item.x - origin[0], item.y - origin[1]))
    sign = -1.0 if away else 1.0
    return _unit((sign * (entity.x - origin[0]), sign * (entity.y - origin[1])))


def _field_vector(
    origin: tuple[float, float],
    entities: Iterable[ImitationBlob | ImitationPoint],
    *,
    away: bool = False,
) -> tuple[float, float]:
    x = 0.0
    y = 0.0
    sign = -1.0 if away else 1.0
    for entity in entities:
        dx = entity.x - origin[0]
        dy = entity.y - origin[1]
        distance_squared = dx * dx + dy * dy
        if distance_squared <= EPS:
            continue
        scale = sign / (distance_squared + 0.25)
        x += dx * scale
        y += dy * scale
    return _unit((x, y))


def _relations(
    observation: ImitationObservation,
) -> tuple[list[ImitationBlob], list[ImitationBlob], list[ImitationBlob]]:
    predators: list[ImitationBlob] = []
    prey: list[ImitationBlob] = []
    neutral: list[ImitationBlob] = []
    for other in observation.visible_blobs:
        can_eat_us = any(
            other.radius * other.radius
            >= own.radius * own.radius * EAT_SIZE_RATIO
            for own in observation.own_blobs
        )
        can_be_eaten = any(
            own.radius * own.radius
            >= other.radius * other.radius * EAT_SIZE_RATIO
            for own in observation.own_blobs
        )
        if can_eat_us:
            predators.append(other)
        elif can_be_eaten:
            prey.append(other)
        else:
            neutral.append(other)
    return predators, prey, neutral


def direction_feature_vectors(
    observation: ImitationObservation,
    previous_direction: tuple[float, float] = (0.0, 0.0),
) -> tuple[tuple[float, float], ...]:
    center = _mass_center(observation.own_blobs)
    predators, prey, neutral = _relations(observation)
    edible_viruses = [
        virus
        for virus in observation.visible_viruses
        if any(
            own.radius * own.radius
            > virus.radius * virus.radius * EAT_SIZE_RATIO
            for own in observation.own_blobs
        )
    ]
    dangerous_viruses = [
        virus for virus in observation.visible_viruses if virus not in edible_viruses
    ]

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
        _unit((observation.arena_size / 2.0 - center[0], observation.arena_size / 2.0 - center[1])),
        wall,
        _nearest_vector(center, observation.visible_food),
        _field_vector(center, observation.visible_food),
        _nearest_vector(center, prey),
        _field_vector(center, prey),
        _nearest_vector(center, predators, away=True),
        _field_vector(center, predators, away=True),
        _nearest_vector(center, neutral, away=True),
        _nearest_vector(center, edible_viruses),
        _nearest_vector(center, dangerous_viruses, away=True),
    )
    assert len(vectors) == len(FEATURE_NAMES)
    return vectors


def split_feature_values(
    observation: ImitationObservation,
    previous_direction: tuple[float, float] = (0.0, 0.0),
    proposed_direction: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, ...]:
    center = _mass_center(observation.own_blobs)
    predators, prey, _ = _relations(observation)
    nearest_prey = min(
        prey,
        key=lambda item: math.hypot(item.x - center[0], item.y - center[1]),
        default=None,
    )
    nearest_predator = min(
        predators,
        key=lambda item: math.hypot(item.x - center[0], item.y - center[1]),
        default=None,
    )
    largest_radius = max((blob.radius for blob in observation.own_blobs), default=0.0)
    total_mass = sum(blob.radius * blob.radius for blob in observation.own_blobs)
    edible_viruses = [
        virus
        for virus in observation.visible_viruses
        if any(
            own.radius * own.radius
            > virus.radius * virus.radius * EAT_SIZE_RATIO
            for own in observation.own_blobs
        )
    ]
    nearest_virus = min(
        edible_viruses,
        key=lambda item: math.hypot(item.x - center[0], item.y - center[1]),
        default=None,
    )
    wall_distance = min(
        center[0], center[1], observation.arena_size - center[0], observation.arena_size - center[1]
    ) / max(observation.arena_size, EPS)
    previous = _unit(previous_direction)
    proposed = _unit(proposed_direction)
    turn_amount = 1.0 - max(-1.0, min(1.0, previous[0] * proposed[0] + previous[1] * proposed[1]))

    values = (
        1.0,
        observation.round_number / max(observation.max_rounds, 1),
        math.log1p(total_mass),
        len(observation.own_blobs) / 16.0,
        largest_radius / 10.0,
        (
            sum(blob.merge_cooldown <= 0 for blob in observation.own_blobs)
            / max(len(observation.own_blobs), 1)
        ),
        1.0 if nearest_prey is not None else 0.0,
        (
            math.hypot(nearest_prey.x - center[0], nearest_prey.y - center[1]) / 20.0
            if nearest_prey is not None
            else 2.0
        ),
        (largest_radius / max(nearest_prey.radius, EPS) if nearest_prey is not None else 0.0),
        1.0 if nearest_predator is not None else 0.0,
        (
            math.hypot(nearest_predator.x - center[0], nearest_predator.y - center[1]) / 20.0
            if nearest_predator is not None
            else 2.0
        ),
        (nearest_predator.radius / max(largest_radius, EPS) if nearest_predator is not None else 0.0),
        1.0 if nearest_virus is not None else 0.0,
        (
            math.hypot(nearest_virus.x - center[0], nearest_virus.y - center[1]) / 20.0
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
    weights = (
        profile.regime_direction_weights[observation_regime(observation)]
        if profile.regime_direction_weights
        else profile.direction_weights
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
        angle = round((angle - profile.angle_offset) / step) * step + profile.angle_offset
        direction = (math.cos(angle), math.sin(angle))
    return direction


def predict_split(
    profile: ReplayProfile,
    observation: ImitationObservation,
    previous_direction: tuple[float, float],
    proposed_direction: tuple[float, float],
) -> tuple[bool, float]:
    values = split_feature_values(observation, previous_direction, proposed_direction)
    score = sum(weight * value for weight, value in zip(profile.split_weights, values))
    can_split = any(blob.radius * blob.radius >= 2.0 for blob in observation.own_blobs)
    return can_split and score >= profile.split_threshold, score


def observation_from_context(context: StrategyContext) -> ImitationObservation:
    game = context.game
    own_blobs = tuple(
        ImitationBlob(
            x=blob.pos[0],
            y=blob.pos[1],
            radius=blob.radius,
            player_id=game.state.me.player_id,
            team_id=getattr(game.state.me, "team_id", -1),
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
            team_id=blob.team_id,
            blob_id=blob.blob_id,
            merge_cooldown=blob.merge_cooldown,
        )
        for blob in game.state.visible_blobs
    )
    visible_food = tuple(
        ImitationPoint(food.pos[0], food.pos[1], entity_id=food.food_id)
        for food in game.state.visible_food
    )
    visible_viruses = tuple(
        ImitationPoint(virus.pos[0], virus.pos[1], virus.radius, virus.virus_id)
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

    def choose(self, context: StrategyContext) -> StrategyDecision:
        observation = observation_from_context(context)
        direction = predict_direction(self.profile, observation, self._previous_direction)
        split, split_score = predict_split(
            self.profile,
            observation,
            self._previous_direction,
            direction,
        )
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
