from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from helper.game import Game
from lib.config.player import (
    BASE_PLAYER_SPEED,
    EAT_SIZE_RATIO,
    MIN_PLAYER_SPEED,
    PLAYER_SPEED_RADIUS_FACTOR,
)
from lib.models.blob_model import BlobModel, VisibleBlobModel
from lib.models.food_model import FoodModel
from lib.models.virus_model import VirusModel
from simulation.rules import (
    can_consume_virus as _feature_can_consume_virus,
    movement_speed as _feature_movement_speed,
)
@dataclass(frozen=True)
class BlobRelation:
    blob: VisibleBlobModel
    nearest_own_blob: BlobModel
    distance: float
    can_eat_us: bool
    can_be_eaten: bool

    @property
    def danger_margin(self) -> float:
        return self.distance - self.blob.radius


@dataclass(frozen=True)
class VisibleFeatures:
    own_blobs: tuple[BlobModel, ...]
    predators: tuple[BlobRelation, ...]
    prey: tuple[BlobRelation, ...]
    neutral: tuple[BlobRelation, ...]
    nearest_food: FoodModel | None
    nearest_food_distance: float | None
    nearest_predator: BlobRelation | None
    nearest_prey: BlobRelation | None
    nearest_virus_distance: float | None


def squared_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt(squared_distance(a, b))


def vector_from_to(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (b[0] - a[0], b[1] - a[1])


def vector_magnitude(vector: tuple[float, float]) -> float:
    return math.hypot(vector[0], vector[1])


def normalise(vector: tuple[float, float]) -> tuple[float, float]:
    magnitude = vector_magnitude(vector)
    if magnitude == 0.0 or not math.isfinite(magnitude):
        return (0.0, 0.0)
    return (vector[0] / magnitude, vector[1] / magnitude)


def can_eat_player_blob(
    eater_radius: float,
    target_radius: float,
    radius_margin: float = 1.0,
) -> bool:
    """Return the engine's mass-ratio eating rule in radius coordinates."""

    return eater_radius * eater_radius >= (
        target_radius
        * target_radius
        * EAT_SIZE_RATIO
        * radius_margin
        * radius_margin
    )


def player_speed(radius: float) -> float:
    return _feature_movement_speed(
        radius,
        base_speed=BASE_PLAYER_SPEED,
        radius_factor=PLAYER_SPEED_RADIUS_FACTOR,
        minimum_speed=MIN_PLAYER_SPEED,
    )


def can_consume_virus(blob_radius: float, virus_radius: float) -> bool:
    """Return the engine's strict mass-ratio rule for consuming a virus."""

    return _feature_can_consume_virus(
        blob_radius,
        virus_radius,
        eat_size_ratio=EAT_SIZE_RATIO,
    )


def virus_center_clearance(
    blob_pos: tuple[float, float],
    blob_radius: float,
    virus_pos: tuple[float, float],
) -> float:
    """Return signed clearance from a virus center to the blob boundary.

    Since agario-kit 2026.1.13, a virus is hit only when its center lies inside
    the blob.  The virus's own radius affects the mass threshold, but no longer
    expands the collision boundary.  A value at or below zero is therefore a
    geometric hit; callers must check ``can_consume_virus`` separately.
    """

    return distance(blob_pos, virus_pos) - blob_radius


def nearest_to_any_blob(
    own_blobs: tuple[BlobModel, ...],
    items: Iterable[FoodModel | VirusModel],
) -> tuple[FoodModel | VirusModel | None, float | None]:
    nearest_item = None
    nearest_distance_squared = None
    for item in items:
        for blob in own_blobs:
            distance_squared = squared_distance(blob.pos, item.pos)
            if (
                nearest_distance_squared is None
                or distance_squared < nearest_distance_squared
            ):
                nearest_item = item
                nearest_distance_squared = distance_squared
    return (
        nearest_item,
        math.sqrt(nearest_distance_squared)
        if nearest_distance_squared is not None
        else None,
    )


def _own_blobs(game: Game) -> tuple[BlobModel, ...]:
    return tuple(game.state.me.blobs.values())


def _relation_for_blob(
    own_blobs: tuple[BlobModel, ...],
    visible_blob: VisibleBlobModel,
) -> BlobRelation:
    # A split player can have a large blob that is safe beside a small blob the
    # same enemy can consume.  Measure threats from the nearest *vulnerable*
    # blob; using the nearest arbitrary blob can point the shared movement in
    # exactly the wrong escape direction.  For prey, use the nearest blob that
    # is actually large enough to eat it.
    nearest_by_kind: list[tuple[float, BlobModel] | None] = [None, None, None]
    can_eat_us = False
    can_be_eaten = False
    for own_blob in own_blobs:
        vulnerable = can_eat_player_blob(visible_blob.radius, own_blob.radius)
        capable = can_eat_player_blob(own_blob.radius, visible_blob.radius)
        can_eat_us |= vulnerable
        can_be_eaten |= capable
        kind = 0 if vulnerable else (1 if capable else 2)
        candidate = (squared_distance(own_blob.pos, visible_blob.pos), own_blob)
        current = nearest_by_kind[kind]
        if current is None or candidate[0] < current[0]:
            nearest_by_kind[kind] = candidate

    nearest = next(item for item in nearest_by_kind if item is not None)
    return BlobRelation(
        blob=visible_blob,
        nearest_own_blob=nearest[1],
        distance=math.sqrt(nearest[0]),
        can_eat_us=can_eat_us,
        can_be_eaten=can_be_eaten,
    )


def extract_predator_relations(game: Game) -> tuple[BlobRelation, ...]:
    """Extract only predator relations for cheap emergency decisions."""

    own_blobs = _own_blobs(game)
    if not own_blobs:
        return ()
    predators: list[BlobRelation] = []
    for visible_blob in game.state.visible_blobs:
        relation = _relation_for_blob(own_blobs, visible_blob)
        if relation.can_eat_us:
            predators.append(relation)
    return tuple(predators)


def extract_visible_features(game: Game) -> VisibleFeatures:
    own_blobs = _own_blobs(game)
    if not own_blobs:
        return VisibleFeatures(
            own_blobs=(),
            predators=(),
            prey=(),
            neutral=(),
            nearest_food=None,
            nearest_food_distance=None,
            nearest_predator=None,
            nearest_prey=None,
            nearest_virus_distance=None,
        )

    predators_list: list[BlobRelation] = []
    prey_list: list[BlobRelation] = []
    neutral_list: list[BlobRelation] = []
    for visible_blob in game.state.visible_blobs:
        relation = _relation_for_blob(own_blobs, visible_blob)
        if relation.can_eat_us:
            predators_list.append(relation)
        elif relation.can_be_eaten:
            prey_list.append(relation)
        else:
            neutral_list.append(relation)

    # Keep the categories disjoint.  In particular, a blob that can eat one of
    # our small fragments is not "safe prey" merely because a different, large
    # fragment could eat it.
    predators = tuple(predators_list)
    prey = tuple(prey_list)
    neutral = tuple(neutral_list)
    nearest_food, nearest_food_distance = nearest_to_any_blob(
        own_blobs,
        game.state.visible_food,
    )
    _, nearest_virus_distance = nearest_to_any_blob(
        own_blobs,
        game.state.visible_viruses,
    )

    return VisibleFeatures(
        own_blobs=own_blobs,
        predators=predators,
        prey=prey,
        neutral=neutral,
        nearest_food=nearest_food if isinstance(nearest_food, FoodModel) else None,
        nearest_food_distance=nearest_food_distance,
        nearest_predator=min(predators, key=lambda relation: relation.danger_margin)
        if predators
        else None,
        nearest_prey=min(prey, key=lambda relation: relation.distance) if prey else None,
        nearest_virus_distance=nearest_virus_distance,
    )
