from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from helper.game import Game
from lib.config.player import EAT_SIZE_RATIO
from lib.models.blob_model import BlobModel, VisibleBlobModel
from lib.models.food_model import FoodModel
from lib.models.virus_model import VirusModel
from simulation.rules import can_consume_virus as _feature_can_consume_virus


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
    nearest_virus: VirusModel | None
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


def _nearest_own_blob(
    own_blobs: tuple[BlobModel, ...],
    visible_blob: VisibleBlobModel,
) -> tuple[BlobModel, float]:
    nearest = min(
        own_blobs,
        key=lambda own_blob: squared_distance(own_blob.pos, visible_blob.pos),
    )
    return nearest, distance(nearest.pos, visible_blob.pos)


def _relation_for_blob(
    own_blobs: tuple[BlobModel, ...],
    visible_blob: VisibleBlobModel,
) -> BlobRelation:
    vulnerable_blobs = tuple(
        own_blob
        for own_blob in own_blobs
        if can_eat_player_blob(visible_blob.radius, own_blob.radius)
    )
    capable_eaters = tuple(
        own_blob
        for own_blob in own_blobs
        if can_eat_player_blob(own_blob.radius, visible_blob.radius)
    )

    # A split player can have a large blob that is safe beside a small blob the
    # same enemy can consume.  Measure threats from the nearest *vulnerable*
    # blob; using the nearest arbitrary blob can point the shared movement in
    # exactly the wrong escape direction.  For prey, use the nearest blob that
    # is actually large enough to eat it.
    relevant_own_blobs = vulnerable_blobs or capable_eaters or own_blobs
    nearest_own_blob, blob_distance = _nearest_own_blob(relevant_own_blobs, visible_blob)
    return BlobRelation(
        blob=visible_blob,
        nearest_own_blob=nearest_own_blob,
        distance=blob_distance,
        can_eat_us=bool(vulnerable_blobs),
        can_be_eaten=bool(capable_eaters),
    )


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
            nearest_virus=None,
            nearest_virus_distance=None,
        )

    relations = [
        _relation_for_blob(own_blobs, visible_blob)
        for visible_blob in game.state.visible_blobs
    ]
    predators = tuple(relation for relation in relations if relation.can_eat_us)
    # Keep the categories disjoint.  In particular, a blob that can eat one of
    # our small fragments is not "safe prey" merely because a different, large
    # fragment could eat it.
    prey = tuple(
        relation
        for relation in relations
        if relation.can_be_eaten and not relation.can_eat_us
    )
    neutral = tuple(
        relation
        for relation in relations
        if not relation.can_eat_us and not relation.can_be_eaten
    )
    nearest_food, nearest_food_distance = nearest_to_any_blob(
        own_blobs,
        game.state.visible_food,
    )
    nearest_virus, nearest_virus_distance = nearest_to_any_blob(
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
        nearest_virus=nearest_virus if isinstance(nearest_virus, VirusModel) else None,
        nearest_virus_distance=nearest_virus_distance,
    )
