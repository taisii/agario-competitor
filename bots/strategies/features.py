from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from helper.game import Game
from lib.config.player import EAT_SIZE_RATIO
from lib.models.blob_model import BlobModel, VisibleBlobModel
from lib.models.food_model import FoodModel
from lib.models.virus_model import VirusModel


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


def nearest_by_position(
    origin: tuple[float, float],
    items: Iterable[FoodModel | VirusModel],
) -> tuple[FoodModel | VirusModel | None, float | None]:
    nearest = None
    nearest_distance = None
    for item in items:
        item_distance = distance(origin, item.pos)
        if nearest_distance is None or item_distance < nearest_distance:
            nearest = item
            nearest_distance = item_distance
    return nearest, nearest_distance


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
    nearest_own_blob, blob_distance = _nearest_own_blob(own_blobs, visible_blob)
    can_eat_us = any(
        visible_blob.radius >= own_blob.radius * EAT_SIZE_RATIO
        for own_blob in own_blobs
    )
    can_be_eaten = any(
        own_blob.radius >= visible_blob.radius * EAT_SIZE_RATIO
        for own_blob in own_blobs
    )
    return BlobRelation(
        blob=visible_blob,
        nearest_own_blob=nearest_own_blob,
        distance=blob_distance,
        can_eat_us=can_eat_us,
        can_be_eaten=can_be_eaten,
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
    prey = tuple(relation for relation in relations if relation.can_be_eaten)
    neutral = tuple(
        relation
        for relation in relations
        if not relation.can_eat_us and not relation.can_be_eaten
    )
    my_position = (game.state.me.x, game.state.me.y)
    nearest_food, nearest_food_distance = nearest_by_position(
        my_position,
        game.state.visible_food,
    )
    nearest_virus, nearest_virus_distance = nearest_by_position(
        my_position,
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
