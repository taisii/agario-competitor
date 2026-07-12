from __future__ import annotations

"""Behavioral reconstruction of competition team 22 from official replays."""

import math

from strategies.base import StrategyContext, StrategyDecision
from strategies.features import BlobRelation, extract_visible_features, normalise


ANGLE_STEP_DEGREES = 15.0
DANGER_SURFACE_GAP = 10.45
PREY_CHASE_DISTANCE = 3.7
COMPARISON_EPSILON = 1e-9

CENTER_WEIGHT = 4.0
PREDATOR_ESCAPE_WEIGHT = 2.0
INERTIA_WEIGHT = 1.0


class ReplayTeam22Strategy:
    """Imitate team 22's food/prey pursuit and quantised danger movement.

    Across the six available team-22 replays, ordinary moves were raw vectors
    to the nearest food (or, rarely, a nearby edible blob). Danger moves were
    unit vectors on a 15-degree grid and never requested a split.
    """

    name = "replay_team_22"

    def __init__(self) -> None:
        self.previous_direction = (1.0, 0.0)

    def choose(self, context: StrategyContext) -> StrategyDecision:
        state = context.game.state
        features = extract_visible_features(context.game)
        if not features.own_blobs:
            return self._decision(
                self.previous_direction,
                reason="dead_fallback",
            )

        dangerous = [
            relation
            for relation in features.predators
            if self._surface_gap(relation)
            <= DANGER_SURFACE_GAP + COMPARISON_EPSILON
        ]
        if dangerous:
            predator = min(dangerous, key=self._surface_gap)
            away = normalise(
                (
                    predator.nearest_own_blob.pos[0] - predator.blob.pos[0],
                    predator.nearest_own_blob.pos[1] - predator.blob.pos[1],
                )
            )
            arena_center = float(state.map.size) / 2.0
            toward_center = normalise(
                (arena_center - float(state.me.x), arena_center - float(state.me.y))
            )
            desired = (
                CENTER_WEIGHT * toward_center[0]
                + PREDATOR_ESCAPE_WEIGHT * away[0]
                + INERTIA_WEIGHT * self.previous_direction[0],
                CENTER_WEIGHT * toward_center[1]
                + PREDATOR_ESCAPE_WEIGHT * away[1]
                + INERTIA_WEIGHT * self.previous_direction[1],
            )
            if desired == (0.0, 0.0):
                desired = away if away != (0.0, 0.0) else self.previous_direction
            direction = _nearest_grid_direction(desired)
            return self._decision(
                direction,
                target_kind="escape",
                target_id=_blob_id(predator),
                reason="replay_danger_grid",
                score=_direction_score(direction, toward_center, away, self.previous_direction),
                diagnostics={
                    "surface_gap": self._surface_gap(predator),
                    "angle_step_degrees": ANGLE_STEP_DEGREES,
                },
            )

        prey = [
            relation
            for relation in features.prey
            if relation.distance <= PREY_CHASE_DISTANCE
        ]
        if prey:
            target = min(prey, key=lambda relation: relation.distance)
            direction = (
                target.blob.pos[0] - target.nearest_own_blob.pos[0],
                target.blob.pos[1] - target.nearest_own_blob.pos[1],
            )
            return self._decision(
                direction,
                target_kind="prey",
                target_id=_blob_id(target),
                reason="replay_nearby_edible_enemy",
                diagnostics={"prey_distance": target.distance},
            )

        if features.nearest_food is not None:
            target_food = features.nearest_food
            origin = min(
                features.own_blobs,
                key=lambda blob: math.dist(blob.pos, target_food.pos),
            )
            direction = (
                target_food.pos[0] - origin.pos[0],
                target_food.pos[1] - origin.pos[1],
            )
            return self._decision(
                direction,
                target_kind="food",
                target_id=str(target_food.food_id),
                reason="replay_nearest_food",
                diagnostics={
                    "food_distance": math.dist(origin.pos, target_food.pos),
                    "origin_blob_id": origin.blob_id,
                },
            )

        return self._decision(
            self.previous_direction,
            reason="replay_inertia_fallback",
        )

    @staticmethod
    def _surface_gap(relation: BlobRelation) -> float:
        return (
            relation.distance
            - relation.nearest_own_blob.radius
            - relation.blob.radius
        )

    def _decision(
        self,
        direction: tuple[float, float],
        *,
        target_kind: str = "none",
        target_id: str | None = None,
        reason: str,
        score: float | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> StrategyDecision:
        unit = normalise(direction)
        if unit != (0.0, 0.0):
            self.previous_direction = unit
        return StrategyDecision(
            direction=direction,
            split=False,
            target_kind=target_kind,
            target_id=target_id,
            reason=reason,
            score=score,
            diagnostics=diagnostics or {},
        )


def _nearest_grid_direction(direction: tuple[float, float]) -> tuple[float, float]:
    angle = math.degrees(math.atan2(direction[1], direction[0])) % 360.0
    grid_index = math.floor(angle / ANGLE_STEP_DEGREES + 0.5)
    snapped_radians = math.radians((grid_index * ANGLE_STEP_DEGREES) % 360.0)
    return (math.cos(snapped_radians), math.sin(snapped_radians))


def _direction_score(
    direction: tuple[float, float],
    center: tuple[float, float],
    away: tuple[float, float],
    inertia: tuple[float, float],
) -> float:
    return (
        CENTER_WEIGHT * _dot(direction, center)
        + PREDATOR_ESCAPE_WEIGHT * _dot(direction, away)
        + INERTIA_WEIGHT * _dot(direction, inertia)
    )


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _blob_id(relation: BlobRelation) -> str:
    return f"{relation.blob.player_id}:{relation.blob.blob_id}"
