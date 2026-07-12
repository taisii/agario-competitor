"""Greedy strategies that choose one immediate food, prey, or escape target."""

from __future__ import annotations

from strategies.base import StrategyContext, StrategyDecision
from strategies.features import (
    BlobRelation,
    extract_visible_features,
    normalise,
    vector_from_to,
)


class FoodGreedyStrategy:
    name = "food_greedy"

    def choose(self, context: StrategyContext) -> StrategyDecision:
        state = context.game.state
        own_blobs = tuple(state.me.blobs.values())
        if state.visible_food and own_blobs:
            origin_blob, target = min(
                (
                    (blob, food)
                    for blob in own_blobs
                    for food in state.visible_food
                ),
                key=lambda pair: (pair[1].pos[0] - pair[0].pos[0]) ** 2
                + (pair[1].pos[1] - pair[0].pos[1]) ** 2,
            )
            return StrategyDecision(
                direction=vector_from_to(origin_blob.pos, target.pos),
                target_kind="food",
                target_id=str(target.food_id),
                reason="nearest_food",
            )

        return StrategyDecision(direction=(1.0, 0.0), reason="fallback_east")

class SurvivalGreedyStrategy:
    name = "survival_greedy"

    def __init__(self, danger_margin: float = 3.0) -> None:
        self.danger_margin = danger_margin

    def choose(self, context: StrategyContext) -> StrategyDecision:
        features = extract_visible_features(context.game)

        escape = self._escape_vector(features.predators)
        if escape is not None:
            return StrategyDecision(
                direction=escape,
                target_kind="escape",
                target_id=self._blob_id(features.nearest_predator),
                reason="predator_near",
                diagnostics={
                    "nearest_predator_margin": features.nearest_predator.danger_margin
                    if features.nearest_predator
                    else None,
                },
            )

        if features.nearest_prey is not None:
            return StrategyDecision(
                direction=vector_from_to(
                    features.nearest_prey.nearest_own_blob.pos,
                    features.nearest_prey.blob.pos,
                ),
                target_kind="prey",
                target_id=self._blob_id(features.nearest_prey),
                reason="nearest_safe_prey",
                diagnostics={"prey_distance": features.nearest_prey.distance},
            )

        if features.nearest_food is not None:
            origin_blob = min(
                features.own_blobs,
                key=lambda blob: (
                    (blob.pos[0] - features.nearest_food.pos[0]) ** 2
                    + (blob.pos[1] - features.nearest_food.pos[1]) ** 2
                ),
            )
            return StrategyDecision(
                direction=vector_from_to(origin_blob.pos, features.nearest_food.pos),
                target_kind="food",
                target_id=str(features.nearest_food.food_id),
                reason="nearest_food",
                diagnostics={"food_distance": features.nearest_food_distance},
            )

        return StrategyDecision(direction=(1.0, 0.0), reason="fallback_east")

    def _escape_vector(
        self,
        predators: tuple[BlobRelation, ...],
    ) -> tuple[float, float] | None:
        x = 0.0
        y = 0.0
        has_danger = False
        for relation in predators:
            if relation.danger_margin > self.danger_margin:
                continue
            has_danger = True
            away_x = relation.nearest_own_blob.pos[0] - relation.blob.pos[0]
            away_y = relation.nearest_own_blob.pos[1] - relation.blob.pos[1]
            weight = 1.0 / max(relation.danger_margin + 0.25, 0.25)
            x += away_x * weight
            y += away_y * weight

        if not has_danger:
            return None
        return normalise((x, y))

    def _blob_id(self, relation: BlobRelation | None) -> str | None:
        if relation is None:
            return None
        return f"{relation.blob.player_id}:{relation.blob.blob_id}"
