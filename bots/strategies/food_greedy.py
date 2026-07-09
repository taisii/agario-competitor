from __future__ import annotations

from strategies.base import StrategyContext, StrategyDecision
from strategies.features import vector_from_to


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
