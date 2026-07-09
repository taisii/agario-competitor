from __future__ import annotations

from strategies.base import StrategyContext, StrategyDecision
from strategies.features import vector_from_to


class FoodGreedyStrategy:
    name = "food_greedy"

    def choose(self, context: StrategyContext) -> StrategyDecision:
        state = context.game.state
        origin = (state.me.x, state.me.y)
        if state.visible_food:
            target = min(
                state.visible_food,
                key=lambda food: (food.pos[0] - origin[0]) ** 2
                + (food.pos[1] - origin[1]) ** 2,
            )
            return StrategyDecision(
                direction=vector_from_to(origin, target.pos),
                target_kind="food",
                target_id=str(target.food_id),
                reason="nearest_food",
            )

        return StrategyDecision(direction=(1.0, 0.0), reason="fallback_east")
