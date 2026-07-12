from __future__ import annotations

"""Replay-derived nearest-food strategy for official team 2.

Across 2,281 observed turns in matches 11679 and 11724, team 2 never split.
Choosing the visible food nearest to any real fragment reproduced 99.65% of
recorded headings within 30 degrees and had zero median angular error.
"""

import math

from strategies.base import StrategyContext, StrategyDecision
from strategies.features import extract_visible_features, normalise


class ReplayTeam2Strategy:
    """Team-2 policy: always steer the nearest fragment to nearest food."""

    name = "replay_team_2"

    def __init__(self) -> None:
        self.previous_direction = (1.0, 0.0)

    def choose(self, context: StrategyContext) -> StrategyDecision:
        features = extract_visible_features(context.game)
        if features.own_blobs and features.nearest_food is not None:
            food = features.nearest_food
            origin = min(
                features.own_blobs,
                key=lambda blob: math.dist(blob.pos, food.pos),
            )
            direction = (
                food.pos[0] - origin.pos[0],
                food.pos[1] - origin.pos[1],
            )
            unit = normalise(direction)
            if unit != (0.0, 0.0):
                self.previous_direction = unit
            return StrategyDecision(
                direction=direction,
                split=False,
                target_kind="food",
                target_id=str(food.food_id),
                reason="team2_nearest_food",
                diagnostics={
                    "origin_blob_id": origin.blob_id,
                    "food_distance": math.dist(origin.pos, food.pos),
                    "source_matches": (11679, 11724),
                },
            )

        return StrategyDecision(
            direction=self.previous_direction,
            split=False,
            reason="team2_inertia_fallback",
            diagnostics={"source_matches": (11679, 11724)},
        )
