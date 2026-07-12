from __future__ import annotations

"""Replay-derived nearest-food policy for official team 27.

Across matches 11716, 11719, and 11752, team 27 issued 3,089 moves and
never split.  For 3,012 of the 3,084 turns with visible food, its raw command
was exactly the vector from the nearest real fragment to the nearest food.
The same rule is within 30 degrees of 99.29% of recorded non-zero headings.
"""

import math

from strategies.base import StrategyContext, StrategyDecision
from strategies.features import extract_visible_features, normalise
from strategies.replay_profiles import PROFILES


TEAM_ID = 27
PROFILE = PROFILES[TEAM_ID]


class ReplayTeam27Strategy:
    """Team-27 policy: pursue the food nearest to any actual fragment."""

    name = "replay_team_27"

    def __init__(self) -> None:
        self._previous_direction = (1.0, 0.0)

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
                self._previous_direction = unit
            return StrategyDecision(
                direction=direction,
                split=False,
                target_kind="food",
                target_id=str(food.food_id),
                reason="team27_nearest_fragment_food",
                diagnostics={
                    "origin_blob_id": origin.blob_id,
                    "food_distance": math.dist(origin.pos, food.pos),
                    "source_matches": PROFILE.source_matches,
                    "profile_validation_passed": PROFILE.validation_passed,
                },
            )

        return StrategyDecision(
            direction=self._previous_direction,
            split=False,
            reason="team27_inertia_fallback",
            diagnostics={
                "source_matches": PROFILE.source_matches,
                "profile_validation_passed": PROFILE.validation_passed,
            },
        )
