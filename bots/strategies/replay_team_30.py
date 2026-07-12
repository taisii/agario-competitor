from __future__ import annotations

"""Exact replay-derived policy for official team 30.

Team 30 appeared in matches 11724 and 11756.  All 1,750 commands were the
raw vector from the nearest real fragment to the nearest visible food, and no
command requested a split.
"""

import math

from strategies.base import StrategyContext, StrategyDecision
from strategies.features import extract_visible_features, normalise
from strategies.replay_profiles import PROFILES


TEAM_ID = 30
PROFILE = PROFILES[TEAM_ID]


class ReplayTeam30Strategy:
    """Team-30 policy: nearest-fragment food pursuit without splitting."""

    name = "replay_team_30"

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
                reason="team30_nearest_fragment_food",
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
            reason="team30_inertia_fallback",
            diagnostics={
                "source_matches": PROFILE.source_matches,
                "profile_validation_passed": PROFILE.validation_passed,
            },
        )
