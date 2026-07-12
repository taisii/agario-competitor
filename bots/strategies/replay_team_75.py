from __future__ import annotations

"""Replay-derived nearest-food policy for official team 75.

Team 75 appeared in matches 11697, 11698, and 11719.  Of 3,137 commands
issued with visible food, 3,119 were exactly the raw vector from the nearest
real fragment to the nearest food; every command was within 30 degrees of
that rule.  The team never requested a split in 3,142 observed turns.
"""

import math

from strategies.base import StrategyContext, StrategyDecision
from strategies.features import extract_visible_features, normalise
from strategies.replay_profiles import PROFILES


TEAM_ID = 75
PROFILE = PROFILES[TEAM_ID]


class ReplayTeam75Strategy:
    """Team-75 policy: nearest-fragment food pursuit without splitting."""

    name = "replay_team_75"

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
                reason="team75_nearest_fragment_food",
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
            reason="team75_inertia_fallback",
            diagnostics={
                "source_matches": PROFILE.source_matches,
                "profile_validation_passed": PROFILE.validation_passed,
            },
        )
