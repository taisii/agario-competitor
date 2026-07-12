from __future__ import annotations

"""Replay-derived nearest-food policy for official team 27.

Across matches 11716, 11719, and 11752, team 27 issued 3,089 moves and
never split.  For 3,012 of the 3,084 turns with visible food, its raw command
was exactly the vector from the nearest real fragment to the nearest food.
The same rule is within 30 degrees of 99.29% of recorded non-zero headings.
"""

from strategies.replay_opponent_policies import (
    NearestFragmentFoodProfile,
    NearestFragmentFoodStrategy,
)
from strategies.replay_profiles import PROFILES


TEAM_ID = 27
PROFILE = PROFILES[TEAM_ID]


class ReplayTeam27Strategy(NearestFragmentFoodStrategy):
    """Team-27 policy: pursue the food nearest to any actual fragment."""

    name = "replay_team_27"
    profile = NearestFragmentFoodProfile(
        source_matches=PROFILE.source_matches,
        move_reason="team27_nearest_fragment_food",
        fallback_reason="team27_inertia_fallback",
        validation_passed=PROFILE.validation_passed,
    )
