from __future__ import annotations

"""Exact replay-derived policy for official team 30.

Team 30 appeared in matches 11724 and 11756.  All 1,750 commands were the
raw vector from the nearest real fragment to the nearest visible food, and no
command requested a split.
"""

from strategies.replay_opponent_policies import (
    NearestFragmentFoodProfile,
    NearestFragmentFoodStrategy,
)
from strategies.replay_profiles import PROFILES


TEAM_ID = 30
PROFILE = PROFILES[TEAM_ID]


class ReplayTeam30Strategy(NearestFragmentFoodStrategy):
    """Team-30 policy: nearest-fragment food pursuit without splitting."""

    name = "replay_team_30"
    profile = NearestFragmentFoodProfile(
        source_matches=PROFILE.source_matches,
        move_reason="team30_nearest_fragment_food",
        fallback_reason="team30_inertia_fallback",
        validation_passed=PROFILE.validation_passed,
    )
