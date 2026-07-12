from __future__ import annotations

"""Replay-derived nearest-food policy for official team 75.

Team 75 appeared in matches 11697, 11698, and 11719.  Of 3,137 commands
issued with visible food, 3,119 were exactly the raw vector from the nearest
real fragment to the nearest food; every command was within 30 degrees of
that rule.  The team never requested a split in 3,142 observed turns.
"""

from strategies.replay_opponent_policies import (
    NearestFragmentFoodProfile,
    NearestFragmentFoodStrategy,
)
from strategies.replay_profiles import PROFILES


TEAM_ID = 75
PROFILE = PROFILES[TEAM_ID]


class ReplayTeam75Strategy(NearestFragmentFoodStrategy):
    """Team-75 policy: nearest-fragment food pursuit without splitting."""

    name = "replay_team_75"
    profile = NearestFragmentFoodProfile(
        source_matches=PROFILE.source_matches,
        move_reason="team75_nearest_fragment_food",
        fallback_reason="team75_inertia_fallback",
        validation_passed=PROFILE.validation_passed,
    )
