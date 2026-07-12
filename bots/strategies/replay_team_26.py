from __future__ import annotations

"""Exact nearest-food opponent inferred from team 26's official replays.

All 2,230 recorded actions point at the nearest visible food, including rounds
with predators and prey in view.  No split was submitted.  The fitted profile
is retained here because its autonomous leave-one-match-out error is below
0.04 degrees at p75 while preserving the replay evaluator's observation rules.
"""

from strategies.replay_imitation import ProfiledReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam26Strategy(ProfiledReplayImitationStrategy):
    name = "replay_team_26"
    replay_profile = PROFILES[26]
