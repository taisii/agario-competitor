from __future__ import annotations

"""Stateful imitation of team 3's five heterogeneous replay traces.

Movement still differs materially between matches. Newer traces reveal a
large-close-prey split gate with an 18-round re-arm interval, which improves
held-out split reproduction without claiming an overall exact copy. Shadow
validation remains marked failed on the profile and is reported as such.
"""

from strategies.replay_imitation import ProfiledReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam3Strategy(ProfiledReplayImitationStrategy):
    name = "replay_team_3"
    replay_profile = PROFILES[3]
