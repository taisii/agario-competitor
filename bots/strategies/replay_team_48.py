from __future__ import annotations

"""Best-fit imitation of team 48's three official replay traces.

The profile mixes food, prey, predator, wall, and inertia fields and models a
frequent high-mass split policy.  Both direction and split timing vary enough
between source matches that validation fails; this wrapper preserves that
status explicitly.
"""

from strategies.replay_imitation import ProfiledReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam48Strategy(ProfiledReplayImitationStrategy):
    name = "replay_team_48"
    replay_profile = PROFILES[48]
