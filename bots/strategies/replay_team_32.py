from __future__ import annotations

"""Best-fit imitation of team 32's three official replay traces.

The fitted direction mixes food, prey, center, wall, inertia, and predator
escape terms.  Neither direction nor split timing transfers cleanly between
the three matches, so this wrapper intentionally retains the profile's failed
validation status rather than presenting it as an exact clone.
"""

from strategies.replay_imitation import ProfiledReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam32Strategy(ProfiledReplayImitationStrategy):
    name = "replay_team_32"
    replay_profile = PROFILES[32]
