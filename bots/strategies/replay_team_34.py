from __future__ import annotations

"""Single-trace best-fit imitation for official team 34.

Only match 11681 is available.  The profile follows food in clear views and
mixes inertia, wall, prey, and predator fields elsewhere, but its final 20%
time holdout fails.  This wrapper therefore preserves the failed validation
and must not be interpreted as evidence of cross-match generalisation.
"""

from strategies.replay_imitation import ProfiledReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam34Strategy(ProfiledReplayImitationStrategy):
    name = "replay_team_34"
    replay_profile = PROFILES[34]
