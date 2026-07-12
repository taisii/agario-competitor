from __future__ import annotations

"""Single-trace best-fit imitation for official team 34.

Only match 11681 is available.  The profile follows food in clear views and
mixes inertia, wall, prey, and predator fields elsewhere, but its final 20%
time holdout fails.  This wrapper therefore preserves the failed validation
and must not be interpreted as evidence of cross-match generalisation.
"""

from strategies.replay_imitation import ReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam34Strategy(ReplayImitationStrategy):
    name = "replay_team_34"

    def __init__(self) -> None:
        super().__init__(PROFILES[34])
        self.name = "replay_team_34"
