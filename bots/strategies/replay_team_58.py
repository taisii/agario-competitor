from __future__ import annotations

"""Best-fit stateful imitation of team 58's two replay traces.

The fitted direction is dominated by previous-heading inertia with smaller
food, prey, predator, wall, and virus corrections.  Cross-match validation is
poor for both direction and sparse split timing, which remains explicit on
the profile used by this dedicated wrapper.
"""

from strategies.replay_imitation import ReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam58Strategy(ReplayImitationStrategy):
    name = "replay_team_58"

    def __init__(self) -> None:
        super().__init__(PROFILES[58])
        self.name = "replay_team_58"
