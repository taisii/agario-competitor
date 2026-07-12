from __future__ import annotations

"""Best-fit imitation of team 49's four official replay traces.

The direction profile has clear safe-food, prey-chase, and predator-escape
regimes.  Split timing varies from 2 to 25 events per match and does not meet
the replay gate, so the dedicated wrapper preserves failed validation.
"""

from strategies.replay_imitation import ReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam49Strategy(ReplayImitationStrategy):
    name = "replay_team_49"

    def __init__(self) -> None:
        super().__init__(PROFILES[49])
        self.name = "replay_team_49"
