from __future__ import annotations

"""Stateful imitation of team 49's five official replay traces.

The direction profile has clear safe-food, prey-chase, and predator-escape
regimes. Its split command is an edge-triggered close-prey rule with an
18-round re-arm interval; modelling that internal cooldown raises held-out
split F1 from 0.22 to 0.74. Direction still narrowly misses the strict gate,
so the profile correctly remains marked as failed overall.
"""

from strategies.replay_imitation import ReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam49Strategy(ReplayImitationStrategy):
    name = "replay_team_49"

    def __init__(self) -> None:
        super().__init__(PROFILES[49])
        self.name = "replay_team_49"
