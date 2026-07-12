from __future__ import annotations

"""Best-fit imitation of team 12's three official replay traces.

The safe regime is strongly food-driven, while predator/prey regimes mix
inertia with local target fields. Split events use a close-prey gate with a
15-round re-arm interval, raising held-out split F1 substantially. Direction
still differs between matches, so failed validation remains explicit.
"""

from strategies.replay_imitation import ReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam12Strategy(ReplayImitationStrategy):
    name = "replay_team_12"

    def __init__(self) -> None:
        super().__init__(PROFILES[12])
        self.name = "replay_team_12"
