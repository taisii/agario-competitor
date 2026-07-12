from __future__ import annotations

"""Nearest-food imitation fitted from team 28's two official matches.

The recorded bot pursued the nearest visible food in every populated regime,
including predator, prey, and edible-virus views, and never requested a split.
"""

from strategies.replay_imitation import ReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam28Strategy(ReplayImitationStrategy):
    name = "replay_team_28"

    def __init__(self) -> None:
        super().__init__(PROFILES[28])
        self.name = "replay_team_28"
