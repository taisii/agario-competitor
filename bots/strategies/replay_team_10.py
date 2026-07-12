from __future__ import annotations

"""Replay-derived opponent strategy for official team 10.

The shared fitted policy passes the full 3,900-action direction shadow gate,
but its 18 sparse split commands do not have a stable rule across the three
source matches.  This wrapper deliberately preserves that measured profile
and its failed validation flag instead of claiming an unsupported hand rule.
"""

from strategies.replay_imitation import ReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam10Strategy(ReplayImitationStrategy):
    """Team-10 profile with autonomous previous-direction feedback."""

    name = "replay_team_10"

    def __init__(self) -> None:
        super().__init__(PROFILES[10])
        self.name = type(self).name
