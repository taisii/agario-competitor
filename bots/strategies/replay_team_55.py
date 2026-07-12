from __future__ import annotations

"""Replay-derived opponent strategy for official team 55.

Team 55 appears in two source matches with 2,590 observed actions.  Every
recorded command lies on a 24-heading grid and all split flags are false.  Its
heading changes are highly stochastic between matches, so the fitted movement
mixture is retained with its failed validation status rather than presented as
an exact reconstruction.
"""

from strategies.replay_imitation import ReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam55Strategy(ReplayImitationStrategy):
    """Team-55 fitted policy, quantized to its observed 15-degree headings."""

    name = "replay_team_55"

    def __init__(self) -> None:
        super().__init__(PROFILES[55])
        self.name = type(self).name
