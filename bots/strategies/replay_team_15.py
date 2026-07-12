from __future__ import annotations

"""Replay-derived opponent strategy for official team 15.

Team 15 has five source matches and 6,909 observed actions.  Its movement is
strongly inertial but the target mixture and 99 split commands do not
generalize between matches.  This wrapper keeps the measured autonomous
profile and its failed validation status instead of inventing a false rule.
"""

from strategies.replay_imitation import ReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam15Strategy(ReplayImitationStrategy):
    """Team-15 fitted policy with autonomous previous-direction feedback."""

    name = "replay_team_15"

    def __init__(self) -> None:
        super().__init__(PROFILES[15])
        self.name = type(self).name
