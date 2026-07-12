from __future__ import annotations

"""Replay-derived opponent strategy for official team 14.

Seven source matches show prey-chase and predator-escape regimes. Split events
use a close-prey gate with a 15-round re-arm interval; this substantially
improves held-out event reproduction while the direction policy remains below
the strict cross-match gate.
"""

from strategies.replay_imitation import ReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam14Strategy(ReplayImitationStrategy):
    """Team-14 fitted policy with autonomous previous-direction feedback."""

    name = "replay_team_14"

    def __init__(self) -> None:
        super().__init__(PROFILES[14])
        self.name = type(self).name
