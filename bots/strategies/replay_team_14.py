from __future__ import annotations

"""Replay-derived opponent strategy for official team 14.

The four source matches show stable prey-chase and predator-escape regimes,
but safe movement and the 30 sparse split commands do not generalize across
matches.  This wrapper preserves the measured autonomous profile and its
failed validation flag rather than replacing it with an unsupported rule.
"""

from strategies.replay_imitation import ReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam14Strategy(ReplayImitationStrategy):
    """Team-14 fitted policy with autonomous previous-direction feedback."""

    name = "replay_team_14"

    def __init__(self) -> None:
        super().__init__(PROFILES[14])
        self.name = type(self).name
