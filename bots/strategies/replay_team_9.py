from __future__ import annotations

"""Stateful replay-derived opponent policy for official team 9.

The earlier wrapper used a synthetic hash to reproduce only the observed split
frequency. Applying that hash to the source traces produced split F1 0.065.
The fitted profile now uses the replay-supported close-prey condition and a
90-round re-arm interval, raising held-out split reproduction while keeping
runtime and validation behavior identical.
"""

from strategies.replay_imitation import ReplayImitationStrategy
from strategies.replay_profiles import PROFILES

PROFILE = PROFILES[9]


class ReplayTeam9Strategy(ReplayImitationStrategy):
    """Team-9 field movement with a deterministic sparse-prey split gate."""

    name = "replay_team_9"

    def __init__(self) -> None:
        super().__init__(PROFILE)
        self.name = type(self).name
