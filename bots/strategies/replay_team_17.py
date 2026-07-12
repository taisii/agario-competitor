from __future__ import annotations

"""Opponent reconstructed from team 17's three official replay traces.

Across 3,810 decisions the bot never split.  Its direction is almost exactly
nearest-food pursuit when safe and the aggregate predator escape field when a
predator is visible; the fitted regime profile preserves the small wall,
inertia, and neutral-player corrections around those two dominant rules.
"""

from strategies.replay_imitation import ReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam17Strategy(ReplayImitationStrategy):
    name = "replay_team_17"

    def __init__(self) -> None:
        super().__init__(PROFILES[17])
        self.name = "replay_team_17"
