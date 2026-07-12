from __future__ import annotations

"""Replay-derived opponent strategy for official team 59.

Team 59 appears in three source matches with 3,900 observed actions.  Its
movement is primarily a mixture of the previous heading and nearby food, with
prey and predator fields affecting contested views.  All 19 observed split
commands occur in one match, so the fitted policy retains its failed
cross-match validation status.
"""

from strategies.replay_imitation import ReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam59Strategy(ReplayImitationStrategy):
    """Team-59 fitted policy with autonomous previous-direction feedback."""

    name = "replay_team_59"

    def __init__(self) -> None:
        super().__init__(PROFILES[59])
        self.name = type(self).name
