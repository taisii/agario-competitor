from __future__ import annotations

"""Best-fit imitation of team 3's two heterogeneous replay traces.

The two official matches differ materially in both movement and split rate, so
this strategy intentionally exposes the fitted aggregate profile rather than
claiming one exact hand-written rule.  Shadow validation remains marked failed
on the profile and is reported as such by the surrounding evaluation tooling.
"""

from strategies.replay_imitation import ReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam3Strategy(ReplayImitationStrategy):
    name = "replay_team_3"

    def __init__(self) -> None:
        super().__init__(PROFILES[3])
        self.name = "replay_team_3"
