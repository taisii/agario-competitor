from __future__ import annotations

"""Best-fit stateful imitation of team 58's seven replay traces.

The fitted direction is dominated by previous-heading inertia with smaller
food, prey, predator, wall, and virus corrections. Its stable probabilistic
16-heading quantization is reproduced from regime-specific replay rates.
Cross-match validation remains below the strict exact-command gate.
"""

from strategies.replay_imitation import ReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam58Strategy(ReplayImitationStrategy):
    name = "replay_team_58"

    def __init__(self) -> None:
        super().__init__(PROFILES[58])
        self.name = "replay_team_58"
