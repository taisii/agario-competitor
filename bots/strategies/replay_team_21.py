from __future__ import annotations

"""Direction-accurate replay imitation for official team 21.

The four source matches share a stable direction policy: nearest food when
safe and predator escape when threatened.  Split frequency is highly
match-dependent, so the aggregate profile is retained with validation marked
failed even though its autonomous direction leave-one-match-out score passes.
"""

from strategies.replay_imitation import ReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam21Strategy(ReplayImitationStrategy):
    name = "replay_team_21"

    def __init__(self) -> None:
        super().__init__(PROFILES[21])
        self.name = "replay_team_21"
