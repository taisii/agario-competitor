from __future__ import annotations

"""Best-fit imitation of team 77 across all three official replay traces.

Safe movement is overwhelmingly nearest-food driven and changes heading
quickly when the food target changes.  Prey and predator regimes retain more
heading inertia.  No split was observed in any of the 3,570 samples, so the
fitted profile intentionally disables splitting.  Direction generalization
still fails two of the three leave-one-match-out folds.
"""

from strategies.replay_imitation import ReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam77Strategy(ReplayImitationStrategy):
    name = "replay_team_77"

    def __init__(self) -> None:
        super().__init__(PROFILES[77])
        self.name = "replay_team_77"
