from __future__ import annotations

"""Best-fit imitation of team 12's three official replay traces.

The safe regime is strongly food-driven, while predator/prey regimes mix
inertia with local target fields. Split events use a close-prey gate with a
15-round re-arm interval, raising held-out split F1 substantially. Direction
still differs between matches, so failed validation remains explicit.
"""

from strategies.replay_imitation import ProfiledReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam12Strategy(ProfiledReplayImitationStrategy):
    name = "replay_team_12"
    replay_profile = PROFILES[12]
