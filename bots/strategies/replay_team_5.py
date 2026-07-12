from __future__ import annotations

"""Best-fit stateful imitation of team 5's two official match traces.

Team 5 primarily preserves its previous direction and applies smaller food,
prey, wall, and predator-field corrections.  Split timing differs sharply
between the two matches, so this wrapper keeps the fitted aggregate profile
and deliberately preserves its failed validation status.
"""

from strategies.replay_imitation import ProfiledReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam5Strategy(ProfiledReplayImitationStrategy):
    name = "replay_team_5"
    replay_profile = PROFILES[5]
