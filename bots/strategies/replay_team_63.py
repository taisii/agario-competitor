from __future__ import annotations

"""Best-fit stateful imitation of team 63 across all three replay traces.

The fitted policy is strongly heading-persistent, with food-field guidance in
safe states and smaller prey, predator, virus, and wall corrections.  Its
cross-match direction and split validation remain below the acceptance gates,
which is preserved on the shared profile rather than hidden by this wrapper.
"""

from strategies.replay_imitation import ProfiledReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam63Strategy(ProfiledReplayImitationStrategy):
    name = "replay_team_63"
    replay_profile = PROFILES[63]
