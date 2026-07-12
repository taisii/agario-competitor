from __future__ import annotations

"""Nearest-food imitation fitted from team 28's two official matches.

The recorded bot pursued the nearest visible food in every populated regime,
including predator, prey, and edible-virus views, and never requested a split.
"""

from strategies.replay_imitation import ProfiledReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam28Strategy(ProfiledReplayImitationStrategy):
    name = "replay_team_28"
    replay_profile = PROFILES[28]
