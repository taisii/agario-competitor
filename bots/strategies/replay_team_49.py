from __future__ import annotations

"""Stateful imitation of team 49's five official replay traces.

The direction profile has clear safe-food, prey-chase, and predator-escape
regimes. Its split command is an edge-triggered close-prey rule with an
18-round re-arm interval; modelling that internal cooldown raises held-out
split F1 from 0.22 to 0.74. Direction still narrowly misses the strict gate,
even after team-specific ridge tuning improves its median from 13.89° to
13.52° and its within-30° rate from 69.43% to 69.84%, so the profile correctly
remains marked as failed overall.
"""

from strategies.replay_imitation import ProfiledReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam49Strategy(ProfiledReplayImitationStrategy):
    name = "replay_team_49"
    replay_profile = PROFILES[49]
