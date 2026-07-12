from __future__ import annotations

"""Replay-derived opponent strategy for official team 24.

Team 24 appears in three source matches with 3,900 observed actions.  Its
movement combines inertia with food, prey, and predator fields, while its 26
split commands vary substantially between matches.  The fitted policy is kept
with its failed validation status so callers do not mistake an approximate
opponent model for a proven reconstruction.
"""

from strategies.replay_imitation import ProfiledReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam24Strategy(ProfiledReplayImitationStrategy):
    """Team-24 fitted policy with autonomous previous-direction feedback."""

    name = "replay_team_24"

    replay_profile = PROFILES[24]
