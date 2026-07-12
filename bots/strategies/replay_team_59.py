from __future__ import annotations

"""Replay-derived opponent strategy for official team 59.

Team 59 appears in five source matches. Its
movement is primarily a mixture of the previous heading and nearby food, with
prey and predator fields affecting contested views. Newer traces reveal a
close-prey split gate with a 17-round re-arm interval, improving held-out split
F1 while preserving a failed overall verdict for the inconsistent old traces.
"""

from strategies.replay_imitation import ProfiledReplayImitationStrategy
from strategies.replay_profiles import PROFILES


class ReplayTeam59Strategy(ProfiledReplayImitationStrategy):
    """Team-59 fitted policy with autonomous previous-direction feedback."""

    name = "replay_team_59"

    replay_profile = PROFILES[59]
