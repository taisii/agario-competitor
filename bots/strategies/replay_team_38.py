from __future__ import annotations

"""Discrete persistent random-walk imitation of official team 38.

All three traces use unit headings on a 22.5 degree grid.  Consecutive moves
hold 55.3% of the time, turn one bin about 20.9%, turn two bins about 6.7%,
and otherwise jump farther.  Split is a rare 15/1726 event while mass-eligible.
The official random seed is not recorded, so this strategy reproduces those
statistics with a deterministic local stream rather than claiming exact turns.
"""

from strategies.replay_opponent_policies import (
    DiscreteRandomWalkProfile,
    DiscreteRandomWalkStrategy,
    HeadingTransition,
)


_SPLIT_RATE = 15.0 / 1726.0
_OBSERVED_INITIAL_BINS = {3: 6, 6: 2, 7: 4}


PROFILE = DiscreteRandomWalkProfile(
    team_id=38,
    seed_salt=0x3C6EF372FE94F82B,
    split_salt=0xA54FF53A5F1D36F1,
    split_rate=_SPLIT_RATE,
    observed_initial_bins=tuple(_OBSERVED_INITIAL_BINS.items()),
    transitions=(
        HeadingTransition(0.553, None, "hold_heading"),
        HeadingTransition(0.654, 1, "turn_left_one_bin"),
        HeadingTransition(0.762, -1, "turn_right_one_bin"),
        HeadingTransition(0.797, 2, "turn_left_two_bins"),
        HeadingTransition(0.829, -2, "turn_right_two_bins"),
    ),
)


class ReplayTeam38Strategy(DiscreteRandomWalkStrategy):
    name = "replay_team_38"
    profile = PROFILE
