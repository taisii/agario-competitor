from __future__ import annotations

"""Discrete persistent random-walk imitation of official team 6.

Both replay traces use exactly sixteen headings spaced by 22.5 degrees.  About
60.6% of consecutive actions keep the same heading, 15.7% turn one bin left,
14.6% turn one bin right, and the remaining 9.1% jump to another bin.  Split
was requested on 5 of 163 mass-eligible observations and never below the
engine threshold, represented here by a deterministic 3.1% eligible-state
roll.  The random stream is local and reproducible, but its hidden official
seed is unknowable, so exact shadow timing is not claimed.
"""

from strategies.replay_opponent_policies import (
    DiscreteRandomWalkProfile,
    DiscreteRandomWalkStrategy,
    HeadingTransition,
)


_SPLIT_RATE = 0.031


PROFILE = DiscreteRandomWalkProfile(
    team_id=6,
    seed_salt=0x6A09E667F3BCC909,
    split_salt=0xBB67AE8584CAA73B,
    split_rate=_SPLIT_RATE,
    parity_origin_player_id=2,
    transitions=(
        HeadingTransition(0.606, None, "hold_heading"),
        HeadingTransition(0.763, 1, "turn_left_one_bin"),
        HeadingTransition(0.909, -1, "turn_right_one_bin"),
    ),
)


class ReplayTeam6Strategy(DiscreteRandomWalkStrategy):
    name = "replay_team_6"
    profile = PROFILE
