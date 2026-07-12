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

import math

from strategies.base import StrategyContext, StrategyDecision


_MASK_64 = (1 << 64) - 1
_GOLDEN_RATIO_64 = 0x9E3779B97F4A7C15
_HEADING_BINS = 16
_SPLIT_RATE = 0.031


class ReplayTeam6Strategy:
    name = "replay_team_6"

    def __init__(self) -> None:
        self._heading_bin: int | None = None
        self._direction_rng = 0
        self._split_rng = 0

    def choose(self, context: StrategyContext) -> StrategyDecision:
        state = context.game.state
        player_id = int(getattr(state.me, "player_id", 0))
        if self._heading_bin is None:
            # Matches both observed initial headings: player 2 -> east and
            # player 7 -> west.  Other slots receive the same parity pattern.
            self._heading_bin = ((player_id - 2) * 8) % _HEADING_BINS
            seed = self._mix64(0x6A09E667F3BCC909 ^ player_id)
            self._direction_rng = seed
            self._split_rng = seed ^ 0xBB67AE8584CAA73B
            reason = "initial_discrete_heading"
        else:
            reason = self._advance_heading()

        can_split = any(
            blob.radius * blob.radius >= 2.0
            for blob in state.me.blobs.values()
        )
        split_roll = self._split_fraction() if can_split else None
        split = bool(split_roll is not None and split_roll < _SPLIT_RATE)
        angle = self._heading_bin * math.tau / _HEADING_BINS
        direction = (math.cos(angle), math.sin(angle))
        return StrategyDecision(
            direction=direction,
            split=split,
            target_kind="discrete_random_walk",
            target_id="6",
            reason=reason,
            diagnostics={
                "source_team_id": 6,
                "heading_bin": self._heading_bin,
                "heading_bins": _HEADING_BINS,
                "split_roll": split_roll,
                "split_rate_when_eligible": _SPLIT_RATE,
            },
        )

    def _advance_heading(self) -> str:
        assert self._heading_bin is not None
        roll = self._direction_fraction()
        if roll < 0.606:
            return "hold_heading"
        if roll < 0.763:
            self._heading_bin = (self._heading_bin + 1) % _HEADING_BINS
            return "turn_left_one_bin"
        if roll < 0.909:
            self._heading_bin = (self._heading_bin - 1) % _HEADING_BINS
            return "turn_right_one_bin"
        self._heading_bin = int(self._direction_fraction() * _HEADING_BINS) % _HEADING_BINS
        return "jump_heading"

    def _direction_fraction(self) -> float:
        self._direction_rng = (
            self._direction_rng + _GOLDEN_RATIO_64
        ) & _MASK_64
        return self._mix64(self._direction_rng) / float(1 << 64)

    def _split_fraction(self) -> float:
        self._split_rng = (self._split_rng + _GOLDEN_RATIO_64) & _MASK_64
        return self._mix64(self._split_rng) / float(1 << 64)

    @staticmethod
    def _mix64(value: int) -> int:
        value ^= value >> 30
        value = (value * 0xBF58476D1CE4E5B9) & _MASK_64
        value ^= value >> 27
        value = (value * 0x94D049BB133111EB) & _MASK_64
        return (value ^ (value >> 31)) & _MASK_64
