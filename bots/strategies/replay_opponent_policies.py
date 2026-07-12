from __future__ import annotations

"""Reusable implementations for replay-derived opponent families."""

import math
from dataclasses import dataclass

from strategies.base import StrategyContext, StrategyDecision
from strategies.features import extract_visible_features, normalise


@dataclass(frozen=True)
class NearestFragmentFoodProfile:
    source_matches: tuple[int, ...]
    move_reason: str
    fallback_reason: str
    validation_passed: bool | None = None


class NearestFragmentFoodStrategy:
    """Pursue the food nearest to any real fragment and never split."""

    profile: NearestFragmentFoodProfile

    def __init__(self) -> None:
        self._previous_direction = (1.0, 0.0)

    def choose(self, context: StrategyContext) -> StrategyDecision:
        features = extract_visible_features(context.game)
        profile = self.profile
        diagnostics: dict[str, object]
        if features.own_blobs and features.nearest_food is not None:
            food = features.nearest_food
            origin = min(
                features.own_blobs,
                key=lambda blob: math.dist(blob.pos, food.pos),
            )
            direction = (
                food.pos[0] - origin.pos[0],
                food.pos[1] - origin.pos[1],
            )
            unit = normalise(direction)
            if unit != (0.0, 0.0):
                self._previous_direction = unit
            diagnostics = {
                "origin_blob_id": origin.blob_id,
                "food_distance": math.dist(origin.pos, food.pos),
                "source_matches": profile.source_matches,
            }
            if profile.validation_passed is not None:
                diagnostics["profile_validation_passed"] = profile.validation_passed
            return StrategyDecision(
                direction=direction,
                split=False,
                target_kind="food",
                target_id=str(food.food_id),
                reason=profile.move_reason,
                diagnostics=diagnostics,
            )

        diagnostics = {"source_matches": profile.source_matches}
        if profile.validation_passed is not None:
            diagnostics["profile_validation_passed"] = profile.validation_passed
        return StrategyDecision(
            direction=self._previous_direction,
            split=False,
            reason=profile.fallback_reason,
            diagnostics=diagnostics,
        )


@dataclass(frozen=True)
class HeadingTransition:
    upper_bound: float
    offset: int | None
    reason: str


@dataclass(frozen=True)
class DiscreteRandomWalkProfile:
    team_id: int
    seed_salt: int
    split_salt: int
    split_rate: float
    transitions: tuple[HeadingTransition, ...]
    observed_initial_bins: tuple[tuple[int, int], ...] = ()
    parity_origin_player_id: int | None = None
    heading_bins: int = 16


class DiscreteRandomWalkStrategy:
    """Deterministic local stream matching a team's random-walk statistics."""

    profile: DiscreteRandomWalkProfile
    _MASK_64 = (1 << 64) - 1
    _GOLDEN_RATIO_64 = 0x9E3779B97F4A7C15

    def __init__(self) -> None:
        self._heading_bin: int | None = None
        self._direction_rng = 0
        self._split_rng = 0

    def choose(self, context: StrategyContext) -> StrategyDecision:
        state = context.game.state
        player_id = int(getattr(state.me, "player_id", 0))
        profile = self.profile
        if self._heading_bin is None:
            seed = self._mix64(profile.seed_salt ^ player_id)
            self._direction_rng = seed
            self._split_rng = seed ^ profile.split_salt
            observed = dict(profile.observed_initial_bins)
            if profile.parity_origin_player_id is not None:
                self._heading_bin = (
                    (player_id - profile.parity_origin_player_id)
                    * (profile.heading_bins // 2)
                ) % profile.heading_bins
            else:
                self._heading_bin = observed.get(
                    player_id,
                    seed % profile.heading_bins,
                )
            reason = "initial_discrete_heading"
        else:
            reason = self._advance_heading()

        can_split = any(
            blob.radius * blob.radius >= 2.0
            for blob in state.me.blobs.values()
        )
        split_roll = self._split_fraction() if can_split else None
        split = bool(split_roll is not None and split_roll < profile.split_rate)
        angle = self._heading_bin * math.tau / profile.heading_bins
        return StrategyDecision(
            direction=(math.cos(angle), math.sin(angle)),
            split=split,
            target_kind="discrete_random_walk",
            target_id=str(profile.team_id),
            reason=reason,
            diagnostics={
                "source_team_id": profile.team_id,
                "heading_bin": self._heading_bin,
                "heading_bins": profile.heading_bins,
                "split_roll": split_roll,
                "split_rate_when_eligible": profile.split_rate,
            },
        )

    def _advance_heading(self) -> str:
        assert self._heading_bin is not None
        profile = self.profile
        roll = self._direction_fraction()
        for transition in profile.transitions:
            if roll >= transition.upper_bound:
                continue
            if transition.offset is not None:
                self._heading_bin = (
                    self._heading_bin + transition.offset
                ) % profile.heading_bins
            return transition.reason
        self._heading_bin = int(
            self._direction_fraction() * profile.heading_bins
        ) % profile.heading_bins
        return "jump_heading"

    def _direction_fraction(self) -> float:
        self._direction_rng = (
            self._direction_rng + self._GOLDEN_RATIO_64
        ) & self._MASK_64
        return self._mix64(self._direction_rng) / float(1 << 64)

    def _split_fraction(self) -> float:
        self._split_rng = (
            self._split_rng + self._GOLDEN_RATIO_64
        ) & self._MASK_64
        return self._mix64(self._split_rng) / float(1 << 64)

    @classmethod
    def _mix64(cls, value: int) -> int:
        value ^= value >> 30
        value = (value * 0xBF58476D1CE4E5B9) & cls._MASK_64
        value ^= value >> 27
        value = (value * 0x94D049BB133111EB) & cls._MASK_64
        return (value ^ (value >> 31)) & cls._MASK_64
