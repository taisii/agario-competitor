from __future__ import annotations

"""Reusable implementations for replay-derived opponent families."""

import math
from dataclasses import dataclass

from strategies.base import StrategyContext, StrategyDecision
from strategies.features import normalise, squared_distance
from strategies.randomness import GOLDEN_RATIO_64, MASK_64, mix64, unit_interval


@dataclass(frozen=True)
class NearestFragmentFoodProfile:
    team_id: int
    source_matches: tuple[int, ...]
    move_reason: str
    fallback_reason: str
    validation_passed: bool | None = None


class NearestFragmentFoodStrategy:
    """Pursue the food nearest to any real fragment and never split."""

    def __init__(self, profile: NearestFragmentFoodProfile) -> None:
        self.name = f"replay_team_{profile.team_id}"
        self.profile = profile
        self._previous_direction = (1.0, 0.0)

    def choose(self, context: StrategyContext) -> StrategyDecision:
        state = context.game.state
        own_blobs = tuple(state.me.blobs.values())
        profile = self.profile
        diagnostics: dict[str, object]
        nearest = min(
            (
                (squared_distance(blob.pos, food.pos), food.food_id, blob.blob_id, blob, food)
                for food in state.visible_food
                for blob in own_blobs
            ),
            default=None,
        )
        if nearest is not None:
            distance_squared, _, _, origin, food = nearest
            direction = (
                food.pos[0] - origin.pos[0],
                food.pos[1] - origin.pos[1],
            )
            unit = normalise(direction)
            if unit != (0.0, 0.0):
                self._previous_direction = unit
            diagnostics = {
                "origin_blob_id": origin.blob_id,
                "food_distance": math.sqrt(distance_squared),
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

    def __init__(self, profile: DiscreteRandomWalkProfile) -> None:
        self.name = f"replay_team_{profile.team_id}"
        self.profile = profile
        self._heading_bin: int | None = None
        self._direction_rng = 0
        self._split_rng = 0

    def choose(self, context: StrategyContext) -> StrategyDecision:
        state = context.game.state
        player_id = int(getattr(state.me, "player_id", 0))
        profile = self.profile
        if self._heading_bin is None:
            seed = mix64(profile.seed_salt ^ player_id)
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
            self._direction_rng + GOLDEN_RATIO_64
        ) & MASK_64
        return unit_interval(self._direction_rng)

    def _split_fraction(self) -> float:
        self._split_rng = (
            self._split_rng + GOLDEN_RATIO_64
        ) & MASK_64
        return unit_interval(self._split_rng)


_NEAREST_FOOD_SETTINGS = {
    2: ((11679, 11724), "team2_nearest_food", "team2_inertia_fallback", None),
    27: (None, "team27_nearest_fragment_food", "team27_inertia_fallback", True),
    30: (None, "team30_nearest_fragment_food", "team30_inertia_fallback", True),
    75: (None, "team75_nearest_fragment_food", "team75_inertia_fallback", True),
}

_RANDOM_WALK_PROFILES = {
    6: DiscreteRandomWalkProfile(
        team_id=6,
        seed_salt=0x6A09E667F3BCC909,
        split_salt=0xBB67AE8584CAA73B,
        split_rate=0.031,
        parity_origin_player_id=2,
        transitions=(
            HeadingTransition(0.606, None, "hold_heading"),
            HeadingTransition(0.763, 1, "turn_left_one_bin"),
            HeadingTransition(0.909, -1, "turn_right_one_bin"),
        ),
    ),
    38: DiscreteRandomWalkProfile(
        team_id=38,
        seed_salt=0x3C6EF372FE94F82B,
        split_salt=0xA54FF53A5F1D36F1,
        split_rate=15.0 / 1726.0,
        observed_initial_bins=((3, 6), (6, 2), (7, 4)),
        transitions=(
            HeadingTransition(0.553, None, "hold_heading"),
            HeadingTransition(0.654, 1, "turn_left_one_bin"),
            HeadingTransition(0.762, -1, "turn_right_one_bin"),
            HeadingTransition(0.797, 2, "turn_left_two_bins"),
            HeadingTransition(0.829, -2, "turn_right_two_bins"),
        ),
    ),
}


def create_profiled_opponent_strategy(
    team_id: int,
) -> NearestFragmentFoodStrategy | DiscreteRandomWalkStrategy:
    if team_id in _RANDOM_WALK_PROFILES:
        return DiscreteRandomWalkStrategy(_RANDOM_WALK_PROFILES[team_id])

    from strategies.replay_profiles import PROFILES

    source_matches, move_reason, fallback_reason, validation_passed = (
        _NEAREST_FOOD_SETTINGS[team_id]
    )
    fitted = PROFILES[team_id]
    return NearestFragmentFoodStrategy(
        NearestFragmentFoodProfile(
            team_id=team_id,
            source_matches=source_matches or fitted.source_matches,
            move_reason=move_reason,
            fallback_reason=fallback_reason,
            validation_passed=validation_passed,
        )
    )
