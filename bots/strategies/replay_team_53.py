from __future__ import annotations

"""Replay-derived opponent for team 53.

Direction choice is the fitted regime policy shared by the replay tooling:
nearest food in an empty view, nearest prey when safe, and the aggregate
predator escape field whenever a predator is visible.  Official team 53 only
split on a small fraction of otherwise valid prey chases, so that latent
choice is represented by a deterministic 4.2% roll over eligible states.
"""

import math

from strategies.base import StrategyContext, StrategyDecision
from strategies.replay_imitation import (
    EAT_SIZE_RATIO,
    ImitationBlob,
    ImitationObservation,
    observation_from_context,
    predict_direction,
)
from strategies.replay_profiles import PROFILES


_MASK_64 = (1 << 64) - 1
_SPLIT_RATE = 0.042
_MAX_SPLIT_PREY_DISTANCE = 12.0


class ReplayTeam53Strategy:
    name = "replay_team_53"

    def __init__(self) -> None:
        self.profile = PROFILES[53]
        self._previous_direction = (0.0, 0.0)

    def choose(self, context: StrategyContext) -> StrategyDecision:
        observation = observation_from_context(context)
        direction = predict_direction(
            self.profile,
            observation,
            self._previous_direction,
        )
        split, split_roll = self._split_decision(observation)
        self._previous_direction = direction
        regime = self._regime(observation)
        return StrategyDecision(
            direction=direction,
            split=split,
            target_kind=regime,
            target_id="53",
            reason=f"team_53_{regime}",
            diagnostics={
                "source_team_id": 53,
                "source_matches": self.profile.source_matches,
                "split_roll": split_roll,
                "split_rate": _SPLIT_RATE,
                "direction_shadow_median_degrees": 0.751,
                "direction_shadow_within_30_rate": 0.858,
            },
        )

    def _split_decision(
        self,
        observation: ImitationObservation,
    ) -> tuple[bool, float | None]:
        candidate = self._split_candidate(observation)
        if candidate is None:
            return (False, None)
        own, prey, distance = candidate
        roll = self._split_roll(
            round_number=observation.round_number,
            player_id=own.player_id,
            own_radius=own.radius,
            prey_radius=prey.radius,
            prey_distance=distance,
        )
        return (roll < _SPLIT_RATE, roll)

    def _split_candidate(
        self,
        observation: ImitationObservation,
    ) -> tuple[ImitationBlob, ImitationBlob, float] | None:
        if len(observation.own_blobs) != 1:
            return None
        own = observation.own_blobs[0]
        if own.radius * own.radius < 2.0:
            return None

        predators, prey = self._relations(observation)
        if predators or not prey:
            return None
        nearest = min(
            prey,
            key=lambda blob: math.hypot(blob.x - own.x, blob.y - own.y),
        )
        distance = math.hypot(nearest.x - own.x, nearest.y - own.y)
        child_mass = own.radius * own.radius / 2.0
        if child_mass < nearest.radius * nearest.radius * EAT_SIZE_RATIO:
            return None
        if distance > _MAX_SPLIT_PREY_DISTANCE:
            return None
        return (own, nearest, distance)

    def _relations(
        self,
        observation: ImitationObservation,
    ) -> tuple[list[ImitationBlob], list[ImitationBlob]]:
        predators: list[ImitationBlob] = []
        prey: list[ImitationBlob] = []
        for other in observation.visible_blobs:
            can_eat_us = any(
                other.radius * other.radius
                >= own.radius * own.radius * EAT_SIZE_RATIO
                for own in observation.own_blobs
            )
            can_be_eaten = any(
                own.radius * own.radius
                >= other.radius * other.radius * EAT_SIZE_RATIO
                for own in observation.own_blobs
            )
            if can_eat_us:
                predators.append(other)
            elif can_be_eaten:
                prey.append(other)
        return predators, prey

    def _regime(self, observation: ImitationObservation) -> str:
        predators, prey = self._relations(observation)
        if predators:
            return "predator_escape"
        if prey:
            return "prey_chase"
        return "food_chase"

    @classmethod
    def _split_roll(
        cls,
        *,
        round_number: int,
        player_id: int,
        own_radius: float,
        prey_radius: float,
        prey_distance: float,
    ) -> float:
        value = (
            0x35D1B54A32D192ED
            ^ (round_number * 0x9E3779B97F4A7C15)
            ^ (player_id * 0xD6E8FEB86659FD93)
            ^ (round(own_radius * 1_000_000) * 0x8EBC6AF09C88C6E3)
            ^ (round(prey_radius * 1_000_000) * 0xA0761D6478BD642F)
            ^ (round(prey_distance * 1_000_000) * 0x589965CC75374CC3)
        ) & _MASK_64
        return cls._mix64(value) / float(1 << 64)

    @staticmethod
    def _mix64(value: int) -> int:
        value ^= value >> 30
        value = (value * 0xBF58476D1CE4E5B9) & _MASK_64
        value ^= value >> 27
        value = (value * 0x94D049BB133111EB) & _MASK_64
        return (value ^ (value >> 31)) & _MASK_64
