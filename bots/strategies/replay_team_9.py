from __future__ import annotations

"""Replay-derived opponent policy for official team 9.

Team 9 appeared in five official matches and submitted 32 splits in 6,820
turns.  Every split used one blob, had visible prey and no predator, and all
were covered by an engine-valid child/prey ratio within 16 units.  Only 5.6%
of such eligible observations split, so the unobserved choice is represented
by a deterministic sparse roll.
"""

import math

from strategies.base import StrategyContext, StrategyDecision
from strategies.replay_imitation import (
    EAT_SIZE_RATIO,
    ImitationBlob,
    ImitationObservation,
    _mass_center,
    _relations,
    _unit,
    observation_from_context,
    predict_direction,
)
from strategies.replay_profiles import PROFILES


TEAM_ID = 9
PROFILE = PROFILES[TEAM_ID]
MAX_SPLIT_PREY_DISTANCE = 16.0
SPLIT_RATE = 0.056

_MASK_64 = (1 << 64) - 1


class ReplayTeam9Strategy:
    """Fitted field policy with sparse, safe prey splits."""

    name = "replay_team_9"

    def __init__(self) -> None:
        self.profile = PROFILE
        self._previous_direction = (0.0, 0.0)
        self._last_round: int | None = None

    def choose(self, context: StrategyContext) -> StrategyDecision:
        return self.choose_observation(observation_from_context(context))

    def choose_observation(
        self,
        observation: ImitationObservation,
    ) -> StrategyDecision:
        respawned = (
            self._last_round is not None
            and observation.round_number > self._last_round + 1
        )
        if respawned:
            self._previous_direction = (0.0, 0.0)

        fitted = predict_direction(
            self.profile,
            observation,
            self._previous_direction,
        )
        candidate = self._split_candidate(observation)
        split_roll = None
        if candidate is not None:
            own, prey, distance = candidate
            split_roll = self._split_roll(
                round_number=observation.round_number,
                player_id=own.player_id,
                own_radius=own.radius,
                prey_radius=prey.radius,
                prey_distance=distance,
            )
        split = split_roll is not None and split_roll < SPLIT_RATE
        if split and candidate is not None:
            center = _mass_center(observation.own_blobs)
            prey = candidate[1]
            direction = _unit((prey.x - center[0], prey.y - center[1]))
            reason = "team9_sparse_prey_split"
            target_kind = "prey"
            target_id = f"{prey.player_id}:{prey.blob_id}"
        else:
            direction = fitted
            reason = "team9_fitted_field"
            target_kind = "replay_imitation"
            target_id = "9"

        self._previous_direction = direction
        self._last_round = observation.round_number
        return StrategyDecision(
            direction=direction,
            split=split,
            target_kind=target_kind,
            target_id=target_id,
            reason=reason,
            diagnostics={
                "source_matches": self.profile.source_matches,
                "split_roll": split_roll,
                "split_rate": SPLIT_RATE,
                "respawned": respawned,
                "validation_passed": self.profile.validation_passed,
            },
        )

    @staticmethod
    def _split_candidate(
        observation: ImitationObservation,
    ) -> tuple[ImitationBlob, ImitationBlob, float] | None:
        if len(observation.own_blobs) != 1:
            return None
        own = observation.own_blobs[0]
        if own.radius * own.radius < 2.0:
            return None
        predators, prey, _ = _relations(observation)
        if predators or not prey:
            return None
        target = min(prey, key=lambda blob: math.hypot(blob.x - own.x, blob.y - own.y))
        distance = math.hypot(target.x - own.x, target.y - own.y)
        child_mass = own.radius * own.radius / 2.0
        if child_mass < target.radius * target.radius * EAT_SIZE_RATIO:
            return None
        if distance > MAX_SPLIT_PREY_DISTANCE:
            return None
        return (own, target, distance)

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
            0x09D1B54A32D192ED
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
