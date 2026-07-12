from __future__ import annotations

"""Replay-derived opponent policy for official team 16.

Team 16 appeared in four official matches and split only in its one strong
growth run.  All 20 splits used one blob, visible edible prey, no predator,
and an engine-capable child within 21 units.  Those events are 20.2% of the
99 observations satisfying the same visible conditions, so the latent choice
is represented by a deterministic sparse roll.
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
from strategies.randomness import MASK_64, unit_interval


TEAM_ID = 16
PROFILE = PROFILES[TEAM_ID]
MAX_SPLIT_PREY_DISTANCE = 21.0
SPLIT_RATE = 0.202

class ReplayTeam16Strategy:
    """Fitted continuous direction policy with sparse prey splits."""

    name = "replay_team_16"

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

        fitted = predict_direction(self.profile, observation, self._previous_direction)
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
            reason = "team16_sparse_prey_split"
            target_kind = "prey"
            target_id = f"{prey.player_id}:{prey.blob_id}"
        else:
            direction = fitted
            reason = "team16_fitted_direction"
            target_kind = "replay_imitation"
            target_id = "16"

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
            0x16D1B54A32D192ED
            ^ (round_number * 0x9E3779B97F4A7C15)
            ^ (player_id * 0xD6E8FEB86659FD93)
            ^ (round(own_radius * 1_000_000) * 0x8EBC6AF09C88C6E3)
            ^ (round(prey_radius * 1_000_000) * 0xA0761D6478BD642F)
            ^ (round(prey_distance * 1_000_000) * 0x589965CC75374CC3)
        ) & MASK_64
        return unit_interval(value)
