from __future__ import annotations

"""Replay-derived opponent policy for official team 31.

Team 31 appeared in three official matches.  Its ordinary movement was
mostly keyboard-like: 82--87% of commands landed exactly on a 16-direction
grid, with the previous heading repeated on 42--52% of transitions.  The
fitted replay direction is therefore quantized to the same grid.

All 21 split commands targeted visible prey with no visible predator.  The
prey was close enough (at most 11.87 units) for a child of the largest blob
to eat it.  Split frequency depended strongly on an attack already being in
progress: 14/266 eligible one-blob observations, 5/17 two-blob observations,
and 2/2 four-blob observations.  A deterministic roll preserves those three
observed rates without introducing nondeterminism into local benchmarks.
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


TEAM_ID = 31
PROFILE = PROFILES[TEAM_ID]
DIRECTION_BINS = 16
MAX_SPLIT_PREY_DISTANCE = 12.0
SINGLE_BLOB_SPLIT_RATE = 14.0 / 266.0
TWO_BLOB_SPLIT_RATE = 5.0 / 17.0
FOUR_BLOB_SPLIT_RATE = 1.0

_MASK_64 = (1 << 64) - 1


class ReplayTeam31Strategy:
    """Quantized fitted movement with replay-observed chained prey attacks."""

    name = "replay_team_31"

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
        direction = self._quantize_direction(fitted)
        candidate = self._split_candidate(observation)
        split_rate = self._split_rate(len(observation.own_blobs))
        split_roll = None
        split = False
        if candidate is not None:
            own, prey, distance = candidate
            split_roll = self._split_roll(
                round_number=observation.round_number,
                player_id=own.player_id,
                blob_count=len(observation.own_blobs),
                own_radius=own.radius,
                prey_radius=prey.radius,
                prey_distance=distance,
            )
            split = split_roll < split_rate

        if split and candidate is not None:
            center = _mass_center(observation.own_blobs)
            prey = candidate[1]
            direction = _unit((prey.x - center[0], prey.y - center[1]))
            reason = "team31_chained_prey_split"
            target_kind = "prey"
            target_id = f"{prey.player_id}:{prey.blob_id}"
        else:
            reason = "team31_quantized_fitted_direction"
            target_kind = "replay_imitation"
            target_id = str(TEAM_ID)

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
                "direction_bins": DIRECTION_BINS,
                "split_rate": split_rate,
                "split_roll": split_roll,
                "respawned": respawned,
                "validation_passed": self.profile.validation_passed,
            },
        )

    @staticmethod
    def _quantize_direction(
        direction: tuple[float, float],
    ) -> tuple[float, float]:
        angle = math.atan2(direction[1], direction[0])
        step = math.tau / DIRECTION_BINS
        snapped = round(angle / step) * step
        return (math.cos(snapped), math.sin(snapped))

    @staticmethod
    def _split_rate(blob_count: int) -> float:
        if blob_count >= 4:
            return FOUR_BLOB_SPLIT_RATE
        if blob_count >= 2:
            return TWO_BLOB_SPLIT_RATE
        return SINGLE_BLOB_SPLIT_RATE

    @staticmethod
    def _split_candidate(
        observation: ImitationObservation,
    ) -> tuple[ImitationBlob, ImitationBlob, float] | None:
        if not observation.own_blobs:
            return None
        predators, prey, _ = _relations(observation)
        if predators or not prey:
            return None

        center = _mass_center(observation.own_blobs)
        candidates: list[tuple[float, ImitationBlob, ImitationBlob]] = []
        for target in prey:
            distance = math.hypot(target.x - center[0], target.y - center[1])
            if distance > MAX_SPLIT_PREY_DISTANCE:
                continue
            capable = [
                own
                for own in observation.own_blobs
                if own.radius * own.radius >= 2.0
                and own.radius * own.radius / 2.0
                >= target.radius * target.radius * EAT_SIZE_RATIO
            ]
            if capable:
                candidates.append(
                    (distance, max(capable, key=lambda blob: blob.radius), target)
                )
        if not candidates:
            return None
        distance, own, target = min(candidates, key=lambda item: item[0])
        return (own, target, distance)

    @classmethod
    def _split_roll(
        cls,
        *,
        round_number: int,
        player_id: int,
        blob_count: int,
        own_radius: float,
        prey_radius: float,
        prey_distance: float,
    ) -> float:
        value = (
            0x31D1B54A32D192ED
            ^ (round_number * 0x9E3779B97F4A7C15)
            ^ (player_id * 0xD6E8FEB86659FD93)
            ^ (blob_count * 0xDB4F0B9175AE2165)
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
