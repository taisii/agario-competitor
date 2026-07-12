from __future__ import annotations

"""Replay-derived opponent policy for official team 56.

Team 56 appeared in three official matches and submitted 223 split commands.
The commands form bursts rather than independent target attacks: 97 commands
started a burst and 126 continued one from the preceding turn.  No split was
observed below log1p(total_mass) 1.2.  Above that floor, both burst-onset and
continuation rates rose with mass.  The fitted visible-state classifier picks
burst onsets, while measured mass-dependent continuation rates determine how
long an active burst persists.

Outside split turns, 91--94% of directions were exactly on a 24-direction
grid.  Only 43% of split-turn directions used that grid; the fitted continuous
direction is therefore retained while a split is active.  A small observed
subset aimed exactly at child-edible prey, reproduced by a separate aim roll.
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
    predict_split,
)
from strategies.replay_profiles import PROFILES


TEAM_ID = 56
PROFILE = PROFILES[TEAM_ID]
DIRECTION_BINS = 24

MIN_SPLIT_TOTAL_MASS = math.expm1(1.2)
MID_SPLIT_TOTAL_MASS = math.expm1(2.0)
HIGH_SPLIT_TOTAL_MASS = math.expm1(2.5)

LOW_MASS_ONSET_RATE = 8.0 / 891.0
MID_MASS_ONSET_RATE = 6.0 / 144.0
HIGH_MASS_ONSET_RATE = 83.0 / 1182.0
LOW_MASS_CONTINUATION_RATE = 1.0 / 7.0
MID_MASS_CONTINUATION_RATE = 3.0 / 7.0
HIGH_MASS_CONTINUATION_RATE = 122.0 / 209.0

MAX_AIMED_PREY_DISTANCE = 14.1
PREY_AIM_RATE = 16.0 / 112.0

_MASK_64 = (1 << 64) - 1


class ReplayTeam56Strategy:
    """Mass-dependent split bursts over a fitted 24-direction policy."""

    name = "replay_team_56"

    def __init__(self) -> None:
        self.profile = PROFILE
        self._previous_direction = (0.0, 0.0)
        self._last_round: int | None = None
        self._last_split = False

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
            self._last_split = False

        fitted = predict_direction(
            self.profile,
            observation,
            self._previous_direction,
        )
        total_mass = sum(
            blob.radius * blob.radius for blob in observation.own_blobs
        )
        split_rate = self._split_probability(total_mass, self._last_split)
        fitted_onset, split_score = predict_split(
            self.profile,
            observation,
            self._previous_direction,
            fitted,
        )
        split_roll = self._state_roll(
            salt=0x56D1B54A32D192ED,
            observation=observation,
            total_mass=total_mass,
        )
        if total_mass < MIN_SPLIT_TOTAL_MASS:
            split = False
        elif self._last_split:
            split = split_roll < split_rate
        else:
            split = fitted_onset

        prey_candidate = self._aimed_prey_candidate(observation)
        prey_aim_roll = None
        if split and prey_candidate is not None:
            prey_aim_roll = self._state_roll(
                salt=0xA0761D6478BD642F,
                observation=observation,
                total_mass=total_mass,
            )

        if (
            split
            and prey_candidate is not None
            and prey_aim_roll is not None
            and prey_aim_roll < PREY_AIM_RATE
        ):
            center = _mass_center(observation.own_blobs)
            prey = prey_candidate[1]
            direction = _unit((prey.x - center[0], prey.y - center[1]))
            reason = "team56_aimed_split_burst"
            target_kind = "prey"
            target_id = f"{prey.player_id}:{prey.blob_id}"
        elif split:
            direction = fitted
            reason = (
                "team56_split_burst_continuation"
                if self._last_split
                else "team56_split_burst_onset"
            )
            target_kind = "split_burst"
            target_id = None
        else:
            direction = self._quantize_direction(fitted)
            reason = "team56_quantized_fitted_direction"
            target_kind = "replay_imitation"
            target_id = str(TEAM_ID)

        self._previous_direction = direction
        self._last_round = observation.round_number
        self._last_split = split
        return StrategyDecision(
            direction=direction,
            split=split,
            target_kind=target_kind,
            target_id=target_id,
            reason=reason,
            diagnostics={
                "source_matches": self.profile.source_matches,
                "total_mass": total_mass,
                "split_rate": split_rate,
                "split_score": split_score,
                "fitted_onset": fitted_onset,
                "split_roll": split_roll,
                "prey_aim_roll": prey_aim_roll,
                "respawned": respawned,
                "validation_passed": self.profile.validation_passed,
            },
        )

    @staticmethod
    def _quantize_direction(
        direction: tuple[float, float],
    ) -> tuple[float, float]:
        step = math.tau / DIRECTION_BINS
        angle = round(math.atan2(direction[1], direction[0]) / step) * step
        return (math.cos(angle), math.sin(angle))

    @staticmethod
    def _split_probability(total_mass: float, continuing: bool) -> float:
        if total_mass < MIN_SPLIT_TOTAL_MASS:
            return 0.0
        if total_mass < MID_SPLIT_TOTAL_MASS:
            return (
                LOW_MASS_CONTINUATION_RATE
                if continuing
                else LOW_MASS_ONSET_RATE
            )
        if total_mass < HIGH_SPLIT_TOTAL_MASS:
            return (
                MID_MASS_CONTINUATION_RATE
                if continuing
                else MID_MASS_ONSET_RATE
            )
        return (
            HIGH_MASS_CONTINUATION_RATE
            if continuing
            else HIGH_MASS_ONSET_RATE
        )

    @staticmethod
    def _aimed_prey_candidate(
        observation: ImitationObservation,
    ) -> tuple[ImitationBlob, ImitationBlob, float] | None:
        if not observation.own_blobs:
            return None
        _, prey, _ = _relations(observation)
        if not prey:
            return None
        center = _mass_center(observation.own_blobs)
        candidates: list[tuple[float, ImitationBlob, ImitationBlob]] = []
        for target in prey:
            distance = math.hypot(target.x - center[0], target.y - center[1])
            if distance > MAX_AIMED_PREY_DISTANCE + 1e-9:
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
    def _state_roll(
        cls,
        *,
        salt: int,
        observation: ImitationObservation,
        total_mass: float,
    ) -> float:
        player_id = (
            observation.own_blobs[0].player_id
            if observation.own_blobs
            else -1
        )
        value = (
            salt
            ^ (observation.round_number * 0x9E3779B97F4A7C15)
            ^ (player_id * 0xD6E8FEB86659FD93)
            ^ (len(observation.own_blobs) * 0xDB4F0B9175AE2165)
            ^ (round(total_mass * 1_000_000) * 0x8EBC6AF09C88C6E3)
        ) & _MASK_64
        return cls._mix64(value) / float(1 << 64)

    @staticmethod
    def _mix64(value: int) -> int:
        value ^= value >> 30
        value = (value * 0xBF58476D1CE4E5B9) & _MASK_64
        value ^= value >> 27
        value = (value * 0x94D049BB133111EB) & _MASK_64
        return (value ^ (value >> 31)) & _MASK_64
