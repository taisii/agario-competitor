from __future__ import annotations

"""Replay-derived opponent policy for official team 4.

Team 4 emitted unit vectors on all 3,868 observed turns.  Its direction is
best represented by the fitted food-field, inertia, predator, and edible-virus
regimes.  All 20 split commands came from the single high-growth replay and
shared a safe-farming state: one merge-ready blob, mass at least 7.18, no
visible predator, and no currently edible virus.  The exact latent trigger is
not observable, so its observed 18% eligible-state rate is reproduced with a
deterministic local roll.
"""

from lib.config.player import EAT_SIZE_RATIO
from simulation.rules import can_consume_virus
from strategies.base import StrategyContext, StrategyDecision
from strategies.replay_imitation import (
    ImitationObservation,
    _relations,
    observation_from_context,
    predict_direction,
)
from strategies.replay_profiles import PROFILES
from strategies.randomness import MASK_64, unit_interval


TEAM_ID = 4
PROFILE = PROFILES[TEAM_ID]
MIN_FARM_SPLIT_MASS = 7.0
FARM_SPLIT_RATE = 0.18

class ReplayTeam4Strategy:
    """Stateful fitted direction policy with sparse safe-farming splits."""

    name = "replay_team_4"

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

        direction = predict_direction(
            self.profile,
            observation,
            self._previous_direction,
        )
        split, split_roll = self._split_decision(observation)
        self._previous_direction = direction
        self._last_round = observation.round_number

        return StrategyDecision(
            direction=direction,
            split=split,
            target_kind="replay_imitation",
            target_id="4",
            reason="team4_sparse_farm_split" if split else "team4_fitted_field",
            diagnostics={
                "source_matches": self.profile.source_matches,
                "split_roll": split_roll,
                "split_rate": FARM_SPLIT_RATE,
                "respawned": respawned,
                "validation_passed": self.profile.validation_passed,
            },
        )

    def _split_decision(
        self,
        observation: ImitationObservation,
    ) -> tuple[bool, float | None]:
        if not self._farm_split_candidate(observation):
            return (False, None)
        own = observation.own_blobs[0]
        roll = self._split_roll(
            round_number=observation.round_number,
            player_id=own.player_id,
            radius=own.radius,
        )
        return (roll < FARM_SPLIT_RATE, roll)

    @staticmethod
    def _farm_split_candidate(observation: ImitationObservation) -> bool:
        if len(observation.own_blobs) != 1:
            return False
        own = observation.own_blobs[0]
        if own.radius * own.radius < MIN_FARM_SPLIT_MASS:
            return False
        if own.merge_cooldown > 0:
            return False
        predators, _, _ = _relations(observation)
        if predators:
            return False
        edible_virus = any(
            can_consume_virus(
                own.radius,
                virus.radius,
                eat_size_ratio=EAT_SIZE_RATIO,
            )
            for virus in observation.visible_viruses
        )
        return not edible_virus

    @classmethod
    def _split_roll(
        cls,
        *,
        round_number: int,
        player_id: int,
        radius: float,
    ) -> float:
        value = (
            0x04D1B54A32D192ED
            ^ (round_number * 0x9E3779B97F4A7C15)
            ^ (player_id * 0xD6E8FEB86659FD93)
            ^ (round(radius * 1_000_000) * 0x8EBC6AF09C88C6E3)
        ) & MASK_64
        return unit_interval(value)
