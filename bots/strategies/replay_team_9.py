from __future__ import annotations

"""Replay-derived opponent strategy for official team 9.

Across 29 recent matches all 222 split commands had visible prey.  A rule
fitted only on the first 18 matches generalises to the 11 later matches better
than a prey-independent pseudo-random stream: it requires one merge-ready
blob of radius at least 2, a sufficiently smaller prey within 15 units, no
visible predator, and a 15-round rearm interval.  Its exact held-out split F1
is 0.437 (0.508 with a +/-2-round tolerance); the rejected Bernoulli surrogate
scored 0.011 exact F1 on the same held-out cohort.
"""

from strategies.base import StrategyContext, StrategyDecision
from strategies.replay_imitation import (
    ImitationObservation,
    ReplayImitationStrategy,
    observation_from_context,
    predict_direction,
    split_feature_values,
)
from strategies.replay_profiles import PROFILES


MIN_SPLIT_RADIUS = 2.0
MAX_PREY_DISTANCE = 15.0
MIN_PREY_RADIUS_RATIO = 3.0
SPLIT_REARM_ROUNDS = 15


class ReplayTeam9Strategy(ReplayImitationStrategy):
    """Team-9 fitted movement with its prey-split state machine."""

    name = "replay_team_9"

    def __init__(self) -> None:
        super().__init__(PROFILES[9])
        self._last_observed_round: int | None = None
        self._last_player_id: int | None = None

    def choose(self, context: StrategyContext) -> StrategyDecision:
        observation = observation_from_context(context)
        trace_reset = self._begin_observation(observation)
        direction = predict_direction(
            self.profile,
            observation,
            self._previous_direction,
        )
        split, split_reason = self._split_decision(
            observation,
            last_split_round=self._last_split_round,
        )
        if split:
            self._last_split_round = observation.round_number
        self._previous_direction = direction

        return StrategyDecision(
            direction=direction,
            split=split,
            target_kind="replay_imitation",
            target_id="9",
            reason=split_reason if split else "team9_fitted_direction",
            diagnostics={
                "source_matches": self.profile.source_matches,
                "split_rule": split_reason,
                "last_split_round": self._last_split_round,
                "trace_reset": trace_reset,
                "split_model": "deterministic_prey_geometry",
                "split_exact_match_claimed": False,
                "direction_median_error": self.profile.direction_median_error,
                "direction_within_30_rate": self.profile.direction_within_30_rate,
            },
        )

    def _begin_observation(self, observation: ImitationObservation) -> bool:
        """Reset history when one strategy instance is reused for a new trace."""

        player_id = (
            observation.own_blobs[0].player_id if observation.own_blobs else None
        )
        trace_reset = self._last_observed_round is not None and (
            observation.round_number < self._last_observed_round
            or player_id != self._last_player_id
        )
        if trace_reset:
            self._previous_direction = (0.0, 0.0)
            self._last_split_round = -10_000
        self._last_observed_round = observation.round_number
        self._last_player_id = player_id
        return trace_reset

    @staticmethod
    def _split_decision(
        observation: ImitationObservation,
        *,
        last_split_round: int = -10_000,
    ) -> tuple[bool, str]:
        if len(observation.own_blobs) != 1:
            return (False, "requires_single_blob")

        own = observation.own_blobs[0]
        if own.radius < MIN_SPLIT_RADIUS:
            return (False, "below_observed_radius_floor")
        if own.merge_cooldown > 0:
            return (False, "merge_cooldown_active")
        if observation.round_number - last_split_round < SPLIT_REARM_ROUNDS:
            return (False, "split_rearming")

        values = split_feature_values(observation)
        if values[9] > 0.5:
            return (False, "predator_visible")
        if values[6] <= 0.5:
            return (False, "no_visible_prey")
        if values[7] * 20.0 > MAX_PREY_DISTANCE:
            return (False, "prey_too_far")
        if values[8] < MIN_PREY_RADIUS_RATIO:
            return (False, "prey_too_large")
        return (True, "single_blob_prey_split")
