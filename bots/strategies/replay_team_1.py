from __future__ import annotations

"""Replay-derived opponent strategy for official team 1.

Twenty current matches show a resource-first movement policy and a stateful
split rhythm.  With no visible predator, team 1 aims exactly at the nearest
food or prey.  Splits are edge-triggered when at least one fragment is
merge-ready, the largest blob is large enough, at most eight fragments exist,
and prey or a consumable virus is visible.  A short rearm interval models the
observed split/recombine cycle without firing on every frame while the same
resource remains visible.
"""

from strategies.base import StrategyContext, StrategyDecision
from strategies.replay_imitation import (
    ImitationObservation,
    ReplayImitationStrategy,
    analyze_observation,
    observation_from_context,
    predict_direction,
)
from strategies.replay_profiles import PROFILES


MAX_SPLIT_BLOB_COUNT = 8
MIN_SPLIT_RADIUS = 2.0
SPLIT_REARM_ROUNDS = 15


class ReplayTeam1Strategy(ReplayImitationStrategy):
    """Team-1 resource tracking with its replay-observed split state machine."""

    name = "replay_team_1"

    def __init__(self) -> None:
        super().__init__(PROFILES[1])
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
            target_id="1",
            reason=split_reason if split else "team1_resource_direction",
            diagnostics={
                "source_matches": self.profile.source_matches,
                "split_rule": split_reason,
                "last_split_round": self._last_split_round,
                "trace_reset": trace_reset,
                "validation_passed": self.profile.validation_passed,
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
        if len(observation.own_blobs) > MAX_SPLIT_BLOB_COUNT:
            return (False, "fragment_cap")
        if observation.round_number - last_split_round < SPLIT_REARM_ROUNDS:
            return (False, "split_rearming")
        if not any(blob.radius >= MIN_SPLIT_RADIUS for blob in observation.own_blobs):
            return (False, "no_split_sized_blob")
        if not any(blob.merge_cooldown <= 0 for blob in observation.own_blobs):
            return (False, "no_merge_ready_blob")

        analysis = analyze_observation(observation)
        if not analysis.prey and not analysis.edible_viruses:
            return (False, "no_visible_resource")
        return (True, "merge_ready_resource_split")
