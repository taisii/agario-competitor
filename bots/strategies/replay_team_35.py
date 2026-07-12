from __future__ import annotations

"""Replay-derived opponent strategy for official team 35.

Team 35 issued 41 split commands across 6,280 observed turns.  Every split
had exactly one merge-ready blob, radius at least 2.054, visible prey, and no
visible predator.  A radius floor of 2.0 reproduces all 41 commands with one
false positive across the five source matches, substantially outperforming
the generic fitted split classifier while retaining its direction model.
"""

from strategies.base import StrategyContext, StrategyDecision
from strategies.replay_imitation import (
    ImitationObservation,
    ReplayImitationStrategy,
    _relations,
    observation_from_context,
    predict_direction,
)
from strategies.replay_profiles import PROFILES


MIN_SPLIT_RADIUS = 2.0


class ReplayTeam35Strategy(ReplayImitationStrategy):
    """Team-35 fitted movement with its replay-observed split invariant."""

    name = "replay_team_35"

    def __init__(self) -> None:
        super().__init__(PROFILES[35])
        self.name = type(self).name

    def choose(self, context: StrategyContext) -> StrategyDecision:
        observation = observation_from_context(context)
        direction = predict_direction(
            self.profile,
            observation,
            self._previous_direction,
        )
        split, split_reason = self._split_decision(observation)
        self._previous_direction = direction

        return StrategyDecision(
            direction=direction,
            split=split,
            target_kind="replay_imitation",
            target_id="35",
            reason=split_reason if split else "team35_fitted_direction",
            diagnostics={
                "source_matches": self.profile.source_matches,
                "split_rule": split_reason,
                "direction_median_error": self.profile.direction_median_error,
                "direction_within_30_rate": self.profile.direction_within_30_rate,
            },
        )

    @staticmethod
    def _split_decision(
        observation: ImitationObservation,
    ) -> tuple[bool, str]:
        if len(observation.own_blobs) != 1:
            return (False, "requires_single_blob")

        own = observation.own_blobs[0]
        if own.radius < MIN_SPLIT_RADIUS:
            return (False, "below_observed_radius_floor")
        if own.merge_cooldown > 0:
            return (False, "merge_cooldown_active")

        predators, prey, _ = _relations(observation)
        if predators:
            return (False, "predator_visible")
        if not prey:
            return (False, "no_visible_prey")
        return (True, "single_blob_prey_split")
