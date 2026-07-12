from __future__ import annotations

"""Replay-derived opponent strategy for official team 25.

The four source matches contain 5,309 actions and no split command.  In safe
observations the recorded heading follows the nearest-food vector; whenever a
predator is visible it switches to the inverse-distance predator escape field.
That two-regime rule passes every source-match shadow gate independently.
"""

from strategies.base import StrategyContext, StrategyDecision
from strategies.replay_imitation import (
    ImitationObservation,
    _field_vector,
    _mass_center,
    _nearest_vector,
    _relations,
    _unit,
    observation_from_context,
)


SOURCE_MATCHES = (11719, 11725, 11752, 11756)


class ReplayTeam25Strategy:
    """Team-25 nearest-food policy with predator-field escape."""

    name = "replay_team_25"

    def __init__(self) -> None:
        self._previous_direction = (1.0, 0.0)

    def choose(self, context: StrategyContext) -> StrategyDecision:
        observation = observation_from_context(context)
        direction, target_kind, reason = self._direction(observation)
        direction = _unit(direction)
        if direction == (0.0, 0.0):
            direction = self._previous_direction
            target_kind = "none"
            reason = "team25_inertia_fallback"
        self._previous_direction = direction

        predators, prey, _ = _relations(observation)
        return StrategyDecision(
            direction=direction,
            split=self._split_decision(observation),
            target_kind=target_kind,
            target_id="25",
            reason=reason,
            diagnostics={
                "source_matches": SOURCE_MATCHES,
                "predator_count": len(predators),
                "prey_count": len(prey),
                "visible_food_count": len(observation.visible_food),
            },
        )

    @staticmethod
    def _direction(
        observation: ImitationObservation,
    ) -> tuple[tuple[float, float], str, str]:
        center = _mass_center(observation.own_blobs)
        predators, _, _ = _relations(observation)
        if predators:
            return (
                _field_vector(center, predators, away=True),
                "escape",
                "team25_predator_field",
            )
        if observation.visible_food:
            return (
                _nearest_vector(center, observation.visible_food),
                "food",
                "team25_nearest_food",
            )
        return ((0.0, 0.0), "none", "team25_no_target")

    @staticmethod
    def _split_decision(observation: ImitationObservation) -> bool:
        del observation
        return False
