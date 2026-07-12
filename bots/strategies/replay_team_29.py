from __future__ import annotations

"""Replay-derived nearest-food strategy for official team 29.

All 4,304 observed commands follow the nearest-food heading within 30 degrees,
including observations containing predators, prey, and edible viruses.  No
split command appears in any of the four source matches.
"""

from strategies.base import StrategyContext, StrategyDecision
from strategies.replay_imitation import (
    ImitationObservation,
    _mass_center,
    _nearest_vector,
    _unit,
    observation_from_context,
)


SOURCE_MATCHES = (11646, 11679, 11745, 11757)


class ReplayTeam29Strategy:
    """Team-29 policy: nearest food remains authoritative in every regime."""

    name = "replay_team_29"

    def __init__(self) -> None:
        self._previous_direction = (1.0, 0.0)

    def choose(self, context: StrategyContext) -> StrategyDecision:
        observation = observation_from_context(context)
        direction = self._direction(observation)
        if direction == (0.0, 0.0):
            direction = self._previous_direction
            reason = "team29_inertia_fallback"
            target_kind = "none"
        else:
            reason = "team29_nearest_food"
            target_kind = "food"
        self._previous_direction = direction

        return StrategyDecision(
            direction=direction,
            split=False,
            target_kind=target_kind,
            target_id="29",
            reason=reason,
            diagnostics={
                "source_matches": SOURCE_MATCHES,
                "visible_food_count": len(observation.visible_food),
            },
        )

    @staticmethod
    def _direction(observation: ImitationObservation) -> tuple[float, float]:
        if not observation.visible_food:
            return (0.0, 0.0)
        center = _mass_center(observation.own_blobs)
        return _unit(_nearest_vector(center, observation.visible_food))
