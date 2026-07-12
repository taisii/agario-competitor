from __future__ import annotations

"""Behavioral reconstruction of official competition team 68.

Across the available matches, movement follows the nearest visible prey,
escapes a nearest predator more than 1.4 times the largest own radius, and
otherwise follows the nearest food.  That explicit rule passes the combined
direction shadow gate.  Split timing is less stable, so it uses the fitted
profile and retains the profile's failed validation marker.
"""

import math
from collections.abc import Sequence

from strategies.base import StrategyContext, StrategyDecision
from strategies.replay_imitation import (
    ImitationBlob,
    ImitationObservation,
    _mass_center,
    _nearest_vector,
    _relations,
    _unit,
    observation_from_context,
    predict_split,
)
from strategies.replay_profiles import PROFILES


SOURCE_MATCHES = PROFILES[68].source_matches
PREDATOR_ESCAPE_RADIUS_RATIO = 1.4


class ReplayTeam68Strategy:
    """Team-68 prey/food pursuit with selective predator escape."""

    name = "replay_team_68"

    def __init__(self) -> None:
        self.profile = PROFILES[68]
        self._previous_direction = (1.0, 0.0)

    def choose(self, context: StrategyContext) -> StrategyDecision:
        observation = observation_from_context(context)
        direction, target_kind, reason, predator_ratio = self._direction(observation)
        direction = _unit(direction)
        if direction == (0.0, 0.0):
            direction = self._previous_direction
            target_kind = "none"
            reason = "team68_inertia_fallback"

        split, split_score = predict_split(
            self.profile,
            observation,
            self._previous_direction,
            direction,
        )
        self._previous_direction = direction
        return StrategyDecision(
            direction=direction,
            split=split,
            target_kind=target_kind,
            target_id="68",
            reason=reason,
            diagnostics={
                "source_matches": SOURCE_MATCHES,
                "predator_radius_ratio": predator_ratio,
                "predator_escape_radius_ratio": PREDATOR_ESCAPE_RADIUS_RATIO,
                "split_score": split_score,
                "validation_passed": self.profile.validation_passed,
            },
        )

    @staticmethod
    def _direction(
        observation: ImitationObservation,
    ) -> tuple[tuple[float, float], str, str, float | None]:
        center = _mass_center(observation.own_blobs)
        predators, prey, _ = _relations(observation)

        if prey:
            return (
                _nearest_vector(center, prey),
                "prey",
                "team68_nearest_prey",
                None,
            )

        nearest_predator = _nearest(center, predators)
        largest_radius = max((blob.radius for blob in observation.own_blobs), default=0.0)
        predator_ratio = (
            nearest_predator.radius / largest_radius
            if nearest_predator is not None and largest_radius > 0.0
            else None
        )
        if (
            nearest_predator is not None
            and predator_ratio is not None
            and predator_ratio > PREDATOR_ESCAPE_RADIUS_RATIO
        ):
            return (
                _nearest_vector(center, (nearest_predator,), away=True),
                "escape",
                "team68_strong_predator_escape",
                predator_ratio,
            )

        if observation.visible_food:
            return (
                _nearest_vector(center, observation.visible_food),
                "food",
                "team68_nearest_food",
                predator_ratio,
            )
        return ((0.0, 0.0), "none", "team68_no_target", predator_ratio)


def _nearest(
    origin: tuple[float, float], blobs: Sequence[ImitationBlob]
) -> ImitationBlob | None:
    return min(
        blobs,
        key=lambda blob: math.hypot(blob.x - origin[0], blob.y - origin[1]),
        default=None,
    )
