from __future__ import annotations

"""Replay-derived opponent strategy for official team 51.

Five official recordings contain 6,670 team-51 commands.  Outside danger,
roughly four fifths of its non-zero headings lie on a 22.5-degree grid and its
median change from the previous command is only 1.77 degrees.  In danger the
grid largely disappears and the fitted predator field is more faithful.

All nine observed split commands share a much sharper rule: one blob, no
visible predator, a nearby edible prey, and a command aimed exactly at that
prey.  The split gate below encodes that rule directly.
"""

import math

from strategies.base import StrategyContext, StrategyDecision
from strategies.replay_imitation import (
    EAT_SIZE_RATIO,
    ImitationBlob,
    ImitationObservation,
    ReplayImitationStrategy,
    _mass_center,
    _relations,
    _unit,
    observation_from_context,
    predict_direction,
)
from strategies.replay_profiles import PROFILES


SAFE_ANGLE_BINS = 16
SAFE_ANGLE_STEP = math.tau / SAFE_ANGLE_BINS

# Replay bounds.  The smallest split mass was 2.701, prey distances ranged
# from 3.88 to 6.39, and the smallest wall clearance was 7.49.
MIN_SPLIT_MASS = 2.65
MAX_SPLIT_PREY_DISTANCE = 6.50
MIN_SPLIT_RADIUS_RATIO = 1.80
MIN_SPLIT_WALL_CLEARANCE = 7.00
SPLIT_SAFETY_DISTANCE = 14.0


def _snap_safe_heading(direction: tuple[float, float]) -> tuple[float, float]:
    direction = _unit(direction)
    if direction == (0.0, 0.0):
        return direction
    angle = round(math.atan2(direction[1], direction[0]) / SAFE_ANGLE_STEP) * SAFE_ANGLE_STEP
    return (math.cos(angle), math.sin(angle))


class ReplayTeam51Strategy(ReplayImitationStrategy):
    """Hybrid grid/continuous policy reconstructed from team 51 replays."""

    name = "replay_team_51"

    def __init__(self) -> None:
        super().__init__(PROFILES[51])
        self.name = type(self).name
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
            # The engine does not query a dead player.  A round gap therefore
            # marks a respawn, where retaining a pre-death heading creates a
            # persistent shadow-replay error.
            self._previous_direction = (0.0, 0.0)

        if not observation.own_blobs:
            direction = self._previous_direction or (1.0, 0.0)
            self._last_round = observation.round_number
            return StrategyDecision(
                direction=direction,
                split=False,
                target_kind="replay_imitation",
                target_id="51",
                reason="dead_fallback",
            )

        split_target = self._safe_split_target(observation)
        if split_target is not None:
            center = _mass_center(observation.own_blobs)
            direction = _unit((split_target.x - center[0], split_target.y - center[1]))
            split = True
            reason = "team51_exact_prey_split"
            target_kind = "prey"
            target_id = f"{split_target.player_id}:{split_target.blob_id}"
        else:
            predators, _, _ = _relations(observation)
            fitted = predict_direction(
                self.profile,
                observation,
                self._previous_direction,
            )
            # Safe movement was predominantly on a 16-heading grid.  Predator
            # movement was predominantly continuous, so preserve the fitted
            # escape vector without snapping it.
            direction = fitted if predators else _snap_safe_heading(fitted)
            if direction == (0.0, 0.0):
                direction = self._previous_direction or (1.0, 0.0)
            split = False
            reason = "team51_predator_field" if predators else "team51_safe_grid_inertia"
            target_kind = "escape" if predators else "replay_imitation"
            target_id = None

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
                "safe_angle_bins": SAFE_ANGLE_BINS,
                "respawned": respawned,
                "validation_passed": self.profile.validation_passed,
            },
        )

    def _safe_split_target(
        self,
        observation: ImitationObservation,
    ) -> ImitationBlob | None:
        predators, prey, _ = _relations(observation)
        if predators or len(observation.own_blobs) != 1:
            return None

        own = observation.own_blobs[0]
        own_mass = own.radius * own.radius
        if own_mass < MIN_SPLIT_MASS:
            return None

        center = _mass_center(observation.own_blobs)
        wall_clearance = min(
            center[0],
            center[1],
            observation.arena_size - center[0],
            observation.arena_size - center[1],
        )
        if wall_clearance < MIN_SPLIT_WALL_CLEARANCE:
            return None

        child_radius = own.radius / math.sqrt(2.0)
        candidates = []
        for target in prey:
            distance = math.hypot(target.x - own.x, target.y - own.y)
            if distance > MAX_SPLIT_PREY_DISTANCE:
                continue
            if own.radius / max(target.radius, 1e-9) < MIN_SPLIT_RADIUS_RATIO:
                continue
            if child_radius * child_radius < target.radius * target.radius * EAT_SIZE_RATIO:
                continue
            if not self._child_is_safe(own, child_radius, target, observation):
                continue
            candidates.append((distance, target))

        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    @staticmethod
    def _child_is_safe(
        own: ImitationBlob,
        child_radius: float,
        target: ImitationBlob,
        observation: ImitationObservation,
    ) -> bool:
        for other in observation.visible_blobs:
            if other is target:
                continue
            can_eat_child = (
                other.radius * other.radius
                >= child_radius * child_radius * EAT_SIZE_RATIO
            )
            if can_eat_child and math.hypot(other.x - own.x, other.y - own.y) < SPLIT_SAFETY_DISTANCE:
                return False
        return True
