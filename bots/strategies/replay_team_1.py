from __future__ import annotations

"""Replay-derived opponent strategy for official team 1.

Direction uses the fitted two-match policy.  Team 1 issued only nine split
commands in 2,350 turns; eight of those aligned an eligible split child with a
reachable prey blob.  The dedicated split gate models that geometric action
instead of the generic classifier, which produced substantially more false
split commands on this sparse trace.
"""

import math

from lib.config.player import (
    BASE_PLAYER_SPEED,
    MIN_PLAYER_SPEED,
    PLAYER_SPEED_RADIUS_FACTOR,
)
from simulation.rules import movement_speed
from strategies.base import StrategyContext, StrategyDecision
from strategies.replay_imitation import (
    EAT_SIZE_RATIO,
    ImitationBlob,
    ImitationObservation,
    ReplayImitationStrategy,
    _relations,
    _unit,
    observation_from_context,
    predict_direction,
)
from strategies.replay_profiles import PROFILES


MAX_BLOB_COUNT = 16
SPLIT_MIN_MASS = 2.0
SPLIT_EJECT_SPEED = 1.6
def _speed(radius: float) -> float:
    return movement_speed(
        radius,
        base_speed=BASE_PLAYER_SPEED,
        radius_factor=PLAYER_SPEED_RADIUS_FACTOR,
        minimum_speed=MIN_PLAYER_SPEED,
    )


class ReplayTeam1Strategy(ReplayImitationStrategy):
    """Team-1 fitted direction with a replay-derived split corridor gate."""

    name = "replay_team_1"

    def __init__(self) -> None:
        super().__init__(PROFILES[1])
        self.name = type(self).name

    def choose(self, context: StrategyContext) -> StrategyDecision:
        observation = observation_from_context(context)
        direction = predict_direction(
            self.profile,
            observation,
            self._previous_direction,
        )
        split, split_reason = self._split_decision(observation, direction)
        self._previous_direction = direction

        return StrategyDecision(
            direction=direction,
            split=split,
            target_kind="replay_imitation",
            target_id="1",
            reason=split_reason if split else "team1_fitted_direction",
            diagnostics={
                "source_matches": self.profile.source_matches,
                "split_rule": split_reason,
                "validation_passed": self.profile.validation_passed,
            },
        )

    @classmethod
    def _split_decision(
        cls,
        observation: ImitationObservation,
        direction: tuple[float, float],
    ) -> tuple[bool, str]:
        if len(observation.own_blobs) >= MAX_BLOB_COUNT:
            return (False, "blob_cap")
        direction = _unit(direction)
        if direction == (0.0, 0.0):
            return (False, "zero_direction")

        _, prey, _ = _relations(observation)
        for own in observation.own_blobs:
            if own.radius * own.radius < SPLIT_MIN_MASS:
                continue
            child_radius = own.radius / math.sqrt(2.0)
            reach = 3.0 * child_radius + SPLIT_EJECT_SPEED + _speed(child_radius)
            for target in prey:
                if (
                    child_radius * child_radius
                    < target.radius * target.radius * EAT_SIZE_RATIO
                ):
                    continue
                if cls._inside_split_corridor(
                    own,
                    target,
                    child_radius,
                    reach,
                    direction,
                ):
                    return (True, "reachable_prey_split")
        return (False, "no_reachable_split_prey")

    @staticmethod
    def _inside_split_corridor(
        own: ImitationBlob,
        target: ImitationBlob,
        child_radius: float,
        reach: float,
        direction: tuple[float, float],
    ) -> bool:
        dx = target.x - own.x
        dy = target.y - own.y
        forward = dx * direction[0] + dy * direction[1]
        lateral = abs(dx * direction[1] - dy * direction[0])
        return -0.1 <= forward <= reach and lateral <= child_radius
