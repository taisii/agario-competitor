from __future__ import annotations

"""Replay-derived opponent strategy for official team 44.

Team 44 used a 32-heading action grid in every one of the 9,260 observed
turns.  Its next heading stayed within two grid steps of the previous heading
82.8% of the time, and none of its 131 split commands were issued while a
predator was visible.  This specialised policy keeps those invariants instead
of relying solely on the generic fitted split classifier.
"""

import math

from strategies.base import StrategyContext, StrategyDecision
from strategies.replay_imitation import (
    EAT_SIZE_RATIO,
    ImitationBlob,
    ImitationObservation,
    ImitationPoint,
    ReplayImitationStrategy,
    _field_vector,
    _relations,
    _unit,
    observation_from_context,
    predict_direction,
)
from strategies.replay_profiles import PROFILES


ANGLE_BINS = 32
ANGLE_STEP = math.tau / ANGLE_BINS
MAX_TURN_STEPS = 2
MAX_BLOB_COUNT = 16

# The smallest observed team-44 split happened at total mass 2.588.  Use a
# slightly lower floor to tolerate one round of engine mass decay.
MIN_OBSERVED_SPLIT_MASS = 2.55
SAFE_FARM_SPLIT_MASS = 2.60
SAFE_FARM_MIN_FOOD = 5
SPLIT_SAFETY_DISTANCE = 14.0

SPLIT_EJECT_SPEED = 1.6
BASE_PLAYER_SPEED = 1.1
MIN_PLAYER_SPEED = 0.25
PLAYER_SPEED_RADIUS_FACTOR = 0.08


def _speed(radius: float) -> float:
    return max(
        MIN_PLAYER_SPEED,
        BASE_PLAYER_SPEED / (1.0 + radius * PLAYER_SPEED_RADIUS_FACTOR),
    )


def _quantize_32(direction: tuple[float, float]) -> tuple[float, float]:
    direction = _unit(direction)
    if direction == (0.0, 0.0):
        return (1.0, 0.0)
    angle = round(math.atan2(direction[1], direction[0]) / ANGLE_STEP) * ANGLE_STEP
    return (math.cos(angle), math.sin(angle))


def _angle_difference(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    first_angle = math.atan2(first[1], first[0])
    second_angle = math.atan2(second[1], second[0])
    return abs((first_angle - second_angle + math.pi) % math.tau - math.pi)


class ReplayTeam44Strategy(ReplayImitationStrategy):
    """Team-44 policy with replay-observed inertia and tactical split gates."""

    name = "replay_team_44"

    def __init__(self) -> None:
        super().__init__(PROFILES[44])
        self.name = type(self).name

    def choose(self, context: StrategyContext) -> StrategyDecision:
        observation = observation_from_context(context)
        fitted_direction = predict_direction(
            self.profile,
            observation,
            self._previous_direction,
        )
        direction = self._inertial_direction(fitted_direction)
        split, split_reason = self._split_decision(observation, direction)
        self._previous_direction = direction

        return StrategyDecision(
            direction=direction,
            split=split,
            target_kind="replay_imitation",
            target_id="44",
            reason=split_reason if split else "team44_inertial_heading",
            diagnostics={
                "source_matches": self.profile.source_matches,
                "angle_bins": ANGLE_BINS,
                "max_turn_steps": MAX_TURN_STEPS,
                "split_rule": split_reason,
                "validation_passed": self.profile.validation_passed,
            },
        )

    def _inertial_direction(
        self,
        fitted_direction: tuple[float, float],
    ) -> tuple[float, float]:
        target = _quantize_32(fitted_direction)
        previous = _unit(self._previous_direction)
        if previous == (0.0, 0.0):
            return target

        previous = _quantize_32(previous)
        previous_angle = math.atan2(previous[1], previous[0])
        target_angle = math.atan2(target[1], target[0])
        delta = (target_angle - previous_angle + math.pi) % math.tau - math.pi
        turn_limit = MAX_TURN_STEPS * ANGLE_STEP
        limited_delta = max(-turn_limit, min(turn_limit, delta))
        return _quantize_32(
            (
                math.cos(previous_angle + limited_delta),
                math.sin(previous_angle + limited_delta),
            )
        )

    def _split_decision(
        self,
        observation: ImitationObservation,
        direction: tuple[float, float],
    ) -> tuple[bool, str]:
        predators, prey, _ = _relations(observation)
        if predators:
            return (False, "predator_visible")

        eligible = tuple(
            blob
            for blob in observation.own_blobs
            if blob.radius * blob.radius >= MIN_OBSERVED_SPLIT_MASS
        )
        if not eligible:
            return (False, "below_observed_mass_floor")
        if len(observation.own_blobs) >= MAX_BLOB_COUNT:
            # The replay contains 23 redundant commands at the blob cap.  They
            # cannot affect movement, so suppress them in the playable clone.
            return (False, "blob_cap")

        direction = _unit(direction)
        if self._split_capture_available(eligible, prey, direction, observation):
            return (True, "safe_split_capture")
        if self._edible_virus_split_available(eligible, observation, direction):
            return (True, "edible_virus_split")
        if self._safe_farm_split_available(observation, direction):
            return (True, "safe_farm_split")
        return (False, "no_safe_split_opportunity")

    def _split_capture_available(
        self,
        eligible: tuple[ImitationBlob, ...],
        prey: list[ImitationBlob],
        direction: tuple[float, float],
        observation: ImitationObservation,
    ) -> bool:
        for own in eligible:
            child_radius = own.radius / math.sqrt(2.0)
            if not self._child_is_safe(own, child_radius, observation):
                continue
            reach = 3.0 * child_radius + SPLIT_EJECT_SPEED + _speed(child_radius)
            for target in prey:
                if child_radius * child_radius < target.radius * target.radius * EAT_SIZE_RATIO:
                    continue
                if self._lies_in_split_corridor(own, target, child_radius, reach, direction):
                    return True
        return False

    def _edible_virus_split_available(
        self,
        eligible: tuple[ImitationBlob, ...],
        observation: ImitationObservation,
        direction: tuple[float, float],
    ) -> bool:
        for own in eligible:
            child_radius = own.radius / math.sqrt(2.0)
            if not self._child_is_safe(own, child_radius, observation):
                continue
            reach = 3.0 * child_radius + SPLIT_EJECT_SPEED + _speed(child_radius)
            for virus in observation.visible_viruses:
                if child_radius * child_radius <= virus.radius * virus.radius * EAT_SIZE_RATIO:
                    continue
                if self._lies_in_split_corridor(own, virus, child_radius, reach, direction):
                    return True
        return False

    def _safe_farm_split_available(
        self,
        observation: ImitationObservation,
        direction: tuple[float, float],
    ) -> bool:
        if len(observation.own_blobs) != 1:
            return False
        own = observation.own_blobs[0]
        if own.radius * own.radius < SAFE_FARM_SPLIT_MASS:
            return False
        if len(observation.visible_food) < SAFE_FARM_MIN_FOOD:
            return False

        child_radius = own.radius / math.sqrt(2.0)
        if not self._child_is_safe(own, child_radius, observation):
            return False
        food_field = _field_vector((own.x, own.y), observation.visible_food)
        return (
            food_field != (0.0, 0.0)
            and _angle_difference(direction, food_field) <= math.radians(45.0)
        )

    @staticmethod
    def _lies_in_split_corridor(
        own: ImitationBlob,
        target: ImitationBlob | ImitationPoint,
        child_radius: float,
        reach: float,
        direction: tuple[float, float],
    ) -> bool:
        target_x = float(getattr(target, "x"))
        target_y = float(getattr(target, "y"))
        dx = target_x - own.x
        dy = target_y - own.y
        forward = dx * direction[0] + dy * direction[1]
        lateral = abs(dx * direction[1] - dy * direction[0])
        return -0.1 <= forward <= reach and lateral <= child_radius

    @staticmethod
    def _child_is_safe(
        own: ImitationBlob,
        child_radius: float,
        observation: ImitationObservation,
    ) -> bool:
        for other in observation.visible_blobs:
            can_eat_child = (
                other.radius * other.radius
                >= child_radius * child_radius * EAT_SIZE_RATIO
            )
            if can_eat_child and math.hypot(other.x - own.x, other.y - own.y) < SPLIT_SAFETY_DISTANCE:
                return False
        return True
