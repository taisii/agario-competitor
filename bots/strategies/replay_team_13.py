from __future__ import annotations

"""Replay-derived opponent policy for official team 13.

Team 13 appeared in three matches and issued one split in 3,690 turns.  The
direction profile passes the autonomous shadow gate.  The sole split has a
unique observable signature: one blob with mass 6.98, no predator or edible
virus, and edible prey 3.4 units away.  A mass floor of 6.5 and distance cap
of 3.5 isolate that event from every other recorded observation.
"""

import math

from lib.config.player import EAT_SIZE_RATIO
from simulation.rules import can_consume_virus
from strategies.base import StrategyContext, StrategyDecision
from strategies.replay_imitation import (
    ImitationBlob,
    ImitationObservation,
    _mass_center,
    _relations,
    _unit,
    observation_from_context,
    predict_direction,
)
from strategies.replay_profiles import PROFILES


TEAM_ID = 13
PROFILE = PROFILES[TEAM_ID]
MIN_SPLIT_MASS = 6.5
MAX_SPLIT_PREY_DISTANCE = 3.5


class ReplayTeam13Strategy:
    """Fitted direction policy with one replay-isolated prey split rule."""

    name = "replay_team_13"

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

        split_target = self._split_target(observation)
        if split_target is not None:
            center = _mass_center(observation.own_blobs)
            direction = _unit((split_target.x - center[0], split_target.y - center[1]))
            split = True
            reason = "team13_close_prey_split"
            target_kind = "prey"
            target_id = f"{split_target.player_id}:{split_target.blob_id}"
        else:
            direction = predict_direction(
                self.profile,
                observation,
                self._previous_direction,
            )
            split = False
            reason = "team13_fitted_direction"
            target_kind = "replay_imitation"
            target_id = "13"

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
                "respawned": respawned,
                "profile_validation_passed": self.profile.validation_passed,
            },
        )

    @staticmethod
    def _split_target(observation: ImitationObservation) -> ImitationBlob | None:
        if len(observation.own_blobs) != 1:
            return None
        own = observation.own_blobs[0]
        if own.radius * own.radius < MIN_SPLIT_MASS:
            return None
        predators, prey, _ = _relations(observation)
        if predators or not prey:
            return None
        edible_virus = any(
            can_consume_virus(
                own.radius,
                virus.radius,
                eat_size_ratio=EAT_SIZE_RATIO,
            )
            for virus in observation.visible_viruses
        )
        if edible_virus:
            return None

        target = min(prey, key=lambda blob: math.hypot(blob.x - own.x, blob.y - own.y))
        distance = math.hypot(target.x - own.x, target.y - own.y)
        child_mass = own.radius * own.radius / 2.0
        if child_mass < target.radius * target.radius * EAT_SIZE_RATIO:
            return None
        return target if distance <= MAX_SPLIT_PREY_DISTANCE else None
