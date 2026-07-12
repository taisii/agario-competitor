from __future__ import annotations

"""Stateful imitation of team 39 inferred from six official replays.

Team 39 submitted unit vectors, retained its previous heading until a
stochastic-looking turn, continuously bent away from visible predators, and
never requested a split.  The pseudo-random component below is deliberately
local and deterministic so benchmark reruns remain reproducible.
"""

import math

from lib.models.blob_model import BlobModel
from strategies.base import StrategyContext, StrategyDecision
from strategies.features import BlobRelation, extract_visible_features, normalise
from strategies.randomness import GOLDEN_RATIO_64, MASK_64, mix64, unit_interval


class ReplayTeam39Strategy:
    name = "replay_team_39"

    def __init__(
        self,
        *,
        heading_refresh_rate: float = 0.23,
        wall_buffer: float = 3.0,
    ) -> None:
        if not 0.0 <= heading_refresh_rate <= 1.0:
            raise ValueError("heading_refresh_rate must be between zero and one")
        if wall_buffer < 0.0:
            raise ValueError("wall_buffer cannot be negative")
        self._heading_refresh_rate = heading_refresh_rate
        self._wall_buffer = wall_buffer
        self._heading: tuple[float, float] | None = None
        self._rng_state = 0
        self._last_round: int | None = None

    def choose(self, context: StrategyContext) -> StrategyDecision:
        state = context.game.state
        features = extract_visible_features(context.game)
        round_number = int(getattr(state, "round", 0))

        if not features.own_blobs:
            return StrategyDecision(
                direction=self._heading or (1.0, 0.0),
                split=False,
                reason="dead_fallback",
            )

        respawned = (
            self._last_round is not None and round_number > self._last_round + 1
        )
        if self._heading is None or respawned:
            self._reset_heading(context, round_number)
            reason = "respawn_heading" if respawned else "initial_heading"
        elif features.predators:
            self._heading = self._predator_avoidance(features.predators)
            reason = "predator_avoidance"
        elif self._random_fraction() < self._heading_refresh_rate:
            self._heading = self._random_unit_vector()
            reason = "heading_refresh"
        else:
            reason = "inertia"

        arena_size = float(getattr(state.map, "size", 60.0) or 60.0)
        reflected = self._reflect_from_walls(
            self._heading,
            features.own_blobs,
            arena_size,
        )
        if reflected != self._heading:
            self._heading = reflected
            reason = "wall_reflection"

        self._last_round = round_number
        nearest_predator = features.nearest_predator
        return StrategyDecision(
            direction=self._heading,
            split=False,
            target_kind="escape" if features.predators else "heading",
            target_id=(
                f"{nearest_predator.blob.player_id}:{nearest_predator.blob.blob_id}"
                if nearest_predator is not None
                else None
            ),
            reason=reason,
            diagnostics={
                "source_team_id": 39,
                "predator_count": len(features.predators),
                "heading_refresh_rate": self._heading_refresh_rate,
                "wall_buffer": self._wall_buffer,
            },
        )

    def _reset_heading(self, context: StrategyContext, round_number: int) -> None:
        state = context.game.state
        own_blobs = tuple(state.me.blobs.values())
        total_mass = sum(blob.radius * blob.radius for blob in own_blobs)
        center_x = sum(
            blob.pos[0] * blob.radius * blob.radius for blob in own_blobs
        ) / max(total_mass, 1e-9)
        center_y = sum(
            blob.pos[1] * blob.radius * blob.radius for blob in own_blobs
        ) / max(total_mass, 1e-9)
        player_id = int(getattr(state.me, "player_id", 0))
        seed = (
            0x27D4EB2F165667C5
            ^ (player_id * 0xD6E8FEB86659FD93)
            ^ (round_number * 0xA0761D6478BD642F)
            ^ (round(center_x * 1_000_000) * 0xE7037ED1A0B428DB)
            ^ (round(center_y * 1_000_000) * 0x8EBC6AF09C88C6E3)
        ) & MASK_64
        self._rng_state = mix64(seed)
        self._heading = self._random_unit_vector()

    def _predator_avoidance(
        self,
        predators: tuple[BlobRelation, ...],
    ) -> tuple[float, float]:
        nearest = min(predators, key=lambda relation: relation.danger_margin)
        nearest_away = normalise(
            (
                nearest.nearest_own_blob.pos[0] - nearest.blob.pos[0],
                nearest.nearest_own_blob.pos[1] - nearest.blob.pos[1],
            )
        )
        field_x = 0.0
        field_y = 0.0
        for relation in predators:
            away = normalise(
                (
                    relation.nearest_own_blob.pos[0] - relation.blob.pos[0],
                    relation.nearest_own_blob.pos[1] - relation.blob.pos[1],
                )
            )
            weight = 1.0 / max(relation.danger_margin + 0.25, 0.25)
            field_x += away[0] * weight
            field_y += away[1] * weight
        field = normalise((field_x, field_y))
        previous = self._heading or nearest_away
        # Replay-fitted coefficients: inertia 0.454, nearest escape 0.536,
        # aggregate predator field 0.040.  The nearest escape can therefore
        # override a directly opposed stale heading.
        return normalise(
            (
                0.454 * previous[0]
                + 0.536 * nearest_away[0]
                + 0.040 * field[0],
                0.454 * previous[1]
                + 0.536 * nearest_away[1]
                + 0.040 * field[1],
            )
        )

    def _reflect_from_walls(
        self,
        direction: tuple[float, float],
        own_blobs: tuple[BlobModel, ...],
        arena_size: float,
    ) -> tuple[float, float]:
        left = min(blob.pos[0] - blob.radius for blob in own_blobs)
        right = min(arena_size - blob.pos[0] - blob.radius for blob in own_blobs)
        bottom = min(blob.pos[1] - blob.radius for blob in own_blobs)
        top = min(arena_size - blob.pos[1] - blob.radius for blob in own_blobs)
        x, y = direction
        if left <= self._wall_buffer and x < 0.0:
            x = -x
        if right <= self._wall_buffer and x > 0.0:
            x = -x
        if bottom <= self._wall_buffer and y < 0.0:
            y = -y
        if top <= self._wall_buffer and y > 0.0:
            y = -y
        return normalise((x, y))

    def _random_fraction(self) -> float:
        self._rng_state = (self._rng_state + GOLDEN_RATIO_64) & MASK_64
        return unit_interval(self._rng_state)

    def _random_unit_vector(self) -> tuple[float, float]:
        angle = math.tau * self._random_fraction()
        return (math.cos(angle), math.sin(angle))
