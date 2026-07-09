from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from lib.config.arena import ARENA_SIZE
from lib.config.player import (
    BASE_PLAYER_SPEED,
    EAT_SIZE_RATIO,
    FOOD_RADIUS,
    MIN_PLAYER_SPEED,
    PLAYER_SPEED_RADIUS_FACTOR,
)
from lib.models.food_model import FoodModel
from lib.models.virus_model import VirusModel
from strategies.base import StrategyContext, StrategyDecision
from strategies.features import normalise, squared_distance, vector_from_to


@dataclass(frozen=True)
class SimBlob:
    player_id: int
    blob_id: int
    x: float
    y: float
    radius: float

    @property
    def key(self) -> tuple[int, int]:
        return (self.player_id, self.blob_id)

    @property
    def mass(self) -> float:
        return self.radius * self.radius


@dataclass(frozen=True)
class BeamNode:
    x: float
    y: float
    radius: float
    score: float
    first_direction: tuple[float, float]
    last_direction: tuple[float, float]
    blobs: tuple[SimBlob, ...]
    eaten_food_ids: frozenset[int] = field(default_factory=frozenset)
    captured_blob_ids: frozenset[tuple[int, int]] = field(default_factory=frozenset)


@dataclass(frozen=True)
class RolloutResult:
    node: BeamNode
    fatal: bool
    reason: str


class BeamSurvivalStrategy:
    name = "beam_survival"

    def __init__(
        self,
        depth: int | None = None,
        width: int | None = None,
        angular_samples: int | None = None,
    ) -> None:
        self.depth = depth if depth is not None else int(os.environ.get("BOT_BEAM_DEPTH", "5"))
        self.width = width if width is not None else int(os.environ.get("BOT_BEAM_WIDTH", "4"))
        self.angular_samples = (
            angular_samples
            if angular_samples is not None
            else int(os.environ.get("BOT_BEAM_ANGULAR_SAMPLES", "8"))
        )
        self.turn_penalty_weight = float(os.environ.get("BOT_BEAM_TURN_PENALTY_WEIGHT", "1.4"))
        self.keep_direction_candidate = _env_bool("BOT_BEAM_KEEP_DIRECTION_CANDIDATE", default=True)
        self.previous_direction: tuple[float, float] = (1.0, 0.0)

    def choose(self, context: StrategyContext) -> StrategyDecision:
        state = context.game.state
        arena_size = state.map.size or ARENA_SIZE
        start = BeamNode(
            x=state.me.x,
            y=state.me.y,
            radius=state.me.radius,
            score=0.0,
            first_direction=(1.0, 0.0),
            last_direction=self.previous_direction,
            blobs=tuple(
                SimBlob(
                    player_id=blob.player_id,
                    blob_id=blob.blob_id,
                    x=blob.pos[0],
                    y=blob.pos[1],
                    radius=blob.radius,
                )
                for blob in state.visible_blobs
            ),
        )
        foods = tuple(state.visible_food)
        viruses = tuple(state.visible_viruses)

        beam = [start]
        best_rejected: RolloutResult | None = None
        for depth_index in range(self.depth):
            candidates: list[BeamNode] = []
            for node in beam:
                for direction in self._candidate_directions(node, foods, arena_size):
                    result = self._step(
                        node=node,
                        direction=direction,
                        foods=foods,
                        viruses=viruses,
                        arena_size=arena_size,
                        is_first_step=depth_index == 0,
                    )
                    if result.fatal:
                        if best_rejected is None or result.node.score > best_rejected.node.score:
                            best_rejected = result
                        continue
                    candidates.append(result.node)

            if not candidates:
                fallback = best_rejected.node if best_rejected is not None else start
                fallback_direction = normalise(fallback.first_direction)
                self.previous_direction = fallback_direction
                return StrategyDecision(
                    direction=fallback_direction,
                    target_kind="escape",
                    reason="all_beam_paths_fatal",
                    score=fallback.score,
                    diagnostics={
                        "beam_depth": self.depth,
                        "beam_width": self.width,
                        "turn_penalty_weight": self.turn_penalty_weight,
                        "keep_direction_candidate": self.keep_direction_candidate,
                    },
                )

            candidates.sort(key=lambda node: node.score, reverse=True)
            beam = candidates[: self.width]

        best = max(beam, key=lambda node: node.score)
        chosen_direction = normalise(best.first_direction)
        self.previous_direction = chosen_direction
        return StrategyDecision(
            direction=chosen_direction,
            target_kind="beam",
            reason="beam_rollout",
            score=best.score,
            diagnostics={
                "beam_depth": self.depth,
                "beam_width": self.width,
                "beam_angular_samples": self.angular_samples,
                "turn_penalty_weight": self.turn_penalty_weight,
                "keep_direction_candidate": self.keep_direction_candidate,
                "projected_x": best.x,
                "projected_y": best.y,
                "projected_radius": best.radius,
                "projected_captured_blobs": len(best.captured_blob_ids),
                "projected_eaten_food": len(best.eaten_food_ids),
            },
        )

    def _candidate_directions(
        self,
        node: BeamNode,
        foods: tuple[FoodModel, ...],
        arena_size: float,
    ) -> tuple[tuple[float, float], ...]:
        directions = []
        if self.keep_direction_candidate:
            directions.append(node.last_direction)
        directions.extend(
            (
                math.cos(2.0 * math.pi * index / self.angular_samples),
                math.sin(2.0 * math.pi * index / self.angular_samples),
            )
            for index in range(self.angular_samples)
        )

        predators = [
            blob
            for blob in node.blobs
            if blob.radius >= node.radius * EAT_SIZE_RATIO
        ]
        if predators:
            away_x = 0.0
            away_y = 0.0
            for predator in predators:
                margin = max(math.hypot(node.x - predator.x, node.y - predator.y) - predator.radius, 0.2)
                away_x += (node.x - predator.x) / margin
                away_y += (node.y - predator.y) / margin
            directions.append(normalise((away_x, away_y)))

        edible_blobs = [
            blob
            for blob in node.blobs
            if node.radius >= blob.radius * EAT_SIZE_RATIO
        ]
        if edible_blobs:
            target = min(
                edible_blobs,
                key=lambda blob: squared_distance((node.x, node.y), (blob.x, blob.y)),
            )
            directions.append(normalise(vector_from_to((node.x, node.y), (target.x, target.y))))

        uneaten_food = [
            food for food in foods if food.food_id not in node.eaten_food_ids
        ]
        if uneaten_food:
            food = min(
                uneaten_food,
                key=lambda item: squared_distance((node.x, node.y), item.pos),
            )
            directions.append(normalise(vector_from_to((node.x, node.y), food.pos)))

        center = (arena_size / 2.0, arena_size / 2.0)
        directions.append(normalise(vector_from_to((node.x, node.y), center)))

        deduped: dict[tuple[int, int], tuple[float, float]] = {}
        for direction in directions:
            unit = normalise(direction)
            if unit == (0.0, 0.0):
                continue
            key = (round(unit[0] * 1000), round(unit[1] * 1000))
            deduped[key] = unit
        return tuple(deduped.values())

    def _step(
        self,
        *,
        node: BeamNode,
        direction: tuple[float, float],
        foods: tuple[FoodModel, ...],
        viruses: tuple[VirusModel, ...],
        arena_size: float,
        is_first_step: bool,
    ) -> RolloutResult:
        unit = normalise(direction)
        speed = self._movement_speed(node.radius)
        unclamped_x = node.x + unit[0] * speed
        unclamped_y = node.y + unit[1] * speed
        next_x = self._clamp(unclamped_x, node.radius, arena_size - node.radius)
        next_y = self._clamp(unclamped_y, node.radius, arena_size - node.radius)

        moved_blobs = tuple(
            self._predict_blob(blob, own_x=next_x, own_y=next_y, own_radius=node.radius)
            for blob in node.blobs
        )
        radius = node.radius
        score = node.score
        eaten_food_ids = set(node.eaten_food_ids)
        captured_blob_ids = set(node.captured_blob_ids)
        surviving_blobs: list[SimBlob] = []

        wall_penalty = self._wall_penalty(next_x, next_y, radius, arena_size)
        score -= wall_penalty
        if next_x != unclamped_x or next_y != unclamped_y:
            score -= 25.0

        score -= self._turn_penalty(node.last_direction, unit)

        for food in foods:
            if food.food_id in eaten_food_ids:
                continue
            dist2 = squared_distance((next_x, next_y), food.pos)
            if dist2 <= radius * radius:
                eaten_food_ids.add(food.food_id)
                radius = math.sqrt(radius * radius + FOOD_RADIUS * FOOD_RADIUS)
                score += 3.0

        nearest_food_distance = self._nearest_food_distance(
            next_x,
            next_y,
            foods,
            eaten_food_ids,
        )
        if nearest_food_distance is not None:
            score += 0.8 / (nearest_food_distance + 1.0)

        for virus in viruses:
            score -= self._virus_penalty(next_x, next_y, radius, virus)

        for blob in moved_blobs:
            dist = math.hypot(next_x - blob.x, next_y - blob.y)
            if blob.radius >= radius * EAT_SIZE_RATIO:
                margin = dist - blob.radius
                if margin <= 0.1:
                    return RolloutResult(
                        node=self._with_score(node, direction, score - 1_000.0, is_first_step),
                        fatal=True,
                        reason="predator_collision",
                    )
                score -= 18.0 / (margin + 0.35)
                surviving_blobs.append(blob)
                continue

            if radius >= blob.radius * EAT_SIZE_RATIO:
                if dist <= radius:
                    captured_blob_ids.add(blob.key)
                    radius = math.sqrt(radius * radius + blob.radius * blob.radius)
                    score += 20.0 * blob.mass
                    continue
                score += 2.0 / (max(dist - radius, 0.0) + 1.0)
                surviving_blobs.append(blob)
                continue

            overlap_margin = dist - (radius + blob.radius)
            if overlap_margin < 1.0:
                score -= (1.0 - overlap_margin) * 1.5
            surviving_blobs.append(blob)

        next_node = BeamNode(
            x=next_x,
            y=next_y,
            radius=radius,
            score=score,
            first_direction=unit if is_first_step else node.first_direction,
            last_direction=unit,
            blobs=tuple(surviving_blobs),
            eaten_food_ids=frozenset(eaten_food_ids),
            captured_blob_ids=frozenset(captured_blob_ids),
        )
        return RolloutResult(node=next_node, fatal=False, reason="")

    def _predict_blob(
        self,
        blob: SimBlob,
        *,
        own_x: float,
        own_y: float,
        own_radius: float,
    ) -> SimBlob:
        if blob.radius >= own_radius * EAT_SIZE_RATIO:
            direction = normalise((own_x - blob.x, own_y - blob.y))
        elif own_radius >= blob.radius * EAT_SIZE_RATIO:
            direction = normalise((blob.x - own_x, blob.y - own_y))
        else:
            direction = (0.0, 0.0)

        speed = self._movement_speed(blob.radius)
        return SimBlob(
            player_id=blob.player_id,
            blob_id=blob.blob_id,
            x=blob.x + direction[0] * speed,
            y=blob.y + direction[1] * speed,
            radius=blob.radius,
        )

    def _movement_speed(self, radius: float) -> float:
        return max(
            MIN_PLAYER_SPEED,
            BASE_PLAYER_SPEED / (1.0 + radius * PLAYER_SPEED_RADIUS_FACTOR),
        )

    def _wall_penalty(
        self,
        x: float,
        y: float,
        radius: float,
        arena_size: float,
    ) -> float:
        margin = min(x - radius, y - radius, arena_size - radius - x, arena_size - radius - y)
        if margin >= 4.0:
            return 0.0
        return (4.0 - margin) ** 2 * 1.8

    def _virus_penalty(
        self,
        x: float,
        y: float,
        radius: float,
        virus: VirusModel,
    ) -> float:
        can_consume_virus = radius * radius > virus.radius * virus.radius * EAT_SIZE_RATIO
        if not can_consume_virus:
            return 0.0
        distance = math.hypot(x - virus.pos[0], y - virus.pos[1])
        margin = distance - (radius + virus.radius)
        if margin >= 3.0:
            return 0.0
        return (3.0 - margin) ** 2 * 2.0

    def _nearest_food_distance(
        self,
        x: float,
        y: float,
        foods: tuple[FoodModel, ...],
        eaten_food_ids: set[int],
    ) -> float | None:
        distances = [
            math.hypot(x - food.pos[0], y - food.pos[1])
            for food in foods
            if food.food_id not in eaten_food_ids
        ]
        return min(distances) if distances else None

    def _with_score(
        self,
        node: BeamNode,
        direction: tuple[float, float],
        score: float,
        is_first_step: bool,
    ) -> BeamNode:
        return BeamNode(
            x=node.x,
            y=node.y,
            radius=node.radius,
            score=score,
            first_direction=normalise(direction) if is_first_step else node.first_direction,
            last_direction=node.last_direction,
            blobs=node.blobs,
            eaten_food_ids=node.eaten_food_ids,
            captured_blob_ids=node.captured_blob_ids,
        )

    def _clamp(self, value: float, lower: float, upper: float) -> float:
        return min(max(value, lower), upper)

    def _turn_penalty(
        self,
        previous_direction: tuple[float, float],
        next_direction: tuple[float, float],
    ) -> float:
        previous = normalise(previous_direction)
        current = normalise(next_direction)
        if previous == (0.0, 0.0) or current == (0.0, 0.0):
            return 0.0
        dot = max(-1.0, min(1.0, previous[0] * current[0] + previous[1] * current[1]))
        return (1.0 - dot) * self.turn_penalty_weight


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}
