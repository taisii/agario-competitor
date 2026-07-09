from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace

from lib.config.arena import ARENA_SIZE
from lib.config.player import (
    BASE_PLAYER_SPEED,
    EAT_SIZE_RATIO,
    FOOD_RADIUS,
    MASS_DECAY_RATE,
    MIN_PLAYER_SPEED,
    PLAYER_SPEED_RADIUS_FACTOR,
    STARTING_RADIUS,
)
from lib.models.food_model import FoodModel
from lib.models.virus_model import VirusModel
from strategies.base import StrategyContext, StrategyDecision
from strategies.features import can_eat_player_blob, normalise, squared_distance


@dataclass(frozen=True)
class SimOwnBlob:
    blob_id: int
    x: float
    y: float
    radius: float

    @property
    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def mass(self) -> float:
        return self.radius * self.radius


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
    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def mass(self) -> float:
        return self.radius * self.radius


@dataclass(frozen=True)
class BeamNode:
    own_blobs: tuple[SimOwnBlob, ...]
    enemies: tuple[SimBlob, ...]
    score: float
    first_direction: tuple[float, float]
    last_direction: tuple[float, float]
    eaten_food_ids: frozenset[int] = field(default_factory=frozenset)
    captured_blob_ids: frozenset[tuple[int, int]] = field(default_factory=frozenset)

    @property
    def total_mass(self) -> float:
        return sum(blob.mass for blob in self.own_blobs)

    @property
    def center(self) -> tuple[float, float]:
        if not self.own_blobs:
            return (0.0, 0.0)
        mass = self.total_mass
        return (
            sum(blob.x * blob.mass for blob in self.own_blobs) / mass,
            sum(blob.y * blob.mass for blob in self.own_blobs) / mass,
        )


@dataclass(frozen=True)
class RolloutResult:
    node: BeamNode
    fatal: bool
    reason: str


class BeamSurvivalStrategy:
    """Conservative multi-blob beam search.

    The previous implementation represented a split player as one aggregate
    circle at the mass centroid. That virtual circle could eat enemies that no
    real fragment could eat and could miss threats to a small fragment. This
    rollout keeps every own blob while still applying one shared move, matching
    the engine's control model.
    """

    name = "beam_survival"

    def __init__(
        self,
        depth: int | None = None,
        width: int | None = None,
        angular_samples: int | None = None,
    ) -> None:
        self.depth = max(
            1,
            depth if depth is not None else int(os.environ.get("BOT_BEAM_DEPTH", "5")),
        )
        self.width = max(
            1,
            width if width is not None else int(os.environ.get("BOT_BEAM_WIDTH", "4")),
        )
        self.angular_samples = max(
            4,
            angular_samples
            if angular_samples is not None
            else int(os.environ.get("BOT_BEAM_ANGULAR_SAMPLES", "8")),
        )
        self.turn_penalty_weight = float(
            os.environ.get("BOT_BEAM_TURN_PENALTY_WEIGHT", "1.4")
        )
        self.keep_direction_candidate = _env_bool(
            "BOT_BEAM_KEEP_DIRECTION_CANDIDATE",
            default=True,
        )
        self.previous_direction: tuple[float, float] = (1.0, 0.0)

    def choose(self, context: StrategyContext) -> StrategyDecision:
        state = context.game.state
        own_blobs = tuple(
            SimOwnBlob(
                blob_id=blob.blob_id,
                x=blob.pos[0],
                y=blob.pos[1],
                radius=blob.radius,
            )
            for blob in state.me.blobs.values()
        )
        if not own_blobs:
            return StrategyDecision(
                direction=self.previous_direction,
                reason="dead_fallback",
            )

        start = BeamNode(
            own_blobs=own_blobs,
            enemies=tuple(
                SimBlob(
                    player_id=blob.player_id,
                    blob_id=blob.blob_id,
                    x=blob.pos[0],
                    y=blob.pos[1],
                    radius=blob.radius,
                )
                for blob in state.visible_blobs
            ),
            score=0.0,
            first_direction=self.previous_direction,
            last_direction=self.previous_direction,
        )
        foods = tuple(state.visible_food)
        viruses = tuple(state.visible_viruses)
        arena_size = state.map.size or ARENA_SIZE

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
                direction = normalise(fallback.first_direction) or self.previous_direction
                self.previous_direction = direction
                return StrategyDecision(
                    direction=direction,
                    target_kind="escape",
                    reason="all_beam_paths_fatal",
                    score=fallback.score,
                    diagnostics=self._diagnostics(fallback),
                )

            candidates.sort(key=lambda node: node.score, reverse=True)
            beam = candidates[: self.width]

        best = max(beam, key=lambda node: node.score)
        direction = normalise(best.first_direction) or self.previous_direction
        self.previous_direction = direction
        return StrategyDecision(
            direction=direction,
            target_kind="beam",
            reason="beam_rollout",
            score=best.score,
            diagnostics=self._diagnostics(best),
        )

    def _candidate_directions(
        self,
        node: BeamNode,
        foods: tuple[FoodModel, ...],
        arena_size: float,
    ) -> tuple[tuple[float, float], ...]:
        directions: list[tuple[float, float]] = []
        if self.keep_direction_candidate:
            directions.append(node.last_direction)
        directions.extend(
            (
                math.cos(2.0 * math.pi * index / self.angular_samples),
                math.sin(2.0 * math.pi * index / self.angular_samples),
            )
            for index in range(self.angular_samples)
        )

        escape = self._escape_vector(node)
        if escape != (0.0, 0.0):
            directions.append(escape)

        prey_pair = self._nearest_prey_pair(node)
        if prey_pair is not None:
            own, enemy = prey_pair
            directions.append(normalise((enemy.x - own.x, enemy.y - own.y)))

        food_pair = self._nearest_food_pair(node, foods)
        if food_pair is not None:
            own, food = food_pair
            directions.append(normalise((food.pos[0] - own.x, food.pos[1] - own.y)))

        center = node.center
        directions.append(
            normalise((arena_size / 2.0 - center[0], arena_size / 2.0 - center[1]))
        )
        return _dedupe_directions(directions)

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
        unit = normalise(direction) or node.last_direction
        score = node.score - self._turn_penalty(node.last_direction, unit)
        own_blobs: list[SimOwnBlob] = []
        hit_wall = False
        for blob in node.own_blobs:
            unclamped_x = blob.x + unit[0] * _speed(blob.radius)
            unclamped_y = blob.y + unit[1] * _speed(blob.radius)
            x = _clamp(unclamped_x, blob.radius, arena_size - blob.radius)
            y = _clamp(unclamped_y, blob.radius, arena_size - blob.radius)
            if x != unclamped_x or y != unclamped_y:
                hit_wall = True
            own_blobs.append(
                replace(blob, x=x, y=y, radius=_decayed_radius(blob.radius))
            )
        primary = max(own_blobs, key=lambda blob: blob.radius)
        score -= self._wall_penalty(primary, arena_size)
        if hit_wall:
            score -= 25.0

        enemies = [
            self._predict_enemy(enemy, own_blobs, arena_size)
            for enemy in node.enemies
        ]
        eaten_food_ids = set(node.eaten_food_ids)
        captured_blob_ids = set(node.captured_blob_ids)

        for food in foods:
            if food.food_id in eaten_food_ids:
                continue
            candidates = [
                index
                for index, blob in enumerate(own_blobs)
                if squared_distance(blob.pos, food.pos) <= blob.radius * blob.radius
            ]
            if not candidates:
                continue
            index = max(candidates, key=lambda item: own_blobs[item].radius)
            eater = own_blobs[index]
            own_blobs[index] = replace(
                eater,
                radius=math.sqrt(eater.mass + FOOD_RADIUS * FOOD_RADIUS),
            )
            eaten_food_ids.add(food.food_id)
            score += 3.0

        for virus in viruses:
            for blob in own_blobs:
                if blob.mass <= virus.radius * virus.radius * EAT_SIZE_RATIO:
                    continue
                clearance = math.dist(blob.pos, virus.pos) - blob.radius - virus.radius
                if clearance <= 0.0:
                    score -= 220.0 + 30.0 * blob.mass
                elif clearance < 3.0:
                    score -= (3.0 - clearance) ** 2 * 2.0

        own_blobs, enemies, interaction_score = self._resolve_interactions(
            own_blobs,
            enemies,
            captured_blob_ids,
        )
        score += interaction_score
        if not own_blobs:
            dead = self._replace_node(
                node,
                own_blobs=(),
                enemies=tuple(enemies),
                score=score - 10_000.0,
                direction=unit,
                is_first_step=is_first_step,
                eaten_food_ids=eaten_food_ids,
                captured_blob_ids=captured_blob_ids,
            )
            return RolloutResult(dead, fatal=True, reason="all_blobs_eaten")

        score += self._position_score(own_blobs, enemies, foods, eaten_food_ids)
        next_node = self._replace_node(
            node,
            own_blobs=tuple(own_blobs),
            enemies=tuple(enemies),
            score=score,
            direction=unit,
            is_first_step=is_first_step,
            eaten_food_ids=eaten_food_ids,
            captured_blob_ids=captured_blob_ids,
        )
        return RolloutResult(next_node, fatal=False, reason="")

    def _predict_enemy(
        self,
        enemy: SimBlob,
        own_blobs: list[SimOwnBlob],
        arena_size: float,
    ) -> SimBlob:
        vulnerable = [
            blob
            for blob in own_blobs
            if can_eat_player_blob(enemy.radius, blob.radius)
        ]
        hunters = [
            blob
            for blob in own_blobs
            if can_eat_player_blob(blob.radius, enemy.radius)
        ]
        if vulnerable:
            target = min(vulnerable, key=lambda blob: squared_distance(enemy.pos, blob.pos))
            direction = normalise((target.x - enemy.x, target.y - enemy.y))
        elif hunters:
            hunter = min(hunters, key=lambda blob: squared_distance(enemy.pos, blob.pos))
            direction = normalise((enemy.x - hunter.x, enemy.y - hunter.y))
        else:
            direction = (0.0, 0.0)
        radius = _decayed_radius(enemy.radius)
        return replace(
            enemy,
            x=_clamp(enemy.x + direction[0] * _speed(enemy.radius), radius, arena_size - radius),
            y=_clamp(enemy.y + direction[1] * _speed(enemy.radius), radius, arena_size - radius),
            radius=radius,
        )

    def _resolve_interactions(
        self,
        own_blobs: list[SimOwnBlob],
        enemies: list[SimBlob],
        captured_blob_ids: set[tuple[int, int]],
    ) -> tuple[list[SimOwnBlob], list[SimBlob], float]:
        own_by_id = {blob.blob_id: blob for blob in own_blobs}
        enemy_by_id = {blob.key: blob for blob in enemies}
        score = 0.0

        while True:
            actors = [
                (blob.radius, 0, blob.blob_id)
                for blob in own_by_id.values()
            ] + [
                (blob.radius, 1, blob.key)
                for blob in enemy_by_id.values()
            ]
            actors.sort(key=lambda item: -item[0])
            changed = False
            for _radius, team, key in actors:
                if team == 0:
                    eater = own_by_id.get(key)
                    if eater is None:
                        continue
                    targets = sorted(enemy_by_id.values(), key=lambda blob: -blob.radius)
                    for target in targets:
                        if not _can_eat(eater.radius, target.radius, eater.pos, target.pos):
                            continue
                        own_by_id[eater.blob_id] = replace(
                            eater,
                            radius=math.sqrt(eater.mass + target.mass),
                        )
                        del enemy_by_id[target.key]
                        captured_blob_ids.add(target.key)
                        score += 20.0 * target.mass
                        changed = True
                        break
                else:
                    eater = enemy_by_id.get(key)
                    if eater is None:
                        continue
                    targets = sorted(own_by_id.values(), key=lambda blob: -blob.radius)
                    for target in targets:
                        if not _can_eat(eater.radius, target.radius, eater.pos, target.pos):
                            continue
                        enemy_by_id[eater.key] = replace(
                            eater,
                            radius=math.sqrt(eater.mass + target.mass),
                        )
                        del own_by_id[target.blob_id]
                        score -= 280.0 + 55.0 * target.mass
                        changed = True
                        break
                if changed:
                    break
            if not changed:
                break

        return list(own_by_id.values()), list(enemy_by_id.values()), score

    def _position_score(
        self,
        own_blobs: list[SimOwnBlob],
        enemies: list[SimBlob],
        foods: tuple[FoodModel, ...],
        eaten_food_ids: set[int],
    ) -> float:
        score = 0.0
        for own in own_blobs:
            for enemy in enemies:
                distance = math.dist(own.pos, enemy.pos)
                if can_eat_player_blob(enemy.radius, own.radius):
                    margin = distance - enemy.radius
                    score -= 20.0 / (max(margin, 0.0) + 0.35)
                elif can_eat_player_blob(own.radius, enemy.radius):
                    score += min(4.0, enemy.mass / (max(distance - own.radius, 0.0) + 1.0))

        remaining_food = [food for food in foods if food.food_id not in eaten_food_ids]
        if remaining_food:
            nearest_distance = min(
                math.dist(own.pos, food.pos)
                for own in own_blobs
                for food in remaining_food
            )
            score += 0.8 / (nearest_distance + 1.0)
        return score

    def _wall_penalty(self, blob: SimOwnBlob, arena_size: float) -> float:
        margin = min(
            blob.x - blob.radius,
            blob.y - blob.radius,
            arena_size - blob.radius - blob.x,
            arena_size - blob.radius - blob.y,
        )
        if margin >= 4.0:
            return 0.0
        return (4.0 - margin) ** 2 * 1.8

    def _escape_vector(self, node: BeamNode) -> tuple[float, float]:
        x = 0.0
        y = 0.0
        for own in node.own_blobs:
            for enemy in node.enemies:
                if not can_eat_player_blob(enemy.radius, own.radius):
                    continue
                margin = max(math.dist(own.pos, enemy.pos) - enemy.radius, 0.2)
                away = normalise((own.x - enemy.x, own.y - enemy.y))
                weight = own.mass / margin
                x += away[0] * weight
                y += away[1] * weight
        return normalise((x, y))

    def _nearest_prey_pair(
        self,
        node: BeamNode,
    ) -> tuple[SimOwnBlob, SimBlob] | None:
        pairs = [
            (own, enemy)
            for own in node.own_blobs
            for enemy in node.enemies
            if can_eat_player_blob(own.radius, enemy.radius)
        ]
        if not pairs:
            return None
        return min(
            pairs,
            key=lambda pair: squared_distance(pair[0].pos, pair[1].pos),
        )

    def _nearest_food_pair(
        self,
        node: BeamNode,
        foods: tuple[FoodModel, ...],
    ) -> tuple[SimOwnBlob, FoodModel] | None:
        pairs = [
            (own, food)
            for own in node.own_blobs
            for food in foods
            if food.food_id not in node.eaten_food_ids
        ]
        if not pairs:
            return None
        return min(
            pairs,
            key=lambda pair: squared_distance(pair[0].pos, pair[1].pos),
        )

    def _replace_node(
        self,
        node: BeamNode,
        *,
        own_blobs: tuple[SimOwnBlob, ...],
        enemies: tuple[SimBlob, ...],
        score: float,
        direction: tuple[float, float],
        is_first_step: bool,
        eaten_food_ids: set[int],
        captured_blob_ids: set[tuple[int, int]],
    ) -> BeamNode:
        return BeamNode(
            own_blobs=own_blobs,
            enemies=enemies,
            score=score,
            first_direction=direction if is_first_step else node.first_direction,
            last_direction=direction,
            eaten_food_ids=frozenset(eaten_food_ids),
            captured_blob_ids=frozenset(captured_blob_ids),
        )

    def _turn_penalty(
        self,
        previous_direction: tuple[float, float],
        next_direction: tuple[float, float],
    ) -> float:
        previous = normalise(previous_direction)
        current = normalise(next_direction)
        dot = max(-1.0, min(1.0, previous[0] * current[0] + previous[1] * current[1]))
        return (1.0 - dot) * self.turn_penalty_weight

    def _diagnostics(self, node: BeamNode) -> dict[str, object]:
        return {
            "beam_depth": self.depth,
            "beam_width": self.width,
            "beam_angular_samples": self.angular_samples,
            "turn_penalty_weight": self.turn_penalty_weight,
            "keep_direction_candidate": self.keep_direction_candidate,
            "projected_blob_count": len(node.own_blobs),
            "projected_total_mass": node.total_mass if node.own_blobs else 0.0,
            "projected_captured_blobs": len(node.captured_blob_ids),
            "projected_eaten_food": len(node.eaten_food_ids),
        }


def _can_eat(
    eater_radius: float,
    target_radius: float,
    eater_pos: tuple[float, float],
    target_pos: tuple[float, float],
) -> bool:
    return (
        can_eat_player_blob(eater_radius, target_radius)
        and squared_distance(eater_pos, target_pos) <= eater_radius * eater_radius
    )


def _speed(radius: float) -> float:
    return max(
        MIN_PLAYER_SPEED,
        BASE_PLAYER_SPEED / (1.0 + radius * PLAYER_SPEED_RADIUS_FACTOR),
    )


def _decayed_radius(radius: float) -> float:
    mass = radius * radius
    minimum_mass = STARTING_RADIUS * STARTING_RADIUS
    if mass <= minimum_mass:
        return radius
    return math.sqrt(max(minimum_mass, mass * (1.0 - MASS_DECAY_RATE)))


def _dedupe_directions(
    directions: list[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    deduped: dict[tuple[int, int], tuple[float, float]] = {}
    for direction in directions:
        unit = normalise(direction)
        if unit == (0.0, 0.0):
            continue
        deduped[(round(unit[0] * 1000), round(unit[1] * 1000))] = unit
    return tuple(deduped.values())


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}
