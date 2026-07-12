from __future__ import annotations

"""Beam-search strategy family and its configurable scoring profiles."""

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
from strategies.features import (
    can_eat_player_blob,
    normalise,
    squared_distance,
    virus_center_clearance,
)


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
            grown_radius = math.sqrt(eater.mass + FOOD_RADIUS * FOOD_RADIUS)
            own_blobs[index] = replace(
                eater,
                x=_clamp(eater.x, grown_radius, arena_size - grown_radius),
                y=_clamp(eater.y, grown_radius, arena_size - grown_radius),
                radius=grown_radius,
            )
            eaten_food_ids.add(food.food_id)
            score += 3.0

        for virus in viruses:
            for blob in own_blobs:
                if blob.mass <= virus.radius * virus.radius * EAT_SIZE_RATIO:
                    continue
                clearance = virus_center_clearance(
                    blob.pos,
                    blob.radius,
                    virus.pos,
                )
                if clearance <= 0.0:
                    score -= 220.0 + 30.0 * blob.mass
                elif clearance < 3.0:
                    score -= (3.0 - clearance) ** 2 * 2.0

        own_blobs, enemies, interaction_score = self._resolve_interactions(
            own_blobs,
            enemies,
            captured_blob_ids,
            arena_size,
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
        arena_size: float,
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
                        grown_radius = math.sqrt(eater.mass + target.mass)
                        own_by_id[eater.blob_id] = replace(
                            eater,
                            x=_clamp(eater.x, grown_radius, arena_size - grown_radius),
                            y=_clamp(eater.y, grown_radius, arena_size - grown_radius),
                            radius=grown_radius,
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
                        grown_radius = math.sqrt(eater.mass + target.mass)
                        enemy_by_id[eater.key] = replace(
                            eater,
                            x=_clamp(eater.x, grown_radius, arena_size - grown_radius),
                            y=_clamp(eater.y, grown_radius, arena_size - grown_radius),
                            radius=grown_radius,
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



import math
import os
from dataclasses import dataclass, field

from lib.config.arena import ARENA_SIZE, MAX_BLOB_COUNT
from lib.config.player import (
    BASE_PLAYER_SPEED,
    EAT_SIZE_RATIO,
    FOOD_RADIUS,
    MASS_DECAY_RATE,
    MERGE_ATTRACTION_SPEED,
    MIN_PLAYER_SPEED,
    PLAYER_SPEED_RADIUS_FACTOR,
    SAME_PLAYER_OVERLAP_EPSILON,
    SPLIT_COOLDOWN_FRAMES,
    SPLIT_EJECT_DRAG,
    SPLIT_EJECT_SPEED,
    SPLIT_MIN_MASS,
    STARTING_RADIUS,
)
from lib.models.food_model import FoodModel
from lib.models.virus_model import VirusModel
from strategies.base import StrategyContext, StrategyDecision
from strategies.features import (
    can_eat_player_blob,
    normalise,
    squared_distance,
    virus_center_clearance,
)


SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class Action:
    direction: tuple[float, float]
    split: bool = False
    reason: str = "move"


@dataclass(frozen=True)
class HunterSimOwnBlob:
    blob_id: int
    x: float
    y: float
    radius: float
    merge_cooldown: int = 0
    eject_vx: float = 0.0
    eject_vy: float = 0.0

    @property
    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def mass(self) -> float:
        return self.radius * self.radius


@dataclass(frozen=True)
class SimEnemyBlob:
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
class HunterBeamNode:
    own_blobs: tuple[HunterSimOwnBlob, ...]
    enemies: tuple[SimEnemyBlob, ...]
    score: float
    first_direction: tuple[float, float]
    first_split: bool
    last_direction: tuple[float, float]
    eaten_food_ids: frozenset[int] = field(default_factory=frozenset)
    captured_blob_ids: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    consumed_virus_ids: frozenset[int] = field(default_factory=frozenset)

    @property
    def primary(self) -> HunterSimOwnBlob:
        return max(self.own_blobs, key=lambda blob: blob.radius)

    @property
    def total_mass(self) -> float:
        return sum(blob.mass for blob in self.own_blobs)


@dataclass(frozen=True)
class StepResult:
    node: HunterBeamNode
    fatal: bool
    reason: str


class BeamHunterStrategy:
    name = "beam_hunter"

    def __init__(
        self,
        depth: int | None = None,
        width: int | None = None,
        angular_samples: int | None = None,
    ) -> None:
        self.depth = depth if depth is not None else int(os.environ.get("BOT_HUNTER_BEAM_DEPTH", "4"))
        self.width = width if width is not None else int(os.environ.get("BOT_HUNTER_BEAM_WIDTH", "8"))
        self.angular_samples = (
            angular_samples
            if angular_samples is not None
            else int(os.environ.get("BOT_HUNTER_BEAM_ANGULAR_SAMPLES", "12"))
        )
        self.previous_direction: tuple[float, float] = (1.0, 0.0)

    def choose(self, context: StrategyContext) -> StrategyDecision:
        state = context.game.state
        own_blobs = tuple(
            HunterSimOwnBlob(
                blob_id=blob.blob_id,
                x=blob.pos[0],
                y=blob.pos[1],
                radius=blob.radius,
                merge_cooldown=blob.merge_cooldown,
            )
            for blob in state.me.blobs.values()
        )
        if not own_blobs:
            return StrategyDecision(direction=self.previous_direction, reason="dead_fallback")

        start = HunterBeamNode(
            own_blobs=own_blobs,
            enemies=tuple(
                SimEnemyBlob(
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
            first_split=False,
            last_direction=self.previous_direction,
        )
        foods = tuple(state.visible_food)
        viruses = tuple(state.visible_viruses)
        arena_size = state.map.size or ARENA_SIZE

        beam = [start]
        best_rejected: StepResult | None = None
        for depth_index in range(max(self.depth, 1)):
            candidates: list[HunterBeamNode] = []
            for node in beam:
                for action in self._candidate_actions(node, foods, viruses, arena_size):
                    result = self._step(
                        node=node,
                        action=action,
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
                fallback = best_rejected.node if best_rejected else start
                direction = normalise(fallback.first_direction)
                if direction == (0.0, 0.0):
                    direction = self.previous_direction
                self.previous_direction = direction
                return StrategyDecision(
                    direction=direction,
                    split=False,
                    target_kind="escape",
                    reason="all_hunter_beam_paths_fatal",
                    score=fallback.score,
                    diagnostics=self._diagnostics(fallback, best_rejected.reason if best_rejected else None),
                )

            candidates.sort(key=lambda node: node.score, reverse=True)
            beam = candidates[: max(self.width, 1)]

        best = max(beam, key=lambda node: node.score)
        direction = normalise(best.first_direction)
        if direction == (0.0, 0.0):
            direction = self.previous_direction
        self.previous_direction = direction
        return StrategyDecision(
            direction=direction,
            split=best.first_split,
            target_kind="beam",
            reason="hunter_beam_rollout",
            score=best.score,
            diagnostics=self._diagnostics(best, None),
        )

    def _candidate_actions(
        self,
        node: HunterBeamNode,
        foods: tuple[FoodModel, ...],
        viruses: tuple[VirusModel, ...],
        arena_size: float,
    ) -> tuple[Action, ...]:
        primary = node.primary
        directions: list[tuple[float, float]] = [node.last_direction]
        directions.extend(
            (
                math.cos(2.0 * math.pi * index / self.angular_samples),
                math.sin(2.0 * math.pi * index / self.angular_samples),
            )
            for index in range(max(self.angular_samples, 1))
        )

        threat = self._threat_vector(node)
        has_close_threat = threat != (0.0, 0.0)
        if has_close_threat:
            directions.append(threat)

        prey_targets = self._prey_targets(node)
        directions.extend(
            normalise((target[0] - primary.x, target[1] - primary.y))
            for target in prey_targets[:3]
        )

        food = self._nearest_food(primary, foods, node.eaten_food_ids)
        if food is not None:
            directions.append(normalise((food.pos[0] - primary.x, food.pos[1] - primary.y)))

        virus_away = self._virus_avoidance_vector(node, viruses)
        if virus_away != (0.0, 0.0):
            directions.append(virus_away)

        wall = self._wall_vector(primary, arena_size)
        if wall != (0.0, 0.0):
            directions.append(wall)
        directions.append(normalise((arena_size / 2.0 - primary.x, arena_size / 2.0 - primary.y)))

        deduped_directions = self._dedupe_directions(directions)
        actions = [Action(direction=direction) for direction in deduped_directions]

        if not has_close_threat:
            for target in prey_targets[:2]:
                direction = normalise((target[0] - primary.x, target[1] - primary.y))
                if direction != (0.0, 0.0):
                    actions.append(Action(direction=direction, split=True, reason="split_prey"))

        return tuple(self._dedupe_actions(actions))

    def _step(
        self,
        *,
        node: HunterBeamNode,
        action: Action,
        foods: tuple[FoodModel, ...],
        viruses: tuple[VirusModel, ...],
        arena_size: float,
        is_first_step: bool,
    ) -> StepResult:
        direction = normalise(action.direction)
        if direction == (0.0, 0.0):
            direction = node.last_direction

        score = node.score - self._turn_penalty(node.last_direction, direction)
        own_blobs = list(node.own_blobs)
        if action.split:
            own_blobs = self._apply_split(own_blobs, direction, arena_size)
            score -= 3.0 + len(own_blobs) * 0.25

        own_blobs = [
            self._move_own_blob(blob, direction, arena_size)
            for blob in own_blobs
        ]
        own_blobs = self._apply_mass_decay(own_blobs)
        enemies = tuple(self._predict_enemy(enemy, own_blobs, arena_size) for enemy in node.enemies)

        eaten_food_ids = set(node.eaten_food_ids)
        captured_blob_ids = set(node.captured_blob_ids)
        consumed_virus_ids = set(node.consumed_virus_ids)

        score += self._resolve_food(
            own_blobs,
            foods,
            eaten_food_ids,
            arena_size,
        )
        virus_penalty, consumed_virus_ids = self._virus_penalty(
            own_blobs,
            viruses,
            consumed_virus_ids,
        )
        score -= virus_penalty

        own_blobs, enemies, interaction_score = self._resolve_blob_interactions(
            own_blobs,
            enemies,
            captured_blob_ids,
            arena_size,
        )
        score += interaction_score
        if not own_blobs:
            rejected = self._replace_node(
                node=node,
                own_blobs=(),
                enemies=enemies,
                score=score - 1_000.0,
                action=action,
                direction=direction,
                is_first_step=is_first_step,
                eaten_food_ids=eaten_food_ids,
                captured_blob_ids=captured_blob_ids,
                consumed_virus_ids=consumed_virus_ids,
            )
            return StepResult(node=rejected, fatal=True, reason="all_blobs_eaten")

        score += self._position_score(own_blobs, enemies, foods, eaten_food_ids, viruses, arena_size)
        next_node = self._replace_node(
            node=node,
            own_blobs=tuple(own_blobs),
            enemies=tuple(enemies),
            score=score,
            action=action,
            direction=direction,
            is_first_step=is_first_step,
            eaten_food_ids=eaten_food_ids,
            captured_blob_ids=captured_blob_ids,
            consumed_virus_ids=consumed_virus_ids,
        )
        return StepResult(node=next_node, fatal=False, reason="")

    def _apply_split(
        self,
        own_blobs: list[HunterSimOwnBlob],
        direction: tuple[float, float],
        arena_size: float,
    ) -> list[HunterSimOwnBlob]:
        if len(own_blobs) >= MAX_BLOB_COUNT:
            return own_blobs

        starting_ids = [blob.blob_id for blob in sorted(own_blobs, key=lambda item: item.blob_id)]
        next_id = max(starting_ids, default=-1) + 1
        by_id = {blob.blob_id: blob for blob in own_blobs}
        for blob_id in starting_ids:
            if len(by_id) >= MAX_BLOB_COUNT:
                break
            blob = by_id[blob_id]
            if blob.mass < SPLIT_MIN_MASS:
                continue
            child_radius = math.sqrt(blob.mass / 2.0)
            parent = HunterSimOwnBlob(
                blob_id=blob.blob_id,
                x=blob.x,
                y=blob.y,
                radius=child_radius,
                merge_cooldown=SPLIT_COOLDOWN_FRAMES,
                eject_vx=blob.eject_vx,
                eject_vy=blob.eject_vy,
            )
            child_x = self._clamp(
                blob.x + direction[0] * (parent.radius + child_radius + SAME_PLAYER_OVERLAP_EPSILON),
                child_radius,
                arena_size - child_radius,
            )
            child_y = self._clamp(
                blob.y + direction[1] * (parent.radius + child_radius + SAME_PLAYER_OVERLAP_EPSILON),
                child_radius,
                arena_size - child_radius,
            )
            child = HunterSimOwnBlob(
                blob_id=next_id,
                x=child_x,
                y=child_y,
                radius=child_radius,
                merge_cooldown=SPLIT_COOLDOWN_FRAMES,
                eject_vx=direction[0] * SPLIT_EJECT_SPEED,
                eject_vy=direction[1] * SPLIT_EJECT_SPEED,
            )
            by_id[blob_id] = parent
            by_id[next_id] = child
            next_id += 1
        return list(by_id.values())

    def _move_own_blob(
        self,
        blob: HunterSimOwnBlob,
        direction: tuple[float, float],
        arena_size: float,
    ) -> HunterSimOwnBlob:
        x = blob.x + direction[0] * _hunter_speed(blob.radius) + blob.eject_vx
        y = blob.y + direction[1] * _hunter_speed(blob.radius) + blob.eject_vy
        return HunterSimOwnBlob(
            blob_id=blob.blob_id,
            x=self._clamp(x, blob.radius, arena_size - blob.radius),
            y=self._clamp(y, blob.radius, arena_size - blob.radius),
            radius=blob.radius,
            merge_cooldown=max(0, blob.merge_cooldown - 1),
            eject_vx=blob.eject_vx * SPLIT_EJECT_DRAG if abs(blob.eject_vx) >= 1e-4 else 0.0,
            eject_vy=blob.eject_vy * SPLIT_EJECT_DRAG if abs(blob.eject_vy) >= 1e-4 else 0.0,
        )

    def _apply_mass_decay(self, own_blobs: list[HunterSimOwnBlob]) -> list[HunterSimOwnBlob]:
        min_mass = STARTING_RADIUS * STARTING_RADIUS
        decayed = []
        for blob in own_blobs:
            if blob.mass <= min_mass:
                decayed.append(blob)
                continue
            radius = math.sqrt(max(min_mass, blob.mass * (1.0 - MASS_DECAY_RATE)))
            decayed.append(
                HunterSimOwnBlob(
                    blob_id=blob.blob_id,
                    x=blob.x,
                    y=blob.y,
                    radius=radius,
                    merge_cooldown=blob.merge_cooldown,
                    eject_vx=blob.eject_vx,
                    eject_vy=blob.eject_vy,
                )
            )
        return decayed

    def _predict_enemy(
        self,
        enemy: SimEnemyBlob,
        own_blobs: list[HunterSimOwnBlob],
        arena_size: float,
    ) -> SimEnemyBlob:
        edible_own = [
            blob
            for blob in own_blobs
            if can_eat_player_blob(enemy.radius, blob.radius)
        ]
        if edible_own:
            target = min(edible_own, key=lambda blob: squared_distance(enemy.pos, blob.pos))
            direction = normalise((target.x - enemy.x, target.y - enemy.y))
        else:
            threatening_own = [
                blob
                for blob in own_blobs
                if can_eat_player_blob(blob.radius, enemy.radius)
            ]
            if threatening_own:
                hunter = min(threatening_own, key=lambda blob: squared_distance(enemy.pos, blob.pos))
                direction = normalise((enemy.x - hunter.x, enemy.y - hunter.y))
            else:
                direction = (0.0, 0.0)

        x = self._clamp(enemy.x + direction[0] * _hunter_speed(enemy.radius), enemy.radius, arena_size - enemy.radius)
        y = self._clamp(enemy.y + direction[1] * _hunter_speed(enemy.radius), enemy.radius, arena_size - enemy.radius)
        return SimEnemyBlob(
            player_id=enemy.player_id,
            blob_id=enemy.blob_id,
            x=x,
            y=y,
            radius=enemy.radius,
        )

    def _resolve_food(
        self,
        own_blobs: list[HunterSimOwnBlob],
        foods: tuple[FoodModel, ...],
        eaten_food_ids: set[int],
        arena_size: float,
    ) -> float:
        score = 0.0
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
            own_blobs[index] = self._replace_own_radius(
                eater,
                math.sqrt(eater.mass + FOOD_RADIUS * FOOD_RADIUS),
                arena_size,
            )
            eaten_food_ids.add(food.food_id)
            score += 4.0
        return score

    def _virus_penalty(
        self,
        own_blobs: list[HunterSimOwnBlob],
        viruses: tuple[VirusModel, ...],
        consumed_virus_ids: set[int],
    ) -> tuple[float, set[int]]:
        penalty = 0.0
        for virus in viruses:
            if virus.virus_id in consumed_virus_ids:
                continue
            for blob in own_blobs:
                can_consume = blob.mass > virus.radius * virus.radius * EAT_SIZE_RATIO
                clearance = virus_center_clearance(
                    blob.pos,
                    blob.radius,
                    virus.pos,
                )
                if can_consume and clearance <= 0.0:
                    consumed_virus_ids.add(virus.virus_id)
                    penalty += 140.0 + 12.0 * blob.mass
                    break
                if can_consume:
                    if clearance < 4.0:
                        penalty += (4.0 - clearance) ** 2 * (1.2 + blob.radius * 0.2)
        return penalty, consumed_virus_ids

    def _resolve_blob_interactions(
        self,
        own_blobs: list[HunterSimOwnBlob],
        enemies: tuple[SimEnemyBlob, ...],
        captured_blob_ids: set[tuple[int, int]],
        arena_size: float,
    ) -> tuple[list[HunterSimOwnBlob], tuple[SimEnemyBlob, ...], float]:
        score = 0.0
        survivors = list(own_blobs)
        remaining_enemies = list(enemies)

        for enemy in list(remaining_enemies):
            eaten_indices = [
                index
                for index, own in enumerate(survivors)
                if can_eat_player_blob(enemy.radius, own.radius)
                and squared_distance(enemy.pos, own.pos) <= enemy.radius * enemy.radius
            ]
            for index in sorted(eaten_indices, reverse=True):
                own = survivors.pop(index)
                score -= 280.0 + 55.0 * own.mass
            if not survivors:
                return [], tuple(remaining_enemies), score

        for own_index, own in sorted(
            list(enumerate(survivors)),
            key=lambda item: item[1].radius,
            reverse=True,
        ):
            if own_index >= len(survivors):
                continue
            current = survivors[own_index]
            still_remaining: list[SimEnemyBlob] = []
            for enemy in remaining_enemies:
                if (
                    can_eat_player_blob(current.radius, enemy.radius)
                    and squared_distance(current.pos, enemy.pos) <= current.radius * current.radius
                ):
                    current = self._replace_own_radius(
                        current,
                        math.sqrt(current.mass + enemy.mass),
                        arena_size,
                    )
                    captured_blob_ids.add(enemy.key)
                    score += 32.0 * enemy.mass
                else:
                    still_remaining.append(enemy)
            survivors[own_index] = current
            remaining_enemies = still_remaining

        return survivors, tuple(remaining_enemies), score

    def _position_score(
        self,
        own_blobs: list[HunterSimOwnBlob],
        enemies: tuple[SimEnemyBlob, ...],
        foods: tuple[FoodModel, ...],
        eaten_food_ids: set[int],
        viruses: tuple[VirusModel, ...],
        arena_size: float,
    ) -> float:
        score = 0.0
        primary = max(own_blobs, key=lambda blob: blob.radius)
        score += primary.mass * 0.08
        score -= (len(own_blobs) - 1) * 0.8

        score -= self._wall_penalty(primary, arena_size)
        for enemy in enemies:
            distance = math.dist(primary.pos, enemy.pos)
            if can_eat_player_blob(enemy.radius, primary.radius):
                split_capable = (
                    enemy.mass >= SPLIT_MIN_MASS
                    and can_eat_player_blob(enemy.radius / SQRT2, primary.radius)
                )
                danger_reach = enemy.radius + _hunter_speed(enemy.radius) * 4.0 + 1.2
                if split_capable:
                    danger_reach += 6.5
                margin = max(distance - danger_reach, 0.05)
                score -= 28.0 / margin
            elif can_eat_player_blob(primary.radius, enemy.radius):
                margin = max(distance - primary.radius, 0.0)
                score += min(8.0, 2.5 * enemy.mass / (margin + 1.0))
            else:
                if distance < primary.radius + enemy.radius + 8.0:
                    score -= 3.0 / (distance + 1.0)

        nearest_food = self._nearest_food(primary, foods, eaten_food_ids)
        if nearest_food is not None:
            score += 1.4 / (math.dist(primary.pos, nearest_food.pos) + 1.0)

        for virus in viruses:
            if primary.mass > virus.radius * virus.radius * EAT_SIZE_RATIO:
                clearance = virus_center_clearance(
                    primary.pos,
                    primary.radius,
                    virus.pos,
                )
                if clearance < 5.0:
                    score -= (5.0 - clearance) ** 2 * 0.9
        return score

    def _prey_targets(self, node: HunterBeamNode) -> list[tuple[float, float]]:
        primary = node.primary
        prey = [
            enemy
            for enemy in node.enemies
            if can_eat_player_blob(primary.radius, enemy.radius)
        ]
        scored: list[tuple[float, tuple[float, float]]] = []
        for enemy in prey:
            distance = math.dist(primary.pos, enemy.pos)
            cluster = [
                other for other in prey if math.dist(enemy.pos, other.pos) <= 4.0
            ]
            if len(cluster) >= 3:
                mass = sum(blob.mass for blob in cluster)
                aim = (
                    sum(blob.x for blob in cluster) / len(cluster),
                    sum(blob.y for blob in cluster) / len(cluster),
                )
                score = mass / (distance + 1.0) * 2.0
            else:
                aim = enemy.pos
                score = enemy.mass / (distance + 1.0)
            split_ready = (
                node.total_mass >= SPLIT_MIN_MASS
                and can_eat_player_blob(primary.radius / SQRT2, enemy.radius)
                and len(node.own_blobs) < MAX_BLOB_COUNT
            )
            if split_ready:
                score *= 1.5
            scored.append((score, aim))
        scored.sort(reverse=True, key=lambda item: item[0])
        return [aim for _, aim in scored]

    def _threat_vector(self, node: HunterBeamNode) -> tuple[float, float]:
        x = 0.0
        y = 0.0
        for own in node.own_blobs:
            for enemy in node.enemies:
                if not can_eat_player_blob(enemy.radius, own.radius):
                    continue
                distance = math.dist(own.pos, enemy.pos)
                reach = enemy.radius + _hunter_speed(enemy.radius) * 4.0 + 1.2
                if (
                    enemy.mass >= SPLIT_MIN_MASS
                    and can_eat_player_blob(enemy.radius / SQRT2, own.radius)
                ):
                    reach += 6.5
                if distance >= reach:
                    continue
                away = normalise((own.x - enemy.x, own.y - enemy.y))
                severity = (reach - distance) / max(reach, 1e-9)
                x += away[0] * severity * own.mass
                y += away[1] * severity * own.mass
        return normalise((x, y))

    def _virus_avoidance_vector(
        self,
        node: HunterBeamNode,
        viruses: tuple[VirusModel, ...],
    ) -> tuple[float, float]:
        x = 0.0
        y = 0.0
        for own in node.own_blobs:
            for virus in viruses:
                if own.mass <= virus.radius * virus.radius * EAT_SIZE_RATIO:
                    continue
                keep = own.radius + 1.5 + own.radius * 0.5
                distance = math.dist(own.pos, virus.pos)
                if distance >= keep:
                    continue
                away = normalise((own.x - virus.pos[0], own.y - virus.pos[1]))
                severity = (keep - distance) / keep
                x += away[0] * severity
                y += away[1] * severity
        return normalise((x, y))

    def _wall_vector(self, blob: HunterSimOwnBlob, arena_size: float) -> tuple[float, float]:
        margin = blob.radius + 3.5
        x = max(0.0, (margin - blob.x) / margin)
        x -= max(0.0, (blob.x - (arena_size - margin)) / margin)
        y = max(0.0, (margin - blob.y) / margin)
        y -= max(0.0, (blob.y - (arena_size - margin)) / margin)
        return normalise((x, y))

    def _wall_penalty(self, blob: HunterSimOwnBlob, arena_size: float) -> float:
        margin = min(
            blob.x - blob.radius,
            blob.y - blob.radius,
            arena_size - blob.radius - blob.x,
            arena_size - blob.radius - blob.y,
        )
        if margin >= 4.0:
            return 0.0
        return (4.0 - margin) ** 2 * 2.0

    def _nearest_food(
        self,
        blob: HunterSimOwnBlob,
        foods: tuple[FoodModel, ...],
        eaten_food_ids: frozenset[int] | set[int],
    ) -> FoodModel | None:
        available = [food for food in foods if food.food_id not in eaten_food_ids]
        if not available:
            return None
        return min(available, key=lambda food: squared_distance(blob.pos, food.pos))

    def _replace_node(
        self,
        *,
        node: HunterBeamNode,
        own_blobs: tuple[HunterSimOwnBlob, ...] | list[HunterSimOwnBlob],
        enemies: tuple[SimEnemyBlob, ...],
        score: float,
        action: Action,
        direction: tuple[float, float],
        is_first_step: bool,
        eaten_food_ids: set[int],
        captured_blob_ids: set[tuple[int, int]],
        consumed_virus_ids: set[int],
    ) -> HunterBeamNode:
        return HunterBeamNode(
            own_blobs=tuple(own_blobs),
            enemies=enemies,
            score=score,
            first_direction=direction if is_first_step else node.first_direction,
            first_split=action.split if is_first_step else node.first_split,
            last_direction=direction,
            eaten_food_ids=frozenset(eaten_food_ids),
            captured_blob_ids=frozenset(captured_blob_ids),
            consumed_virus_ids=frozenset(consumed_virus_ids),
        )

    def _diagnostics(self, node: HunterBeamNode, rejected_reason: str | None) -> dict[str, object]:
        primary = node.primary if node.own_blobs else None
        return {
            "beam_depth": self.depth,
            "beam_width": self.width,
            "beam_angular_samples": self.angular_samples,
            "first_split": node.first_split,
            "projected_blob_count": len(node.own_blobs),
            "projected_primary_radius": primary.radius if primary else None,
            "projected_total_mass": node.total_mass if node.own_blobs else 0.0,
            "projected_captured_blobs": len(node.captured_blob_ids),
            "projected_eaten_food": len(node.eaten_food_ids),
            "projected_virus_hits": len(node.consumed_virus_ids),
            "rejected_reason": rejected_reason,
        }

    def _dedupe_directions(
        self,
        directions: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        deduped: dict[tuple[int, int], tuple[float, float]] = {}
        for direction in directions:
            unit = normalise(direction)
            if unit == (0.0, 0.0):
                continue
            deduped[(round(unit[0] * 1000), round(unit[1] * 1000))] = unit
        return list(deduped.values())

    def _dedupe_actions(self, actions: list[Action]) -> list[Action]:
        deduped: dict[tuple[int, int, bool], Action] = {}
        for action in actions:
            direction = normalise(action.direction)
            if direction == (0.0, 0.0):
                continue
            key = (round(direction[0] * 1000), round(direction[1] * 1000), action.split)
            deduped[key] = Action(direction=direction, split=action.split, reason=action.reason)
        return list(deduped.values())

    def _replace_own_radius(
        self,
        blob: HunterSimOwnBlob,
        radius: float,
        arena_size: float,
    ) -> HunterSimOwnBlob:
        return HunterSimOwnBlob(
            blob_id=blob.blob_id,
            x=self._clamp(blob.x, radius, arena_size - radius),
            y=self._clamp(blob.y, radius, arena_size - radius),
            radius=radius,
            merge_cooldown=blob.merge_cooldown,
            eject_vx=blob.eject_vx,
            eject_vy=blob.eject_vy,
        )

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
        return (1.0 - dot) * 0.9

    def _clamp(self, value: float, lower: float, upper: float) -> float:
        return min(max(value, lower), upper)


def _hunter_speed(radius: float) -> float:
    return max(MIN_PLAYER_SPEED, BASE_PLAYER_SPEED / (1.0 + radius * PLAYER_SPEED_RADIUS_FACTOR))



import os
from dataclasses import replace

from beam_search_core import (
    BeamSearchPlanner,
    StrategyConfig,
    config_from_json_env,
    extract_world,
    log_decision_if_requested,
    profile_config,
)
from strategies.base import StrategyContext, StrategyDecision


class BeamRlProfileStrategy:
    profile_name = "balanced"
    name = "beam_rl_balanced"

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = _submission_safe_config(config or profile_config(self.profile_name))
        self.planner = BeamSearchPlanner(self.config)

    def choose(self, context: StrategyContext) -> StrategyDecision:
        raw_world = extract_world(context.game)
        world = _limited_world(raw_world)
        action = self.planner.choose_action(world)
        log_decision_if_requested(world, action, self.config)
        return StrategyDecision(
            direction=(action.dx, action.dy),
            split=action.split,
            target_kind="beam_rl",
            reason=action.label or self.config.name,
            diagnostics={
                "profile": self.config.name,
                "horizon": self.config.horizon,
                "beam_width": self.config.beam_width,
                "action_bins": self.config.action_bins,
                "allow_split": self.config.allow_split,
                "world_blob_count": world.blob_count,
                "world_enemy_count": len(world.enemies),
                "world_food_count": len(world.food),
                "raw_enemy_count": len(raw_world.enemies),
                "raw_food_count": len(raw_world.food),
            },
        )


class BeamRlBalancedStrategy(BeamRlProfileStrategy):
    profile_name = "balanced"
    name = "beam_rl_balanced"


class BeamRlSurvivalStrategy(BeamRlProfileStrategy):
    profile_name = "survival"
    name = "beam_rl_survival"


class BeamRlFarmerStrategy(BeamRlProfileStrategy):
    profile_name = "farmer"
    name = "beam_rl_farmer"


class BeamRlHunterStrategy(BeamRlProfileStrategy):
    profile_name = "hunter"
    name = "beam_rl_hunter"


class BeamRlOpportunistStrategy(BeamRlProfileStrategy):
    profile_name = "opportunist"
    name = "beam_rl_opportunist"


class BeamRlValueStrategy(BeamRlProfileStrategy):
    name = "beam_rl_value"

    def __init__(self) -> None:
        config = replace(profile_config("balanced"), name="rl_value", weight_learned_value=1.0)
        super().__init__(config=config)


class BeamRlTunedStrategy(BeamRlProfileStrategy):
    name = "beam_rl_tuned"

    def __init__(self) -> None:
        default = replace(profile_config("balanced"), name="tuned")
        super().__init__(config=config_from_json_env(default))


def _submission_safe_config(config: StrategyConfig) -> StrategyConfig:
    horizon = int(os.environ.get("BOT_BEAM_RL_HORIZON", min(config.horizon, 3)))
    beam_width = int(os.environ.get("BOT_BEAM_RL_WIDTH", min(config.beam_width, 3)))
    action_bins = int(os.environ.get("BOT_BEAM_RL_ACTION_BINS", min(config.action_bins, 10)))
    max_extra = int(os.environ.get("BOT_BEAM_RL_MAX_EXTRA", min(config.max_extra_candidates, 6)))
    return replace(
        config,
        horizon=max(1, horizon),
        beam_width=max(1, beam_width),
        action_bins=max(4, action_bins),
        max_extra_candidates=max(0, max_extra),
    )


def _limited_world(world):
    center = world.center
    max_food = int(os.environ.get("BOT_BEAM_RL_MAX_FOOD", "24"))
    max_enemies = int(os.environ.get("BOT_BEAM_RL_MAX_ENEMIES", "12"))
    max_viruses = int(os.environ.get("BOT_BEAM_RL_MAX_VIRUSES", "6"))

    food = tuple(
        sorted(
            world.food,
            key=lambda item: (item.pos.x - center.x) ** 2 + (item.pos.y - center.y) ** 2,
        )[:max_food]
    )
    enemies = tuple(
        sorted(
            world.enemies,
            key=lambda blob: (blob.pos.x - center.x) ** 2 + (blob.pos.y - center.y) ** 2,
        )[:max_enemies]
    )
    viruses = tuple(
        sorted(
            world.viruses,
            key=lambda virus: (virus.pos.x - center.x) ** 2 + (virus.pos.y - center.y) ** 2,
        )[:max_viruses]
    )
    return replace(world, food=food, enemies=enemies, viruses=viruses)
