from __future__ import annotations

"""Robust receding-horizon strategy used by the submission entry point.

The strategy deliberately separates prediction from evaluation:

* public MovePlayer events provide an opponent's most recent direction;
* predators are still rolled forward adversarially, so a stale direction never
  makes an unsafe line look safe;
* own movement, splitting, eject velocity, decay, food, and eating follow the
  engine's update order closely;
* every first action competes in the beam before locally-steered continuations
  are explored.

Only standard-library code and public agario-kit models are used at runtime.
"""

import math
import os
from dataclasses import dataclass, field, replace

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
from lib.interface.events.moves.move_player import MovePlayer
from lib.models.food_model import FoodModel
from lib.models.virus_model import VirusModel
from strategies.base import StrategyContext, StrategyDecision
from strategies.features import normalise, squared_distance


SQRT2 = math.sqrt(2.0)
TAU = 2.0 * math.pi
EPSILON = 1e-9


@dataclass(frozen=True)
class Action:
    direction: tuple[float, float]
    split: bool = False
    reason: str = "move"


@dataclass(frozen=True)
class OwnBlob:
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
class EnemyBlob:
    player_id: int
    blob_id: int
    x: float
    y: float
    radius: float
    direction: tuple[float, float] = (0.0, 0.0)
    stale_rounds: int = 0

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
class EnemyTrack:
    player_id: int
    blob_id: int
    x: float
    y: float
    radius: float
    direction: tuple[float, float]
    last_seen_round: int


@dataclass(frozen=True)
class SearchNode:
    own_blobs: tuple[OwnBlob, ...]
    enemies: tuple[EnemyBlob, ...]
    score: float
    first_direction: tuple[float, float]
    first_split: bool
    first_reason: str
    last_direction: tuple[float, float]
    eaten_food_ids: frozenset[int] = field(default_factory=frozenset)
    captured_enemy_ids: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    consumed_virus_ids: frozenset[int] = field(default_factory=frozenset)
    projected_food: int = 0
    projected_captures: int = 0
    min_safety_margin: float = math.inf

    @property
    def total_mass(self) -> float:
        return sum(blob.mass for blob in self.own_blobs)

    @property
    def primary(self) -> OwnBlob:
        return max(self.own_blobs, key=lambda blob: blob.radius)

    @property
    def center(self) -> tuple[float, float]:
        total_mass = self.total_mass
        if total_mass <= EPSILON:
            return self.primary.pos
        return (
            sum(blob.x * blob.mass for blob in self.own_blobs) / total_mass,
            sum(blob.y * blob.mass for blob in self.own_blobs) / total_mass,
        )


@dataclass(frozen=True)
class StepResult:
    node: SearchNode
    fatal: bool = False


class ChampionStrategy:
    """Threat-first beam search with exact public-rule tactical simulation."""

    name = "champion"

    def __init__(
        self,
        depth: int | None = None,
        width: int | None = None,
        angular_samples: int | None = None,
    ) -> None:
        # These defaults stay comfortably below the engine's eight-second
        # cumulative budget while still searching substantially more first
        # actions than the older beam strategies.
        self.depth = depth if depth is not None else int(os.environ.get("BOT_CHAMPION_DEPTH", "3"))
        self.width = width if width is not None else int(os.environ.get("BOT_CHAMPION_WIDTH", "4"))
        self.angular_samples = (
            angular_samples
            if angular_samples is not None
            else int(os.environ.get("BOT_CHAMPION_ANGLES", "18"))
        )
        self.max_food = int(os.environ.get("BOT_CHAMPION_MAX_FOOD", "24"))
        self.max_enemies = int(os.environ.get("BOT_CHAMPION_MAX_ENEMIES", "12"))
        self.previous_direction: tuple[float, float] = (1.0, 0.0)
        self.last_moves: dict[int, tuple[tuple[float, float], bool]] = {}
        self.enemy_tracks: dict[tuple[int, int], EnemyTrack] = {}

    def choose(self, context: StrategyContext) -> StrategyDecision:
        state = context.game.state
        own_blobs = tuple(
            OwnBlob(
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

        round_number = int(state.round)
        arena_size = float(state.map.size or ARENA_SIZE)
        self._read_public_moves(context)
        enemies = self._update_enemy_memory(context, own_blobs, arena_size)

        center = self._mass_center(own_blobs)
        foods = tuple(
            sorted(
                state.visible_food,
                key=lambda food: squared_distance(center, food.pos),
            )[: self.max_food]
        )
        viruses = tuple(state.visible_viruses)
        enemies = tuple(
            sorted(
                enemies,
                key=lambda enemy: squared_distance(center, enemy.pos),
            )[: self.max_enemies]
        )
        # Clustering is quadratic in the number of food items. It depends only
        # on the authoritative current observation, so compute it once rather
        # than once for every rollout node.
        food_targets = tuple(self._food_targets(center, list(foods)))

        start = SearchNode(
            own_blobs=own_blobs,
            enemies=enemies,
            score=0.0,
            first_direction=self.previous_direction,
            first_split=False,
            first_reason="keep",
            last_direction=self.previous_direction,
        )
        rank_position = self._rank_position(state.rankings, state.me.player_id)
        progress = round_number / max(1, int(state.max_rounds))
        safety_weight = self._safety_weight(rank_position, progress)
        aggression = self._aggression(rank_position, progress)

        beam = [start]
        best_rejected: SearchNode | None = None
        reached_depth = 0
        for depth_index in range(max(1, self.depth)):
            candidates: list[SearchNode] = []
            for node in beam:
                actions = self._candidate_actions(
                    node=node,
                    foods=foods,
                    food_targets=food_targets,
                    viruses=viruses,
                    arena_size=arena_size,
                    first_step=depth_index == 0,
                    allow_split=not (progress >= 0.78 and rank_position <= 2),
                )
                for action in actions:
                    result = self._step(
                        node=node,
                        action=action,
                        foods=foods,
                        viruses=viruses,
                        arena_size=arena_size,
                        first_step=depth_index == 0,
                        safety_weight=safety_weight,
                        aggression=aggression,
                    )
                    if result.fatal:
                        if best_rejected is None or result.node.score > best_rejected.score:
                            best_rejected = result.node
                        continue
                    candidates.append(result.node)

            if not candidates:
                break
            beam = self._prune(candidates)
            reached_depth = depth_index + 1

        if beam and reached_depth:
            best = max(beam, key=self._terminal_score)
            reason = best.first_reason
        elif best_rejected is not None:
            best = best_rejected
            reason = "least_bad_escape"
        else:
            best = start
            reason = "no_search_result"

        direction = normalise(best.first_direction) or self.previous_direction
        self.previous_direction = direction
        return StrategyDecision(
            direction=direction,
            split=best.first_split,
            target_kind="escape" if "escape" in reason else ("prey" if "prey" in reason else "beam"),
            reason=reason,
            score=self._terminal_score(best),
            diagnostics={
                "depth": reached_depth,
                "width": self.width,
                "angles": self.angular_samples,
                "rank": rank_position,
                "progress": round(progress, 4),
                "visible_or_tracked_enemies": len(enemies),
                "projected_food": best.projected_food,
                "projected_captures": best.projected_captures,
                "projected_blob_count": len(best.own_blobs),
                "min_safety_margin": best.min_safety_margin,
            },
        )

    def _read_public_moves(self, context: StrategyContext) -> None:
        for event in context.query.update.values():
            if not isinstance(event, MovePlayer):
                continue
            direction = normalise(event.direction.to_vector())
            self.last_moves[event.player_id] = (direction, bool(event.split))

    def _update_enemy_memory(
        self,
        context: StrategyContext,
        own_blobs: tuple[OwnBlob, ...],
        arena_size: float,
    ) -> tuple[EnemyBlob, ...]:
        state = context.game.state
        round_number = int(state.round)
        visible_keys: set[tuple[int, int]] = set()

        # Advance unseen tracks with the exact direction publicly sent on the
        # preceding round. Visible data below always replaces this estimate.
        advanced: dict[tuple[int, int], EnemyTrack] = {}
        for key, track in self.enemy_tracks.items():
            direction = self.last_moves.get(track.player_id, (track.direction, False))[0]
            speed = _speed(track.radius)
            radius = _decayed_radius(track.radius)
            advanced[key] = replace(
                track,
                x=_clamp(track.x + direction[0] * speed, radius, arena_size - radius),
                y=_clamp(track.y + direction[1] * speed, radius, arena_size - radius),
                radius=radius,
                direction=direction,
            )

        for blob in state.visible_blobs:
            key = (blob.player_id, blob.blob_id)
            visible_keys.add(key)
            direction = self.last_moves.get(blob.player_id, ((0.0, 0.0), False))[0]
            advanced[key] = EnemyTrack(
                player_id=blob.player_id,
                blob_id=blob.blob_id,
                x=blob.pos[0],
                y=blob.pos[1],
                radius=blob.radius,
                direction=direction,
                last_seen_round=round_number,
            )

        view_center = tuple(state.view_center)
        half_view = float(state.vision_size) / 2.0
        kept: dict[tuple[int, int], EnemyTrack] = {}
        enemies: list[EnemyBlob] = []
        largest_own = max(blob.radius for blob in own_blobs)
        for key, track in advanced.items():
            stale_rounds = round_number - track.last_seen_round
            if stale_rounds > 10:
                continue
            should_be_visible = (
                abs(track.x - view_center[0]) <= half_view + track.radius
                and abs(track.y - view_center[1]) <= half_view + track.radius
            )
            if key not in visible_keys and should_be_visible:
                # The estimate contradicted the authoritative current view.
                continue
            kept[key] = track
            # Stale prey is never chased. Retain only potentially dangerous
            # stale blobs close enough to matter within this search horizon.
            if stale_rounds and track.radius < largest_own * EAT_SIZE_RATIO:
                continue
            enemies.append(
                EnemyBlob(
                    player_id=track.player_id,
                    blob_id=track.blob_id,
                    x=track.x,
                    y=track.y,
                    radius=track.radius,
                    direction=track.direction,
                    stale_rounds=stale_rounds,
                )
            )
        self.enemy_tracks = kept
        return tuple(enemies)

    def _candidate_actions(
        self,
        *,
        node: SearchNode,
        foods: tuple[FoodModel, ...],
        food_targets: tuple[tuple[float, float], ...],
        viruses: tuple[VirusModel, ...],
        arena_size: float,
        first_step: bool,
        allow_split: bool,
    ) -> tuple[Action, ...]:
        actions: list[Action] = []
        if first_step:
            actions.extend(
                Action(
                    (math.cos(TAU * index / self.angular_samples), math.sin(TAU * index / self.angular_samples)),
                    reason="angle",
                )
                for index in range(max(8, self.angular_samples))
            )
        else:
            actions.append(Action(node.last_direction, reason="continue"))
            for angle in (-math.pi / 6, -math.pi / 12, math.pi / 12, math.pi / 6):
                actions.append(Action(_rotate(node.last_direction, angle), reason="steer"))

        actions.append(Action(node.last_direction, reason="keep"))
        escape = self._escape_vector(node)
        if escape != (0.0, 0.0):
            actions.append(Action(escape, reason="escape"))
            # Tangential options often escape a wall/predator pincer better
            # than a pure potential-field vector.
            actions.append(Action(_rotate(escape, math.pi / 8), reason="escape_tangent"))
            actions.append(Action(_rotate(escape, -math.pi / 8), reason="escape_tangent"))

        center = node.center
        wall = self._wall_vector(node.primary, arena_size)
        if wall != (0.0, 0.0):
            actions.append(Action(wall, reason="wall_escape"))
        actions.append(Action(normalise((arena_size / 2.0 - center[0], arena_size / 2.0 - center[1])), reason="center"))

        available_food = [food for food in foods if food.food_id not in node.eaten_food_ids]
        if available_food:
            nearest_food = min(
                available_food,
                key=lambda food: squared_distance(center, food.pos),
            )
            actions.append(
                Action(
                    normalise((nearest_food.pos[0] - center[0], nearest_food.pos[1] - center[1])),
                    reason="nearest_food",
                )
            )
        target_limit = 4 if first_step else 2
        for target in food_targets[:target_limit]:
            actions.append(Action(normalise((target[0] - center[0], target[1] - center[1])), reason="food_cluster"))

        prey = [
            enemy
            for enemy in node.enemies
            if enemy.stale_rounds == 0
            and any(own.radius >= enemy.radius * EAT_SIZE_RATIO for own in node.own_blobs)
        ]
        prey.sort(key=lambda enemy: squared_distance(center, enemy.pos))
        for enemy in prey[: 3 if first_step else 2]:
            intercept = self._intercept_direction(node.primary, enemy)
            actions.append(Action(intercept, reason="prey"))
            if allow_split and self._split_can_capture(node, enemy, intercept):
                actions.append(Action(intercept, split=True, reason="split_prey"))

        if allow_split:
            farm_action = self._safe_farm_split(node, available_food, food_targets)
            if farm_action is not None:
                actions.append(farm_action)

        return self._dedupe_actions(actions)

    def _step(
        self,
        *,
        node: SearchNode,
        action: Action,
        foods: tuple[FoodModel, ...],
        viruses: tuple[VirusModel, ...],
        arena_size: float,
        first_step: bool,
        safety_weight: float,
        aggression: float,
    ) -> StepResult:
        direction = normalise(action.direction)
        own_blobs = list(node.own_blobs)
        score = node.score
        if action.split:
            own_blobs = self._apply_split(own_blobs, direction, arena_size)
            score -= 6.0 + max(0, len(own_blobs) - len(node.own_blobs)) * 1.5

        own_blobs = [self._move_own(blob, direction, arena_size) for blob in own_blobs]
        own_blobs = self._apply_attraction(own_blobs, arena_size)
        enemies = self._move_enemies(node.enemies, own_blobs, arena_size)

        eaten_food_ids = set(node.eaten_food_ids)
        captured_enemy_ids = set(node.captured_enemy_ids)
        consumed_virus_ids = set(node.consumed_virus_ids)
        projected_food = node.projected_food
        projected_captures = node.projected_captures

        own_blobs = [replace(blob, radius=_decayed_radius(blob.radius)) for blob in own_blobs]
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
            own_blobs[index] = replace(eater, radius=math.sqrt(eater.mass + FOOD_RADIUS * FOOD_RADIUS))
            eaten_food_ids.add(food.food_id)
            projected_food += 1
            score += 7.0

        virus_penalty = 0.0
        for virus in viruses:
            if virus.virus_id in consumed_virus_ids:
                continue
            for blob in own_blobs:
                if not _can_consume_virus(blob.radius, virus.radius):
                    continue
                clearance = math.dist(blob.pos, virus.pos) - blob.radius - virus.radius
                if clearance <= 0.0:
                    consumed_virus_ids.add(virus.virus_id)
                    virus_penalty += 240.0 + blob.mass * 18.0
                    break
                if clearance < 4.0:
                    virus_penalty += (4.0 - clearance) ** 2 * 2.4
        score -= virus_penalty * safety_weight

        own_blobs, enemies, interaction_score, captures = self._resolve_interactions(
            own_blobs,
            enemies,
            captured_enemy_ids,
        )
        projected_captures += captures
        score += interaction_score * aggression
        if not own_blobs:
            dead = self._replace_node(
                node=node,
                own_blobs=(),
                enemies=enemies,
                score=score - 100_000.0,
                direction=direction,
                action=action,
                first_step=first_step,
                eaten_food_ids=eaten_food_ids,
                captured_enemy_ids=captured_enemy_ids,
                consumed_virus_ids=consumed_virus_ids,
                projected_food=projected_food,
                projected_captures=projected_captures,
                min_safety_margin=-math.inf,
            )
            return StepResult(dead, fatal=True)

        risk_penalty, min_margin, unavoidable = self._risk_score(own_blobs, enemies, safety_weight)
        score -= risk_penalty
        score += self._position_value(own_blobs, enemies, foods, eaten_food_ids, arena_size, aggression)
        score -= self._turn_cost(node.last_direction, direction)
        score -= max(0, len(own_blobs) - 1) * 0.65

        next_node = self._replace_node(
            node=node,
            own_blobs=tuple(own_blobs),
            enemies=enemies,
            score=score,
            direction=direction,
            action=action,
            first_step=first_step,
            eaten_food_ids=eaten_food_ids,
            captured_enemy_ids=captured_enemy_ids,
            consumed_virus_ids=consumed_virus_ids,
            projected_food=projected_food,
            projected_captures=projected_captures,
            min_safety_margin=min(node.min_safety_margin, min_margin),
        )
        # A negative split margin means a rational predator can consume every
        # remaining fragment immediately. Reject only when all own blobs are in
        # that envelope; otherwise the mass-loss penalty already prices it.
        return StepResult(next_node, fatal=unavoidable)

    def _apply_split(
        self,
        blobs: list[OwnBlob],
        direction: tuple[float, float],
        arena_size: float,
    ) -> list[OwnBlob]:
        result = list(blobs)
        starting_ids = [blob.blob_id for blob in sorted(blobs, key=lambda blob: blob.blob_id)]
        next_id = max(starting_ids, default=-1) + 1
        by_id = {blob.blob_id: blob for blob in result}
        for blob_id in starting_ids:
            if len(by_id) >= MAX_BLOB_COUNT:
                break
            blob = by_id.get(blob_id)
            if blob is None or blob.mass < SPLIT_MIN_MASS:
                continue
            child_radius = blob.radius / SQRT2
            parent = replace(blob, radius=child_radius, merge_cooldown=SPLIT_COOLDOWN_FRAMES)
            child = OwnBlob(
                blob_id=next_id,
                x=_clamp(
                    blob.x + direction[0] * (2.0 * child_radius + SAME_PLAYER_OVERLAP_EPSILON),
                    child_radius,
                    arena_size - child_radius,
                ),
                y=_clamp(
                    blob.y + direction[1] * (2.0 * child_radius + SAME_PLAYER_OVERLAP_EPSILON),
                    child_radius,
                    arena_size - child_radius,
                ),
                radius=child_radius,
                merge_cooldown=SPLIT_COOLDOWN_FRAMES,
                eject_vx=direction[0] * SPLIT_EJECT_SPEED,
                eject_vy=direction[1] * SPLIT_EJECT_SPEED,
            )
            by_id[blob_id] = parent
            by_id[next_id] = child
            next_id += 1
        return list(by_id.values())

    def _move_own(self, blob: OwnBlob, direction: tuple[float, float], arena_size: float) -> OwnBlob:
        x = blob.x + direction[0] * _speed(blob.radius) + blob.eject_vx
        y = blob.y + direction[1] * _speed(blob.radius) + blob.eject_vy
        return OwnBlob(
            blob_id=blob.blob_id,
            x=_clamp(x, blob.radius, arena_size - blob.radius),
            y=_clamp(y, blob.radius, arena_size - blob.radius),
            radius=blob.radius,
            merge_cooldown=max(0, blob.merge_cooldown - 1),
            eject_vx=_damped(blob.eject_vx),
            eject_vy=_damped(blob.eject_vy),
        )

    def _apply_attraction(self, blobs: list[OwnBlob], arena_size: float) -> list[OwnBlob]:
        if len(blobs) <= 1:
            return blobs
        total_mass = sum(blob.mass for blob in blobs)
        center_x = sum(blob.x * blob.mass for blob in blobs) / total_mass
        center_y = sum(blob.y * blob.mass for blob in blobs) / total_mass
        result: list[OwnBlob] = []
        for blob in blobs:
            dx = center_x - blob.x
            dy = center_y - blob.y
            distance = math.hypot(dx, dy)
            if distance <= EPSILON:
                result.append(blob)
                continue
            step = min(MERGE_ATTRACTION_SPEED, distance)
            result.append(
                replace(
                    blob,
                    x=_clamp(blob.x + dx / distance * step, blob.radius, arena_size - blob.radius),
                    y=_clamp(blob.y + dy / distance * step, blob.radius, arena_size - blob.radius),
                )
            )
        return result

    def _move_enemies(
        self,
        enemies: tuple[EnemyBlob, ...],
        own_blobs: list[OwnBlob],
        arena_size: float,
    ) -> tuple[EnemyBlob, ...]:
        moved: list[EnemyBlob] = []
        for enemy in enemies:
            target = min(own_blobs, key=lambda own: squared_distance(enemy.pos, own.pos))
            observed = normalise(enemy.direction)
            if enemy.radius >= target.radius * EAT_SIZE_RATIO:
                adversarial = normalise((target.x - enemy.x, target.y - enemy.y))
                direction = normalise((
                    adversarial[0] * 0.82 + observed[0] * 0.18,
                    adversarial[1] * 0.82 + observed[1] * 0.18,
                ))
            elif any(own.radius >= enemy.radius * EAT_SIZE_RATIO for own in own_blobs):
                hunter = min(own_blobs, key=lambda own: squared_distance(enemy.pos, own.pos))
                flee = normalise((enemy.x - hunter.x, enemy.y - hunter.y))
                direction = normalise((
                    flee[0] * 0.62 + observed[0] * 0.38,
                    flee[1] * 0.62 + observed[1] * 0.38,
                ))
            else:
                direction = observed
            speed = _speed(enemy.radius)
            moved.append(
                replace(
                    enemy,
                    x=_clamp(enemy.x + direction[0] * speed, enemy.radius, arena_size - enemy.radius),
                    y=_clamp(enemy.y + direction[1] * speed, enemy.radius, arena_size - enemy.radius),
                    radius=_decayed_radius(enemy.radius),
                    direction=direction,
                )
            )
        return tuple(moved)

    def _resolve_interactions(
        self,
        own_blobs: list[OwnBlob],
        enemies: tuple[EnemyBlob, ...],
        captured_enemy_ids: set[tuple[int, int]],
    ) -> tuple[list[OwnBlob], tuple[EnemyBlob, ...], float, int]:
        survivors = list(own_blobs)
        remaining = list(enemies)
        score = 0.0
        captures = 0

        # The engine resolves larger eaters first. Handle enemy consumption
        # before own captures so optimistic mutual-eat states cannot survive.
        for enemy in sorted(remaining, key=lambda item: item.radius, reverse=True):
            eaten = [
                index
                for index, own in enumerate(survivors)
                if enemy.radius >= own.radius * EAT_SIZE_RATIO
                and squared_distance(enemy.pos, own.pos) <= enemy.radius * enemy.radius
            ]
            for index in sorted(eaten, reverse=True):
                lost = survivors.pop(index)
                score -= 520.0 + lost.mass * 90.0
            if not survivors:
                return [], tuple(remaining), score, captures

        updated: list[OwnBlob] = []
        for own in sorted(survivors, key=lambda item: item.radius, reverse=True):
            current = own
            still_remaining: list[EnemyBlob] = []
            for enemy in remaining:
                if (
                    current.radius >= enemy.radius * EAT_SIZE_RATIO
                    and squared_distance(current.pos, enemy.pos) <= current.radius * current.radius
                ):
                    current = replace(current, radius=math.sqrt(current.mass + enemy.mass))
                    captured_enemy_ids.add(enemy.key)
                    score += 52.0 * enemy.mass + 40.0
                    captures += 1
                else:
                    still_remaining.append(enemy)
            remaining = still_remaining
            updated.append(current)
        return updated, tuple(remaining), score, captures

    def _risk_score(
        self,
        own_blobs: list[OwnBlob],
        enemies: tuple[EnemyBlob, ...],
        safety_weight: float,
    ) -> tuple[float, float, bool]:
        penalty = 0.0
        min_margin = math.inf
        endangered_blob_ids: set[int] = set()
        for own in own_blobs:
            for enemy in enemies:
                if enemy.radius < own.radius * EAT_SIZE_RATIO:
                    continue
                distance = math.dist(own.pos, enemy.pos)
                normal_margin = distance - enemy.radius
                danger_radius = enemy.radius
                if _can_split_eat(enemy.radius, own.radius):
                    danger_radius = max(danger_radius, _split_attack_reach(enemy.radius))
                split_margin = distance - danger_radius
                margin = min(normal_margin, split_margin)
                min_margin = min(min_margin, margin)
                uncertainty = 1.0 + enemy.stale_rounds * 0.12
                if margin <= 0.0:
                    endangered_blob_ids.add(own.blob_id)
                    penalty += (440.0 + 75.0 * own.mass + min(180.0, -margin * 35.0)) * uncertainty
                elif margin < 9.0:
                    penalty += (75.0 / (margin + 0.25) + (9.0 - margin) * 2.2) * uncertainty
        unavoidable = bool(own_blobs) and len(endangered_blob_ids) == len(own_blobs)
        return penalty * safety_weight, min_margin, unavoidable

    def _position_value(
        self,
        own_blobs: list[OwnBlob],
        enemies: tuple[EnemyBlob, ...],
        foods: tuple[FoodModel, ...],
        eaten_food_ids: set[int],
        arena_size: float,
        aggression: float,
    ) -> float:
        value = 0.0
        primary = max(own_blobs, key=lambda blob: blob.radius)
        clearance = min(
            primary.x - primary.radius,
            primary.y - primary.radius,
            arena_size - primary.radius - primary.x,
            arena_size - primary.radius - primary.y,
        )
        if clearance < 4.5:
            value -= (4.5 - clearance) ** 2 * 2.4

        available = [food for food in foods if food.food_id not in eaten_food_ids]
        if available:
            distances = sorted(math.dist(primary.pos, food.pos) for food in available)
            value += sum(2.0 / (distance + 1.0) for distance in distances[:6])

        for enemy in enemies:
            if enemy.stale_rounds:
                continue
            distance = math.dist(primary.pos, enemy.pos)
            if primary.radius >= enemy.radius * EAT_SIZE_RATIO:
                gap = max(0.0, distance - primary.radius)
                value += aggression * min(9.0, enemy.mass * 3.2 / (gap + 1.0))
            elif not enemy.radius >= primary.radius * EAT_SIZE_RATIO and distance < 8.0:
                value -= (8.0 - distance) * 0.3
        return value

    def _escape_vector(self, node: SearchNode) -> tuple[float, float]:
        x = 0.0
        y = 0.0
        for own in node.own_blobs:
            for enemy in node.enemies:
                if enemy.radius < own.radius * EAT_SIZE_RATIO:
                    continue
                danger_radius = enemy.radius
                if _can_split_eat(enemy.radius, own.radius):
                    danger_radius = max(danger_radius, _split_attack_reach(enemy.radius))
                distance = math.dist(own.pos, enemy.pos)
                if distance > danger_radius + 8.0:
                    continue
                away = normalise((own.x - enemy.x, own.y - enemy.y))
                severity = max(0.2, danger_radius + 8.0 - distance) / max(distance, 0.25)
                weight = severity * own.mass * (1.0 + enemy.stale_rounds * 0.1)
                x += away[0] * weight
                y += away[1] * weight
        return normalise((x, y))

    def _wall_vector(self, blob: OwnBlob, arena_size: float) -> tuple[float, float]:
        margin = blob.radius + 4.0
        x = max(0.0, (margin - blob.x) / margin)
        x -= max(0.0, (blob.x - (arena_size - margin)) / margin)
        y = max(0.0, (margin - blob.y) / margin)
        y -= max(0.0, (blob.y - (arena_size - margin)) / margin)
        return normalise((x, y))

    def _food_targets(
        self,
        center: tuple[float, float],
        foods: list[FoodModel],
    ) -> list[tuple[float, float]]:
        if not foods:
            return []
        nearest = min(foods, key=lambda food: squared_distance(center, food.pos))
        scored: list[tuple[float, tuple[float, float]]] = []
        for food in foods:
            neighbours = [other for other in foods if squared_distance(food.pos, other.pos) <= 9.0]
            target = (
                sum(other.pos[0] for other in neighbours) / len(neighbours),
                sum(other.pos[1] for other in neighbours) / len(neighbours),
            )
            distance = math.dist(center, target)
            score = (len(neighbours) + 0.5) / (distance + 1.5)
            scored.append((score, target))
        scored.sort(reverse=True)
        targets: list[tuple[float, float]] = [nearest.pos]
        seen: set[tuple[int, int]] = {(round(nearest.pos[0] * 2), round(nearest.pos[1] * 2))}
        for _, target in scored:
            key = (round(target[0] * 2), round(target[1] * 2))
            if key in seen:
                continue
            seen.add(key)
            targets.append(target)
        return targets

    def _intercept_direction(self, own: OwnBlob, enemy: EnemyBlob) -> tuple[float, float]:
        distance = math.dist(own.pos, enemy.pos)
        lookahead = min(3.0, distance / max(_speed(own.radius), 0.1) * 0.3)
        target = (
            enemy.x + enemy.direction[0] * _speed(enemy.radius) * lookahead,
            enemy.y + enemy.direction[1] * _speed(enemy.radius) * lookahead,
        )
        return normalise((target[0] - own.x, target[1] - own.y))

    def _split_can_capture(
        self,
        node: SearchNode,
        enemy: EnemyBlob,
        direction: tuple[float, float],
    ) -> bool:
        if len(node.own_blobs) >= MAX_BLOB_COUNT:
            return False
        eligible = [blob for blob in node.own_blobs if blob.mass >= SPLIT_MIN_MASS]
        if not eligible:
            return False
        for blob in eligible:
            child_radius = blob.radius / SQRT2
            if child_radius < enemy.radius * EAT_SIZE_RATIO:
                continue
            rel = (enemy.x - blob.x, enemy.y - blob.y)
            forward = rel[0] * direction[0] + rel[1] * direction[1]
            lateral = abs(rel[0] * direction[1] - rel[1] * direction[0])
            reach = 2.0 * child_radius + SPLIT_EJECT_SPEED + _speed(child_radius) + child_radius
            if -0.1 <= forward <= reach and lateral <= child_radius:
                return True
        return False

    def _safe_farm_split(
        self,
        node: SearchNode,
        foods: list[FoodModel],
        food_targets: tuple[tuple[float, float], ...],
    ) -> Action | None:
        if len(node.own_blobs) != 1 or node.primary.mass < 2.6 or len(foods) < 5:
            return None
        primary = node.primary
        if any(
            enemy.radius >= primary.radius / SQRT2 * EAT_SIZE_RATIO
            and math.dist(primary.pos, enemy.pos) < 14.0
            for enemy in node.enemies
        ):
            return None
        if not food_targets:
            return None
        target = food_targets[0]
        direction = normalise((target[0] - primary.x, target[1] - primary.y))
        along = 0
        for food in foods:
            rel = (food.pos[0] - primary.x, food.pos[1] - primary.y)
            forward = rel[0] * direction[0] + rel[1] * direction[1]
            lateral = abs(rel[0] * direction[1] - rel[1] * direction[0])
            if 0.0 <= forward <= 9.0 and lateral <= primary.radius * 1.4:
                along += 1
        if along < 4:
            return None
        return Action(direction, split=True, reason="split_farm")

    def _prune(self, candidates: list[SearchNode]) -> list[SearchNode]:
        candidates.sort(key=self._terminal_score, reverse=True)
        kept: list[SearchNode] = []
        seen: set[tuple[int, bool, int, int, int]] = set()
        for node in candidates:
            center = node.center
            first_angle = int(round(math.atan2(node.first_direction[1], node.first_direction[0]) / TAU * 48)) % 48
            key = (
                first_angle,
                node.first_split,
                round(center[0] * 2),
                round(center[1] * 2),
                len(node.own_blobs),
            )
            if key in seen:
                continue
            seen.add(key)
            kept.append(node)
            if len(kept) >= max(1, self.width):
                break
        return kept

    def _terminal_score(self, node: SearchNode) -> float:
        if not node.own_blobs:
            return node.score
        # Mass is counted only at the frontier, avoiding a bias toward gains
        # that happen earlier merely because they are re-added every depth.
        return node.score + math.log1p(node.total_mass) * 5.0

    def _replace_node(
        self,
        *,
        node: SearchNode,
        own_blobs: tuple[OwnBlob, ...] | list[OwnBlob],
        enemies: tuple[EnemyBlob, ...],
        score: float,
        direction: tuple[float, float],
        action: Action,
        first_step: bool,
        eaten_food_ids: set[int],
        captured_enemy_ids: set[tuple[int, int]],
        consumed_virus_ids: set[int],
        projected_food: int,
        projected_captures: int,
        min_safety_margin: float,
    ) -> SearchNode:
        return SearchNode(
            own_blobs=tuple(own_blobs),
            enemies=enemies,
            score=score,
            first_direction=direction if first_step else node.first_direction,
            first_split=action.split if first_step else node.first_split,
            first_reason=action.reason if first_step else node.first_reason,
            last_direction=direction,
            eaten_food_ids=frozenset(eaten_food_ids),
            captured_enemy_ids=frozenset(captured_enemy_ids),
            consumed_virus_ids=frozenset(consumed_virus_ids),
            projected_food=projected_food,
            projected_captures=projected_captures,
            min_safety_margin=min_safety_margin,
        )

    def _dedupe_actions(self, actions: list[Action]) -> tuple[Action, ...]:
        result: list[Action] = []
        seen: set[tuple[int, bool]] = set()
        for action in actions:
            direction = normalise(action.direction)
            if direction == (0.0, 0.0):
                continue
            angle_bin = int(round(math.atan2(direction[1], direction[0]) / TAU * 96)) % 96
            key = (angle_bin, action.split)
            if key in seen:
                continue
            seen.add(key)
            result.append(Action(direction, action.split, action.reason))
        return tuple(result)

    def _mass_center(self, blobs: tuple[OwnBlob, ...]) -> tuple[float, float]:
        total = sum(blob.mass for blob in blobs)
        return (
            sum(blob.x * blob.mass for blob in blobs) / total,
            sum(blob.y * blob.mass for blob in blobs) / total,
        )

    def _rank_position(self, rankings: list[int], player_id: int) -> int:
        try:
            return list(rankings).index(player_id) + 1
        except ValueError:
            return 8

    def _safety_weight(self, rank_position: int, progress: float) -> float:
        if progress >= 0.72 and rank_position <= 2:
            return 1.8
        if progress >= 0.88:
            return 1.35
        return 1.0

    def _aggression(self, rank_position: int, progress: float) -> float:
        if progress >= 0.75 and rank_position >= 5:
            return 1.45
        if progress >= 0.72 and rank_position <= 2:
            return 0.72
        return 1.0

    def _turn_cost(
        self,
        previous: tuple[float, float],
        current: tuple[float, float],
    ) -> float:
        previous = normalise(previous)
        current = normalise(current)
        dot = _clamp(previous[0] * current[0] + previous[1] * current[1], -1.0, 1.0)
        return (1.0 - dot) * 0.35


def _speed(radius: float) -> float:
    return max(MIN_PLAYER_SPEED, BASE_PLAYER_SPEED / (1.0 + radius * PLAYER_SPEED_RADIUS_FACTOR))


def _decayed_radius(radius: float) -> float:
    mass = radius * radius
    minimum = STARTING_RADIUS * STARTING_RADIUS
    if mass <= minimum:
        return radius
    return math.sqrt(max(minimum, mass * (1.0 - MASS_DECAY_RATE)))


def _can_consume_virus(blob_radius: float, virus_radius: float) -> bool:
    return blob_radius * blob_radius > virus_radius * virus_radius * EAT_SIZE_RATIO


def _can_split_eat(predator_radius: float, prey_radius: float) -> bool:
    return (
        predator_radius * predator_radius >= SPLIT_MIN_MASS
        and predator_radius / SQRT2 >= prey_radius * EAT_SIZE_RATIO
    )


def _split_attack_reach(predator_radius: float) -> float:
    """One-round center-distance reach of a directly aimed split attack."""

    child_radius = predator_radius / SQRT2
    return 3.0 * child_radius + SPLIT_EJECT_SPEED + _speed(child_radius)


def _damped(value: float) -> float:
    value *= SPLIT_EJECT_DRAG
    return 0.0 if abs(value) < 1e-4 else value


def _rotate(direction: tuple[float, float], angle: float) -> tuple[float, float]:
    direction = normalise(direction)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        direction[0] * cosine - direction[1] * sine,
        direction[0] * sine + direction[1] * cosine,
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
