from __future__ import annotations

import math
from dataclasses import dataclass

from lib.config.arena import ARENA_SIZE, MAX_BLOB_COUNT
from lib.config.player import (
    EAT_SIZE_RATIO,
    FOOD_RADIUS,
    SPLIT_EJECT_DRAG,
    SPLIT_EJECT_SPEED,
    SPLIT_MIN_MASS,
)
from lib.models.blob_model import BlobModel, VisibleBlobModel
from lib.models.virus_model import VirusModel
from simulation.rules import can_consume_virus
from strategies.base import StrategyContext, StrategyDecision
from strategies.features import can_eat_player_blob, normalise, player_speed


SQRT2 = math.sqrt(2.0)
EPSILON = 1.0e-9
TAU = 2.0 * math.pi
STATIC_HORIZON = 6
GROWTH_CANDIDATE_LIMIT = 4


@dataclass(frozen=True, slots=True)
class GrowthAction:
    direction: tuple[float, float]
    split: bool = False
    reason: str = "explore"


@dataclass(frozen=True, slots=True)
class ProjectedBlob:
    source_index: int
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    radius: float
    speed: float
    split_action: bool = False
    split_ejected: bool = False

    @property
    def mass(self) -> float:
        return self.radius * self.radius


@dataclass(frozen=True, slots=True)
class StageOneValue:
    action: GrowthAction
    expected_growth: float
    paths: tuple[ProjectedBlob, ...]


@dataclass(frozen=True, slots=True)
class RetainedValue:
    action: GrowthAction
    retained_mass: float
    expected_growth: float
    catastrophe: bool
    threatened_fragments: int
    survival_margin: float

    @property
    def total(self) -> float:
        if self.catastrophe:
            # When no feasible action exists, growth is irrelevant.  Maximise
            # time/separation to the closest reachable capture boundary.
            return -1.0e12 + self.survival_margin
        return self.retained_mass + self.expected_growth


class StaticRetainedGrowthStrategy:
    """A bounded two-stage policy for growth under a strict compute budget.

    Stage one treats prey as static routing targets and never constructs an
    opponent response tree.  Stage two applies a one-round reachable envelope
    to only four growth leaders plus every explicit escape candidate.  Safety
    is therefore a catastrophe filter; among feasible actions the objective is
    mass retained by the fragment that actually earns the capture.
    """

    name = "static_retained_growth"

    def __init__(self) -> None:
        self._last_direction = (1.0, 0.0)

    def choose(self, context: StrategyContext) -> StrategyDecision:
        state = context.game.state
        own = tuple(state.me.blobs.values())
        if not own:
            return StrategyDecision(
                direction=self._last_direction,
                reason="dead_fallback",
            )

        arena_size = float(state.map.size or ARENA_SIZE)
        enemies = tuple(state.visible_blobs)
        foods = tuple(state.visible_food)
        viruses = tuple(state.visible_viruses)
        actions = self._candidate_actions(
            own=own,
            enemies=enemies,
            foods=foods,
            viruses=viruses,
            arena_size=arena_size,
        )
        stage_one = tuple(
            self._stage_one_value(
                action=action,
                own=own,
                enemies=enemies,
                foods=foods,
                viruses=viruses,
                arena_size=arena_size,
            )
            for action in actions
        )
        growth_leaders = sorted(
            stage_one,
            key=lambda value: (
                -value.expected_growth,
                value.action.split,
                value.action.reason,
            ),
        )[:GROWTH_CANDIDATE_LIMIT]
        escape_rows = tuple(
            value
            for value in stage_one
            if value.action.reason.startswith("escape")
            or value.action.reason in {"keep", "wall_escape"}
        )
        finalists = self._dedupe_stage_one((*growth_leaders, *escape_rows))
        retained = tuple(
            self._retained_value(
                stage_one=value,
                own=own,
                enemies=enemies,
                foods=foods,
                viruses=viruses,
                arena_size=arena_size,
            )
            for value in finalists
        )
        best = max(
            retained,
            key=lambda value: (
                value.total,
                not value.action.split,
                value.action.reason,
            ),
        )
        direction = normalise(best.action.direction) or self._last_direction
        self._last_direction = direction
        return StrategyDecision(
            direction=direction,
            split=best.action.split,
            target_kind=(
                "prey"
                if "prey" in best.action.reason
                else "escape"
                if "escape" in best.action.reason
                else "growth"
            ),
            reason=best.action.reason,
            score=best.total,
            diagnostics={
                "stage_one_candidates": len(stage_one),
                "stage_two_candidates": len(finalists),
                "expected_growth": round(best.expected_growth, 6),
                "retained_mass": round(best.retained_mass, 6),
                "catastrophe": best.catastrophe,
                "threatened_fragments": best.threatened_fragments,
                "survival_margin": round(best.survival_margin, 6),
            },
        )

    def _candidate_actions(
        self,
        *,
        own: tuple[BlobModel, ...],
        enemies: tuple[VisibleBlobModel, ...],
        foods: tuple[object, ...],
        viruses: tuple[VirusModel, ...],
        arena_size: float,
    ) -> tuple[GrowthAction, ...]:
        total_mass = sum(_mass(blob.radius) for blob in own)
        center = (
            sum(blob.pos[0] * _mass(blob.radius) for blob in own) / total_mass,
            sum(blob.pos[1] * _mass(blob.radius) for blob in own) / total_mass,
        )
        actions: list[GrowthAction] = [
            GrowthAction(self._last_direction, reason="keep")
        ]

        # A small angular cover prevents sparse static targets from making the
        # policy immobile without recreating a wide search tree.
        actions.extend(
            GrowthAction((math.cos(TAU * index / 8), math.sin(TAU * index / 8)))
            for index in range(8)
        )

        if foods:
            nearest = min(foods, key=lambda food: _distance_sq(center, food.pos))
            actions.append(
                GrowthAction(
                    normalise((nearest.pos[0] - center[0], nearest.pos[1] - center[1])),
                    reason="nearest_food",
                )
            )
            food_x = food_y = 0.0
            for food in foods:
                dx = food.pos[0] - center[0]
                dy = food.pos[1] - center[1]
                weight = 1.0 / (dx * dx + dy * dy + 1.0)
                food_x += dx * weight
                food_y += dy * weight
            actions.append(GrowthAction(normalise((food_x, food_y)), reason="food_field"))

        prey_rows: list[tuple[float, BlobModel, VisibleBlobModel]] = []
        for enemy in enemies:
            best_source: BlobModel | None = None
            best_value = 0.0
            for blob in own:
                if not can_eat_player_blob(blob.radius, enemy.radius):
                    continue
                gap = max(0.0, math.dist(blob.pos, enemy.pos) - blob.radius)
                value = _mass(enemy.radius) * math.exp(
                    -gap / max(player_speed(blob.radius) * STATIC_HORIZON, EPSILON)
                )
                if best_source is None or value > best_value:
                    best_source = blob
                    best_value = value
            if best_source is not None:
                prey_rows.append((best_value, best_source, enemy))
        prey_rows.sort(key=lambda row: -row[0])
        for _, source, enemy in prey_rows[:4]:
            direction = normalise(
                (enemy.pos[0] - source.pos[0], enemy.pos[1] - source.pos[1])
            )
            actions.append(GrowthAction(direction, reason="prey"))
            if self._split_capture_is_robust(
                source=source,
                enemy=enemy,
                own_count=len(own),
            ):
                actions.append(GrowthAction(direction, split=True, reason="split_prey"))

        for virus in viruses:
            harvesters = [
                blob
                for blob in own
                if can_consume_virus(
                    blob.radius,
                    virus.radius,
                    eat_size_ratio=EAT_SIZE_RATIO,
                )
            ]
            if not harvesters:
                continue
            source = min(harvesters, key=lambda blob: math.dist(blob.pos, virus.pos))
            actions.append(
                GrowthAction(
                    normalise(
                        (virus.pos[0] - source.pos[0], virus.pos[1] - source.pos[1])
                    ),
                    reason="virus_harvest",
                )
            )

        escape = self._escape_vector(own, enemies)
        if escape != (0.0, 0.0):
            actions.extend(
                GrowthAction(_rotate(escape, angle), reason=reason)
                for angle, reason in (
                    (0.0, "escape"),
                    (math.pi / 8, "escape_tangent"),
                    (-math.pi / 8, "escape_tangent"),
                    (math.pi / 2, "escape_wide"),
                    (-math.pi / 2, "escape_wide"),
                )
            )

        wall = self._wall_escape(own, arena_size)
        if wall != (0.0, 0.0):
            actions.append(GrowthAction(wall, reason="wall_escape"))
        actions.append(
            GrowthAction(
                normalise((arena_size / 2 - center[0], arena_size / 2 - center[1])),
                reason="center",
            )
        )
        return self._dedupe_actions(actions)

    def _stage_one_value(
        self,
        *,
        action: GrowthAction,
        own: tuple[BlobModel, ...],
        enemies: tuple[VisibleBlobModel, ...],
        foods: tuple[object, ...],
        viruses: tuple[VirusModel, ...],
        arena_size: float,
    ) -> StageOneValue:
        paths = self._project_paths(
            own=own,
            action=action,
            arena_size=arena_size,
            horizon=STATIC_HORIZON,
        )
        expected_growth = self._static_growth(
            paths=paths,
            enemies=enemies,
            foods=foods,
            viruses=viruses,
            own_count=len(own),
            arena_size=arena_size,
        )
        return StageOneValue(action, expected_growth, paths)

    def _retained_value(
        self,
        *,
        stage_one: StageOneValue,
        own: tuple[BlobModel, ...],
        enemies: tuple[VisibleBlobModel, ...],
        foods: tuple[object, ...],
        viruses: tuple[VirusModel, ...],
        arena_size: float,
    ) -> RetainedValue:
        one_step = self._project_paths(
            own=own,
            action=stage_one.action,
            arena_size=arena_size,
            horizon=1,
        )
        safe_by_path: list[bool] = []
        path_margins: list[float] = []
        threatened_fragments = 0
        for path in one_step:
            margins = tuple(
                self._predator_margin(path, enemy)
                for enemy in enemies
                if can_eat_player_blob(enemy.radius, path.radius)
            )
            worst_margin = min(margins, default=math.inf)
            threatened = worst_margin <= 0.0
            safe_by_path.append(not threatened)
            path_margins.append(worst_margin)
            if threatened:
                threatened_fragments += 1

        retained_mass = sum(
            path.mass for path, safe in zip(one_step, safe_by_path, strict=True) if safe
        )
        catastrophe = not any(safe_by_path)
        expected_growth = (
            stage_one.expected_growth
            if all(safe_by_path)
            else self._static_growth(
                paths=stage_one.paths,
                enemies=enemies,
                foods=foods,
                viruses=viruses,
                own_count=len(own),
                arena_size=arena_size,
                safe_by_path=tuple(safe_by_path),
            )
        )
        return RetainedValue(
            action=stage_one.action,
            retained_mass=retained_mass,
            expected_growth=expected_growth,
            catastrophe=catastrophe,
            threatened_fragments=threatened_fragments,
            survival_margin=max(
                (margin for margin in path_margins if math.isfinite(margin)),
                default=arena_size,
            ),
        )

    def _static_growth(
        self,
        *,
        paths: tuple[ProjectedBlob, ...],
        enemies: tuple[VisibleBlobModel, ...],
        foods: tuple[object, ...],
        viruses: tuple[VirusModel, ...],
        own_count: int,
        arena_size: float,
        safe_by_path: tuple[bool, ...] | None = None,
    ) -> float:
        retained_path_ids = (
            None
            if safe_by_path is None
            else {
                id(path)
                for path, safe in zip(paths, safe_by_path, strict=True)
                if safe
            }
        )

        def retained(path: ProjectedBlob) -> bool:
            return retained_path_ids is None or id(path) in retained_path_ids

        values: list[float] = []
        for enemy in enemies:
            best = 0.0
            for path in paths:
                if not retained(path) or not can_eat_player_blob(
                    path.radius, enemy.radius
                ):
                    continue
                value = _prey_growth_value(
                    path=path,
                    enemy=enemy,
                    arena_size=arena_size,
                )
                best = max(best, value)
            if best > 0.0:
                values.append(best)

        food_mass = FOOD_RADIUS * FOOD_RADIUS
        for food in foods:
            best = 0.0
            for path in paths:
                if not retained(path):
                    continue
                gap = max(
                    0.0,
                    _point_segment_distance(
                        food.pos,
                        (path.start_x, path.start_y),
                        (path.end_x, path.end_y),
                    )
                    - path.radius,
                )
                best = max(
                    best,
                    food_mass * math.exp(-gap / max(path.speed * STATIC_HORIZON, 1.0)),
                )
            if best > 0.0:
                values.append(best)

        for virus in viruses:
            best = 0.0
            for path in paths:
                if not retained(path) or not can_consume_virus(
                    path.radius,
                    virus.radius,
                    eat_size_ratio=EAT_SIZE_RATIO,
                ):
                    continue
                gap = max(
                    0.0,
                    _point_segment_distance(
                        virus.pos,
                        (path.start_x, path.start_y),
                        (path.end_x, path.end_y),
                    )
                    - path.radius,
                )
                retention = self._virus_retention(
                    path=path,
                    virus=virus,
                    enemies=enemies,
                    own_count=own_count,
                )
                best = max(
                    best,
                    _mass(virus.radius)
                    * retention
                    * math.exp(-gap / max(path.speed * STATIC_HORIZON, 1.0)),
                )
            if best > 0.0:
                values.append(best)

        values.sort(reverse=True)
        return sum(value * weight for value, weight in zip(values[:3], (1.0, 0.25, 0.1)))

    @staticmethod
    def _virus_retention(
        *,
        path: ProjectedBlob,
        virus: VirusModel,
        enemies: tuple[VisibleBlobModel, ...],
        own_count: int,
    ) -> float:
        piece_count = max(1, MAX_BLOB_COUNT - own_count + 1)
        piece_radius = math.sqrt((_mass(path.radius) + _mass(virus.radius)) / piece_count)
        for enemy in enemies:
            if not can_eat_player_blob(enemy.radius, piece_radius):
                continue
            danger = enemy.radius + player_speed(enemy.radius)
            if _can_split_eat(enemy.radius, piece_radius):
                danger = max(danger, _split_attack_reach(enemy.radius))
            if math.dist((path.end_x, path.end_y), enemy.pos) <= danger:
                return 0.0
        return 1.0

    @staticmethod
    def _predator_margin(
        path: ProjectedBlob,
        enemy: VisibleBlobModel,
    ) -> float:
        danger = enemy.radius + player_speed(enemy.radius)
        if _can_split_eat(enemy.radius, path.radius):
            danger = max(danger, _split_attack_reach(enemy.radius))
        return _point_segment_distance(
            enemy.pos,
            (path.start_x, path.start_y),
            (path.end_x, path.end_y),
        ) - danger

    @staticmethod
    def _project_paths(
        *,
        own: tuple[BlobModel, ...],
        action: GrowthAction,
        arena_size: float,
        horizon: int,
    ) -> tuple[ProjectedBlob, ...]:
        direction = normalise(action.direction)
        drag_sum = (
            float(horizon)
            if SPLIT_EJECT_DRAG >= 1.0 - EPSILON
            else (1.0 - SPLIT_EJECT_DRAG**horizon) / (1.0 - SPLIT_EJECT_DRAG)
        )
        remaining_slots = MAX_BLOB_COUNT - len(own)
        paths: list[ProjectedBlob] = []
        for source_index, blob in enumerate(own):
            can_split = (
                action.split
                and remaining_slots > 0
                and _mass(blob.radius) >= SPLIT_MIN_MASS
            )
            if can_split:
                remaining_slots -= 1
                radius = blob.radius / SQRT2
                speed = player_speed(radius)
                paths.append(
                    _projected_blob(
                        source_index=source_index,
                        blob=blob,
                        radius=radius,
                        speed=speed,
                        direction=direction,
                        arena_size=arena_size,
                        horizon=horizon,
                        drag_sum=drag_sum,
                        split_action=True,
                    )
                )
                placement = 2.0 * radius
                child_start = (
                    _clamp(blob.pos[0] + direction[0] * placement, radius, arena_size - radius),
                    _clamp(blob.pos[1] + direction[1] * placement, radius, arena_size - radius),
                )
                child_end = (
                    _clamp(
                        child_start[0]
                        + direction[0] * (speed * horizon + SPLIT_EJECT_SPEED * drag_sum),
                        radius,
                        arena_size - radius,
                    ),
                    _clamp(
                        child_start[1]
                        + direction[1] * (speed * horizon + SPLIT_EJECT_SPEED * drag_sum),
                        radius,
                        arena_size - radius,
                    ),
                )
                paths.append(
                    ProjectedBlob(
                        source_index=source_index,
                        start_x=child_start[0],
                        start_y=child_start[1],
                        end_x=child_end[0],
                        end_y=child_end[1],
                        radius=radius,
                        speed=speed,
                        split_action=True,
                        split_ejected=True,
                    )
                )
            else:
                paths.append(
                    _projected_blob(
                        source_index=source_index,
                        blob=blob,
                        radius=blob.radius,
                        speed=player_speed(blob.radius),
                        direction=direction,
                        arena_size=arena_size,
                        horizon=horizon,
                        drag_sum=drag_sum,
                    )
                )
        return tuple(paths)

    @staticmethod
    def _split_capture_is_robust(
        *,
        source: BlobModel,
        enemy: VisibleBlobModel,
        own_count: int,
    ) -> bool:
        if own_count >= MAX_BLOB_COUNT or _mass(source.radius) < SPLIT_MIN_MASS:
            return False
        child_radius = source.radius / SQRT2
        if not can_eat_player_blob(child_radius, enemy.radius):
            return False
        distance = math.dist(source.pos, enemy.pos)
        # The current split must still connect when the prey uses its entire
        # next move directly away.  Future split option value never becomes a
        # root split event.
        return distance + player_speed(enemy.radius) <= _split_attack_reach(
            source.radius
        )

    @staticmethod
    def _escape_vector(
        own: tuple[BlobModel, ...],
        enemies: tuple[VisibleBlobModel, ...],
    ) -> tuple[float, float]:
        x = y = 0.0
        for blob in own:
            for enemy in enemies:
                if not can_eat_player_blob(enemy.radius, blob.radius):
                    continue
                danger = enemy.radius + player_speed(enemy.radius)
                if _can_split_eat(enemy.radius, blob.radius):
                    danger = max(danger, _split_attack_reach(enemy.radius))
                distance = math.dist(blob.pos, enemy.pos)
                if distance > danger + player_speed(blob.radius) * 2.0:
                    continue
                away = normalise(
                    (blob.pos[0] - enemy.pos[0], blob.pos[1] - enemy.pos[1])
                )
                pressure = max(0.25, danger + 2.0 - distance) * _mass(blob.radius)
                x += away[0] * pressure
                y += away[1] * pressure
        return normalise((x, y))

    @staticmethod
    def _wall_escape(
        own: tuple[BlobModel, ...],
        arena_size: float,
    ) -> tuple[float, float]:
        x = y = 0.0
        for blob in own:
            margin = blob.radius + 2.0
            x += max(0.0, margin - blob.pos[0])
            x -= max(0.0, blob.pos[0] - (arena_size - margin))
            y += max(0.0, margin - blob.pos[1])
            y -= max(0.0, blob.pos[1] - (arena_size - margin))
        return normalise((x, y))

    @staticmethod
    def _dedupe_actions(actions: list[GrowthAction]) -> tuple[GrowthAction, ...]:
        actions.sort(key=StaticRetainedGrowthStrategy._reason_priority)
        seen: set[tuple[bool, int, int]] = set()
        result: list[GrowthAction] = []
        for action in actions:
            direction = normalise(action.direction)
            if direction == (0.0, 0.0):
                continue
            key = (action.split, round(direction[0] * 1000), round(direction[1] * 1000))
            if key in seen:
                continue
            seen.add(key)
            result.append(GrowthAction(direction, action.split, action.reason))
        return tuple(result)

    @staticmethod
    def _reason_priority(action: GrowthAction) -> tuple[int, str]:
        reason = action.reason.removeprefix("split_")
        if "prey" in reason:
            return (0, reason)
        if "virus" in reason:
            return (1, reason)
        if "escape" in reason:
            return (2, reason)
        if "food" in reason:
            return (3, reason)
        if reason == "keep":
            return (4, reason)
        return (5, reason)

    @staticmethod
    def _dedupe_stage_one(
        values: tuple[StageOneValue, ...],
    ) -> tuple[StageOneValue, ...]:
        seen: set[tuple[bool, int, int]] = set()
        result: list[StageOneValue] = []
        for value in values:
            direction = normalise(value.action.direction)
            key = (
                value.action.split,
                round(direction[0] * 1000),
                round(direction[1] * 1000),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(value)
        return tuple(result)


def _projected_blob(
    *,
    source_index: int,
    blob: BlobModel,
    radius: float,
    speed: float,
    direction: tuple[float, float],
    arena_size: float,
    horizon: int,
    drag_sum: float,
    split_action: bool = False,
) -> ProjectedBlob:
    end_x = _clamp(
        blob.pos[0]
        + direction[0] * speed * horizon
        + getattr(blob, "eject_vx", 0.0) * drag_sum,
        radius,
        arena_size - radius,
    )
    end_y = _clamp(
        blob.pos[1]
        + direction[1] * speed * horizon
        + getattr(blob, "eject_vy", 0.0) * drag_sum,
        radius,
        arena_size - radius,
    )
    return ProjectedBlob(
        source_index,
        blob.pos[0],
        blob.pos[1],
        end_x,
        end_y,
        radius,
        speed,
        split_action,
    )


def _can_split_eat(predator_radius: float, prey_radius: float) -> bool:
    return _mass(predator_radius) >= SPLIT_MIN_MASS and can_eat_player_blob(
        predator_radius / SQRT2,
        prey_radius,
    )


def _split_attack_reach(radius: float) -> float:
    child_radius = radius / SQRT2
    return 3.0 * child_radius + SPLIT_EJECT_SPEED + player_speed(child_radius)


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    length_sq = delta_x * delta_x + delta_y * delta_y
    if length_sq <= EPSILON:
        return math.dist(point, start)
    fraction = _clamp(
        ((point[0] - start[0]) * delta_x + (point[1] - start[1]) * delta_y)
        / length_sq,
        0.0,
        1.0,
    )
    closest = (start[0] + delta_x * fraction, start[1] + delta_y * fraction)
    return math.dist(point, closest)


def _prey_growth_value(
    *,
    path: ProjectedBlob,
    enemy: VisibleBlobModel,
    arena_size: float,
) -> float:
    """Value capture inside the horizon, otherwise only one-step progress."""

    delta_x = enemy.pos[0] - path.start_x
    delta_y = enemy.pos[1] - path.start_y
    distance = math.hypot(delta_x, delta_y)
    gap = max(0.0, distance - path.radius)
    if gap <= EPSILON:
        return _mass(enemy.radius)
    target_direction = (delta_x / distance, delta_y / distance)
    path_direction = normalise(
        (path.end_x - path.start_x, path.end_y - path.start_y)
    )
    enemy_speed = player_speed(enemy.radius)
    if path.split_action:
        # A root split is a one-round discontinuous event.  It earns capture
        # value only when the child connects now against a full-speed flee;
        # a possible future split belongs to the movement option value, not to
        # the current split action.
        immediate_closing = (
            (path.speed + (SPLIT_EJECT_SPEED if path.split_ejected else 0.0))
            * max(
                0.0,
                path_direction[0] * target_direction[0]
                + path_direction[1] * target_direction[1],
            )
            - enemy_speed
        )
        return _mass(enemy.radius) if gap <= immediate_closing else 0.0

    own_next = (
        _clamp(
            path.start_x + path_direction[0] * path.speed,
            path.radius,
            arena_size - path.radius,
        ),
        _clamp(
            path.start_y + path_direction[1] * path.speed,
            path.radius,
            arena_size - path.radius,
        ),
    )
    observed = normalise(getattr(enemy, "direction", (0.0, 0.0)))
    enemy_next = (
        _clamp(
            enemy.pos[0] + observed[0] * enemy_speed,
            enemy.radius,
            arena_size - enemy.radius,
        ),
        _clamp(
            enemy.pos[1] + observed[1] * enemy_speed,
            enemy.radius,
            arena_size - enemy.radius,
        ),
    )
    next_gap = max(0.0, math.dist(own_next, enemy_next) - path.radius)
    progress = gap - next_gap
    if progress <= EPSILON:
        return 0.0
    prey_mass = _mass(enemy.radius)
    # Only a one-step contact is realised mass.  Deeper contact under a static
    # opponent is merely route progress; promoting it to full prey mass made a
    # slow walk tie a robust split-now capture.
    if gap <= progress:
        return prey_mass

    scale = max(path.speed * STATIC_HORIZON, EPSILON)
    return prey_mass * (math.exp(-next_gap / scale) - math.exp(-gap / scale))


def _rotate(direction: tuple[float, float], angle: float) -> tuple[float, float]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        direction[0] * cosine - direction[1] * sine,
        direction[0] * sine + direction[1] * cosine,
    )


def _distance_sq(first: tuple[float, float], second: tuple[float, float]) -> float:
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2


def _mass(radius: float) -> float:
    return radius * radius


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
