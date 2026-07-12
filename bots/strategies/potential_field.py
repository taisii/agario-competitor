from __future__ import annotations

import math
import random
from dataclasses import dataclass

from lib.config.arena import ARENA_SIZE, MAX_BLOB_COUNT
from lib.config.player import (
    BASE_PLAYER_SPEED,
    EAT_SIZE_RATIO,
    MIN_PLAYER_SPEED,
    PLAYER_SPEED_RADIUS_FACTOR,
    SPLIT_EJECT_SPEED,
    SPLIT_MIN_MASS,
)
from lib.models.blob_model import BlobModel, VisibleBlobModel
from lib.models.virus_model import VirusModel
from strategies.base import StrategyContext, StrategyDecision
from strategies.features import can_eat_player_blob, normalise, squared_distance


SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class PreyPlan:
    blob: VisibleBlobModel
    aim: tuple[float, float]
    score: float
    split: bool
    cluster_size: int


class PotentialFieldHunterStrategy:
    name = "potential_field_hunter"

    def __init__(self) -> None:
        self._last_direction: tuple[float, float] = (1.0, 0.0)
        self._rng = random.Random(271828)

    def choose(self, context: StrategyContext) -> StrategyDecision:
        state = context.game.state
        own_blobs = tuple(state.me.blobs.values())
        if not own_blobs:
            return StrategyDecision(direction=self._last_direction, reason="dead_fallback")

        arena_size = state.map.size or ARENA_SIZE
        primary = max(own_blobs, key=lambda blob: blob.radius)
        primary_pos = primary.pos
        primary_mass = _mass(primary.radius)
        enemies = tuple(state.visible_blobs)
        viruses = tuple(state.visible_viruses)

        early = state.round < 250
        threat_x, threat_y, threat_count, threat_level = self._threat_vector(
            own_blobs,
            enemies,
        )
        threatened = threat_count > 0

        wall_x, wall_y, wall_proximity = self._wall_vector(primary, arena_size)
        virus_x, virus_y, virus_count = self._virus_vector(own_blobs, viruses)
        shelter_x, shelter_y = self._shelter_vector(
            primary=primary,
            viruses=viruses,
            threatened=threatened,
        )
        food_x, food_y, food_count = self._food_vector(primary_pos, tuple(state.visible_food))
        rival_x, rival_y, rival_count = self._rival_vector(primary, enemies)
        prey_plan = self._best_prey(
            primary=primary,
            own_blob_count=len(own_blobs),
            enemies=enemies,
            viruses=viruses,
            early=early,
        )
        prey_x, prey_y = (
            normalise((prey_plan.aim[0] - primary_pos[0], prey_plan.aim[1] - primary_pos[1]))
            if prey_plan is not None
            else (0.0, 0.0)
        )

        rounds_left = max(0, state.max_rounds - state.round)
        clear_lead_late = False
        if rounds_left <= 200 and enemies:
            biggest_enemy = max(enemy.radius for enemy in enemies)
            clear_lead_late = state.me.radius > biggest_enemy * 1.25

        do_split = bool(
            prey_plan is not None
            and prey_plan.split
            and not threatened
            and not clear_lead_late
        )

        if do_split:
            direction = (prey_x, prey_y)
            reason = "split_prey"
        elif threatened:
            direction = (
                threat_x
                + _lerp(0.45, 2.2, wall_proximity) * normalise((wall_x, wall_y))[0]
                + (1.1 if primary_mass >= 4.0 else 0.55) * normalise((virus_x, virus_y))[0]
                + 0.7 * shelter_x
                + 0.15 * normalise((food_x, food_y))[0],
                threat_y
                + _lerp(0.45, 2.2, wall_proximity) * normalise((wall_x, wall_y))[1]
                + (1.1 if primary_mass >= 4.0 else 0.55) * normalise((virus_x, virus_y))[1]
                + 0.7 * shelter_y
                + 0.15 * normalise((food_x, food_y))[1],
            )
            reason = "flee_threat"
        else:
            diet = _clamp((primary_mass - 2.5) / 7.5, 0.0, 1.0)
            prey_weight = _lerp(1.7, 3.2, diet)
            food_weight = _lerp(1.0, 0.2, diet)
            rival_weight = 1.1
            if early:
                prey_weight = max(prey_weight, 2.6)
                food_weight = max(food_weight, 1.3)
                rival_weight *= 0.4
            if rival_count:
                food_weight = max(food_weight, 0.8)

            direction = (
                prey_weight * prey_x
                + food_weight * normalise((food_x, food_y))[0]
                + _lerp(0.55, 1.7, wall_proximity) * normalise((wall_x, wall_y))[0]
                + (2.0 if primary_mass >= 4.0 else 0.9) * normalise((virus_x, virus_y))[0]
                + rival_weight * normalise((rival_x, rival_y))[0],
                prey_weight * prey_y
                + food_weight * normalise((food_x, food_y))[1]
                + _lerp(0.55, 1.7, wall_proximity) * normalise((wall_x, wall_y))[1]
                + (2.0 if primary_mass >= 4.0 else 0.9) * normalise((virus_x, virus_y))[1]
                + rival_weight * normalise((rival_x, rival_y))[1],
            )
            reason = "potential_mix"

        direction = self._avoid_driving_into_wall(
            direction=direction,
            primary=primary,
            arena_size=arena_size,
            wall_proximity=wall_proximity,
            active=threatened or rival_count > 0,
        )
        direction = self._fallback_direction(direction, primary.pos, arena_size, wall_proximity)
        self._last_direction = direction

        return StrategyDecision(
            direction=direction,
            split=do_split,
            target_kind="prey" if prey_plan else ("escape" if threatened else "potential"),
            target_id=_blob_id(prey_plan.blob) if prey_plan is not None else None,
            reason=reason,
            score=prey_plan.score if prey_plan is not None else None,
            diagnostics={
                "early": early,
                "threat_count": threat_count,
                "threat_level": threat_level,
                "rival_count": rival_count,
                "food_count": food_count,
                "virus_avoid_count": virus_count,
                "wall_proximity": wall_proximity,
                "prey_cluster_size": prey_plan.cluster_size if prey_plan else 0,
                "clear_lead_late": clear_lead_late,
            },
        )

    def _threat_vector(
        self,
        own_blobs: tuple[BlobModel, ...],
        enemies: tuple[VisibleBlobModel, ...],
    ) -> tuple[float, float, int, float]:
        x = 0.0
        y = 0.0
        count = 0
        max_level = 0.0
        for own in own_blobs:
            for enemy in enemies:
                if not can_eat_player_blob(enemy.radius, own.radius):
                    continue
                distance = math.dist(own.pos, enemy.pos)
                split_capable = (
                    _mass(enemy.radius) >= SPLIT_MIN_MASS
                    and can_eat_player_blob(enemy.radius / SQRT2, own.radius)
                )
                reach = enemy.radius + _speed(enemy.radius) * 4.0 + 1.2
                if split_capable:
                    reach += 6.5
                if distance >= reach:
                    continue
                level = (reach - distance) / max(reach, 1e-9)
                if split_capable:
                    level *= 1.5
                away = normalise((own.pos[0] - enemy.pos[0], own.pos[1] - enemy.pos[1]))
                weight = level * _mass(own.radius)
                x += away[0] * weight
                y += away[1] * weight
                count += 1
                max_level = max(max_level, level)
        return (x, y, count, max_level)

    def _wall_vector(
        self,
        primary: BlobModel,
        arena_size: float,
    ) -> tuple[float, float, float]:
        margin = primary.radius + 3.5
        x, y = primary.pos
        left = max(0.0, (margin - x) / margin)
        right = max(0.0, (x - (arena_size - margin)) / margin)
        bottom = max(0.0, (margin - y) / margin)
        top = max(0.0, (y - (arena_size - margin)) / margin)
        return (left - right, bottom - top, max(left, right, bottom, top))

    def _virus_vector(
        self,
        own_blobs: tuple[BlobModel, ...],
        viruses: tuple[VirusModel, ...],
    ) -> tuple[float, float, int]:
        x = 0.0
        y = 0.0
        count = 0
        for own in own_blobs:
            for virus in viruses:
                if not _can_consume_virus(own.radius, virus.radius):
                    continue
                keep_clear = own.radius + 1.5 + own.radius * 0.5
                distance = math.dist(own.pos, virus.pos)
                if distance >= keep_clear:
                    continue
                away = normalise((own.pos[0] - virus.pos[0], own.pos[1] - virus.pos[1]))
                severity = (keep_clear - distance) / keep_clear
                x += away[0] * severity
                y += away[1] * severity
                count += 1
        return (x, y, count)

    def _shelter_vector(
        self,
        *,
        primary: BlobModel,
        viruses: tuple[VirusModel, ...],
        threatened: bool,
    ) -> tuple[float, float]:
        if not threatened or not viruses:
            return (0.0, 0.0)
        if _can_consume_virus(primary.radius, viruses[0].radius):
            return (0.0, 0.0)
        nearest = min(viruses, key=lambda virus: squared_distance(primary.pos, virus.pos))
        return normalise((nearest.pos[0] - primary.pos[0], nearest.pos[1] - primary.pos[1]))

    def _food_vector(
        self,
        origin: tuple[float, float],
        foods: tuple[object, ...],
    ) -> tuple[float, float, int]:
        x = 0.0
        y = 0.0
        for food in foods:
            pos = getattr(food, "pos")
            dx = pos[0] - origin[0]
            dy = pos[1] - origin[1]
            weight = 1.0 / (dx * dx + dy * dy + 1.0)
            x += dx * weight
            y += dy * weight
        return (x, y, len(foods))

    def _rival_vector(
        self,
        primary: BlobModel,
        enemies: tuple[VisibleBlobModel, ...],
    ) -> tuple[float, float, int]:
        x = 0.0
        y = 0.0
        count = 0
        for enemy in enemies:
            if can_eat_player_blob(primary.radius, enemy.radius):
                continue
            if can_eat_player_blob(enemy.radius, primary.radius):
                continue
            reach = primary.radius + enemy.radius + 8.0
            distance = math.dist(primary.pos, enemy.pos)
            if distance >= reach:
                continue
            severity = (reach - distance) / reach
            if enemy.radius >= primary.radius:
                severity *= 1.6
            away = normalise((primary.pos[0] - enemy.pos[0], primary.pos[1] - enemy.pos[1]))
            x += away[0] * severity
            y += away[1] * severity
            count += 1
        return (x, y, count)

    def _best_prey(
        self,
        *,
        primary: BlobModel,
        own_blob_count: int,
        enemies: tuple[VisibleBlobModel, ...],
        viruses: tuple[VirusModel, ...],
        early: bool,
    ) -> PreyPlan | None:
        eatable = [
            enemy
            for enemy in enemies
            if can_eat_player_blob(primary.radius, enemy.radius)
        ]
        if not eatable:
            return None

        chase_margin = 1.02 if early else 1.15
        split_min_mass = 2.0 if early else 2.2
        best: PreyPlan | None = None
        for enemy in eatable:
            distance = math.dist(primary.pos, enemy.pos)
            split_kill = (
                _mass(primary.radius) >= split_min_mass
                and can_eat_player_blob(primary.radius / SQRT2, enemy.radius)
                and distance <= _split_capture_reach(primary.radius)
                and own_blob_count < MAX_BLOB_COUNT
            )
            walkable = can_eat_player_blob(
                primary.radius,
                enemy.radius,
                radius_margin=chase_margin,
            )
            if not split_kill and not walkable:
                continue

            cluster = [
                other
                for other in eatable
                if math.dist(enemy.pos, other.pos) <= 4.0
            ]
            aim = enemy.pos
            score = _mass(enemy.radius) / (distance + 1.0)
            if split_kill:
                score *= 1.6

            if len(cluster) >= 3:
                total_mass = sum(_mass(blob.radius) for blob in cluster)
                score = total_mass / (distance + 1.0) * 2.2
                if split_kill:
                    score *= 1.6
                if viruses and min(math.dist(enemy.pos, virus.pos) for virus in viruses) <= 5.0:
                    score *= 1.3
                aim = (
                    sum(blob.pos[0] for blob in cluster) / len(cluster),
                    sum(blob.pos[1] for blob in cluster) / len(cluster),
                )

            plan = PreyPlan(
                blob=enemy,
                aim=aim,
                score=score,
                split=split_kill,
                cluster_size=len(cluster),
            )
            if best is None or plan.score > best.score:
                best = plan
        return best

    def _avoid_driving_into_wall(
        self,
        *,
        direction: tuple[float, float],
        primary: BlobModel,
        arena_size: float,
        wall_proximity: float,
        active: bool,
    ) -> tuple[float, float]:
        if not active:
            return direction
        margin = primary.radius + 3.5
        x, y = primary.pos
        dx, dy = direction
        if x < margin and dx < 0.0:
            dx = 0.0
        if x > arena_size - margin and dx > 0.0:
            dx = 0.0
        if y < margin and dy < 0.0:
            dy = 0.0
        if y > arena_size - margin and dy > 0.0:
            dy = 0.0
        if dx * dx + dy * dy < 1e-9 and wall_proximity > 0.0:
            center = arena_size / 2.0
            if min(x, arena_size - x) <= min(y, arena_size - y):
                dy = 1.0 if center > y else -1.0
            else:
                dx = 1.0 if center > x else -1.0
        return (dx, dy)

    def _fallback_direction(
        self,
        direction: tuple[float, float],
        position: tuple[float, float],
        arena_size: float,
        wall_proximity: float,
    ) -> tuple[float, float]:
        unit = normalise(direction)
        if unit != (0.0, 0.0):
            return unit

        center_direction = normalise((arena_size / 2.0 - position[0], arena_size / 2.0 - position[1]))
        jitter = self._rng.uniform(-0.55, 0.55)
        center_weight = 0.4 + 0.5 * wall_proximity
        base = (
            self._last_direction[0] * (1.0 - center_weight) + center_direction[0] * center_weight,
            self._last_direction[1] * (1.0 - center_weight) + center_direction[1] * center_weight,
        )
        cos_jitter = math.cos(jitter)
        sin_jitter = math.sin(jitter)
        return normalise(
            (
                base[0] * cos_jitter - base[1] * sin_jitter,
                base[0] * sin_jitter + base[1] * cos_jitter,
            )
        ) or self._last_direction


def _speed(radius: float) -> float:
    return max(MIN_PLAYER_SPEED, BASE_PLAYER_SPEED / (1.0 + radius * PLAYER_SPEED_RADIUS_FACTOR))


def _mass(radius: float) -> float:
    return radius * radius


def _split_capture_reach(radius: float) -> float:
    """Maximum one-round center distance for a directly aimed split capture."""

    child_radius = radius / SQRT2
    return 3.0 * child_radius + SPLIT_EJECT_SPEED + _speed(child_radius)


def _can_consume_virus(blob_radius: float, virus_radius: float) -> bool:
    return _mass(blob_radius) > _mass(virus_radius) * EAT_SIZE_RATIO


def _lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * _clamp(t, 0.0, 1.0)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _blob_id(blob: VisibleBlobModel) -> str:
    return f"{blob.player_id}:{blob.blob_id}"
