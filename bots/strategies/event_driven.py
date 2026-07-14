from __future__ import annotations

import math
from dataclasses import dataclass

from lib.config.arena import ARENA_SIZE, MAX_BLOB_COUNT
from lib.config.player import (
    EAT_SIZE_RATIO,
    FOOD_RADIUS,
    SAME_PLAYER_OVERLAP_EPSILON,
    SPLIT_EJECT_SPEED,
    SPLIT_MIN_MASS,
)
from lib.models.blob_model import BlobModel, VisibleBlobModel
from simulation.rules import can_consume_virus
from strategies.base import StrategyContext, StrategyDecision
from strategies.features import can_eat_player_blob, normalise, player_speed


SQRT2 = math.sqrt(2.0)
TAU = 2.0 * math.pi
STATIC_RESOURCE_HORIZON = 8
STATIC_DIRECTION_SAMPLES = 8
STATIC_FOOD_LIMIT = 12
STATIC_ROUTING_BLOB_LIMIT = 4
SAFETY_DIRECTION_SAMPLES = 16
EPSILON = 1.0e-9
SAFETY_DIRECTIONS = tuple(
    (
        math.cos(TAU * index / SAFETY_DIRECTION_SAMPLES),
        math.sin(TAU * index / SAFETY_DIRECTION_SAMPLES),
    )
    for index in range(SAFETY_DIRECTION_SAMPLES)
)
SAFETY_DIRECTION_INDEX = {
    (round(direction[0] * 1000), round(direction[1] * 1000)): index
    for index, direction in enumerate(SAFETY_DIRECTIONS)
}


@dataclass(frozen=True, slots=True)
class EventOpportunity:
    kind: str
    key: tuple[object, ...]
    target_pos: tuple[float, float]
    direction: tuple[float, float]
    value: float
    split: bool = False
    player_id: int | None = None
    target_radius: float | None = None


@dataclass(slots=True)
class TrackedTarget:
    kind: str
    pos: tuple[float, float]
    player_id: int | None = None
    radius: float | None = None


@dataclass(frozen=True, slots=True)
class ProjectedFragment:
    source_index: int
    pos: tuple[float, float]
    radius: float

    @property
    def mass(self) -> float:
        return self.radius * self.radius


@dataclass(frozen=True, slots=True)
class EnemyProjectionGeometry:
    unsplit_radius: float
    unsplit_speed: float
    split_radius: float | None
    split_speed: float
    split_launch: float


EnemyEaterSeed = tuple[float, float, float, int]
EnemyModeProjections = tuple[
    tuple[EnemyEaterSeed, ...],
    tuple[EnemyEaterSeed, ...],
]


@dataclass(frozen=True, slots=True)
class EnemyReachabilityCache:
    geometries: tuple[EnemyProjectionGeometry, ...]
    fixed_projections: tuple[tuple[EnemyModeProjections, ...], ...]


@dataclass(frozen=True, slots=True)
class StaticBlobGeometry:
    pos: tuple[float, float]
    radius: float
    travel: float


@dataclass(frozen=True, slots=True)
class StaticSegment:
    start: tuple[float, float]
    end: tuple[float, float]
    delta: tuple[float, float]
    length_squared: float
    radius: float


@dataclass(frozen=True, slots=True)
class ShieldResult:
    direction: tuple[float, float]
    split: bool
    retained_mass: float
    catastrophe: bool


class EventDrivenStaticSearchStrategy:
    """Static growth, local opportunities, and catastrophe safety.

    Opponents never enter the H8 resource projection. Executable prey and
    virus events are valued from their current one-round capture geometry;
    there is no opponent trajectory tree. A one-round reachable-set shield
    validates the selected command, including post-split fragments and
    capture cascades, on every turn.
    """

    name = "event_driven_static_search"

    def __init__(self) -> None:
        self._last_direction = (1.0, 0.0)
        self._opportunity_key: tuple[object, ...] | None = None
        self._committed_split_target: TrackedTarget | None = None
        self._blocked_split_target: TrackedTarget | None = None
        self._probed_split_target: TrackedTarget | None = None
        self._blocked_split_until_round = -1
        self._safety_overrides = 0

    def choose(self, context: StrategyContext) -> StrategyDecision:
        state = context.game.state
        own = tuple(state.me.blobs.values())
        if not own:
            return StrategyDecision(
                direction=self._last_direction, reason="dead_fallback"
            )

        arena_size = float(state.map.size or ARENA_SIZE)
        enemies = tuple(state.visible_blobs)
        round_number = int(state.round)
        self._committed_split_target = _refresh_tracked_prey(
            self._committed_split_target,
            enemies,
        )
        self._probed_split_target = _refresh_tracked_prey(
            self._probed_split_target,
            enemies,
        )
        if round_number >= self._blocked_split_until_round:
            self._blocked_split_target = None
        else:
            self._blocked_split_target = _refresh_tracked_prey(
                self._blocked_split_target,
                enemies,
            )
        backbone_direction, backbone_score = _static_backbone_direction(
            own=own,
            foods=tuple(state.visible_food),
            previous=self._last_direction,
            arena_size=arena_size,
        )
        opportunity = _best_event_opportunity(
            own=own,
            enemies=enemies,
            viruses=tuple(state.visible_viruses),
            arena_size=arena_size,
            route_value=backbone_score,
            previous_key=self._opportunity_key,
            own_player_id=int(state.me.player_id),
            rankings=tuple(int(player_id) for player_id in state.rankings),
            view_center=state.view_center,
            vision_size=float(state.vision_size),
            suppressed_split_targets=tuple(
                target
                for target in (
                    self._committed_split_target,
                    self._blocked_split_target,
                )
                if target is not None
            ),
            suppressed_probe_targets=(
                (self._probed_split_target,)
                if self._probed_split_target is not None
                else ()
            ),
        )
        topology_stage = _topology_stage(len(own))
        nominal = (
            StrategyDecision(
                direction=opportunity.direction,
                split=opportunity.split,
                target_kind=opportunity.kind,
                reason=opportunity.kind,
                score=opportunity.value,
            )
            if opportunity is not None
            else StrategyDecision(
                direction=backbone_direction,
                target_kind="food",
                reason="static_backbone",
                score=backbone_score,
            )
        )
        split_once = nominal.split
        shield = _shield_action(
            own=own,
            enemies=enemies,
            nominal=normalise(nominal.direction),
            split=split_once,
            arena_size=arena_size,
        )
        overridden = (
            shield.direction != normalise(nominal.direction)
            or shield.split != split_once
        )
        if overridden:
            self._safety_overrides += 1
        self._last_direction = shield.direction
        rejected_split = (
            opportunity is not None
            and nominal.split
            and split_once
            and not shield.split
        )
        if rejected_split:
            # Capture reach alone is insufficient: another predator may make
            # the post-split state losing.  Do not keep pursuing a split-only
            # target after the safety layer vetoes the actual split.
            self._blocked_split_target = _target_from_opportunity(opportunity)
            self._blocked_split_until_round = round_number + 5
        elif opportunity is not None and opportunity.kind == "prey_probe":
            # The visible fragment is valuable, but leaderboard mass and the
            # current view boundary permit a same-player hidden blob to eat a
            # split child before the next observation.  Approach once without
            # committing, then suppress repeated probes while the target is
            # spatially continuous.
            self._probed_split_target = _target_from_opportunity(opportunity)
        elif opportunity is not None and shield.split:
            self._committed_split_target = _target_from_opportunity(opportunity)
        self._opportunity_key = opportunity.key if opportunity is not None else None

        return StrategyDecision(
            direction=shield.direction,
            split=shield.split,
            target_kind="escape" if overridden else nominal.target_kind,
            target_id=None if overridden else nominal.target_id,
            reason="reachable_safety_override" if overridden else nominal.reason,
            score=nominal.score,
            diagnostics={
                **nominal.diagnostics,
                "replan_reason": "event_started" if opportunity is not None else None,
                "event_kind": opportunity.kind if opportunity is not None else None,
                "topology_stage": topology_stage,
                "planner_calls": 0,
                "safety_overrides": self._safety_overrides,
                "static_backbone_score": round(backbone_score, 6),
                "shield_retained_mass": round(shield.retained_mass, 6),
                "shield_catastrophe": shield.catastrophe,
            },
        )


def _static_backbone_direction(
    *,
    own: tuple[BlobModel, ...],
    foods: tuple[object, ...],
    previous: tuple[float, float],
    arena_size: float,
) -> tuple[tuple[float, float], float]:
    previous = normalise(previous) or (1.0, 0.0)
    route_own = _routing_blobs(own)
    route_geometry = tuple(
        StaticBlobGeometry(
            pos=blob.pos,
            radius=blob.radius,
            travel=player_speed(blob.radius) * STATIC_RESOURCE_HORIZON,
        )
        for blob in route_own
    )
    candidates = [previous]
    candidates.extend(
        (
            math.cos(TAU * index / STATIC_DIRECTION_SAMPLES),
            math.sin(TAU * index / STATIC_DIRECTION_SAMPLES),
        )
        for index in range(STATIC_DIRECTION_SAMPLES)
    )
    selected_foods = tuple(
        sorted(
            foods,
            key=lambda food: min(
                max(0.0, math.dist(blob.pos, food.pos) - blob.radius)
                for blob in route_own
            ),
        )[:STATIC_FOOD_LIMIT]
    )
    start_gaps = tuple(
        tuple(
            max(0.0, math.dist(blob.pos, food.pos) - blob.radius)
            for blob in route_geometry
        )
        for food in selected_foods
    )
    for food in selected_foods[:6]:
        source = min(
            route_own, key=lambda blob: math.dist(blob.pos, food.pos) - blob.radius
        )
        candidates.append(
            normalise((food.pos[0] - source.pos[0], food.pos[1] - source.pos[1]))
        )
    candidates.extend(
        _static_food_sweep_directions(
            own=route_own,
            foods=selected_foods,
            previous=previous,
            horizon=STATIC_RESOURCE_HORIZON,
            limit=2,
            travel_by_blob=tuple(blob.travel for blob in route_geometry),
        )
    )
    candidates.extend(_food_density_directions(route_own, selected_foods))

    best_direction = previous
    best_score = -math.inf
    seen: set[tuple[int, int]] = set()
    for direction in candidates:
        direction = normalise(direction)
        if direction == (0.0, 0.0):
            continue
        key = (round(direction[0] * 1000), round(direction[1] * 1000))
        if key in seen:
            continue
        seen.add(key)
        score = _static_direction_value(
            own=route_geometry,
            foods=selected_foods,
            start_gaps=start_gaps,
            direction=direction,
            arena_size=arena_size,
        )
        score += 0.01 * (direction[0] * previous[0] + direction[1] * previous[1])
        if score > best_score:
            best_score = score
            best_direction = direction

    if not selected_foods:
        center = _mass_center(own)
        wall_direction = normalise(
            (arena_size / 2.0 - center[0], arena_size / 2.0 - center[1])
        )
        if _minimum_wall_clearance(own, arena_size) < 1.0:
            best_direction = wall_direction or previous
        best_score = 0.0
    return (best_direction, best_score)


def _routing_blobs(own: tuple[BlobModel, ...]) -> tuple[BlobModel, ...]:
    if len(own) <= STATIC_ROUTING_BLOB_LIMIT:
        return own
    center = _mass_center(own)
    ranked = sorted(
        own,
        key=lambda blob: (
            -blob.radius,
            -math.dist(blob.pos, center),
            blob.blob_id,
        ),
    )
    selected: list[BlobModel] = [ranked[0]]
    while len(selected) < STATIC_ROUTING_BLOB_LIMIT:
        candidate = max(
            (blob for blob in own if blob not in selected),
            key=lambda blob: min(
                math.dist(blob.pos, chosen.pos) for chosen in selected
            ),
        )
        selected.append(candidate)
    return tuple(selected)


def _food_density_directions(
    own: tuple[BlobModel, ...],
    foods: tuple[object, ...],
) -> tuple[tuple[float, float], ...]:
    if not foods:
        return ()
    center = _mass_center(own)
    scored: list[tuple[float, tuple[float, float]]] = []
    for food in foods:
        neighbours = tuple(
            other
            for other in foods
            if (food.pos[0] - other.pos[0]) ** 2 + (food.pos[1] - other.pos[1]) ** 2
            <= 9.0
        )
        target = (
            sum(other.pos[0] for other in neighbours) / len(neighbours),
            sum(other.pos[1] for other in neighbours) / len(neighbours),
        )
        score = (len(neighbours) + 0.5) / (math.dist(center, target) + 1.5)
        scored.append((score, target))
    scored.sort(reverse=True)
    return tuple(
        normalise((target[0] - center[0], target[1] - center[1]))
        for _, target in scored[:3]
    )


def _static_food_sweep_directions(
    *,
    own: tuple[BlobModel, ...],
    foods: tuple[object, ...],
    previous: tuple[float, float],
    horizon: int,
    limit: int,
    travel_by_blob: tuple[float, ...] | None = None,
) -> tuple[tuple[float, float], ...]:
    """Headings whose swept fragment disks cover the most distinct food.

    Each food contributes angular intervals, not a trajectory branch.  The
    intervals from all own fragments are unioned before the sweep so one food
    is counted once even when several fragments can collect it.
    """

    if not own or not foods or horizon <= 0 or limit <= 0:
        return ()
    events: dict[float, float] = {}
    base_score = 0.0
    for food in foods:
        intervals: list[tuple[float, float]] = []
        direction_independent = False
        for blob_index, blob in enumerate(own):
            delta_x = food.pos[0] - blob.pos[0]
            delta_y = food.pos[1] - blob.pos[1]
            distance = math.hypot(delta_x, delta_y)
            if distance <= blob.radius + EPSILON:
                direction_independent = True
                break
            travel = (
                travel_by_blob[blob_index]
                if travel_by_blob is not None
                else player_speed(blob.radius) * horizon
            )
            if travel <= EPSILON or distance > travel + blob.radius:
                continue
            tangent_distance = math.sqrt(
                max(0.0, distance * distance - blob.radius * blob.radius)
            )
            if travel + EPSILON >= tangent_distance:
                half_width = math.asin(_clamp(blob.radius / distance, 0.0, 1.0))
            else:
                cosine = (
                    distance * distance + travel * travel - blob.radius * blob.radius
                ) / (2.0 * distance * travel)
                half_width = math.acos(_clamp(cosine, -1.0, 1.0))
            center = math.atan2(delta_y, delta_x) % TAU
            start = center - half_width
            end = center + half_width
            if start < 0.0:
                intervals.extend(((0.0, end), (start + TAU, TAU)))
            elif end >= TAU:
                intervals.extend(((0.0, end - TAU), (start, TAU)))
            else:
                intervals.append((start, end))
        if direction_independent or not intervals:
            continue

        intervals.sort()
        merged: list[list[float]] = []
        for start, end in intervals:
            if merged and start <= merged[-1][1] + EPSILON:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        for start, end in merged:
            if start <= EPSILON:
                base_score += 1.0
            else:
                events[start] = events.get(start, 0.0) + 1.0
            if end < TAU - EPSILON:
                events[end] = events.get(end, 0.0) - 1.0

    boundaries = sorted({0.0, TAU, *events})
    active = base_score
    segments: list[tuple[float, float]] = []
    for index, start in enumerate(boundaries[:-1]):
        if start > EPSILON:
            active += events.get(start, 0.0)
        end = boundaries[index + 1]
        if active > 0.0 and end - start > EPSILON:
            angle = (start + end) / 2.0
            alignment = math.cos(angle) * previous[0] + math.sin(angle) * previous[1]
            segments.append((active, alignment, angle))
    segments.sort(reverse=True)
    return tuple((math.cos(angle), math.sin(angle)) for _, _, angle in segments[:limit])


def _static_direction_value(
    *,
    own: tuple[StaticBlobGeometry, ...],
    foods: tuple[object, ...],
    start_gaps: tuple[tuple[float, ...], ...],
    direction: tuple[float, float],
    arena_size: float,
) -> float:
    segments = tuple(
        _static_segment(blob, direction=direction, arena_size=arena_size)
        for blob in own
    )
    values: list[float] = []
    food_mass = FOOD_RADIUS * FOOD_RADIUS
    for food_index, food in enumerate(foods):
        best = 0.0
        for blob_index, segment in enumerate(segments):
            start_gap = start_gaps[food_index][blob_index]
            path_gap = max(
                0.0,
                _point_static_segment_distance(food.pos, segment) - segment.radius,
            )
            end_gap = max(
                0.0,
                math.hypot(
                    segment.end[0] - food.pos[0],
                    segment.end[1] - food.pos[1],
                )
                - segment.radius,
            )
            if path_gap <= EPSILON:
                value = food_mass
            else:
                progress = max(0.0, start_gap - end_gap)
                value = food_mass * progress / max(start_gap, 1.0)
            best = max(best, value)
        values.append(best)
    values.sort(reverse=True)
    return sum(
        value * weight
        for value, weight in zip(values[:8], (1.0, 0.8, 0.65, 0.5, 0.4, 0.3, 0.2, 0.1))
    )


def _static_segment(
    blob: StaticBlobGeometry,
    *,
    direction: tuple[float, float],
    arena_size: float,
) -> StaticSegment:
    end = (
        _clamp(
            blob.pos[0] + direction[0] * blob.travel,
            blob.radius,
            arena_size - blob.radius,
        ),
        _clamp(
            blob.pos[1] + direction[1] * blob.travel,
            blob.radius,
            arena_size - blob.radius,
        ),
    )
    delta = (end[0] - blob.pos[0], end[1] - blob.pos[1])
    return StaticSegment(
        start=blob.pos,
        end=end,
        delta=delta,
        length_squared=delta[0] * delta[0] + delta[1] * delta[1],
        radius=blob.radius,
    )


def _point_static_segment_distance(
    point: tuple[float, float],
    segment: StaticSegment,
) -> float:
    if segment.length_squared <= EPSILON:
        return math.hypot(
            point[0] - segment.start[0],
            point[1] - segment.start[1],
        )
    projection = (
        (point[0] - segment.start[0]) * segment.delta[0]
        + (point[1] - segment.start[1]) * segment.delta[1]
    ) / segment.length_squared
    projection = _clamp(projection, 0.0, 1.0)
    closest_x = segment.start[0] + projection * segment.delta[0]
    closest_y = segment.start[1] + projection * segment.delta[1]
    return math.hypot(point[0] - closest_x, point[1] - closest_y)


def _best_event_opportunity(
    *,
    own: tuple[BlobModel, ...],
    enemies: tuple[VisibleBlobModel, ...],
    viruses: tuple[object, ...],
    arena_size: float,
    route_value: float,
    previous_key: tuple[object, ...] | None,
    own_player_id: int,
    rankings: tuple[int, ...],
    view_center: tuple[float, float],
    vision_size: float,
    suppressed_split_targets: tuple[TrackedTarget, ...],
    suppressed_probe_targets: tuple[TrackedTarget, ...],
) -> EventOpportunity | None:
    rows: list[tuple[float, EventOpportunity]] = []
    own_total_mass = sum(blob.radius * blob.radius for blob in own)
    visible_mass_by_player: dict[int, float] = {}
    for enemy in enemies:
        player_id = int(enemy.player_id)
        visible_mass_by_player[player_id] = (
            visible_mass_by_player.get(player_id, 0.0) + enemy.radius * enemy.radius
        )
    for enemy in enemies:
        enemy_mass = enemy.radius * enemy.radius
        prey_speed = player_speed(enemy.radius)
        split_key = (
            "split_prey",
            int(enemy.player_id),
            round(enemy.pos[0] * 2.0),
            round(enemy.pos[1] * 2.0),
            round(enemy.radius * 4.0),
        )
        split_suppressed = any(
            _tracked_prey_matches(target, enemy) for target in suppressed_split_targets
        )
        probe_suppressed = any(
            _tracked_prey_matches(target, enemy) for target in suppressed_probe_targets
        )
        for blob in own:
            distance = math.hypot(
                blob.pos[0] - enemy.pos[0],
                blob.pos[1] - enemy.pos[1],
            )
            gap = max(0.0, distance - blob.radius)
            if not split_suppressed and _split_capture_is_robust(
                blob,
                enemy,
                len(own),
                distance=distance,
                arena_size=arena_size,
            ):
                key = split_key
                value = enemy_mass - route_value
                if key == previous_key:
                    value *= 1.1
                direction = normalise(
                    (enemy.pos[0] - blob.pos[0], enemy.pos[1] - blob.pos[1])
                )
                if direction == (0.0, 0.0):
                    continue
                if _hidden_mass_can_punish_split(
                    source=blob,
                    direction=direction,
                    target_player_id=int(enemy.player_id),
                    visible_target_mass=visible_mass_by_player[int(enemy.player_id)],
                    own_total_mass=own_total_mass,
                    own_player_id=own_player_id,
                    rankings=rankings,
                    view_center=view_center,
                    vision_size=vision_size,
                    arena_size=arena_size,
                ):
                    probe_key = (
                        "prey_probe",
                        int(enemy.player_id),
                        round(enemy.pos[0] * 2.0),
                        round(enemy.pos[1] * 2.0),
                        round(enemy.radius * 4.0),
                    )
                    if value > 0.0 and not probe_suppressed:
                        rows.append(
                            (
                                value / max(gap, 1.0),
                                EventOpportunity(
                                    "prey_probe",
                                    probe_key,
                                    enemy.pos,
                                    direction,
                                    value,
                                    False,
                                    int(enemy.player_id),
                                    enemy.radius,
                                ),
                            )
                        )
                    continue
                rows.append(
                    (
                        value / max(gap, 1.0),
                        EventOpportunity(
                            "split_prey",
                            key,
                            enemy.pos,
                            direction,
                            value,
                            True,
                            int(enemy.player_id),
                            enemy.radius,
                        ),
                    )
                )
                continue
            if not can_eat_player_blob(blob.radius, enemy.radius):
                continue

            direction = normalise(
                (enemy.pos[0] - blob.pos[0], enemy.pos[1] - blob.pos[1])
            )
            if direction == (0.0, 0.0):
                continue
            hunter_end = (
                _clamp(
                    blob.pos[0] + direction[0] * player_speed(blob.radius),
                    blob.radius,
                    arena_size - blob.radius,
                ),
                _clamp(
                    blob.pos[1] + direction[1] * player_speed(blob.radius),
                    blob.radius,
                    arena_size - blob.radius,
                ),
            )
            capture_probability = _capture_probability(
                distance=math.dist(hunter_end, enemy.pos),
                direct_reach=blob.radius,
                prey_speed=prey_speed,
                enemy=enemy,
                hunter_pos=hunter_end,
                arena_size=arena_size,
            )
            if capture_probability < 0.35:
                continue
            key = (
                "prey",
                int(enemy.player_id),
                round(enemy.pos[0] * 2.0),
                round(enemy.pos[1] * 2.0),
                round(enemy.radius * 4.0),
            )
            value = capture_probability * enemy_mass - route_value
            if key == previous_key:
                value *= 1.1
            if value > 0.0:
                rows.append(
                    (
                        value / max(gap + prey_speed, 1.0),
                        EventOpportunity(
                            "prey",
                            key,
                            enemy.pos,
                            direction,
                            value,
                            False,
                            int(enemy.player_id),
                            enemy.radius,
                        ),
                    )
                )

    for virus in viruses:
        harvesters = tuple(
            blob
            for blob in own
            if can_consume_virus(
                blob.radius,
                virus.radius,
                eat_size_ratio=EAT_SIZE_RATIO,
            )
        )
        if not harvesters:
            continue
        hunter = min(harvesters, key=lambda blob: math.dist(blob.pos, virus.pos))
        gap = max(0.0, math.dist(hunter.pos, virus.pos) - hunter.radius)
        if gap > player_speed(hunter.radius) * STATIC_RESOURCE_HORIZON:
            continue
        direction = normalise(
            (virus.pos[0] - hunter.pos[0], virus.pos[1] - hunter.pos[1])
        )
        if direction == (0.0, 0.0):
            continue
        key = (
            "virus",
            round(virus.pos[0] * 2.0),
            round(virus.pos[1] * 2.0),
        )
        value = virus.radius * virus.radius - route_value
        if key == previous_key:
            value *= 1.1
        rows.append(
            (
                value / max(gap, 1.0),
                EventOpportunity(
                    "virus",
                    key,
                    virus.pos,
                    direction,
                    value,
                    False,
                    target_radius=virus.radius,
                ),
            )
        )
    return max(rows, key=lambda row: row[0])[1] if rows else None


def _capture_probability(
    *,
    distance: float,
    direct_reach: float,
    prey_speed: float,
    enemy: VisibleBlobModel,
    hunter_pos: tuple[float, float],
    arena_size: float,
) -> float:
    """Approximate the fraction of one-step prey responses we can capture.

    The prey is not projected along a guessed direction.  Its possible
    one-step displacement is treated as a disk: fully inside the capture disk
    is probability one, fully outside is zero, and the boundary band is a
    smooth opportunity score.  A nearby wall removes part of the escape disk
    without turning this into a minimax rejection.
    """

    if prey_speed <= EPSILON:
        return 1.0 if distance <= direct_reach else 0.0
    probability = _clamp(
        (direct_reach + prey_speed - distance) / (2.0 * prey_speed),
        0.0,
        1.0,
    )
    escape = normalise((enemy.pos[0] - hunter_pos[0], enemy.pos[1] - hunter_pos[1]))
    escape_end = (
        _clamp(
            enemy.pos[0] + escape[0] * prey_speed,
            enemy.radius,
            arena_size - enemy.radius,
        ),
        _clamp(
            enemy.pos[1] + escape[1] * prey_speed,
            enemy.radius,
            arena_size - enemy.radius,
        ),
    )
    available_escape = math.dist(enemy.pos, escape_end)
    if probability > 0.0 and available_escape < prey_speed:
        blocked_fraction = 1.0 - _clamp(
            available_escape / prey_speed,
            0.0,
            1.0,
        )
        probability += (1.0 - probability) * 0.5 * blocked_fraction
    return _clamp(probability, 0.0, 1.0)


def _split_capture_is_robust(
    blob: BlobModel,
    enemy: VisibleBlobModel,
    own_count: int,
    *,
    distance: float,
    arena_size: float,
) -> bool:
    if own_count >= MAX_BLOB_COUNT or blob.radius * blob.radius < SPLIT_MIN_MASS:
        return False
    child_radius = blob.radius / SQRT2
    if not can_eat_player_blob(child_radius, enemy.radius):
        return False
    prey_speed = player_speed(enemy.radius)
    if distance + prey_speed > _split_attack_reach(blob.radius):
        return False
    direction = normalise((enemy.pos[0] - blob.pos[0], enemy.pos[1] - blob.pos[1]))
    if direction == (0.0, 0.0):
        return False
    speed = player_speed(child_radius)
    parent_pos = (
        _clamp(
            blob.pos[0] + direction[0] * speed,
            child_radius,
            arena_size - child_radius,
        ),
        _clamp(
            blob.pos[1] + direction[1] * speed,
            child_radius,
            arena_size - child_radius,
        ),
    )
    child_pos = _split_child_endpoint(
        blob,
        direction=direction,
        arena_size=arena_size,
    )
    hunter_pos = min(
        (parent_pos, child_pos),
        key=lambda pos: math.dist(pos, enemy.pos),
    )
    return (
        _capture_probability(
            distance=math.dist(hunter_pos, enemy.pos),
            direct_reach=child_radius,
            prey_speed=prey_speed,
            enemy=enemy,
            hunter_pos=hunter_pos,
            arena_size=arena_size,
        )
        >= 0.35
    )


def _hidden_mass_can_punish_split(
    *,
    source: BlobModel,
    direction: tuple[float, float],
    target_player_id: int,
    visible_target_mass: float,
    own_total_mass: float,
    own_player_id: int,
    rankings: tuple[int, ...],
    view_center: tuple[float, float],
    vision_size: float,
    arena_size: float,
) -> bool:
    """Whether a still-invisible same-player blob can eat the split child.

    Rankings provide a mass interval, not a point estimate.  For a lower
    ranked target, our total mass is an upper bound on its total mass.  For a
    higher ranked target, only the mass it must hide to outrank us is treated
    as certain enough for a hard veto.  Within that interval, endpoint masses
    suffice: the smallest edible blob maximises normal-move reach, while the
    largest possible blob maximises split reach.
    """

    if vision_size <= EPSILON or own_player_id not in rankings:
        return False
    try:
        own_rank = rankings.index(own_player_id)
        target_rank = rankings.index(target_player_id)
    except ValueError:
        return False

    child_radius = source.radius / SQRT2
    child_mass = child_radius * child_radius
    edible_mass = child_mass * EAT_SIZE_RATIO
    rank_implied_hidden_mass = max(0.0, own_total_mass - visible_target_mass)
    if target_rank > own_rank:
        # The lower-ranked target cannot exceed our total mass.
        hidden_mass_limit = rank_implied_hidden_mass
        if hidden_mass_limit + EPSILON < edible_mass:
            return False
        candidate_masses = (edible_mass, hidden_mass_limit)
    else:
        # A higher-ranked target must hide at least this much mass.  Do not
        # invent an unbounded worst case beyond that evidence.
        if rank_implied_hidden_mass + EPSILON < edible_mass:
            return False
        candidate_masses = (rank_implied_hidden_mass,)

    child_pos = _split_child_endpoint(
        source,
        direction=direction,
        arena_size=arena_size,
    )
    for hidden_mass in candidate_masses:
        hidden_radius = math.sqrt(max(hidden_mass, edible_mass))
        distance_to_unseen_center = _distance_to_unseen_circle_center(
            point=child_pos,
            circle_radius=hidden_radius,
            view_center=view_center,
            vision_size=vision_size,
        )
        if (
            distance_to_unseen_center
            <= _one_step_attack_reach(
                hidden_radius,
                child_radius,
            )
            + EPSILON
        ):
            return True
    return False


def _split_child_endpoint(
    blob: BlobModel,
    *,
    direction: tuple[float, float],
    arena_size: float,
) -> tuple[float, float]:
    child_radius = blob.radius / SQRT2
    launch = (
        2.0 * child_radius
        + SAME_PLAYER_OVERLAP_EPSILON
        + player_speed(child_radius)
        + SPLIT_EJECT_SPEED
    )
    return (
        _clamp(
            blob.pos[0] + direction[0] * launch,
            child_radius,
            arena_size - child_radius,
        ),
        _clamp(
            blob.pos[1] + direction[1] * launch,
            child_radius,
            arena_size - child_radius,
        ),
    )


def _distance_to_unseen_circle_center(
    *,
    point: tuple[float, float],
    circle_radius: float,
    view_center: tuple[float, float],
    vision_size: float,
) -> float:
    """Minimum distance from ``point`` to a center whose circle is unseen.

    The engine sees a circle exactly when it intersects the axis-aligned view
    square.  Invisible centers are therefore outside the square dilated by
    the circle radius.  This is the signed-distance function of that rounded
    square, so corners and wall-clamped view centers need no special cases.
    """

    half_view = vision_size / 2.0
    dx = abs(point[0] - view_center[0]) - half_view
    dy = abs(point[1] - view_center[1]) - half_view
    outside_distance = math.hypot(max(dx, 0.0), max(dy, 0.0))
    inside_distance = min(max(dx, dy), 0.0)
    signed_distance_to_view = outside_distance + inside_distance
    return max(0.0, circle_radius - signed_distance_to_view)


def _target_from_opportunity(opportunity: EventOpportunity) -> TrackedTarget:
    return TrackedTarget(
        kind="prey",
        pos=opportunity.target_pos,
        player_id=opportunity.player_id,
        radius=opportunity.target_radius,
    )


def _tracked_prey_matches(
    target: TrackedTarget,
    enemy: VisibleBlobModel,
) -> bool:
    if target.kind != "prey" or int(enemy.player_id) != target.player_id:
        return False
    if target.radius is not None and abs(enemy.radius - target.radius) > max(
        0.2,
        target.radius * 0.25,
    ):
        return False
    continuity_radius = player_speed(target.radius or enemy.radius) * 2.0 + 0.5
    return math.dist(enemy.pos, target.pos) <= continuity_radius


def _refresh_tracked_prey(
    target: TrackedTarget | None,
    enemies: tuple[VisibleBlobModel, ...],
) -> TrackedTarget | None:
    if target is None:
        return None
    candidates = tuple(
        enemy for enemy in enemies if _tracked_prey_matches(target, enemy)
    )
    if not candidates:
        return None
    current = min(candidates, key=lambda enemy: math.dist(enemy.pos, target.pos))
    target.pos = current.pos
    target.radius = current.radius
    return target


def _tracked_event_direction(
    *,
    context: StrategyContext,
    target: TrackedTarget | None,
    fallback: tuple[float, float],
) -> tuple[float, float] | None:
    if target is None:
        return None
    state = context.game.state
    own = tuple(state.me.blobs.values())
    if target.kind == "escape":
        enemies = tuple(state.visible_blobs)
        escape = _escape_vector(own, enemies)
        return escape if escape != (0.0, 0.0) else None
    if target.kind == "virus":
        candidates = tuple(
            virus
            for virus in state.visible_viruses
            if math.dist(virus.pos, target.pos) <= 1.0
        )
    else:
        candidates = tuple(
            enemy
            for enemy in state.visible_blobs
            if _tracked_prey_matches(target, enemy)
        )
    if not candidates:
        return None
    current = min(candidates, key=lambda item: math.dist(item.pos, target.pos))
    target.pos = current.pos
    if hasattr(current, "radius"):
        target.radius = current.radius
    capable = (
        tuple(
            blob
            for blob in own
            if target.kind == "virus"
            or can_eat_player_blob(blob.radius, current.radius)
        )
        or own
    )
    source = min(capable, key=lambda blob: math.dist(blob.pos, current.pos))
    direction = normalise(
        (current.pos[0] - source.pos[0], current.pos[1] - source.pos[1])
    )
    return direction or normalise(fallback)


def _shield_action(
    *,
    own: tuple[BlobModel, ...],
    enemies: tuple[VisibleBlobModel, ...],
    nominal: tuple[float, float],
    split: bool,
    arena_size: float,
) -> ShieldResult:
    evaluations: dict[
        tuple[bool, int, int],
        tuple[ShieldResult, tuple[ProjectedFragment, ...]],
    ] = {}
    nominal_key = (
        split,
        round(nominal[0] * 1000),
        round(nominal[1] * 1000),
    )
    nominal_fragments = _project_one_step_fragments(
        own=own,
        direction=nominal,
        split=split,
        arena_size=arena_size,
    )
    nominal_result = _evaluate_projected_fragments(
        fragments=nominal_fragments,
        enemies=enemies,
        direction=nominal,
        split=split,
        arena_size=arena_size,
    )
    evaluations[nominal_key] = (nominal_result, nominal_fragments)
    total_mass = sum(blob.radius * blob.radius for blob in own)
    if nominal_result.retained_mass >= total_mass - EPSILON:
        return nominal_result

    directions = [nominal]
    directions.extend(SAFETY_DIRECTIONS)
    escape = _escape_vector(own, enemies)
    if escape != (0.0, 0.0):
        directions.extend(
            (escape, _rotate(escape, math.pi / 2), _rotate(escape, -math.pi / 2))
        )

    enemy_cache = _prepare_enemy_reachability_cache(enemies, arena_size)

    rows: list[tuple[float, float, float, bool, tuple[float, float]]] = []
    seen: set[tuple[bool, int, int]] = set()
    for direction in directions:
        direction = normalise(direction)
        if direction == (0.0, 0.0):
            continue
        modes = (False, split) if split and direction == nominal else (False,)
        for split_mode in modes:
            key = (
                split_mode,
                round(direction[0] * 1000),
                round(direction[1] * 1000),
            )
            if key in seen:
                continue
            seen.add(key)
            cached = evaluations.get(key)
            if cached is None:
                fragments = _project_one_step_fragments(
                    own=own,
                    direction=direction,
                    split=split_mode,
                    arena_size=arena_size,
                )
                evaluated = _evaluate_projected_fragments(
                    fragments=fragments,
                    enemies=enemies,
                    direction=direction,
                    split=split_mode,
                    arena_size=arena_size,
                    enemy_cache=enemy_cache,
                )
                evaluations[key] = (evaluated, fragments)
            else:
                evaluated, fragments = cached
            retained = evaluated.retained_mass
            margin = _minimum_fragment_margin(fragments, enemies)
            alignment = direction[0] * nominal[0] + direction[1] * nominal[1]
            rows.append((retained, margin, alignment, split_mode, direction))

    retained, _margin, _alignment, selected_split, direction = max(
        rows,
        key=lambda row: (
            row[0] > EPSILON,
            round(row[0], 9),
            row[1],
            row[2],
            row[3],
        ),
    )
    return ShieldResult(
        direction=direction,
        split=selected_split,
        retained_mass=retained,
        catastrophe=retained <= EPSILON,
    )


def _evaluate_shield_candidate(
    *,
    own: tuple[BlobModel, ...],
    enemies: tuple[VisibleBlobModel, ...],
    direction: tuple[float, float],
    split: bool,
    arena_size: float,
) -> ShieldResult:
    fragments = _project_one_step_fragments(
        own=own,
        direction=direction,
        split=split,
        arena_size=arena_size,
    )
    return _evaluate_projected_fragments(
        fragments=fragments,
        enemies=enemies,
        direction=direction,
        split=split,
        arena_size=arena_size,
    )


def _evaluate_projected_fragments(
    *,
    fragments: tuple[ProjectedFragment, ...],
    enemies: tuple[VisibleBlobModel, ...],
    direction: tuple[float, float],
    split: bool,
    arena_size: float,
    enemy_cache: EnemyReachabilityCache | None = None,
) -> ShieldResult:
    captured = _capture_cascade(
        fragments,
        enemies,
        arena_size=arena_size,
        enemy_cache=enemy_cache,
    )
    retained = sum(
        fragment.mass
        for index, fragment in enumerate(fragments)
        if index not in captured
    )
    return ShieldResult(
        direction=direction,
        split=split,
        retained_mass=retained,
        catastrophe=retained <= EPSILON,
    )


def _project_one_step_fragments(
    *,
    own: tuple[BlobModel, ...],
    direction: tuple[float, float],
    split: bool,
    arena_size: float,
) -> tuple[ProjectedFragment, ...]:
    fragments: list[ProjectedFragment] = []
    remaining_slots = MAX_BLOB_COUNT - len(own)
    for source_index, blob in enumerate(own):
        can_split = (
            split
            and remaining_slots > 0
            and blob.radius * blob.radius >= SPLIT_MIN_MASS
        )
        if not can_split:
            fragments.append(
                ProjectedFragment(
                    source_index,
                    (
                        _clamp(
                            blob.pos[0] + direction[0] * player_speed(blob.radius),
                            blob.radius,
                            arena_size - blob.radius,
                        ),
                        _clamp(
                            blob.pos[1] + direction[1] * player_speed(blob.radius),
                            blob.radius,
                            arena_size - blob.radius,
                        ),
                    ),
                    blob.radius,
                )
            )
            continue

        remaining_slots -= 1
        radius = blob.radius / SQRT2
        speed = player_speed(radius)
        parent_pos = (
            _clamp(blob.pos[0] + direction[0] * speed, radius, arena_size - radius),
            _clamp(blob.pos[1] + direction[1] * speed, radius, arena_size - radius),
        )
        child_pos = _split_child_endpoint(
            blob,
            direction=direction,
            arena_size=arena_size,
        )
        fragments.extend(
            (
                ProjectedFragment(source_index, parent_pos, radius),
                ProjectedFragment(source_index, child_pos, radius),
            )
        )
    return tuple(fragments)


def _capture_cascade(
    fragments: tuple[ProjectedFragment, ...],
    enemies: tuple[VisibleBlobModel, ...],
    *,
    arena_size: float = ARENA_SIZE,
    enemy_cache: EnemyReachabilityCache | None = None,
) -> frozenset[int]:
    """Worst coherent one-round capture set for each visible enemy blob.

    Enemy motion is selected once per scenario.  Captures can grow an eater
    repeatedly at its fixed post-move position, matching the engine loop, but
    do not grant another movement or another split reach after each meal.
    """

    fragment_count = len(fragments)
    all_fragments_mask = (1 << fragment_count) - 1
    captured_mask = 0
    target_order = tuple(
        sorted(
            range(fragment_count),
            key=lambda index: (
                -fragments[index].radius,
                fragments[index].source_index,
                index,
            ),
        )
    )
    fragment_masses = tuple(fragment.mass for fragment in fragments)
    captured_mass_by_mask = {0: 0.0}
    for enemy_index, enemy in enumerate(enemies):
        available_mask = all_fragments_mask & ~captured_mask
        threatened = tuple(
            index
            for index in range(fragment_count)
            if available_mask & (1 << index)
            if can_eat_player_blob(enemy.radius, fragments[index].radius)
            and math.dist(enemy.pos, fragments[index].pos)
            <= _one_step_attack_reach(enemy.radius, fragments[index].radius)
        )
        if not threatened:
            continue
        geometry = (
            enemy_cache.geometries[enemy_index]
            if enemy_cache is not None
            else _enemy_projection_geometry(enemy)
        )
        fixed_projections = (
            enemy_cache.fixed_projections[enemy_index]
            if enemy_cache is not None
            else None
        )
        best_mask = 0
        best_mass = 0.0
        for direction in _enemy_scenario_directions(enemy, fragments, threatened):
            direction_key = (
                round(direction[0] * 1000),
                round(direction[1] * 1000),
            )
            fixed_index = SAFETY_DIRECTION_INDEX.get(direction_key)
            for split_index, split_mode in enumerate((False, True)):
                if fixed_projections is not None and fixed_index is not None:
                    eaters = fixed_projections[fixed_index][split_index]
                else:
                    eaters = _project_enemy_eaters(
                        enemy=enemy,
                        direction=direction,
                        split=split_mode,
                        arena_size=arena_size,
                        geometry=geometry,
                    )
                if not eaters:
                    continue
                scenario_mask = _capture_mask_at_fixed_positions(
                    eaters=eaters,
                    fragments=fragments,
                    available_mask=available_mask,
                    target_order=target_order,
                    fragment_masses=fragment_masses,
                )
                scenario_mass = captured_mass_by_mask.get(scenario_mask)
                if scenario_mass is None:
                    scenario_mass = sum(
                        mass
                        for index, mass in enumerate(fragment_masses)
                        if scenario_mask & (1 << index)
                    )
                    captured_mass_by_mask[scenario_mask] = scenario_mass
                if scenario_mass > best_mass:
                    best_mask = scenario_mask
                    best_mass = scenario_mass
        captured_mask |= best_mask
    return frozenset(
        index for index in range(fragment_count) if captured_mask & (1 << index)
    )


def _enemy_scenario_directions(
    enemy: VisibleBlobModel,
    fragments: tuple[ProjectedFragment, ...],
    threatened: tuple[int, ...],
) -> tuple[tuple[float, float], ...]:
    directions = list(SAFETY_DIRECTIONS)
    direct = [
        normalise(
            (
                fragments[index].pos[0] - enemy.pos[0],
                fragments[index].pos[1] - enemy.pos[1],
            )
        )
        for index in threatened
    ]
    directions.extend(direct)
    nearest = sorted(
        threatened,
        key=lambda index: math.dist(enemy.pos, fragments[index].pos),
    )[:8]
    for offset, left_index in enumerate(nearest):
        left = normalise(
            (
                fragments[left_index].pos[0] - enemy.pos[0],
                fragments[left_index].pos[1] - enemy.pos[1],
            )
        )
        for right_index in nearest[offset + 1 :]:
            right = normalise(
                (
                    fragments[right_index].pos[0] - enemy.pos[0],
                    fragments[right_index].pos[1] - enemy.pos[1],
                )
            )
            bisector = normalise((left[0] + right[0], left[1] + right[1]))
            if bisector != (0.0, 0.0):
                directions.append(bisector)

    unique: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()
    for direction in directions:
        key = (round(direction[0] * 1000), round(direction[1] * 1000))
        if direction == (0.0, 0.0) or key in seen:
            continue
        seen.add(key)
        unique.append(direction)
    return tuple(unique)


def _project_enemy_eaters(
    *,
    enemy: VisibleBlobModel,
    direction: tuple[float, float],
    split: bool,
    arena_size: float,
    geometry: EnemyProjectionGeometry | None = None,
) -> tuple[EnemyEaterSeed, ...]:
    geometry = geometry or _enemy_projection_geometry(enemy)
    if not split:
        radius = geometry.unsplit_radius
        speed = geometry.unsplit_speed
        return (
            (
                _clamp(
                    enemy.pos[0] + direction[0] * speed,
                    radius,
                    arena_size - radius,
                ),
                _clamp(
                    enemy.pos[1] + direction[1] * speed,
                    radius,
                    arena_size - radius,
                ),
                radius,
                0,
            ),
        )
    if geometry.split_radius is None:
        return ()
    radius = geometry.split_radius
    speed = geometry.split_speed
    parent = (
        _clamp(
            enemy.pos[0] + direction[0] * speed,
            radius,
            arena_size - radius,
        ),
        _clamp(
            enemy.pos[1] + direction[1] * speed,
            radius,
            arena_size - radius,
        ),
        radius,
        0,
    )
    launch = geometry.split_launch
    child = (
        _clamp(
            enemy.pos[0] + direction[0] * launch,
            radius,
            arena_size - radius,
        ),
        _clamp(
            enemy.pos[1] + direction[1] * launch,
            radius,
            arena_size - radius,
        ),
        radius,
        1,
    )
    return (parent, child)


def _enemy_projection_geometry(
    enemy: VisibleBlobModel,
) -> EnemyProjectionGeometry:
    unsplit_radius = enemy.radius
    unsplit_speed = player_speed(unsplit_radius)
    if unsplit_radius * unsplit_radius < SPLIT_MIN_MASS:
        return EnemyProjectionGeometry(
            unsplit_radius,
            unsplit_speed,
            None,
            0.0,
            0.0,
        )
    split_radius = unsplit_radius / SQRT2
    split_speed = player_speed(split_radius)
    split_launch = (
        2.0 * split_radius
        + SAME_PLAYER_OVERLAP_EPSILON
        + split_speed
        + SPLIT_EJECT_SPEED
    )
    return EnemyProjectionGeometry(
        unsplit_radius,
        unsplit_speed,
        split_radius,
        split_speed,
        split_launch,
    )


def _prepare_enemy_reachability_cache(
    enemies: tuple[VisibleBlobModel, ...],
    arena_size: float,
) -> EnemyReachabilityCache:
    geometries = tuple(_enemy_projection_geometry(enemy) for enemy in enemies)
    fixed_projections = tuple(
        tuple(
            (
                _project_enemy_eaters(
                    enemy=enemy,
                    direction=direction,
                    split=False,
                    arena_size=arena_size,
                    geometry=geometry,
                ),
                _project_enemy_eaters(
                    enemy=enemy,
                    direction=direction,
                    split=True,
                    arena_size=arena_size,
                    geometry=geometry,
                ),
            )
            for direction in SAFETY_DIRECTIONS
        )
        for enemy, geometry in zip(enemies, geometries)
    )
    return EnemyReachabilityCache(geometries, fixed_projections)


def _capture_mask_at_fixed_positions(
    *,
    eaters: tuple[EnemyEaterSeed, ...],
    fragments: tuple[ProjectedFragment, ...],
    available_mask: int,
    target_order: tuple[int, ...],
    fragment_masses: tuple[float, ...],
) -> int:
    remaining_mask = available_mask
    captured_mask = 0
    mutable_eaters = [[x, y, radius * radius, order] for x, y, radius, order in eaters]
    while True:
        changed = False
        if len(mutable_eaters) == 2:
            left, right = mutable_eaters
            if right[2] > left[2] or (right[2] == left[2] and right[3] < left[3]):
                eater_order = (1, 0)
            else:
                eater_order = (0, 1)
        else:
            eater_order = (0,)
        for eater_index in eater_order:
            eater = mutable_eaters[eater_index]
            for index in target_order:
                bit = 1 << index
                if not remaining_mask & bit:
                    continue
                fragment = fragments[index]
                eater_mass = eater[2]
                if eater_mass < fragment_masses[index] * EAT_SIZE_RATIO:
                    continue
                if (eater[0] - fragment.pos[0]) ** 2 + (
                    eater[1] - fragment.pos[1]
                ) ** 2 > eater_mass:
                    continue
                eater[2] = eater_mass + fragment_masses[index]
                remaining_mask &= ~bit
                captured_mask |= bit
                changed = True
                break
            if changed:
                break
        if not changed:
            break
    return captured_mask


def _minimum_fragment_margin(
    fragments: tuple[ProjectedFragment, ...],
    enemies: tuple[VisibleBlobModel, ...],
) -> float:
    return min(
        (
            math.dist(fragment.pos, enemy.pos)
            - _one_step_attack_reach(enemy.radius, fragment.radius)
            for fragment in fragments
            for enemy in enemies
            if can_eat_player_blob(enemy.radius, fragment.radius)
        ),
        default=math.inf,
    )


def _one_step_attack_reach(predator_radius: float, prey_radius: float) -> float:
    reach = predator_radius + player_speed(predator_radius)
    if predator_radius * predator_radius >= SPLIT_MIN_MASS and can_eat_player_blob(
        predator_radius / SQRT2, prey_radius
    ):
        reach = max(reach, _split_attack_reach(predator_radius))
    return reach


def _split_attack_reach(radius: float) -> float:
    child_radius = radius / SQRT2
    return 3.0 * child_radius + SPLIT_EJECT_SPEED + player_speed(child_radius)


def _escape_vector(
    own: tuple[BlobModel, ...],
    enemies: tuple[VisibleBlobModel, ...],
) -> tuple[float, float]:
    x = y = 0.0
    for blob in own:
        for enemy in enemies:
            if not can_eat_player_blob(enemy.radius, blob.radius):
                continue
            distance = math.dist(blob.pos, enemy.pos)
            reach = _one_step_attack_reach(enemy.radius, blob.radius)
            if distance > reach + player_speed(blob.radius) * 2.0:
                continue
            away = normalise((blob.pos[0] - enemy.pos[0], blob.pos[1] - enemy.pos[1]))
            weight = blob.radius * blob.radius / max(distance, 0.25)
            x += away[0] * weight
            y += away[1] * weight
    return normalise((x, y))


def _topology_stage(blob_count: int) -> str:
    if blob_count <= 1:
        return "single"
    if blob_count >= MAX_BLOB_COUNT:
        return "cap"
    return "fragmented"


def _mass_center(own: tuple[BlobModel, ...]) -> tuple[float, float]:
    total_mass = sum(blob.radius * blob.radius for blob in own)
    return (
        sum(blob.pos[0] * blob.radius * blob.radius for blob in own) / total_mass,
        sum(blob.pos[1] * blob.radius * blob.radius for blob in own) / total_mass,
    )


def _minimum_wall_clearance(own: tuple[BlobModel, ...], arena_size: float) -> float:
    return min(
        min(
            blob.pos[0] - blob.radius,
            blob.pos[1] - blob.radius,
            arena_size - blob.pos[0] - blob.radius,
            arena_size - blob.pos[1] - blob.radius,
        )
        for blob in own
    )


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= EPSILON:
        return math.dist(point, start)
    t = _clamp(
        ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator,
        0.0,
        1.0,
    )
    return math.dist(point, (start[0] + t * dx, start[1] + t * dy))


def _rotate(direction: tuple[float, float], angle: float) -> tuple[float, float]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        direction[0] * cosine - direction[1] * sine,
        direction[0] * sine + direction[1] * cosine,
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
