from __future__ import annotations

"""Shared simulation and scoring engine for configurable beam-search strategies."""

"""Shared beam-search bot logic for SYNCS/Susquehanna Agario Bot Battle.

The public bot entrypoints in this directory import this module and only set a
StrategyConfig.  The module intentionally uses only the standard library so that
submission bots do not depend on PyTorch/NumPy at match time.

The game engine is only observed through duck-typed objects with .pos, .radius,
.player_id, .blob_id, .merge_cooldown, etc.  That keeps this code compatible with
small engine-side model changes.
"""

from dataclasses import asdict, dataclass, field, replace
from math import atan2, cos, exp, hypot, log1p, pi, sin, sqrt
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

# Current public constants from agario-kit 2026.1.13. Keep local fallbacks so
# the bot can still import if config module names change during the competition.
try:  # pragma: no cover - depends on contest package
    from lib.config.arena import ARENA_SIZE, MAX_BLOB_COUNT, MAX_ROUNDS
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
except Exception:  # pragma: no cover - local testing fallback
    ARENA_SIZE = 60.0
    MAX_BLOB_COUNT = 16
    MAX_ROUNDS = 1400
    BASE_PLAYER_SPEED = 1.1
    EAT_SIZE_RATIO = 1.2
    FOOD_RADIUS = 0.15
    MASS_DECAY_RATE = 0.002
    MERGE_ATTRACTION_SPEED = 0.08
    MIN_PLAYER_SPEED = 0.25
    PLAYER_SPEED_RADIUS_FACTOR = 0.08
    SAME_PLAYER_OVERLAP_EPSILON = 0.0001
    SPLIT_COOLDOWN_FRAMES = 18
    SPLIT_EJECT_SPEED = 1.6
    SPLIT_EJECT_DRAG = 0.82
    SPLIT_MIN_MASS = 2.0
    STARTING_RADIUS = 0.9

EPS = 1e-9
TAU = 2.0 * pi
PLANNING_DT = 1.0
DEFAULT_ACTION_BINS = 32


@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, value: float) -> "Vec2":
        return Vec2(self.x * value, self.y * value)

    def norm2(self) -> float:
        return self.x * self.x + self.y * self.y

    def norm(self) -> float:
        return sqrt(self.norm2())

    def normalized(self) -> "Vec2":
        n = self.norm()
        if n <= EPS:
            return Vec2(1.0, 0.0)
        return Vec2(self.x / n, self.y / n)

    def angle(self) -> float:
        a = atan2(self.y, self.x)
        if a < 0.0:
            a += TAU
        return a

    @staticmethod
    def from_angle(angle: float) -> "Vec2":
        return Vec2(cos(angle), sin(angle))


@dataclass(frozen=True)
class BlobState:
    player_id: int
    blob_id: int
    pos: Vec2
    radius: float
    merge_cooldown: int = 0
    is_self: bool = False
    eject_vx: float = 0.0
    eject_vy: float = 0.0

    @property
    def mass(self) -> float:
        return self.radius * self.radius


@dataclass(frozen=True)
class FoodState:
    pos: Vec2


@dataclass(frozen=True)
class VirusState:
    pos: Vec2
    radius: float


@dataclass(frozen=True)
class WorldState:
    round_number: int
    max_rounds: int
    arena_size: float
    player_id: int
    self_blobs: tuple[BlobState, ...]
    enemies: tuple[BlobState, ...]
    food: tuple[FoodState, ...]
    viruses: tuple[VirusState, ...]
    rankings: tuple[int, ...] = ()
    alive: bool = True

    @property
    def total_mass(self) -> float:
        return sum(blob.mass for blob in self.self_blobs)

    @property
    def radius(self) -> float:
        return sqrt(max(self.total_mass, 0.0))

    @property
    def blob_count(self) -> int:
        return len(self.self_blobs)

    @property
    def center(self) -> Vec2:
        if not self.self_blobs:
            return Vec2(ARENA_SIZE / 2.0, ARENA_SIZE / 2.0)
        mass = self.total_mass
        if mass <= EPS:
            return self.self_blobs[0].pos
        x = sum(b.pos.x * b.mass for b in self.self_blobs) / mass
        y = sum(b.pos.y * b.mass for b in self.self_blobs) / mass
        return Vec2(x, y)

    @property
    def largest_blob(self) -> BlobState:
        if not self.self_blobs:
            return BlobState(self.player_id, 0, self.center, 0.0, 0, True)
        return max(self.self_blobs, key=lambda b: b.radius)


@dataclass(frozen=True)
class Action:
    dx: float
    dy: float
    split: bool = False
    label: str = ""

    @property
    def vec(self) -> Vec2:
        return Vec2(self.dx, self.dy).normalized()

    @staticmethod
    def from_vec(vec: Vec2, split: bool = False, label: str = "") -> "Action":
        v = vec.normalized()
        return Action(v.x, v.y, split, label)


@dataclass(frozen=True)
class SearchNode:
    state: WorldState
    score: float
    first_action: Action
    depth: int


@dataclass(frozen=True)
class StrategyConfig:
    name: str = "balanced"
    horizon: int = 7
    beam_width: int = 8
    action_bins: int = 24
    max_extra_candidates: int = 20
    allow_split: bool = True
    stochastic_angle_jitter: float = 0.0

    weight_mass: float = 1.4
    weight_food: float = 3.2
    weight_food_cluster: float = 1.9
    weight_prey: float = 5.2
    weight_threat: float = 12.0
    weight_close_threat: float = 16.0
    weight_virus: float = 5.0
    weight_wall: float = 2.2
    weight_fragmentation: float = 1.0
    weight_split_risk: float = 9.5
    weight_rank: float = 0.5
    weight_motion_smoothness: float = 0.15
    weight_learned_value: float = 0.0
    value_model_path: str = ""

    threat_margin: float = 1.08
    prey_margin: float = 1.02
    safe_distance_factor: float = 1.45
    food_sigma: float = 2.8
    prey_chase_max_distance: float = 8.0
    split_min_radius_ratio: float = 1.72
    split_reach_bonus: float = 1.4
    split_gain_threshold: float = 0.8
    predicted_enemy_aggression: float = 0.72
    predicted_prey_escape: float = 0.35
    late_game_aggression: float = 0.25
    top_rank_safety_multiplier: float = 1.35
    log_sample_rate: int = 1
    rng_seed: int = 17

    def with_updates(self, values: dict[str, Any]) -> "StrategyConfig":
        valid = set(self.__dataclass_fields__.keys())
        filtered = {k: v for k, v in values.items() if k in valid}
        return replace(self, **filtered)


def profile_config(name: str) -> StrategyConfig:
    """Return one of the hand-built beam-search policy presets."""

    base = StrategyConfig(name=name)
    if name == "survival":
        return base.with_updates(
            {
                "horizon": 9,
                "beam_width": 10,
                "allow_split": True,
                "weight_food": 2.2,
                "weight_food_cluster": 1.4,
                "weight_prey": 2.4,
                "weight_threat": 19.0,
                "weight_close_threat": 28.0,
                "weight_virus": 8.0,
                "weight_wall": 4.3,
                "weight_fragmentation": 2.6,
                "weight_split_risk": 18.0,
                "safe_distance_factor": 1.8,
                "split_gain_threshold": 2.4,
                "top_rank_safety_multiplier": 1.75,
            }
        )
    if name == "farmer":
        return base.with_updates(
            {
                "horizon": 8,
                "beam_width": 9,
                "weight_food": 5.6,
                "weight_food_cluster": 3.4,
                "weight_prey": 2.8,
                "weight_threat": 11.0,
                "weight_close_threat": 15.0,
                "weight_virus": 5.8,
                "weight_wall": 2.4,
                "weight_fragmentation": 1.2,
                "split_gain_threshold": 1.8,
                "prey_chase_max_distance": 5.5,
            }
        )
    if name == "hunter":
        return base.with_updates(
            {
                "horizon": 7,
                "beam_width": 10,
                "weight_food": 2.2,
                "weight_food_cluster": 1.2,
                "weight_prey": 9.6,
                "weight_threat": 12.5,
                "weight_close_threat": 18.0,
                "weight_split_risk": 8.0,
                "weight_fragmentation": 0.8,
                "prey_chase_max_distance": 11.0,
                "split_min_radius_ratio": 1.68,
                "split_reach_bonus": 1.9,
                "split_gain_threshold": 0.2,
                "late_game_aggression": 0.55,
            }
        )
    if name == "opportunist":
        return base.with_updates(
            {
                "horizon": 8,
                "beam_width": 11,
                "action_bins": 32,
                "weight_food": 3.0,
                "weight_food_cluster": 2.0,
                "weight_prey": 7.4,
                "weight_threat": 13.5,
                "weight_close_threat": 20.0,
                "weight_split_risk": 10.5,
                "prey_chase_max_distance": 9.5,
                "split_min_radius_ratio": 1.70,
                "split_reach_bonus": 1.7,
                "late_game_aggression": 0.45,
            }
        )
    if name == "balanced":
        return base.with_updates(
            {
                "horizon": 8,
                "beam_width": 10,
                "action_bins": 28,
                "weight_food": 3.6,
                "weight_food_cluster": 2.1,
                "weight_prey": 5.8,
                "weight_threat": 14.0,
                "weight_close_threat": 21.0,
                "weight_virus": 6.2,
                "weight_wall": 3.0,
                "weight_fragmentation": 1.3,
                "weight_split_risk": 11.0,
                "safe_distance_factor": 1.55,
                "prey_chase_max_distance": 8.0,
            }
        )
    raise ValueError(f"unknown profile: {name}")


def config_from_json_env(default: StrategyConfig) -> StrategyConfig:
    raw = os.environ.get("BOT_CONFIG_JSON")
    if not raw:
        return default
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        path = Path(raw)
        data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("BOT_CONFIG_JSON must be a JSON object or a path to one")
    return default.with_updates(data)


def squared_distance(a: Vec2, b: Vec2) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return dx * dx + dy * dy


def can_eat(my_radius: float, enemy_radius: float, margin: float = 1.0) -> bool:
    return my_radius * my_radius >= (
        enemy_radius * enemy_radius * EAT_SIZE_RATIO * margin * margin
    )


def is_threat(my_radius: float, enemy_radius: float, margin: float = 1.0) -> bool:
    safe_margin = max(margin, EPS)
    return enemy_radius * enemy_radius >= (
        my_radius * my_radius * EAT_SIZE_RATIO / (safe_margin * safe_margin)
    )


def speed_for_radius(radius: float) -> float:
    return max(MIN_PLAYER_SPEED, BASE_PLAYER_SPEED / (1.0 + PLAYER_SPEED_RADIUS_FACTOR * radius))


def clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def _get_pos(obj: Any, default: Vec2 | None = None) -> Vec2:
    pos = getattr(obj, "pos", None)
    if pos is not None:
        return Vec2(float(pos[0]), float(pos[1]))
    if hasattr(obj, "x") and hasattr(obj, "y"):
        return Vec2(float(getattr(obj, "x")), float(getattr(obj, "y")))
    if default is not None and hasattr(obj, "dx") and hasattr(obj, "dy"):
        return Vec2(default.x + float(getattr(obj, "dx")), default.y + float(getattr(obj, "dy")))
    raise AttributeError(f"object has no usable position: {obj!r}")


def _iter_blobs(me: Any) -> list[Any]:
    blobs = getattr(me, "blobs", None)
    if blobs is None:
        return []
    if isinstance(blobs, dict):
        return list(blobs.values())
    return list(blobs)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def extract_world(game: Any) -> WorldState:
    state = game.state
    me = state.me
    player_id = _safe_int(getattr(me, "player_id", 0))
    me_center = _get_pos(me, Vec2(ARENA_SIZE / 2.0, ARENA_SIZE / 2.0))
    state_map = getattr(state, "map", None)
    arena_size = float(getattr(state, "arena_size", getattr(state_map, "size", ARENA_SIZE)))
    self_blobs: list[BlobState] = []
    for i, blob in enumerate(_iter_blobs(me)):
        pos = _get_pos(blob, me_center)
        self_blobs.append(
            BlobState(
                player_id=player_id,
                blob_id=_safe_int(getattr(blob, "blob_id", i), i),
                pos=pos,
                radius=float(getattr(blob, "radius", getattr(me, "radius", 0.9))),
                merge_cooldown=_safe_int(getattr(blob, "merge_cooldown", 0)),
                is_self=True,
            )
        )
    if not self_blobs:
        self_blobs = [
            BlobState(
                player_id=player_id,
                blob_id=0,
                pos=me_center,
                radius=float(getattr(me, "radius", 0.9)),
                is_self=True,
            )
        ]

    enemies: list[BlobState] = []
    for i, blob in enumerate(getattr(state, "visible_blobs", []) or []):
        enemy_player_id = _safe_int(getattr(blob, "player_id", -1), -1)
        if enemy_player_id == player_id:
            continue
        enemies.append(
            BlobState(
                player_id=enemy_player_id,
                blob_id=_safe_int(getattr(blob, "blob_id", i), i),
                pos=_get_pos(blob, me_center),
                radius=float(getattr(blob, "radius", 0.0)),
                merge_cooldown=_safe_int(getattr(blob, "merge_cooldown", 0)),
                is_self=False,
            )
        )

    food = tuple(FoodState(_get_pos(item, me_center)) for item in getattr(state, "visible_food", []) or [])
    viruses = tuple(
        VirusState(_get_pos(item, me_center), float(getattr(item, "radius", 1.5)))
        for item in getattr(state, "visible_viruses", []) or []
    )
    rankings = tuple(int(x) for x in getattr(state, "rankings", []) or [])
    return WorldState(
        round_number=_safe_int(getattr(state, "round", 0)),
        max_rounds=max(1, _safe_int(getattr(state, "max_rounds", MAX_ROUNDS), MAX_ROUNDS)),
        arena_size=arena_size,
        player_id=player_id,
        self_blobs=tuple(self_blobs),
        enemies=tuple(enemies),
        food=food,
        viruses=viruses,
        rankings=rankings,
        alive=bool(getattr(me, "alive", True)),
    )


def wall_clearance(blob: BlobState, arena_size: float) -> float:
    return min(
        blob.pos.x - blob.radius,
        blob.pos.y - blob.radius,
        arena_size - blob.pos.x - blob.radius,
        arena_size - blob.pos.y - blob.radius,
    )


def nearest(items: Sequence[Any], pos: Vec2) -> Any | None:
    if not items:
        return None
    return min(items, key=lambda item: squared_distance(getattr(item, "pos"), pos))


def food_cluster_score(pos: Vec2, food: Sequence[FoodState], sigma: float) -> float:
    if not food:
        return 0.0
    inv_sigma2 = 1.0 / max(sigma * sigma, EPS)
    return sum(exp(-squared_distance(pos, f.pos) * inv_sigma2) for f in food)


def food_score_along_direction(
    pos: Vec2,
    direction: Vec2,
    food: Sequence[FoodState],
    sigma: float,
    lookahead: float = 7.5,
) -> float:
    if not food:
        return 0.0
    u = direction.normalized()
    total = 0.0
    for item in food:
        rel = item.pos - pos
        forward = rel.x * u.x + rel.y * u.y
        if forward < -0.4 or forward > lookahead:
            continue
        lateral2 = max(0.0, rel.norm2() - forward * forward)
        total += exp(-lateral2 / max(sigma * sigma, EPS)) / (1.0 + max(0.0, forward))
    return total


def classify_enemies(state: WorldState, config: StrategyConfig) -> tuple[list[BlobState], list[BlobState], list[BlobState]]:
    threats: list[BlobState] = []
    prey: list[BlobState] = []
    neutral: list[BlobState] = []
    for enemy in state.enemies:
        if any(
            is_threat(blob.radius, enemy.radius, config.threat_margin)
            for blob in state.self_blobs
        ):
            threats.append(enemy)
        elif any(
            can_eat(blob.radius, enemy.radius, config.prey_margin)
            for blob in state.self_blobs
        ):
            prey.append(enemy)
        else:
            neutral.append(enemy)
    return threats, prey, neutral


def combined_escape_vector(
    state: WorldState,
    threats: Sequence[BlobState],
    config: StrategyConfig,
) -> Vec2:
    vec = Vec2(0.0, 0.0)
    for threat in threats:
        for own in state.self_blobs:
            if not is_threat(own.radius, threat.radius, config.threat_margin):
                continue
            rel = threat.pos - own.pos
            d2 = max(rel.norm2(), 0.08)
            # Weight larger and closer threats more. The vector points away
            # from the fragment that is actually vulnerable.
            weight = own.mass * (threat.radius * threat.radius + 0.4) / d2
            vec = vec + rel * -weight
    return vec


def wall_escape_vector(state: WorldState) -> Vec2:
    center = state.center
    arena = state.arena_size
    largest = state.largest_blob
    margin = max(2.5, largest.radius * 2.0)
    x_push = 0.0
    y_push = 0.0
    left = center.x - largest.radius
    right = arena - center.x - largest.radius
    bottom = center.y - largest.radius
    top = arena - center.y - largest.radius
    if left < margin:
        x_push += (margin - left) / margin
    if right < margin:
        x_push -= (margin - right) / margin
    if bottom < margin:
        y_push += (margin - bottom) / margin
    if top < margin:
        y_push -= (margin - top) / margin
    return Vec2(x_push, y_push)


def virus_escape_vector(state: WorldState) -> Vec2:
    if not state.viruses:
        return Vec2(0.0, 0.0)
    vec = Vec2(0.0, 0.0)
    for blob in state.self_blobs:
        for virus in state.viruses:
            if blob.mass <= virus.radius * virus.radius * EAT_SIZE_RATIO:
                continue
            rel = virus.pos - blob.pos
            d = max(rel.norm(), EPS)
            # 2026.1.13 requires center containment; retain 1.5 as an explicit
            # planning buffer rather than treating the virus radius as collision.
            danger_radius = blob.radius + 1.5
            if d < danger_radius:
                vec = vec + rel * (-(danger_radius - d) / (danger_radius * d))
    return vec


def dedupe_actions(actions: Iterable[Action], max_actions: int) -> list[Action]:
    kept: list[Action] = []
    seen: set[tuple[int, bool]] = set()
    for action in actions:
        v = action.vec
        angle_bin = int(round(v.angle() / TAU * 64)) % 64
        key = (angle_bin, action.split)
        if key in seen:
            continue
        seen.add(key)
        kept.append(Action(v.x, v.y, action.split, action.label))
        if len(kept) >= max_actions:
            break
    return kept


def split_is_legal(state: WorldState) -> bool:
    if state.blob_count >= MAX_BLOB_COUNT:
        return False
    largest = state.largest_blob
    return largest.mass >= SPLIT_MIN_MASS and largest.radius > 0.0


def split_can_hit_prey(state: WorldState, action: Action, config: StrategyConfig) -> bool:
    if not config.allow_split or not split_is_legal(state):
        return False
    direction = action.vec
    eligible = [blob for blob in state.self_blobs if blob.mass >= SPLIT_MIN_MASS]
    for blob in eligible:
        split_radius = blob.radius / sqrt(2.0)
        min_target_radius = split_radius / (
            sqrt(EAT_SIZE_RATIO) * max(config.prey_margin, EPS)
        )
        # Initial child offset + this round's movement/ejection + its eating
        # radius. This is the engine's exact one-round center-distance reach.
        reach = (
            3.0 * split_radius
            + SPLIT_EJECT_SPEED
            + speed_for_radius(split_radius) * PLANNING_DT
        )
        for enemy in state.enemies:
            if enemy.radius > min_target_radius:
                continue
            rel = enemy.pos - blob.pos
            forward = rel.x * direction.x + rel.y * direction.y
            if forward < -0.1 or forward > reach:
                continue
            lateral2 = max(0.0, rel.norm2() - forward * forward)
            if lateral2 <= split_radius * split_radius:
                return True
    return False


def generate_actions(state: WorldState, config: StrategyConfig) -> list[Action]:
    rng = random.Random(config.rng_seed + state.round_number)
    center = state.center
    actions: list[Action] = []
    grid_actions: list[Action] = []

    for i in range(config.action_bins):
        angle = TAU * i / config.action_bins
        if config.stochastic_angle_jitter > 0.0:
            angle += rng.uniform(-config.stochastic_angle_jitter, config.stochastic_angle_jitter)
        grid_actions.append(Action.from_vec(Vec2.from_angle(angle), False, "grid"))

    threats, prey, _neutral = classify_enemies(state, config)
    escape = combined_escape_vector(state, threats, config)
    if escape.norm2() > EPS:
        actions.append(Action.from_vec(escape, False, "escape"))
        wall_vec = wall_escape_vector(state)
        if wall_vec.norm2() > EPS:
            actions.append(Action.from_vec((escape * 2.0) + wall_vec, False, "escape_wall"))
    wall_vec = wall_escape_vector(state)
    if wall_vec.norm2() > EPS:
        actions.append(Action.from_vec(wall_vec, False, "wall"))

    virus_vec = virus_escape_vector(state)
    if virus_vec.norm2() > EPS:
        actions.append(Action.from_vec(virus_vec, False, "virus"))

    # Food cluster target directions: pick food with high local density, not only nearest.
    if state.food:
        ranked_food = sorted(
            state.food,
            key=lambda f: (
                -food_cluster_score(f.pos, state.food, config.food_sigma),
                squared_distance(center, f.pos),
            ),
        )
        for f in ranked_food[: min(8, len(ranked_food))]:
            actions.append(Action.from_vec(f.pos - center, False, "food_cluster"))
        nearest_food = min(state.food, key=lambda f: squared_distance(center, f.pos))
        actions.append(Action.from_vec(nearest_food.pos - center, False, "nearest_food"))

    if prey:
        def prey_origin(enemy: BlobState) -> BlobState:
            capable = [
                blob
                for blob in state.self_blobs
                if can_eat(blob.radius, enemy.radius, config.prey_margin)
            ]
            return min(
                capable,
                key=lambda blob: squared_distance(blob.pos, enemy.pos),
            )

        ranked_prey = sorted(
            prey,
            key=lambda enemy: squared_distance(prey_origin(enemy).pos, enemy.pos),
        )
        for enemy in ranked_prey[:6]:
            origin = prey_origin(enemy)
            if (enemy.pos - origin.pos).norm() <= config.prey_chase_max_distance:
                base = Action.from_vec(enemy.pos - origin.pos, False, "prey")
                actions.append(base)
                if split_can_hit_prey(state, base, config):
                    actions.append(Action(base.dx, base.dy, True, "split_prey"))

    # Add split variants for promising directions only.
    if config.allow_split and split_is_legal(state):
        extra: list[Action] = []
        for action in actions:
            if action.split:
                continue
            if split_can_hit_prey(state, action, config):
                extra.append(Action(action.dx, action.dy, True, "split_" + action.label))
        actions.extend(extra[:6])

    # Add the uniform grid after semantic actions so deduplication preserves
    # labels such as escape/prey/food when a grid direction is equivalent.
    actions.extend(grid_actions)

    # Always include safe drift to avoid zero action.
    actions.append(Action(1.0, 0.0, False, "drift"))
    return dedupe_actions(actions, config.action_bins + config.max_extra_candidates)


def _move_blob(blob: BlobState, direction: Vec2, arena_size: float) -> BlobState:
    step = speed_for_radius(blob.radius) * PLANNING_DT
    new_pos = blob.pos + direction * step + Vec2(blob.eject_vx, blob.eject_vy)
    r = blob.radius
    new_pos = Vec2(clamp(new_pos.x, r, arena_size - r), clamp(new_pos.y, r, arena_size - r))
    eject_vx = blob.eject_vx * SPLIT_EJECT_DRAG
    eject_vy = blob.eject_vy * SPLIT_EJECT_DRAG
    return replace(
        blob,
        pos=new_pos,
        merge_cooldown=max(0, blob.merge_cooldown - 1),
        eject_vx=0.0 if abs(eject_vx) < 1e-4 else eject_vx,
        eject_vy=0.0 if abs(eject_vy) < 1e-4 else eject_vy,
    )


def _apply_mass_decay(radius: float) -> float:
    mass = radius * radius
    minimum_mass = STARTING_RADIUS * STARTING_RADIUS
    if mass <= minimum_mass:
        return radius
    return sqrt(max(minimum_mass, mass * (1.0 - MASS_DECAY_RATE)))


def _merge_self_blobs(blobs: list[BlobState], arena_size: float) -> list[BlobState]:
    if len(blobs) <= 1:
        return blobs

    # Match the engine's mass-weighted attraction before checking whether
    # cooldown-free fragments touch and merge.
    total_mass = sum(blob.mass for blob in blobs)
    center = Vec2(
        sum(blob.pos.x * blob.mass for blob in blobs) / total_mass,
        sum(blob.pos.y * blob.mass for blob in blobs) / total_mass,
    )
    attracted: list[BlobState] = []
    for blob in blobs:
        delta = center - blob.pos
        distance = delta.norm()
        if distance <= EPS:
            attracted.append(blob)
            continue
        step = min(MERGE_ATTRACTION_SPEED, distance)
        pos = blob.pos + delta * (step / distance)
        pos = Vec2(
            clamp(pos.x, blob.radius, arena_size - blob.radius),
            clamp(pos.y, blob.radius, arena_size - blob.radius),
        )
        attracted.append(replace(blob, pos=pos))
    blobs = attracted

    changed = True
    while changed:
        changed = False
        ordered = sorted(blobs, key=lambda blob: blob.blob_id)
        for index, blob_a in enumerate(ordered):
            for blob_b in ordered[index + 1 :]:
                if blob_a.merge_cooldown > 0 or blob_b.merge_cooldown > 0:
                    continue
                if (
                    (blob_a.pos - blob_b.pos).norm()
                    > blob_a.radius + blob_b.radius + SAME_PLAYER_OVERLAP_EPSILON
                ):
                    continue
                survivor, consumed = sorted(
                    (blob_a, blob_b),
                    key=lambda blob: (-blob.mass, blob.blob_id),
                )
                mass = survivor.mass + consumed.mass
                radius = sqrt(mass)
                pos = (
                    survivor.pos * survivor.mass + consumed.pos * consumed.mass
                ) * (1.0 / mass)
                pos = Vec2(
                    clamp(pos.x, radius, arena_size - radius),
                    clamp(pos.y, radius, arena_size - radius),
                )
                merged = replace(
                    survivor,
                    pos=pos,
                    radius=radius,
                    merge_cooldown=0,
                    eject_vx=(
                        survivor.eject_vx * survivor.mass
                        + consumed.eject_vx * consumed.mass
                    )
                    / mass,
                    eject_vy=(
                        survivor.eject_vy * survivor.mass
                        + consumed.eject_vy * consumed.mass
                    )
                    / mass,
                )
                blobs = [
                    blob
                    for blob in blobs
                    if blob.blob_id not in {survivor.blob_id, consumed.blob_id}
                ]
                blobs.append(merged)
                changed = True
                break
            if changed:
                break
    return blobs


def _split_blobs(blobs: list[BlobState], action: Action, arena_size: float) -> list[BlobState]:
    if len(blobs) >= MAX_BLOB_COUNT:
        return blobs
    direction = action.vec
    starting_ids = [blob.blob_id for blob in sorted(blobs, key=lambda blob: blob.blob_id)]
    next_id = max(starting_ids, default=-1) + 1
    by_id = {blob.blob_id: blob for blob in blobs}
    for blob_id in starting_ids:
        if len(by_id) >= MAX_BLOB_COUNT:
            break
        blob = by_id[blob_id]
        if blob.mass < SPLIT_MIN_MASS:
            continue
        child_radius = blob.radius / sqrt(2.0)
        parent = replace(
            blob,
            radius=child_radius,
            merge_cooldown=SPLIT_COOLDOWN_FRAMES,
        )
        child_pos = blob.pos + direction * (
            2.0 * child_radius + SAME_PLAYER_OVERLAP_EPSILON
        )
        child_pos = Vec2(
            clamp(child_pos.x, child_radius, arena_size - child_radius),
            clamp(child_pos.y, child_radius, arena_size - child_radius),
        )
        child = BlobState(
            player_id=blob.player_id,
            blob_id=next_id,
            pos=child_pos,
            radius=child_radius,
            merge_cooldown=SPLIT_COOLDOWN_FRAMES,
            is_self=True,
            eject_vx=direction.x * SPLIT_EJECT_SPEED,
            eject_vy=direction.y * SPLIT_EJECT_SPEED,
        )
        by_id[blob_id] = parent
        by_id[next_id] = child
        next_id += 1
    return list(by_id.values())


def _predict_enemies(state: WorldState, config: StrategyConfig) -> tuple[BlobState, ...]:
    next_enemies: list[BlobState] = []
    for enemy in state.enemies:
        direction = Vec2(0.0, 0.0)
        vulnerable = [
            blob
            for blob in state.self_blobs
            if is_threat(blob.radius, enemy.radius, config.threat_margin)
        ]
        hunters = [
            blob
            for blob in state.self_blobs
            if can_eat(blob.radius, enemy.radius, config.prey_margin)
        ]
        if vulnerable:
            target = min(vulnerable, key=lambda blob: squared_distance(enemy.pos, blob.pos))
            direction = (target.pos - enemy.pos).normalized() * config.predicted_enemy_aggression
        elif hunters:
            # Prey tends to run away from us.
            hunter = min(hunters, key=lambda blob: squared_distance(enemy.pos, blob.pos))
            direction = (enemy.pos - hunter.pos).normalized() * config.predicted_prey_escape
        else:
            # Neutral players weakly drift toward nearby visible food.
            close_food = nearest(state.food, enemy.pos)
            if close_food is not None:
                direction = (close_food.pos - enemy.pos).normalized() * 0.25
        # Preserve the configured partial-speed prediction. Re-normalising here
        # used to erase all three aggression/escape/drift multipliers.
        moved = _move_blob(enemy, direction, state.arena_size)
        next_enemies.append(replace(moved, radius=_apply_mass_decay(moved.radius)))
    return tuple(next_enemies)


def _resolve_food(
    blobs: list[BlobState],
    enemies: list[BlobState],
    food: tuple[FoodState, ...],
    arena_size: float = ARENA_SIZE,
) -> tuple[list[BlobState], list[BlobState], list[FoodState], float]:
    by_key = {
        (blob.player_id, blob.blob_id): blob
        for blob in (*blobs, *enemies)
    }
    remaining_food: list[FoodState] = []
    food_gain_mass = 0.0
    for item in food:
        candidates = [
            blob
            for blob in by_key.values()
            if squared_distance(blob.pos, item.pos) <= blob.radius * blob.radius
        ]
        if not candidates:
            remaining_food.append(item)
            continue
        eater = min(
            candidates,
            key=lambda blob: (-blob.radius, blob.player_id, blob.blob_id),
        )
        gain = FOOD_RADIUS * FOOD_RADIUS
        grown_radius = sqrt(eater.mass + gain)
        by_key[(eater.player_id, eater.blob_id)] = replace(
            eater,
            pos=Vec2(
                clamp(eater.pos.x, grown_radius, arena_size - grown_radius),
                clamp(eater.pos.y, grown_radius, arena_size - grown_radius),
            ),
            radius=grown_radius,
        )
        if eater.is_self:
            food_gain_mass += gain

    return (
        [blob for blob in by_key.values() if blob.is_self],
        [blob for blob in by_key.values() if not blob.is_self],
        remaining_food,
        food_gain_mass,
    )


def _resolve_player_eating(
    blobs: list[BlobState],
    enemies: list[BlobState],
    arena_size: float = ARENA_SIZE,
) -> tuple[list[BlobState], list[BlobState], float, int]:
    by_key = {
        (blob.player_id, blob.blob_id): blob
        for blob in (*blobs, *enemies)
    }
    prey_gain_mass = 0.0
    lost_blob_count = 0

    changed = True
    while changed:
        changed = False
        living = sorted(
            by_key,
            key=lambda key: (
                -by_key[key].radius,
                by_key[key].player_id,
                by_key[key].blob_id,
            ),
        )
        for eater_key in living:
            eater = by_key.get(eater_key)
            if eater is None:
                continue
            for target_key in living:
                target = by_key.get(target_key)
                if target is None or target.player_id == eater.player_id:
                    continue
                if not can_eat(eater.radius, target.radius):
                    continue
                if squared_distance(eater.pos, target.pos) > eater.radius * eater.radius:
                    continue
                grown_radius = sqrt(eater.mass + target.mass)
                by_key[eater_key] = replace(
                    eater,
                    pos=Vec2(
                        clamp(eater.pos.x, grown_radius, arena_size - grown_radius),
                        clamp(eater.pos.y, grown_radius, arena_size - grown_radius),
                    ),
                    radius=grown_radius,
                )
                del by_key[target_key]
                if eater.is_self and not target.is_self:
                    prey_gain_mass += target.mass
                elif target.is_self and not eater.is_self:
                    lost_blob_count += 1
                changed = True
                break
            if changed:
                break

    return (
        [blob for blob in by_key.values() if blob.is_self],
        [blob for blob in by_key.values() if not blob.is_self],
        prey_gain_mass,
        lost_blob_count,
    )


def simulate_step(state: WorldState, action: Action, config: StrategyConfig) -> tuple[WorldState, dict[str, float]]:
    direction = action.vec
    blobs = list(state.self_blobs)
    if action.split:
        blobs = _split_blobs(blobs, action, state.arena_size)
    blobs = [_move_blob(b, direction, state.arena_size) for b in blobs]
    # The engine decays mass before resolving food and player collisions.
    blobs = [replace(b, radius=_apply_mass_decay(b.radius)) for b in blobs]
    blobs = _merge_self_blobs(blobs, state.arena_size)

    enemies = list(_predict_enemies(state, config))
    blobs, enemies, remaining_food, food_gain_mass = _resolve_food(
        blobs,
        enemies,
        state.food,
        state.arena_size,
    )
    blobs, enemies, prey_gain_mass, lost_blob_count = _resolve_player_eating(
        blobs,
        enemies,
        state.arena_size,
    )

    decayed_blobs = _merge_self_blobs(blobs, state.arena_size)

    alive = bool(decayed_blobs) and state.alive
    next_state = WorldState(
        round_number=state.round_number + 1,
        max_rounds=state.max_rounds,
        arena_size=state.arena_size,
        player_id=state.player_id,
        self_blobs=tuple(decayed_blobs),
        enemies=tuple(enemies),
        food=tuple(remaining_food),
        viruses=state.viruses,
        rankings=state.rankings,
        alive=alive,
    )
    info = {
        "food_gain_mass": food_gain_mass,
        "prey_gain_mass": prey_gain_mass,
        "lost_blob_count": float(lost_blob_count),
        "split_used": 1.0 if action.split else 0.0,
    }
    return next_state, info


def immediate_threat_risk(state: WorldState, config: StrategyConfig) -> float:
    if not state.self_blobs:
        return 1e6
    risk = 0.0
    for my_blob in state.self_blobs:
        for enemy in state.enemies:
            dist = (enemy.pos - my_blob.pos).norm()
            # Danger if enemy can eat now or after a small position prediction error.
            eat_gap = dist - enemy.radius
            if is_threat(my_blob.radius, enemy.radius, config.threat_margin):
                safe = config.safe_distance_factor * (enemy.radius + my_blob.radius)
                risk += ((max(0.0, safe - dist) / max(safe, EPS)) ** 2) * (enemy.radius / max(my_blob.radius, EPS))
                if eat_gap <= 0.2:
                    risk += 8.0
            elif enemy.radius > my_blob.radius * 0.92:
                risk += 0.15 / (1.0 + max(0.0, dist - my_blob.radius - enemy.radius))
    return risk


def virus_risk(state: WorldState) -> float:
    if not state.viruses:
        return 0.0
    risk = 0.0
    for blob in state.self_blobs:
        for virus in state.viruses:
            if blob.mass <= virus.radius * virus.radius * EAT_SIZE_RATIO:
                continue
            dist = (virus.pos - blob.pos).norm()
            margin = blob.radius + 0.5
            if dist < margin:
                risk += ((margin - dist) / max(margin, EPS)) ** 2 * max(1.0, blob.radius / max(virus.radius, EPS))
    return risk


def wall_risk(state: WorldState) -> float:
    if not state.self_blobs:
        return 100.0
    risk = 0.0
    for blob in state.self_blobs:
        clearance = wall_clearance(blob, state.arena_size)
        margin = max(1.0, blob.radius * 0.8)
        if clearance < margin:
            risk += ((margin - clearance) / margin) ** 2
        if clearance < -0.05:
            risk += 10.0
    return risk


def split_risk(state: WorldState, action: Action, config: StrategyConfig) -> float:
    if not action.split:
        return 0.0
    largest_after = state.largest_blob.radius
    risk = 0.0
    for enemy in state.enemies:
        if is_threat(largest_after, enemy.radius, config.threat_margin):
            dist = (enemy.pos - state.center).norm()
            safe = max(2.5, config.safe_distance_factor * (enemy.radius + largest_after))
            risk += max(0.0, safe - dist) / safe
    return risk + 0.25 * max(0, state.blob_count - 1)


def prey_opportunity(state: WorldState, config: StrategyConfig) -> float:
    value = 0.0
    for enemy in state.enemies:
        capable = [
            blob
            for blob in state.self_blobs
            if can_eat(blob.radius, enemy.radius, config.prey_margin)
        ]
        if not capable:
            continue
        hunter = min(
            capable,
            key=lambda blob: squared_distance(blob.pos, enemy.pos),
        )
        dist = (enemy.pos - hunter.pos).norm()
        if dist > config.prey_chase_max_distance + hunter.radius:
            continue
        value += enemy.mass / (1.0 + max(0.0, dist - hunter.radius))
    return value


def rank_modifier(state: WorldState, config: StrategyConfig) -> float:
    if not state.rankings:
        return 0.0
    try:
        rank_index = list(state.rankings).index(state.player_id)
    except ValueError:
        return 0.0
    n = max(1, len(state.rankings))
    # Positive when near the top; can slightly increase safety weighting elsewhere.
    return 1.0 - rank_index / max(1, n - 1)



_VALUE_MODEL_CACHE: dict[str, tuple[list[str], list[float], float]] = {}


def _load_value_model(config: StrategyConfig) -> tuple[list[str], list[float], float] | None:
    path_value = os.environ.get("BOT_VALUE_MODEL_PATH") or config.value_model_path
    if not path_value:
        return None
    if path_value in _VALUE_MODEL_CACHE:
        return _VALUE_MODEL_CACHE[path_value]
    path = Path(path_value)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        names = [str(x) for x in payload.get("feature_names", FEATURE_NAMES)]
        weights = [float(x) for x in payload.get("weights", [])]
        bias = float(payload.get("bias", 0.0))
    except Exception:
        return None
    model = (names, weights, bias)
    _VALUE_MODEL_CACHE[path_value] = model
    return model


def learned_value_estimate(state: WorldState, config: StrategyConfig) -> float:
    model = _load_value_model(config)
    if model is None:
        return 0.0
    names, weights, bias = model
    feature_map = dict(zip(FEATURE_NAMES, feature_vector(state, config), strict=True))
    value = bias
    for name, weight in zip(names, weights, strict=False):
        value += weight * float(feature_map.get(name, 0.0))
    return value

def evaluate_state(state: WorldState, action: Action, config: StrategyConfig, info: dict[str, float] | None = None) -> float:
    if not state.alive or not state.self_blobs:
        return -1_000_000.0
    info = info or {}
    center = state.center
    total_mass = state.total_mass
    mass_score = config.weight_mass * log1p(total_mass)
    food_cluster = food_cluster_score(center, state.food, config.food_sigma)
    forward_food = food_score_along_direction(center, action.vec, state.food, config.food_sigma)
    food_score = config.weight_food_cluster * food_cluster + config.weight_food * forward_food
    prey_score = config.weight_prey * prey_opportunity(state, config)
    threat = immediate_threat_risk(state, config)
    close_threat = max(0.0, threat - 0.5)
    threat_multiplier = 1.0 + config.top_rank_safety_multiplier * max(0.0, rank_modifier(state, config))
    virus = virus_risk(state)
    wall = wall_risk(state)
    frag = max(0, state.blob_count - 1)
    round_frac = clamp(state.round_number / max(1, state.max_rounds), 0.0, 1.0)
    aggression_bonus = config.late_game_aggression * round_frac * prey_score
    score = (
        mass_score
        + food_score
        + prey_score
        + aggression_bonus
        + config.weight_rank * rank_modifier(state, config)
        + config.weight_learned_value * learned_value_estimate(state, config)
        + config.weight_food * info.get("food_gain_mass", 0.0) * 18.0
        + config.weight_prey * info.get("prey_gain_mass", 0.0)
        - threat_multiplier * (config.weight_threat * threat + config.weight_close_threat * close_threat)
        - config.weight_virus * virus
        - config.weight_wall * wall
        - config.weight_fragmentation * frag
        - config.weight_split_risk * split_risk(state, action, config)
        - 80.0 * info.get("lost_blob_count", 0.0)
    )
    if action.split and info.get("prey_gain_mass", 0.0) < config.split_gain_threshold:
        score -= config.weight_split_risk * (config.split_gain_threshold - info.get("prey_gain_mass", 0.0))
    return score


class BeamSearchPlanner:
    def __init__(self, config: StrategyConfig):
        self.config = config

    def choose_action(self, state: WorldState) -> Action:
        actions = generate_actions(state, self.config)
        if not actions:
            return Action(1.0, 0.0, False, "fallback")

        nodes: list[SearchNode] = []
        for action in actions:
            next_state, info = simulate_step(state, action, self.config)
            score = evaluate_state(next_state, action, self.config, info)
            nodes.append(SearchNode(next_state, score, action, 1))
        nodes.sort(key=lambda n: n.score, reverse=True)
        beam = nodes[: self.config.beam_width]

        for depth in range(2, self.config.horizon + 1):
            expanded: list[SearchNode] = []
            for node in beam:
                for action in generate_actions(node.state, self.config):
                    next_state, info = simulate_step(node.state, action, self.config)
                    step_score = evaluate_state(next_state, action, self.config, info)
                    # Discount later evaluations.  It stabilizes beam choice when
                    # the approximate simulator is wrong far into the horizon.
                    score = node.score + (0.86 ** (depth - 1)) * step_score
                    expanded.append(SearchNode(next_state, score, node.first_action, depth))
            if not expanded:
                break
            expanded.sort(key=lambda n: n.score, reverse=True)
            beam = expanded[: self.config.beam_width]

        best = max(beam, key=lambda n: n.score)
        return self._safety_override(state, best.first_action)

    def evaluate_action(self, state: WorldState, action: Action) -> float:
        next_state, info = simulate_step(state, action, self.config)
        return evaluate_state(next_state, action, self.config, info)

    def is_action_safe(self, state: WorldState, action: Action, min_score: float = -120.0) -> bool:
        if action.split and not split_can_hit_prey(state, action, self.config):
            return False
        next_state, info = simulate_step(state, action, self.config)
        if not next_state.alive:
            return False
        if immediate_threat_risk(next_state, self.config) > 3.0:
            return False
        return evaluate_state(next_state, action, self.config, info) >= min_score

    def _safety_override(self, state: WorldState, action: Action) -> Action:
        # Never allow a predicted-death first step if a non-split escape exists.
        if self.is_action_safe(state, action):
            return action
        candidates = [a for a in generate_actions(state, self.config) if not a.split]
        if not candidates:
            return Action(1.0, 0.0, False, "unsafe_fallback")
        scored = [(self.evaluate_action(state, a), a) for a in candidates]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]


FEATURE_NAMES: tuple[str, ...] = (
    "round_frac",
    "my_radius",
    "total_mass",
    "blob_count",
    "center_x",
    "center_y",
    "wall_left",
    "wall_right",
    "wall_bottom",
    "wall_top",
    "food_count",
    "nearest_food_dx",
    "nearest_food_dy",
    "nearest_food_dist",
    "food_cluster_center",
    "enemy_count",
    "nearest_threat_dx",
    "nearest_threat_dy",
    "nearest_threat_dist",
    "nearest_threat_radius_ratio",
    "nearest_prey_dx",
    "nearest_prey_dy",
    "nearest_prey_dist",
    "nearest_prey_radius_ratio",
    "threat_risk",
    "virus_count",
    "nearest_virus_dx",
    "nearest_virus_dy",
    "nearest_virus_dist",
    "rank_mod",
)


def feature_vector(state: WorldState, config: StrategyConfig | None = None) -> list[float]:
    config = config or StrategyConfig()
    center = state.center
    arena = max(state.arena_size, EPS)
    largest = state.largest_blob
    threats, prey, _neutral = classify_enemies(state, config)
    nearest_food = nearest(state.food, center)
    nearest_threat = nearest(threats, center)
    nearest_prey = nearest(prey, center)
    nearest_virus = nearest(state.viruses, center)

    def rel_features(obj: Any | None, radius_ref: float | None = None) -> tuple[float, float, float, float]:
        if obj is None:
            return (0.0, 0.0, 1.0, 0.0)
        rel = obj.pos - center
        dist = rel.norm()
        ratio = 0.0
        if radius_ref is not None and hasattr(obj, "radius"):
            ratio = float(getattr(obj, "radius")) / max(radius_ref, EPS)
        return (rel.x / arena, rel.y / arena, dist / arena, ratio)

    fd = rel_features(nearest_food)
    td = rel_features(nearest_threat, largest.radius)
    pd = rel_features(nearest_prey, largest.radius)
    vd = rel_features(nearest_virus)
    left = (center.x - largest.radius) / arena
    right = (arena - center.x - largest.radius) / arena
    bottom = (center.y - largest.radius) / arena
    top = (arena - center.y - largest.radius) / arena
    return [
        clamp(state.round_number / max(1, state.max_rounds), 0.0, 1.0),
        largest.radius / arena,
        log1p(state.total_mass) / 6.0,
        state.blob_count / max(1, MAX_BLOB_COUNT),
        center.x / arena,
        center.y / arena,
        left,
        right,
        bottom,
        top,
        min(1.0, len(state.food) / 80.0),
        fd[0],
        fd[1],
        fd[2],
        food_cluster_score(center, state.food, config.food_sigma) / 20.0,
        min(1.0, len(state.enemies) / 16.0),
        td[0],
        td[1],
        td[2],
        td[3],
        pd[0],
        pd[1],
        pd[2],
        pd[3],
        immediate_threat_risk(state, config) / 10.0,
        min(1.0, len(state.viruses) / 8.0),
        vd[0],
        vd[1],
        vd[2],
        rank_modifier(state, config),
    ]


def action_to_label(action: Action, bins: int = DEFAULT_ACTION_BINS) -> int:
    angle_bin = int(round(action.vec.angle() / TAU * bins)) % bins
    return angle_bin + (bins if action.split else 0)


def label_to_action(label: int, bins: int = DEFAULT_ACTION_BINS) -> Action:
    split = label >= bins
    angle_bin = label % bins
    angle = TAU * angle_bin / bins
    v = Vec2.from_angle(angle)
    return Action(v.x, v.y, split, "learned")


def log_decision_if_requested(state: WorldState, action: Action, config: StrategyConfig) -> None:
    path = os.environ.get("BOT_LOG_PATH")
    if not path:
        return
    sample_rate = max(1, int(os.environ.get("BOT_LOG_SAMPLE_RATE", str(config.log_sample_rate))))
    if state.round_number % sample_rate != 0:
        return
    record = {
        "schema": "botbattle-beam-v1",
        "pid": os.getpid(),
        "config_name": config.name,
        "round": state.round_number,
        "max_rounds": state.max_rounds,
        "player_id": state.player_id,
        "alive": state.alive,
        "rankings": list(state.rankings),
        "total_mass": state.total_mass,
        "radius": state.radius,
        "blob_count": state.blob_count,
        "features": feature_vector(state, config),
        "feature_names": list(FEATURE_NAMES),
        "action": {"dx": action.dx, "dy": action.dy, "split": action.split, "label": action.label},
        "action_label_32": action_to_label(action, DEFAULT_ACTION_BINS),
    }
    line = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # Submission bots should not crash due to logging.
        pass


def choose_action_from_game(game: Any, config: StrategyConfig) -> Action:
    state = extract_world(game)
    planner = BeamSearchPlanner(config)
    action = planner.choose_action(state)
    log_decision_if_requested(state, action, config)
    return action


def run_bot(config: StrategyConfig) -> None:  # pragma: no cover - requires contest package runtime
    from helper.game import Game
    from lib.interface.events.moves.move_player import MovePlayer
    from lib.interface.queries.query_move import QueryMovePlayer
    from lib.models.penguin_model import DirectionModel

    game = Game()
    while True:
        query = game.get_next_query()
        match query:
            case QueryMovePlayer():
                action = choose_action_from_game(game, config)
                game.send_move(
                    MovePlayer(
                        player_id=game.state.me.player_id,
                        direction=DirectionModel(x=action.dx, y=action.dy),
                        split=action.split,
                    )
                )
            case _:
                raise RuntimeError(f"Unsupported query type: {type(query)}")


def config_to_json(config: StrategyConfig) -> str:
    return json.dumps(asdict(config), indent=2, sort_keys=True)
