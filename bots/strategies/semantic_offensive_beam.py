from __future__ import annotations

"""A bounded attack-only beam with persistent player-level pursuit.

The base semantic policy already provides broad resource and safety scoring.
This policy spends extra work only on one visible prey player: it compares a
direct pursuit with occupying the prey's centre-side escape lane, then scores
one cheap follow-up step.  Player-level memory survives fragment replacement
and brief vision loss, which lets a chase continue without globally increasing
the search width.
"""

from dataclasses import dataclass, replace
import math
import os

from lib.models.blob_model import BlobModel, VisibleBlobModel
from strategies.base import StrategyContext
from strategies.features import can_eat_player_blob, normalise, player_speed
from strategies.semantic_potential import (
    DIRECTIONAL_HORIZON,
    EPSILON,
    DirectionCandidate,
    PotentialScore,
    SemanticLookaheadStrategy,
    _clamp,
    _dot,
    _project_action_blobs,
)


PURSUIT_MEMORY_TURNS = 6
PURSUIT_MEMORY_DECAY = 4.0
CORNER_INFLUENCE = 15.0
ATTACK_FOLLOWUP_WEIGHT = 0.28
CORNER_TRAP_WEIGHT = 0.30
PURSUIT_COMMITMENT_WEIGHT = 0.12
OWNER_COUNTERATTACK_WEIGHT = 0.45
OFFENSIVE_ROUTE_FAMILIES = frozenset(
    {"pursuit_enemy", "corner_cutoff_enemy", "pursuit_memory"}
)


@dataclass(frozen=True, slots=True)
class PursuitMemory:
    player_id: int
    pos: tuple[float, float]
    velocity: tuple[float, float]
    target_mass: float
    last_seen_round: int


class SemanticOffensiveBeamStrategy(SemanticLookaheadStrategy):
    """Add a two-route attack beam without widening non-attack decisions."""

    name = "semantic_offensive_beam"

    def __init__(self) -> None:
        super().__init__()
        self._pursuit_memory: PursuitMemory | None = None
        self._current_round = 0
        self._corner_enabled = _offensive_environment_enabled(
            "SEMANTIC_OFFENSIVE_CORNER",
            default=False,
        )
        self._memory_enabled = _offensive_environment_enabled(
            "SEMANTIC_OFFENSIVE_MEMORY",
            default=False,
        )
        self._attack_values: dict[
            tuple[str, str | None, bool, int], float
        ] = {}

    def choose(self, context: StrategyContext):
        state = context.game.state
        self._current_round = int(state.round)
        own = tuple(state.me.blobs.values())
        visible_threat = any(
            can_eat_player_blob(enemy.radius, blob.radius)
            for blob in own
            for enemy in state.visible_blobs
        )
        if not self._memory_enabled:
            self._pursuit_memory = None
        elif visible_threat:
            self._pursuit_memory = None
        if (
            self._pursuit_memory is not None
            and self._current_round - self._pursuit_memory.last_seen_round
            > PURSUIT_MEMORY_TURNS
        ):
            self._pursuit_memory = None

        self._attack_values = {}
        decision = super().choose(context)
        if (
            self._memory_enabled
            and not visible_threat
            and decision.target_kind == "prey"
            and decision.target_id is not None
        ):
            player_id, _, raw_blob_id = decision.target_id.partition(":")
            target = next(
                (
                    enemy
                    for enemy in state.visible_blobs
                    if int(enemy.player_id) == int(player_id)
                    and (not raw_blob_id or enemy.blob_id == int(raw_blob_id))
                ),
                None,
            )
            if target is not None:
                velocity = (
                    self._pursuit_memory.velocity
                    if self._pursuit_memory is not None
                    and self._pursuit_memory.player_id == int(player_id)
                    else (0.0, 0.0)
                )
                self._pursuit_memory = PursuitMemory(
                    player_id=int(player_id),
                    pos=target.pos,
                    velocity=velocity,
                    target_mass=target.radius * target.radius,
                    last_seen_round=self._current_round,
                )
        return decision

    def _candidates(self, **kwargs) -> tuple[DirectionCandidate, ...]:
        base = super()._candidates(**kwargs)
        own: tuple[BlobModel, ...] = kwargs["own"]
        enemies: tuple[VisibleBlobModel, ...] = kwargs["enemies"]
        arena_size: float = kwargs["arena_size"]
        previous_directions: dict[
            tuple[int, int], tuple[float, float]
        ] = kwargs["previous_directions"]
        escape_direction: tuple[float, float] = kwargs["escape_direction"]

        if escape_direction != (0.0, 0.0):
            return base

        preferred_player_id = (
            self._pursuit_memory.player_id
            if self._pursuit_memory is not None
            else None
        )
        target = _candidate_attack_target(
            base,
            enemies,
            preferred_player_id=preferred_player_id,
        )
        if target is None:
            target = _attack_target(
                own,
                enemies,
                preferred_player_id=preferred_player_id,
                arena_size=arena_size,
            )
        if target is None:
            memory_candidate = self._memory_candidate(own, arena_size)
            return (*base, memory_candidate) if memory_candidate is not None else base

        hunter = _target_hunter(own, target)
        velocity_direction = previous_directions.get(
            (int(target.player_id), target.blob_id),
            (0.0, 0.0),
        )
        velocity = (
            velocity_direction[0] * player_speed(target.radius),
            velocity_direction[1] * player_speed(target.radius),
        )
        self._pursuit_memory = PursuitMemory(
            player_id=int(target.player_id),
            pos=target.pos,
            velocity=velocity,
            target_mass=target.radius * target.radius,
            last_seen_round=self._current_round,
        )

        target_id = f"{target.player_id}:{target.blob_id}"
        gap = max(
            0.0,
            math.dist(hunter.pos, target.pos) - hunter.radius - target.radius,
        )
        turns = gap / max(player_speed(hunter.radius), EPSILON)
        additions: list[DirectionCandidate] = []
        direct_direction = normalise(
            (target.pos[0] - hunter.pos[0], target.pos[1] - hunter.pos[1])
        )
        trap = _trap_geometry(
            hunter.pos,
            target.pos,
            target.radius,
            arena_size,
        )
        existing_target_directions = tuple(
            candidate.direction
            for candidate in base
            if candidate.target_kind == "prey"
            and candidate.target_id is not None
            and int(candidate.target_id.partition(":")[0]) == int(target.player_id)
        )
        if (
            turns <= 2.0 * DIRECTIONAL_HORIZON
            and direct_direction != (0.0, 0.0)
            and not any(
                _dot(direct_direction, direction) >= 0.995
                for direction in existing_target_directions
            )
        ):
            additions.append(
                DirectionCandidate(
                    family="pursuit_enemy",
                    direction=direct_direction,
                    target_kind="prey",
                    target_id=target_id,
                    target_pos=target.pos,
                    capture_mass=target.radius * target.radius,
                    contact_turns=turns,
                )
            )

        cutoff = _corner_cutoff_point(
            hunter,
            target,
            arena_size=arena_size,
        )
        cutoff_direction = normalise(
            (cutoff[0] - hunter.pos[0], cutoff[1] - hunter.pos[1])
        )
        if (
            self._corner_enabled
            and
            trap > 0.08
            and cutoff_direction != (0.0, 0.0)
            and _dot(cutoff_direction, direct_direction) < 0.999
        ):
            additions.append(
                DirectionCandidate(
                    family="corner_cutoff_enemy",
                    direction=cutoff_direction,
                    target_kind="prey",
                    target_id=target_id,
                    target_pos=target.pos,
                    capture_mass=target.radius * target.radius,
                    contact_turns=turns,
                )
            )
        return (*base, *additions)

    def _memory_candidate(
        self,
        own: tuple[BlobModel, ...],
        arena_size: float,
    ) -> DirectionCandidate | None:
        memory = self._pursuit_memory
        if memory is None:
            return None
        age = self._current_round - memory.last_seen_round
        if age <= 0 or age > PURSUIT_MEMORY_TURNS:
            return None
        center = _mass_center(own)
        prediction_turns = min(float(age), 4.0)
        target_pos = (
            _clamp(
                memory.pos[0] + memory.velocity[0] * prediction_turns,
                0.0,
                arena_size,
            ),
            _clamp(
                memory.pos[1] + memory.velocity[1] * prediction_turns,
                0.0,
                arena_size,
            ),
        )
        direction = normalise(
            (target_pos[0] - center[0], target_pos[1] - center[1])
        )
        if direction == (0.0, 0.0):
            direction = normalise(memory.velocity)
        if direction == (0.0, 0.0):
            return None
        return DirectionCandidate(
            family="pursuit_memory",
            direction=direction,
            target_kind="pursuit",
            target_id=str(memory.player_id),
            target_pos=target_pos,
            capture_mass=memory.target_mass,
        )

    def _score_candidate(self, **kwargs) -> PotentialScore:
        score = super()._score_candidate(**kwargs)
        candidate: DirectionCandidate = kwargs["candidate"]
        if candidate.target_kind not in {"prey", "pursuit"}:
            return score

        attack_value = _attack_beam_value(
            candidate=candidate,
            own=kwargs["own"],
            enemies=kwargs["enemies"],
            arena_size=kwargs["arena_size"],
            previous_directions=kwargs["previous_directions"],
            memory=self._pursuit_memory,
            current_round=self._current_round,
        )
        self._attack_values[
            (
                candidate.family,
                candidate.target_id,
                candidate.split,
                candidate.split_depth,
            )
        ] = attack_value

        # Exact physics may show a small fragment being traded while the attack
        # still secures more enemy mass.  Treat retained mass, not the mere
        # existence of a loss event, as the attack guard.  A remaining negative
        # safety margin still keeps the candidate catastrophic.
        profitable_exchange = (
            score.own_mass_lost > 0.0
            and score.secured_enemy_mass > score.own_mass_lost
            and score.safety_margin > 0.0
        )
        failed_attack_root = (
            candidate.family in OFFENSIVE_ROUTE_FAMILIES
            and attack_value <= EPSILON
        )
        return replace(
            score,
            total=score.total + attack_value,
            intent=score.intent + attack_value,
            catastrophic=(
                (score.catastrophic and not profitable_exchange)
                or failed_attack_root
            ),
        )

    def _refine_scored_candidates(
        self,
        *,
        scored: tuple[tuple[DirectionCandidate, PotentialScore], ...],
        **kwargs,
    ) -> tuple[tuple[DirectionCandidate, PotentialScore], ...]:
        """Keep attack routes outside the existing lookahead root budget.

        The submitted lookahead has four bounded root slots.  Letting a new
        corner route occupy one of those slots silently removes an established
        split-capture root from refinement.  Existing roots therefore run
        through the unchanged lookahead without their attack bonus; the bonus
        is restored afterwards.  New attack routes have their own two-response
        beam and never consume a generic lookahead slot.
        """

        base: list[tuple[DirectionCandidate, PotentialScore]] = []
        offensive: dict[
            tuple[str, str | None, bool, int],
            tuple[DirectionCandidate, PotentialScore],
        ] = {}
        for candidate, score in scored:
            key = _candidate_key(candidate)
            attack_value = self._attack_values.get(key, 0.0)
            if candidate.family in OFFENSIVE_ROUTE_FAMILIES:
                offensive[key] = (candidate, score)
                continue
            base.append(
                (
                    candidate,
                    replace(
                        score,
                        total=score.total - attack_value,
                        intent=score.intent - attack_value,
                    ),
                )
            )

        refined_base = super()._refine_scored_candidates(
            scored=tuple(base),
            **kwargs,
        )
        refined: dict[
            tuple[str, str | None, bool, int],
            tuple[DirectionCandidate, PotentialScore],
        ] = {}
        for candidate, score in refined_base:
            attack_value = self._attack_values.get(
                _candidate_key(candidate),
                0.0,
            )
            refined[_candidate_key(candidate)] = (
                candidate,
                replace(
                    score,
                    total=score.total + attack_value,
                    intent=score.intent + attack_value,
                ),
            )
        refined.update(offensive)
        return tuple(refined[_candidate_key(candidate)] for candidate, _ in scored)

    def _decision_diagnostics(
        self,
        selected: DirectionCandidate,
    ) -> dict[str, object]:
        memory = self._pursuit_memory
        return {
            **super()._decision_diagnostics(selected),
            "offensive_beam": {
                "selected_value": self._attack_values.get(
                    (
                        selected.family,
                        selected.target_id,
                        selected.split,
                        selected.split_depth,
                    ),
                    0.0,
                ),
                "pursuit_player_id": (
                    memory.player_id if memory is not None else None
                ),
                "memory_age": (
                    self._current_round - memory.last_seen_round
                    if memory is not None
                    else None
                ),
                "scored_attack_roots": len(self._attack_values),
            }
        }


def _attack_target(
    own: tuple[BlobModel, ...],
    enemies: tuple[VisibleBlobModel, ...],
    *,
    preferred_player_id: int | None,
    arena_size: float,
) -> VisibleBlobModel | None:
    edible = tuple(
        enemy
        for enemy in enemies
        if any(
            can_eat_player_blob(blob.radius, enemy.radius, radius_margin=1.03)
            for blob in own
        )
    )
    if not edible:
        return None
    own_center = _mass_center(own)

    def value(enemy: VisibleBlobModel) -> tuple[float, float, float]:
        hunter = _target_hunter(own, enemy)
        gap = max(
            0.0,
            math.dist(hunter.pos, enemy.pos) - hunter.radius - enemy.radius,
        )
        speed = max(player_speed(hunter.radius), EPSILON)
        persistence = (
            enemy.radius * enemy.radius
            if preferred_player_id is not None
            and int(enemy.player_id) == preferred_player_id
            else 0.0
        )
        trap = _trap_geometry(
            hunter.pos,
            enemy.pos,
            enemy.radius,
            arena_size,
        )
        center_distance = math.dist(own_center, enemy.pos)
        return (
            enemy.radius * enemy.radius / (1.0 + gap / speed)
            + 0.20 * persistence
            + 0.20 * enemy.radius * enemy.radius * trap,
            -center_distance,
            enemy.radius,
        )

    return max(edible, key=value)


def _candidate_attack_target(
    candidates: tuple[DirectionCandidate, ...],
    enemies: tuple[VisibleBlobModel, ...],
    *,
    preferred_player_id: int | None,
) -> VisibleBlobModel | None:
    enemies_by_id = {
        (int(enemy.player_id), enemy.blob_id): enemy for enemy in enemies
    }
    ranked: list[tuple[float, VisibleBlobModel]] = []
    for candidate in candidates:
        if candidate.target_kind != "prey" or candidate.target_id is None:
            continue
        raw_player_id, _, raw_blob_id = candidate.target_id.partition(":")
        target = enemies_by_id.get((int(raw_player_id), int(raw_blob_id)))
        if target is None:
            continue
        capture_mass = (
            candidate.capture_mass
            if candidate.capture_mass > 0.0
            else target.radius * target.radius
        )
        contact_turns = (
            candidate.contact_turns
            if candidate.contact_turns is not None
            else DIRECTIONAL_HORIZON
        )
        ranked.append(
            (
                capture_mass
                / (1.0 + max(0.0, contact_turns) / DIRECTIONAL_HORIZON),
                target,
            )
        )
    if not ranked:
        return None
    best_value, best_target = max(
        ranked,
        key=lambda item: (item[0], item[1].radius),
    )
    preferred = max(
        (
            item
            for item in ranked
            if preferred_player_id is not None
            and int(item[1].player_id) == preferred_player_id
        ),
        key=lambda item: (item[0], item[1].radius),
        default=None,
    )
    if preferred is not None and preferred[0] >= 0.90 * best_value:
        return preferred[1]
    return best_target


def _candidate_key(
    candidate: DirectionCandidate,
) -> tuple[str, str | None, bool, int]:
    return (
        candidate.family,
        candidate.target_id,
        candidate.split,
        candidate.split_depth,
    )


def _offensive_environment_enabled(name: str, *, default: bool) -> bool:
    fallback = "1" if default else "0"
    return os.environ.get(name, fallback).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _target_hunter(
    own: tuple[BlobModel, ...],
    target: VisibleBlobModel,
) -> BlobModel:
    return min(
        (
            blob
            for blob in own
            if can_eat_player_blob(
                blob.radius,
                target.radius,
                radius_margin=1.03,
            )
        ),
        key=lambda blob: (
            max(0.0, math.dist(blob.pos, target.pos) - blob.radius)
            / max(player_speed(blob.radius), EPSILON),
            -blob.radius,
        ),
    )


def _corner_cutoff_point(
    hunter: BlobModel,
    target: VisibleBlobModel,
    *,
    arena_size: float,
) -> tuple[float, float]:
    center = (arena_size * 0.5, arena_size * 0.5)
    outward = normalise(
        (target.pos[0] - center[0], target.pos[1] - center[1])
    )
    offset = min(
        4.0,
        hunter.radius + target.radius + player_speed(hunter.radius) * 2.0,
    )
    return (
        _clamp(
            target.pos[0] - outward[0] * offset,
            hunter.radius,
            arena_size - hunter.radius,
        ),
        _clamp(
            target.pos[1] - outward[1] * offset,
            hunter.radius,
            arena_size - hunter.radius,
        ),
    )


def _attack_beam_value(
    *,
    candidate: DirectionCandidate,
    own: tuple[BlobModel, ...],
    enemies: tuple[VisibleBlobModel, ...],
    arena_size: float,
    previous_directions: dict[tuple[int, int], tuple[float, float]],
    memory: PursuitMemory | None,
    current_round: int,
) -> float:
    if candidate.target_kind == "pursuit":
        if memory is None or candidate.target_pos is None:
            return 0.0
        age = max(0, current_round - memory.last_seen_round)
        center = _mass_center(own)
        distance = math.dist(center, candidate.target_pos)
        speed = max(
            (player_speed(blob.radius) for blob in own),
            default=EPSILON,
        )
        recency = math.exp(-age / PURSUIT_MEMORY_DECAY)
        return (
            PURSUIT_COMMITMENT_WEIGHT
            * memory.target_mass
            * recency
            / (1.0 + distance / max(speed * DIRECTIONAL_HORIZON, EPSILON))
        )

    if candidate.target_id is None:
        return 0.0
    raw_player_id, _, raw_blob_id = candidate.target_id.partition(":")
    target = next(
        (
            enemy
            for enemy in enemies
            if int(enemy.player_id) == int(raw_player_id)
            and enemy.blob_id == int(raw_blob_id)
        ),
        None,
    )
    if target is None:
        return 0.0

    expected_followup = 0.0
    if candidate.family in {"pursuit_enemy", "corner_cutoff_enemy"}:
        projected = _project_action_blobs(
            own,
            candidate.direction,
            split=candidate.split,
            arena_size=arena_size,
        )
        target_direction = previous_directions.get(
            (int(target.player_id), target.blob_id),
            (0.0, 0.0),
        )
        projected_target = _move_target(
            target,
            target_direction,
            arena_size=arena_size,
        )
        inertial_value = _capture_beam_mass(
            projected,
            projected_target,
            response_direction=target_direction,
            arena_size=arena_size,
        )
        projected_center = _mass_center(projected)
        evasive_direction = normalise(
            (
                projected_target.pos[0] - projected_center[0],
                projected_target.pos[1] - projected_center[1],
            )
        )
        evasive_target = _move_target(
            projected_target,
            evasive_direction,
            arena_size=arena_size,
        )
        evasive_value = _capture_beam_mass(
            projected,
            evasive_target,
            response_direction=None,
            arena_size=arena_size,
        )
        expected_followup = 0.65 * inertial_value + 0.35 * evasive_value

    own_mass = sum(blob.radius * blob.radius for blob in own)
    owner_mass = sum(
        enemy.radius * enemy.radius
        for enemy in enemies
        if int(enemy.player_id) == int(target.player_id)
    )
    counterattack_cost = OWNER_COUNTERATTACK_WEIGHT * max(
        0.0,
        owner_mass - own_mass,
    )
    if candidate.family not in {"pursuit_enemy", "corner_cutoff_enemy"}:
        return -counterattack_cost

    hunter = _target_hunter(own, target)
    target_mass = target.radius * target.radius
    commitment = 0.0
    if memory is not None and memory.player_id == int(target.player_id):
        age = max(0, current_round - memory.last_seen_round)
        commitment = (
            PURSUIT_COMMITMENT_WEIGHT
            * target_mass
            * math.exp(-age / PURSUIT_MEMORY_DECAY)
        )
    trap_value = 0.0
    if candidate.family == "corner_cutoff_enemy":
        trap_value = (
            CORNER_TRAP_WEIGHT
            * target_mass
            * _trap_geometry(
                hunter.pos,
                target.pos,
                target.radius,
                arena_size,
            )
        )
    return (
        ATTACK_FOLLOWUP_WEIGHT * expected_followup
        + trap_value
        + commitment
        - counterattack_cost
    )


def _capture_beam_mass(
    own: tuple[BlobModel, ...],
    target: VisibleBlobModel,
    *,
    response_direction: tuple[float, float] | None,
    arena_size: float,
) -> float:
    target_mass = target.radius * target.radius
    current_own = own
    current_target = target
    for _ in range(2):
        if _captures_target(current_own, current_target):
            return target_mass
        hunter = min(
            (
                blob
                for blob in current_own
                if can_eat_player_blob(
                    blob.radius,
                    current_target.radius,
                    radius_margin=1.03,
                )
            ),
            key=lambda blob: (
                math.dist(blob.pos, current_target.pos) - blob.radius
            ),
            default=None,
        )
        if hunter is None:
            return 0.0
        direct = normalise(
            (
                current_target.pos[0] - hunter.pos[0],
                current_target.pos[1] - hunter.pos[1],
            )
        )
        cutoff_point = _corner_cutoff_point(
            hunter,
            current_target,
            arena_size=arena_size,
        )
        cutoff = normalise(
            (
                cutoff_point[0] - hunter.pos[0],
                cutoff_point[1] - hunter.pos[1],
            )
        )
        next_states: list[
            tuple[float, tuple[BlobModel, ...], VisibleBlobModel]
        ] = []
        for direction in (direct, cutoff):
            if direction == (0.0, 0.0):
                continue
            next_own = _project_action_blobs(
                current_own,
                direction,
                split=False,
                arena_size=arena_size,
            )
            target_direction = response_direction
            if target_direction is None:
                center = _mass_center(next_own)
                target_direction = normalise(
                    (
                        current_target.pos[0] - center[0],
                        current_target.pos[1] - center[1],
                    )
                )
            next_target = _move_target(
                current_target,
                target_direction,
                arena_size=arena_size,
            )
            gap = min(
                (
                    math.dist(blob.pos, next_target.pos)
                    - blob.radius
                    - next_target.radius
                    for blob in next_own
                    if can_eat_player_blob(
                        blob.radius,
                        next_target.radius,
                        radius_margin=1.03,
                    )
                ),
                default=math.inf,
            )
            next_states.append((gap, next_own, next_target))
        if not next_states:
            return 0.0
        _, current_own, current_target = min(
            next_states,
            key=lambda item: item[0],
        )
    return target_mass if _captures_target(current_own, current_target) else 0.0


def _captures_target(
    own: tuple[BlobModel, ...],
    target: VisibleBlobModel,
) -> bool:
    return any(
        can_eat_player_blob(
            blob.radius,
            target.radius,
            radius_margin=1.03,
        )
        and math.dist(blob.pos, target.pos) <= blob.radius + target.radius
        for blob in own
    )


def _move_target(
    target: VisibleBlobModel,
    direction: tuple[float, float],
    *,
    arena_size: float,
) -> VisibleBlobModel:
    speed = player_speed(target.radius)
    return VisibleBlobModel(
        player_id=target.player_id,
        team_id=target.team_id,
        blob_id=target.blob_id,
        pos=(
            _clamp(
                target.pos[0] + direction[0] * speed,
                target.radius,
                arena_size - target.radius,
            ),
            _clamp(
                target.pos[1] + direction[1] * speed,
                target.radius,
                arena_size - target.radius,
            ),
        ),
        radius=target.radius,
        merge_cooldown=target.merge_cooldown,
    )


def _trap_geometry(
    hunter_pos: tuple[float, float],
    target_pos: tuple[float, float],
    target_radius: float,
    arena_size: float,
) -> float:
    x_clearance = min(
        target_pos[0] - target_radius,
        arena_size - target_radius - target_pos[0],
    )
    y_clearance = min(
        target_pos[1] - target_radius,
        arena_size - target_radius - target_pos[1],
    )
    x_pressure = max(0.0, 1.0 - x_clearance / CORNER_INFLUENCE)
    y_pressure = max(0.0, 1.0 - y_clearance / CORNER_INFLUENCE)
    boundary_pressure = 0.65 * max(x_pressure, y_pressure) + 0.35 * min(
        x_pressure,
        y_pressure,
    )
    center = (arena_size * 0.5, arena_size * 0.5)
    outward = normalise(
        (target_pos[0] - center[0], target_pos[1] - center[1])
    )
    chase = normalise(
        (target_pos[0] - hunter_pos[0], target_pos[1] - hunter_pos[1])
    )
    pin_alignment = max(0.0, _dot(chase, outward))
    return boundary_pressure * pin_alignment


def _mass_center(own: tuple[BlobModel, ...]) -> tuple[float, float]:
    total_mass = sum(blob.radius * blob.radius for blob in own)
    if total_mass <= 0.0:
        return (30.0, 30.0)
    return (
        sum(blob.pos[0] * blob.radius * blob.radius for blob in own) / total_mass,
        sum(blob.pos[1] * blob.radius * blob.radius for blob in own) / total_mass,
    )
