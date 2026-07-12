from __future__ import annotations

"""Robust receding-horizon strategy used by the submission entry point.

The strategy deliberately separates prediction from evaluation:

* opponent movement events are optional because current competition payloads
  censor them;
* predators are rolled forward adversarially when no public direction exists,
  so censored or stale movement never makes an unsafe line look safe;
* own movement, splitting, eject velocity, decay, food, and eating follow the
  engine's update order closely;
* semantic actions are ordered by safety and usefulness, allowing the search
  to return its best completed root candidate when the time budget expires.

Only standard-library code and public agario-kit models are used at runtime.
"""

import math
import os
from dataclasses import dataclass, field, replace
from time import perf_counter

from lib.config.arena import ARENA_SIZE, MAX_BLOB_COUNT, VIRUS_SIZE
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
from simulation.rules import (
    can_consume_virus as engine_can_consume_virus,
    circle_intersects_square,
    decayed_mass_after_turns,
    decayed_radius as engine_decayed_radius,
    movement_speed as engine_movement_speed,
    select_largest_first,
    virus_replacement_positions,
)
from strategies.base import StrategyContext, StrategyDecision
from strategies.features import (
    can_eat_player_blob,
    normalise,
    squared_distance,
)


SQRT2 = math.sqrt(2.0)
TAU = 2.0 * math.pi
EPSILON = 1e-9
WALL_TRAP_RISK_SCALE = 0.1
BLOCKED_MOVEMENT_COST = 0.75


def _receding_horizon_setting(name: str, legacy_name: str, default: str) -> str:
    return os.environ.get(name, os.environ.get(legacy_name, default))


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
    merge_cooldown: int = 0

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
    merge_cooldown: int = 0


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
    control_cost: float = 0.0

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


class ThreatAwareRecedingHorizonStrategy:
    """Threat-first beam search with exact public-rule tactical simulation."""

    name = "threat_aware_receding_horizon"

    def __init__(
        self,
        depth: int | None = None,
        width: int | None = None,
        angular_samples: int | None = None,
        endgame_adaptation: bool | None = None,
    ) -> None:
        # These defaults stay comfortably below the engine's eight-second
        # cumulative budget while still searching substantially more first
        # actions than the older beam strategies.
        self.depth = depth if depth is not None else int(
            _receding_horizon_setting(
                "BOT_RECEDING_HORIZON_DEPTH", "BOT_CHAMPION_DEPTH", "3"
            )
        )
        self.width = width if width is not None else int(
            _receding_horizon_setting(
                "BOT_RECEDING_HORIZON_WIDTH", "BOT_CHAMPION_WIDTH", "4"
            )
        )
        self.angular_samples = (
            angular_samples
            if angular_samples is not None
            else int(
                _receding_horizon_setting(
                    "BOT_RECEDING_HORIZON_ANGLES", "BOT_CHAMPION_ANGLES", "18"
                )
            )
        )
        self.max_food = int(
            _receding_horizon_setting(
                "BOT_RECEDING_HORIZON_MAX_FOOD", "BOT_CHAMPION_MAX_FOOD", "24"
            )
        )
        self.max_enemies = int(
            _receding_horizon_setting(
                "BOT_RECEDING_HORIZON_MAX_ENEMIES",
                "BOT_CHAMPION_MAX_ENEMIES",
                "12",
            )
        )
        # The engine's eight-second cumulative limit includes pipe I/O, model
        # validation, and state updates outside this strategy. Reserve nearly
        # half of it for that runtime overhead instead of spending it on search.
        self.compute_budget_seconds = float(
            _receding_horizon_setting(
                "BOT_RECEDING_HORIZON_TOTAL_BUDGET_SECONDS",
                "BOT_CHAMPION_TOTAL_BUDGET_SECONDS",
                "4.2",
            )
        )
        self.max_turn_seconds = float(
            _receding_horizon_setting(
                "BOT_RECEDING_HORIZON_MAX_TURN_SECONDS",
                "BOT_CHAMPION_MAX_TURN_SECONDS",
                "0.003",
            )
        )
        self.min_search_seconds = float(
            _receding_horizon_setting(
                "BOT_RECEDING_HORIZON_MIN_SEARCH_SECONDS",
                "BOT_CHAMPION_MIN_SEARCH_SECONDS",
                "0.00075",
            )
        )
        self.minimum_root_actions = 1
        self.endgame_adaptation = (
            endgame_adaptation
            if endgame_adaptation is not None
            else _receding_horizon_setting(
                "BOT_RECEDING_HORIZON_ENDGAME_ADAPTATION",
                "BOT_CHAMPION_ENDGAME_ADAPTATION",
                "1",
            )
            != "0"
        )
        self.compute_spent_seconds = 0.0
        self._own_player_id = 0
        self.previous_direction: tuple[float, float] = (1.0, 0.0)
        self.last_moves: dict[int, tuple[tuple[float, float], bool]] = {}
        self.enemy_tracks: dict[tuple[int, int], EnemyTrack] = {}

    def choose(self, context: StrategyContext) -> StrategyDecision:
        started_at = perf_counter()
        state = context.game.state
        self._own_player_id = int(state.me.player_id)
        turn_budget = self._turn_budget_seconds(
            round_number=int(state.round),
            max_rounds=int(state.max_rounds),
        )
        try:
            if self._uses_compute_time_bank() and turn_budget < self.min_search_seconds:
                return self._time_budget_fallback(context)
            return self._choose(
                context,
                deadline=started_at + turn_budget,
                turn_budget=turn_budget,
            )
        finally:
            self.compute_spent_seconds += perf_counter() - started_at

    def _choose(
        self,
        context: StrategyContext,
        *,
        deadline: float,
        turn_budget: float,
    ) -> StrategyDecision:
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
        visible_viruses = tuple(state.visible_viruses)
        enemies = self._update_enemy_memory(
            context,
            own_blobs,
            arena_size,
            viruses=visible_viruses,
        )

        center = self._mass_center(own_blobs)
        foods = tuple(
            sorted(
                state.visible_food,
                key=lambda food: squared_distance(center, food.pos),
            )[: self.max_food]
        )
        viruses = visible_viruses
        enemies = tuple(
            sorted(
                enemies,
                key=lambda enemy: self._enemy_priority(
                    enemy,
                    own_blobs,
                    center,
                ),
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
        search_timed_out = False
        search_stop_reason = "complete"
        root_action_count = 0
        root_actions_generated = 0
        evaluated_by_depth: list[int] = []
        transitions_evaluated = 0
        uses_time_bank = self._uses_compute_time_bank()
        transition_budget = self._transition_budget(
            len(own_blobs),
            len(enemies),
        )
        for depth_index in range(max(1, self.depth)):
            depth_stop_reason = self._depth_start_stop_reason(
                depth_index=depth_index,
                transitions_evaluated=transitions_evaluated,
                transition_budget=transition_budget,
                uses_time_bank=uses_time_bank,
                deadline=deadline,
            )
            if depth_stop_reason is not None:
                search_stop_reason = depth_stop_reason
                search_timed_out = depth_stop_reason == "deadline"
                break

            candidates: list[SearchNode] = []
            depth_timed_out = False
            depth_budget_exhausted = False
            evaluated_actions = 0
            action_rows: list[tuple[SearchNode, tuple[Action, ...]]] = []
            for node in beam:
                actions = self._candidate_actions(
                    node=node,
                    foods=foods,
                    food_targets=food_targets,
                    viruses=viruses,
                    arena_size=arena_size,
                    first_step=depth_index == 0,
                    angle_offset=round_number,
                )
                if depth_index == 0:
                    root_actions_generated += len(actions)
                    actions = self._order_root_actions(actions)
                else:
                    actions = self._order_deeper_actions(actions)
                action_limit = self._actions_per_node_limit(depth_index)
                if action_limit is not None:
                    actions = actions[:action_limit]
                action_rows.append((node, actions))
            max_actions = max((len(actions) for _, actions in action_rows), default=0)
            if depth_index == 0:
                root_action_count = sum(len(actions) for _, actions in action_rows)
            for action_index in range(max_actions):
                for node, actions in action_rows:
                    if action_index >= len(actions):
                        continue
                    action = actions[action_index]
                    if (
                        transition_budget is not None
                        and transitions_evaluated >= transition_budget
                    ):
                        depth_budget_exhausted = True
                        break
                    # Evaluate at least one action at each depth, then stop at
                    # the deadline. Deeper parents are expanded round-robin so
                    # the first beam node cannot consume the whole budget.
                    required_actions = (
                        self.minimum_root_actions if depth_index == 0 else 1
                    )
                    if (
                        uses_time_bank
                        and evaluated_actions >= required_actions
                        and perf_counter() >= deadline
                    ):
                        depth_timed_out = True
                        break
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
                    evaluated_actions += 1
                    transitions_evaluated += 1
                    if result.fatal:
                        if best_rejected is None or result.node.score > best_rejected.score:
                            best_rejected = result.node
                        continue
                    candidates.append(result.node)
                if depth_timed_out or depth_budget_exhausted:
                    break

            evaluated_by_depth.append(evaluated_actions)

            if depth_timed_out:
                search_timed_out = True
                search_stop_reason = "deadline"
                if candidates:
                    # All candidates here have the same simulated depth. Keep
                    # useful partial work instead of discarding an interrupted
                    # deeper layer entirely.
                    beam = self._prune(candidates)
                    reached_depth = depth_index + 1
                break
            if depth_budget_exhausted:
                search_stop_reason = "transition_budget"
                if candidates:
                    beam = self._prune(candidates)
                    reached_depth = depth_index + 1
                break
            if not candidates:
                break
            beam = self._prune(candidates)
            reached_depth = depth_index + 1
            if uses_time_bank and perf_counter() >= deadline:
                search_timed_out = True
                search_stop_reason = "deadline"
                break

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
                "endgame_adaptation": self.endgame_adaptation,
                "visible_or_tracked_enemies": len(enemies),
                "projected_food": best.projected_food,
                "projected_captures": best.projected_captures,
                "projected_blob_count": len(best.own_blobs),
                "min_safety_margin": best.min_safety_margin,
                "search_timed_out": search_timed_out,
                "search_stop_reason": search_stop_reason,
                "root_actions_total": root_action_count,
                "root_actions_generated": root_actions_generated,
                "root_actions_evaluated": (
                    evaluated_by_depth[0] if evaluated_by_depth else 0
                ),
                "transitions_evaluated": transitions_evaluated,
                "transitions_by_depth": evaluated_by_depth,
                "transition_budget": transition_budget,
                "turn_budget_ms": round(turn_budget * 1000.0, 3),
                "compute_spent_ms": round(self.compute_spent_seconds * 1000.0, 3),
            },
        )

    @staticmethod
    def _depth_start_stop_reason(
        *,
        depth_index: int,
        transitions_evaluated: int,
        transition_budget: int | None,
        uses_time_bank: bool,
        deadline: float,
    ) -> str | None:
        """Stop before generating a depth whose actions cannot be evaluated."""

        if (
            transition_budget is not None
            and transitions_evaluated >= transition_budget
        ):
            return "transition_budget"
        # Root candidates are needed to produce a decision even if setup used
        # the nominal turn budget. Deeper candidates are optional and can be
        # skipped before their comparatively expensive generation begins.
        if depth_index > 0 and uses_time_bank and perf_counter() >= deadline:
            return "deadline"
        return None

    def _turn_budget_seconds(self, *, round_number: int, max_rounds: int) -> float:
        remaining_budget = max(0.0, self.compute_budget_seconds - self.compute_spent_seconds)
        remaining_rounds = max(1, max_rounds - round_number)
        return min(self.max_turn_seconds, remaining_budget / remaining_rounds)

    def _uses_compute_time_bank(self) -> bool:
        """Whether wall time, rather than fixed work, limits this search."""

        return True

    def _transition_budget(
        self,
        own_blob_count: int,
        enemy_count: int = 0,
    ) -> int | None:
        """Return a deterministic transition quota, or ``None`` for anytime search."""

        return None

    def _time_budget_fallback(self, context: StrategyContext) -> StrategyDecision:
        """Return a cheap visible-state action after the search bank is spent."""

        state = context.game.state
        own_blobs = tuple(state.me.blobs.values())
        if not own_blobs:
            return StrategyDecision(direction=self.previous_direction, reason="time_bank_dead")

        away_x = 0.0
        away_y = 0.0
        for enemy in state.visible_blobs:
            if not any(
                can_eat_player_blob(enemy.radius, own.radius)
                for own in own_blobs
            ):
                continue
            nearest = min(own_blobs, key=lambda own: squared_distance(own.pos, enemy.pos))
            distance = max(math.dist(nearest.pos, enemy.pos), 0.25)
            away_x += (nearest.pos[0] - enemy.pos[0]) * enemy.radius / (distance * distance)
            away_y += (nearest.pos[1] - enemy.pos[1]) * enemy.radius / (distance * distance)

        direction = normalise((away_x, away_y))
        reason = "time_bank_escape"
        if direction == (0.0, 0.0) and state.visible_food:
            center = (state.me.x, state.me.y)
            food = min(state.visible_food, key=lambda item: squared_distance(center, item.pos))
            direction = normalise((food.pos[0] - center[0], food.pos[1] - center[1]))
            reason = "time_bank_food"
        if direction == (0.0, 0.0):
            direction = self.previous_direction
            reason = "time_bank_keep"

        self.previous_direction = direction
        return StrategyDecision(
            direction=direction,
            target_kind="escape" if reason == "time_bank_escape" else "food",
            reason=reason,
            diagnostics={
                "compute_spent_ms": round(self.compute_spent_seconds * 1000.0, 3),
                "compute_budget_ms": round(self.compute_budget_seconds * 1000.0, 3),
            },
        )

    def _read_public_moves(self, context: StrategyContext) -> None:
        for event in context.query.update.values():
            if not isinstance(event, MovePlayer):
                continue
            direction = normalise(event.direction.to_vector())
            self.last_moves[event.player_id] = (direction, bool(event.split))

    def _enemy_priority(
        self,
        enemy: EnemyBlob,
        own_blobs: tuple[OwnBlob, ...],
        center: tuple[float, float],
    ) -> tuple[int, float, float]:
        """Keep threats to current or post-split fragments before harmless blobs."""

        threat_margins: list[float] = []
        for own, candidate_radius in self._exposed_own_radii(own_blobs):
            if not can_eat_player_blob(enemy.radius, candidate_radius):
                continue
            danger_radius = enemy.radius
            if _can_split_eat(enemy.radius, candidate_radius):
                danger_radius = max(danger_radius, _split_attack_reach(enemy.radius))
            threat_margins.append(math.dist(own.pos, enemy.pos) - danger_radius)

        center_distance = math.dist(center, enemy.pos)
        if threat_margins:
            return (0, min(threat_margins), center_distance)
        return (1, center_distance, center_distance)

    def _exposed_own_radii(
        self,
        own_blobs: tuple[OwnBlob, ...],
        viruses: tuple[VirusModel, ...] = (),
    ) -> tuple[tuple[OwnBlob, float], ...]:
        """Radii an enemy may face after one available public transition."""

        exposed: list[tuple[OwnBlob, float]] = []
        virus_piece_count = max(1, MAX_BLOB_COUNT - len(own_blobs) + 1)
        for own in own_blobs:
            exposed.append((own, own.radius))
            if own.mass >= SPLIT_MIN_MASS:
                exposed.append((own, own.radius / SQRT2))
            virus_reachable = any(
                _can_consume_virus(own.radius, virus.radius)
                and max(0.0, math.dist(own.pos, virus.pos) - own.radius)
                <= 18.0 * _speed(own.radius)
                for virus in viruses
            )
            if virus_reachable:
                virus_piece_radius = math.sqrt(
                    (own.mass + VIRUS_SIZE * VIRUS_SIZE)
                    / virus_piece_count
                )
                exposed.append((own, virus_piece_radius))
        return tuple(exposed)

    def _update_enemy_memory(
        self,
        context: StrategyContext,
        own_blobs: tuple[OwnBlob, ...],
        arena_size: float,
        viruses: tuple[VirusModel, ...] = (),
    ) -> tuple[EnemyBlob, ...]:
        state = context.game.state
        round_number = int(state.round)
        visible_keys: set[tuple[int, int]] = set()

        # Current competition payloads censor opponent movement. Advance an
        # unseen predator toward the nearest vulnerable fragment unless an
        # explicit public move is available. Visible data below always replaces
        # this conservative estimate.
        advanced: dict[tuple[int, int], EnemyTrack] = {}
        for key, track in self.enemy_tracks.items():
            observed_move = self.last_moves.get(track.player_id)
            if observed_move is not None:
                direction = observed_move[0]
            else:
                vulnerable = tuple(
                    own
                    for own in own_blobs
                    if can_eat_player_blob(track.radius, own.radius)
                )
                if vulnerable:
                    target = min(
                        vulnerable,
                        key=lambda own: squared_distance((track.x, track.y), own.pos),
                    )
                    direction = normalise((target.x - track.x, target.y - track.y))
                else:
                    direction = track.direction
            speed = _speed(track.radius)
            radius = _decayed_radius(track.radius)
            advanced[key] = replace(
                track,
                x=_clamp(track.x + direction[0] * speed, radius, arena_size - radius),
                y=_clamp(track.y + direction[1] * speed, radius, arena_size - radius),
                radius=radius,
                direction=direction,
                merge_cooldown=max(0, track.merge_cooldown - 1),
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
                merge_cooldown=blob.merge_cooldown,
            )

        view_center = tuple(state.view_center)
        half_view = float(state.vision_size) / 2.0
        kept: dict[tuple[int, int], EnemyTrack] = {}
        enemies: list[EnemyBlob] = []
        for key, track in advanced.items():
            stale_rounds = round_number - track.last_seen_round
            if stale_rounds > 10:
                continue
            should_be_visible = circle_intersects_square(
                circle_x=track.x,
                circle_y=track.y,
                circle_radius=track.radius,
                square_center_x=view_center[0],
                square_center_y=view_center[1],
                square_size=half_view * 2.0,
            )
            if key not in visible_keys and should_be_visible:
                # The estimate contradicted the authoritative current view.
                continue
            kept[key] = track
            # Stale prey is never chased. Retain only potentially dangerous
            # stale blobs, including enemies that become dangerous after our
            # own legal split or virus transition.
            if stale_rounds and not self._stale_enemy_can_threaten_transition(
                track,
                own_blobs,
                viruses,
            ):
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
                    merge_cooldown=track.merge_cooldown,
                )
            )
        self.enemy_tracks = kept
        return tuple(enemies)

    def _stale_enemy_can_threaten_transition(
        self,
        track: EnemyTrack,
        own_blobs: tuple[OwnBlob, ...],
        viruses: tuple[VirusModel, ...],
    ) -> bool:
        # Existing fragments are authoritative state: any predator that can
        # eat one remains relevant, even if it cannot eat our largest blob.
        if any(
            can_eat_player_blob(track.radius, own.radius)
            for own in own_blobs
        ):
            return True

        virus_piece_count = max(1, MAX_BLOB_COUNT - len(own_blobs) + 1)
        for own in own_blobs:
            if own.mass >= SPLIT_MIN_MASS and can_eat_player_blob(
                track.radius,
                own.radius / SQRT2,
            ):
                return True
            virus_reachable = any(
                _can_consume_virus(own.radius, virus.radius)
                and max(0.0, math.dist(own.pos, virus.pos) - own.radius)
                <= 18.0 * _speed(own.radius)
                for virus in viruses
            )
            if not virus_reachable:
                continue
            virus_piece_radius = math.sqrt(
                (own.mass + VIRUS_SIZE * VIRUS_SIZE)
                / virus_piece_count
            )
            # A stale small blob may take one piece, but the replay collapse
            # comes from a large enemy splitting through many pieces.  Preserve
            # that tail risk without treating every unseen prey as a sweeper.
            if _can_split_eat(track.radius, virus_piece_radius):
                return True
        return False

    def _candidate_actions(
        self,
        *,
        node: SearchNode,
        foods: tuple[FoodModel, ...],
        food_targets: tuple[tuple[float, float], ...],
        viruses: tuple[VirusModel, ...],
        arena_size: float,
        first_step: bool,
        angle_offset: int = 0,
    ) -> tuple[Action, ...]:
        actions: list[Action] = []

        # This order is part of the anytime-search contract: if the root is
        # cut short, it must already have considered the tactically important
        # options rather than an arbitrary prefix of the angular grid.
        escape = self._escape_vector(node)
        if escape != (0.0, 0.0):
            actions.append(Action(escape, reason="escape"))
            # Tangential options often escape a wall/predator pincer better
            # than a pure potential-field vector.
            actions.append(Action(_rotate(escape, math.pi / 8), reason="escape_tangent"))
            actions.append(Action(_rotate(escape, -math.pi / 8), reason="escape_tangent"))

        actions.append(Action(node.last_direction, reason="keep" if first_step else "continue"))
        if not first_step:
            for angle in (-math.pi / 6, -math.pi / 12, math.pi / 12, math.pi / 6):
                actions.append(Action(_rotate(node.last_direction, angle), reason="steer"))

        center = node.center
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
            and any(
                can_eat_player_blob(own.radius, enemy.radius)
                for own in node.own_blobs
            )
        ]
        prey.sort(
            key=lambda enemy: self._prey_candidate_priority(
                node,
                enemy,
                arena_size,
            )
        )
        for enemy in prey[: 3 if first_step else 2]:
            intercept = self._intercept_direction(node.primary, enemy)
            actions.append(Action(intercept, reason="prey"))
            if self._split_can_capture(node, enemy, intercept):
                actions.append(Action(intercept, split=True, reason="split_prey"))

        farm_action = self._safe_farm_split(node, available_food, food_targets)
        if farm_action is not None:
            actions.append(farm_action)

        wall = self._wall_vector(node.primary, arena_size)
        if wall != (0.0, 0.0):
            actions.append(Action(wall, reason="wall_escape"))
        actions.append(Action(normalise((arena_size / 2.0 - center[0], arena_size / 2.0 - center[1])), reason="center"))

        if first_step:
            sample_count = max(8, self.angular_samples)
            actions.extend(
                Action(
                    (
                        math.cos(TAU * ((index + angle_offset) % sample_count) / sample_count),
                        math.sin(TAU * ((index + angle_offset) % sample_count) / sample_count),
                    ),
                    reason="angle",
                )
                for index in range(sample_count)
            )

        return self._dedupe_actions(actions)

    def _prey_candidate_priority(
        self,
        node: SearchNode,
        enemy: EnemyBlob,
        arena_size: float,
    ) -> tuple[float, ...]:
        """Cheap candidate ordering hook; lower tuples are explored first."""

        return (squared_distance(node.center, enemy.pos),)

    def _order_root_actions(self, actions: tuple[Action, ...]) -> tuple[Action, ...]:
        return actions

    def _order_deeper_actions(self, actions: tuple[Action, ...]) -> tuple[Action, ...]:
        return actions

    def _actions_per_node_limit(self, depth_index: int) -> int | None:
        return None

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
        split_blob_ids = {blob.blob_id for blob in own_blobs} if action.split else set()

        before_move = own_blobs
        own_blobs = [self._move_own(blob, direction, arena_size) for blob in own_blobs]
        blocked_movement = self._blocked_movement_distance(
            before_move,
            own_blobs,
            direction,
        )
        enemies = self._move_enemies(node.enemies, own_blobs, arena_size)

        eaten_food_ids = set(node.eaten_food_ids)
        captured_enemy_ids = set(node.captured_enemy_ids)
        consumed_virus_ids = set(node.consumed_virus_ids)
        projected_food = node.projected_food
        projected_captures = node.projected_captures

        own_blobs = [replace(blob, radius=_decayed_radius(blob.radius)) for blob in own_blobs]
        own_blobs = self._stabilise_own_blobs(own_blobs, arena_size)
        enemies = self._stabilise_enemy_blobs(enemies, arena_size)

        # Engine order is stabilise -> viruses -> stabilise -> food.  Keeping
        # virus resolution after food can incorrectly push a blob over the
        # consumption threshold or preserve a pre-pop large cell.
        own_blobs, enemies, virus_score, virus_penalty = self._resolve_own_viruses(
            own_blobs=own_blobs,
            enemies=enemies,
            viruses=viruses,
            consumed_virus_ids=consumed_virus_ids,
            arena_size=arena_size,
        )
        score += virus_score
        score -= virus_penalty * safety_weight
        own_blobs = self._stabilise_own_blobs(own_blobs, arena_size)
        enemies = self._stabilise_enemy_blobs(enemies, arena_size)

        all_blobs: dict[tuple[int, int], OwnBlob | EnemyBlob] = {
            (self._own_player_id, blob.blob_id): blob for blob in own_blobs
        }
        all_blobs.update(
            ((enemy.player_id, enemy.blob_id), enemy) for enemy in enemies
        )
        for food in foods:
            if food.food_id in eaten_food_ids:
                continue
            candidates = [
                (key, blob)
                for key, blob in all_blobs.items()
                if squared_distance(blob.pos, food.pos) <= blob.radius * blob.radius
            ]
            winner = select_largest_first(
                candidates,
                radius=lambda candidate: candidate[1].radius,
                player_id=lambda candidate: candidate[0][0],
                blob_id=lambda candidate: candidate[0][1],
            )
            if winner is None:
                continue
            key, eater = winner
            all_blobs[key] = _with_grown_radius(
                eater,
                math.sqrt(eater.mass + FOOD_RADIUS * FOOD_RADIUS),
                arena_size,
            )
            eaten_food_ids.add(food.food_id)
            if key[0] == self._own_player_id:
                projected_food += 1
                score += 7.0

        own_blobs = [
            blob
            for (player_id, _), blob in all_blobs.items()
            if player_id == self._own_player_id
        ]
        enemies = tuple(
            blob
            for (player_id, _), blob in all_blobs.items()
            if player_id != self._own_player_id
        )

        own_blobs, enemies, interaction_score, captures = self._resolve_interactions(
            own_blobs,
            enemies,
            captured_enemy_ids,
            arena_size,
        )
        split_lost_fragment = bool(
            split_blob_ids - {blob.blob_id for blob in own_blobs}
        )
        projected_captures += captures
        score += interaction_score * aggression
        own_blobs = self._stabilise_own_blobs(own_blobs, arena_size)
        enemies = self._stabilise_enemy_blobs(enemies, arena_size)
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

        risk_penalty, min_margin, unavoidable = self._risk_score(
            own_blobs,
            enemies,
            safety_weight,
            arena_size,
        )
        score -= risk_penalty
        score += self._position_value(own_blobs, enemies, foods, eaten_food_ids, arena_size, aggression)
        score -= blocked_movement * BLOCKED_MOVEMENT_COST
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
        # A split is admissible only when every resulting fragment survives the
        # immediate interaction pass and remains outside adversarial split reach.
        unsafe_split = action.split and (split_lost_fragment or min_margin <= 0.0)
        return StepResult(next_node, fatal=unavoidable or unsafe_split)

    def _resolve_own_viruses(
        self,
        *,
        own_blobs: list[OwnBlob],
        enemies: tuple[EnemyBlob, ...] = (),
        viruses: tuple[VirusModel, ...],
        consumed_virus_ids: set[int],
        arena_size: float,
    ) -> tuple[list[OwnBlob], tuple[EnemyBlob, ...], float, float]:
        """Apply the public engine's touching-blob virus transition.

        The threat-aware base assigns a high controllability penalty, while a
        farming subclass can override only the utility terms.  The physical
        state transition itself remains the same.
        """
        penalty = 0.0
        for virus in viruses:
            if virus.virus_id in consumed_virus_ids:
                continue
            collision = self._apply_virus_collision(
                own_blobs=own_blobs,
                enemies=enemies,
                virus=virus,
                arena_size=arena_size,
            )
            if collision is None:
                continue
            consumed_virus_ids.add(virus.virus_id)
            own_blobs, enemies, origin, _ = collision
            if origin is None:
                continue
            penalty += 240.0 + origin.mass * 18.0
        return own_blobs, enemies, 0.0, penalty

    def _apply_virus_collision(
        self,
        *,
        own_blobs: list[OwnBlob],
        enemies: tuple[EnemyBlob, ...],
        virus: VirusModel,
        arena_size: float,
    ) -> tuple[list[OwnBlob], tuple[EnemyBlob, ...], OwnBlob | None, int] | None:
        """Apply one virus collision; return the own origin only if self won."""

        candidates = [
            (True, index, self._own_player_id, blob)
            for index, blob in enumerate(own_blobs)
            if _can_consume_virus(blob.radius, virus.radius)
            and squared_distance(blob.pos, virus.pos) <= blob.radius * blob.radius
        ]
        candidates.extend(
            (False, index, enemy.player_id, enemy)
            for index, enemy in enumerate(enemies)
            if _can_consume_virus(enemy.radius, virus.radius)
            and squared_distance(enemy.pos, virus.pos) <= enemy.radius * enemy.radius
        )
        winner = select_largest_first(
            candidates,
            radius=lambda candidate: candidate[3].radius,
            player_id=lambda candidate: candidate[2],
            blob_id=lambda candidate: candidate[3].blob_id,
        )
        if winner is None:
            return None

        is_own, index, _, origin = winner
        same_player = (
            list(own_blobs)
            if is_own
            else [enemy for enemy in enemies if enemy.player_id == origin.player_id]
        )
        piece_count = max(1, MAX_BLOB_COUNT - len(same_player) + 1)
        piece_radius = math.sqrt(
            (origin.mass + virus.radius * virus.radius) / piece_count
        )
        fragments = self._virus_replacement_fragments(
            origin=origin,
            piece_radius=piece_radius,
            piece_count=piece_count,
            arena_size=arena_size,
            occupied_ids={blob.blob_id for blob in same_player},
        )
        if is_own:
            own_blobs = own_blobs[:index] + list(fragments) + own_blobs[index + 1 :]
            return own_blobs, enemies, origin, piece_count
        enemies = enemies[:index] + tuple(fragments) + enemies[index + 1 :]
        return own_blobs, enemies, None, piece_count

    def _virus_replacement_fragments(
        self,
        *,
        origin: OwnBlob | EnemyBlob,
        piece_radius: float,
        piece_count: int,
        arena_size: float,
        occupied_ids: set[int] | None = None,
    ) -> list[OwnBlob | EnemyBlob]:
        if piece_count <= 1:
            return [
                replace(
                    origin,
                    radius=piece_radius,
                    x=_clamp(origin.x, piece_radius, arena_size - piece_radius),
                    y=_clamp(origin.y, piece_radius, arena_size - piece_radius),
                )
            ]
        positions = virus_replacement_positions(
            center_x=origin.x,
            center_y=origin.y,
            piece_radius=piece_radius,
            piece_count=piece_count,
            overlap_epsilon=SAME_PLAYER_OVERLAP_EPSILON,
        )
        used_ids = set(occupied_ids or ())
        used_ids.add(origin.blob_id)
        next_id = 0
        fragments: list[OwnBlob | EnemyBlob] = []
        for index, (x, y) in enumerate(positions):
            if index == 0:
                blob_id = origin.blob_id
            else:
                while next_id in used_ids:
                    next_id += 1
                blob_id = next_id
                used_ids.add(blob_id)
                next_id += 1
            fragments.append(
                replace(
                    origin,
                    blob_id=blob_id,
                    x=_clamp(x, piece_radius, arena_size - piece_radius),
                    y=_clamp(y, piece_radius, arena_size - piece_radius),
                    radius=piece_radius,
                    merge_cooldown=SPLIT_COOLDOWN_FRAMES,
                    **(
                        {"eject_vx": 0.0, "eject_vy": 0.0}
                        if isinstance(origin, OwnBlob)
                        else {}
                    ),
                )
            )
        return fragments

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

    def _blocked_movement_distance(
        self,
        before: list[OwnBlob],
        after: list[OwnBlob],
        direction: tuple[float, float],
    ) -> float:
        """Return the mass-weighted movement lost when the arena clamps a move."""

        if not before:
            return 0.0
        after_by_id = {blob.blob_id: blob for blob in after}
        total_mass = sum(blob.mass for blob in before)
        if total_mass <= EPSILON:
            return 0.0

        lost = 0.0
        for blob in before:
            moved = after_by_id.get(blob.blob_id)
            if moved is None:
                continue
            intended_dx = direction[0] * _speed(blob.radius) + blob.eject_vx
            intended_dy = direction[1] * _speed(blob.radius) + blob.eject_vy
            intended_distance = math.hypot(intended_dx, intended_dy)
            actual_distance = math.dist(blob.pos, moved.pos)
            lost += blob.mass * max(0.0, intended_distance - actual_distance)
        return lost / total_mass

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

    def _stabilise_own_blobs(
        self,
        blobs: list[OwnBlob],
        arena_size: float,
    ) -> list[OwnBlob]:
        """Apply the exact engine sequence with mutable rollout-local records."""

        if len(blobs) <= 1:
            return blobs

        # [source, x, y, radius, cooldown, eject_vx, eject_vy]
        work = {
            blob.blob_id: [
                blob,
                blob.x,
                blob.y,
                blob.radius,
                blob.merge_cooldown,
                getattr(blob, "eject_vx", 0.0),
                getattr(blob, "eject_vy", 0.0),
            ]
            for blob in blobs
        }

        total_mass = sum(item[3] * item[3] for item in work.values())
        center_x = sum(item[1] * item[3] * item[3] for item in work.values()) / total_mass
        center_y = sum(item[2] * item[3] * item[3] for item in work.values()) / total_mass
        for item in work.values():
            dx = center_x - item[1]
            dy = center_y - item[2]
            distance = math.hypot(dx, dy)
            if distance == 0.0:
                continue
            step = min(MERGE_ATTRACTION_SPEED, distance)
            item[1] = _clamp(
                item[1] + dx / distance * step,
                item[3],
                arena_size - item[3],
            )
            item[2] = _clamp(
                item[2] + dy / distance * step,
                item[3],
                arena_size - item[3],
            )

        def merge_touching() -> None:
            while True:
                merged = False
                ids = sorted(work)
                for first_index, first_id in enumerate(ids):
                    first = work[first_id]
                    for second_id in ids[first_index + 1 :]:
                        second = work[second_id]
                        if first[4] > 0 or second[4] > 0:
                            continue
                        if math.hypot(second[1] - first[1], second[2] - first[2]) > (
                            first[3] + second[3] + SAME_PLAYER_OVERLAP_EPSILON
                        ):
                            continue
                        first_mass = first[3] * first[3]
                        second_mass = second[3] * second[3]
                        if (-first_mass, first_id) <= (-second_mass, second_id):
                            survivor_id, survivor = first_id, first
                            consumed_id, consumed = second_id, second
                        else:
                            survivor_id, survivor = second_id, second
                            consumed_id, consumed = first_id, first
                        survivor_mass = survivor[3] * survivor[3]
                        consumed_mass = consumed[3] * consumed[3]
                        combined_mass = survivor_mass + consumed_mass
                        combined_radius = math.sqrt(combined_mass)
                        survivor[1] = _clamp(
                            (survivor[1] * survivor_mass + consumed[1] * consumed_mass)
                            / combined_mass,
                            combined_radius,
                            arena_size - combined_radius,
                        )
                        survivor[2] = _clamp(
                            (survivor[2] * survivor_mass + consumed[2] * consumed_mass)
                            / combined_mass,
                            combined_radius,
                            arena_size - combined_radius,
                        )
                        survivor[5] = (
                            survivor[5] * survivor_mass + consumed[5] * consumed_mass
                        ) / combined_mass
                        survivor[6] = (
                            survivor[6] * survivor_mass + consumed[6] * consumed_mass
                        ) / combined_mass
                        survivor[3] = combined_radius
                        survivor[4] = 0
                        work[survivor_id] = survivor
                        del work[consumed_id]
                        merged = True
                        break
                    if merged:
                        break
                if not merged:
                    return

        def separate(iterations: int = 4) -> None:
            for _ in range(iterations):
                changed = False
                ids = sorted(work)
                for first_index, first_id in enumerate(ids):
                    first = work[first_id]
                    for second_id in ids[first_index + 1 :]:
                        second = work[second_id]
                        dx = second[1] - first[1]
                        dy = second[2] - first[2]
                        minimum = first[3] + second[3] + SAME_PLAYER_OVERLAP_EPSILON
                        distance = math.hypot(dx, dy)
                        if distance >= minimum:
                            continue
                        if distance == 0.0:
                            nx, ny = (1.0, 0.0)
                        else:
                            nx, ny = (dx / distance, dy / distance)
                        overlap = minimum - distance
                        first_mass = first[3] * first[3]
                        second_mass = second[3] * second[3]
                        pair_mass = first_mass + second_mass
                        first_move = overlap * second_mass / pair_mass
                        second_move = overlap * first_mass / pair_mass
                        first[1] = _clamp(
                            first[1] - nx * first_move,
                            first[3],
                            arena_size - first[3],
                        )
                        first[2] = _clamp(
                            first[2] - ny * first_move,
                            first[3],
                            arena_size - first[3],
                        )
                        second[1] = _clamp(
                            second[1] + nx * second_move,
                            second[3],
                            arena_size - second[3],
                        )
                        second[2] = _clamp(
                            second[2] + ny * second_move,
                            second[3],
                            arena_size - second[3],
                        )
                        changed = True
                if not changed:
                    return

        merge_touching()
        separate()
        merge_touching()
        separate()
        result = []
        for _, item in sorted(work.items()):
            updates = {
                "x": item[1],
                "y": item[2],
                "radius": item[3],
                "merge_cooldown": item[4],
            }
            if isinstance(item[0], OwnBlob):
                updates.update(eject_vx=item[5], eject_vy=item[6])
            result.append(replace(item[0], **updates))
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
            if can_eat_player_blob(enemy.radius, target.radius):
                adversarial = normalise((target.x - enemy.x, target.y - enemy.y))
                direction = normalise((
                    adversarial[0] * 0.82 + observed[0] * 0.18,
                    adversarial[1] * 0.82 + observed[1] * 0.18,
                ))
            elif any(
                can_eat_player_blob(own.radius, enemy.radius)
                for own in own_blobs
            ):
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
                    merge_cooldown=max(0, enemy.merge_cooldown - 1),
                )
            )
        return tuple(moved)

    def _stabilise_enemy_blobs(
        self,
        enemies: tuple[EnemyBlob, ...],
        arena_size: float,
    ) -> tuple[EnemyBlob, ...]:
        """Apply the engine stabilisation sequence independently per player."""

        by_player: dict[int, list[EnemyBlob]] = {}
        for enemy in enemies:
            by_player.setdefault(enemy.player_id, []).append(enemy)

        result: list[EnemyBlob] = []
        for player_id in sorted(by_player):
            group = by_player[player_id]
            group = list(self._apply_attraction(group, arena_size))
            group = list(self._merge_enemy_blobs(tuple(group), arena_size))
            group = list(self._separate_enemy_blobs(tuple(group), arena_size))
            group = list(self._merge_enemy_blobs(tuple(group), arena_size))
            group = list(self._separate_enemy_blobs(tuple(group), arena_size))
            result.extend(group)
        return tuple(sorted(result, key=lambda enemy: enemy.key))

    def _merge_enemy_blobs(
        self,
        enemies: tuple[EnemyBlob, ...],
        arena_size: float,
    ) -> tuple[EnemyBlob, ...]:
        """Merge same-player enemy fragments that can combine this rollout step."""

        by_key = {enemy.key: enemy for enemy in enemies}
        while True:
            merged = False
            keys = sorted(by_key)
            for index, first_key in enumerate(keys):
                for second_key in keys[index + 1 :]:
                    if first_key[0] != second_key[0]:
                        continue
                    first = by_key[first_key]
                    second = by_key[second_key]
                    if first.merge_cooldown > 0 or second.merge_cooldown > 0:
                        continue
                    merge_reach = (
                        first.radius
                        + second.radius
                        + SAME_PLAYER_OVERLAP_EPSILON
                    )
                    if math.dist(first.pos, second.pos) > merge_reach:
                        continue
                    survivor, consumed = sorted(
                        (first, second),
                        key=lambda enemy: (-enemy.mass, enemy.blob_id),
                    )
                    combined_mass = survivor.mass + consumed.mass
                    combined_radius = math.sqrt(combined_mass)
                    combined_direction = normalise(
                        (
                            survivor.direction[0] * survivor.mass
                            + consumed.direction[0] * consumed.mass,
                            survivor.direction[1] * survivor.mass
                            + consumed.direction[1] * consumed.mass,
                        )
                    )
                    combined = replace(
                        survivor,
                        x=_clamp(
                            (survivor.x * survivor.mass + consumed.x * consumed.mass)
                            / combined_mass,
                            combined_radius,
                            arena_size - combined_radius,
                        ),
                        y=_clamp(
                            (survivor.y * survivor.mass + consumed.y * consumed.mass)
                            / combined_mass,
                            combined_radius,
                            arena_size - combined_radius,
                        ),
                        radius=combined_radius,
                        direction=combined_direction,
                        stale_rounds=max(survivor.stale_rounds, consumed.stale_rounds),
                        merge_cooldown=0,
                    )
                    by_key[survivor.key] = combined
                    del by_key[consumed.key]
                    merged = True
                    break
                if merged:
                    break
            if not merged:
                return tuple(by_key[key] for key in sorted(by_key))

    def _separate_enemy_blobs(
        self,
        enemies: tuple[EnemyBlob, ...],
        arena_size: float,
        iterations: int = 4,
    ) -> tuple[EnemyBlob, ...]:
        by_key = {enemy.key: enemy for enemy in enemies}
        for _ in range(iterations):
            changed = False
            keys = sorted(by_key)
            for index, first_key in enumerate(keys):
                for second_key in keys[index + 1 :]:
                    if first_key[0] != second_key[0]:
                        continue
                    first = by_key[first_key]
                    second = by_key[second_key]
                    dx = second.x - first.x
                    dy = second.y - first.y
                    distance = math.hypot(dx, dy)
                    minimum = first.radius + second.radius + SAME_PLAYER_OVERLAP_EPSILON
                    if distance >= minimum:
                        continue
                    if distance <= EPSILON:
                        nx, ny = (1.0, 0.0)
                    else:
                        nx, ny = (dx / distance, dy / distance)
                    overlap = minimum - distance
                    total_mass = first.mass + second.mass
                    first_move = overlap * second.mass / total_mass
                    second_move = overlap * first.mass / total_mass
                    by_key[first_key] = replace(
                        first,
                        x=_clamp(
                            first.x - nx * first_move,
                            first.radius,
                            arena_size - first.radius,
                        ),
                        y=_clamp(
                            first.y - ny * first_move,
                            first.radius,
                            arena_size - first.radius,
                        ),
                    )
                    by_key[second_key] = replace(
                        second,
                        x=_clamp(
                            second.x + nx * second_move,
                            second.radius,
                            arena_size - second.radius,
                        ),
                        y=_clamp(
                            second.y + ny * second_move,
                            second.radius,
                            arena_size - second.radius,
                        ),
                    )
                    changed = True
            if not changed:
                break
        return tuple(by_key[key] for key in sorted(by_key))

    def _future_enemy_envelopes(
        self,
        enemies: tuple[EnemyBlob, ...],
        *,
        horizon: int = 8,
    ) -> tuple[EnemyBlob, ...]:
        """Collapse fragments that can recombine soon into threat envelopes.

        This is a value approximation, not a simulated interaction actor.  It
        makes merge-then-split reach visible to a one-step search without
        pretending that the merge has already happened in the physical state.
        """
        by_player: dict[int, list[EnemyBlob]] = {}
        for enemy in enemies:
            if enemy.merge_cooldown <= horizon:
                by_player.setdefault(enemy.player_id, []).append(enemy)

        envelopes: list[EnemyBlob] = []
        attraction_reach = 2.0 * MERGE_ATTRACTION_SPEED * horizon
        for player_id, fragments in by_player.items():
            remaining = set(range(len(fragments)))
            while remaining:
                component = {remaining.pop()}
                changed = True
                while changed:
                    changed = False
                    for index in tuple(remaining):
                        candidate = fragments[index]
                        if any(
                            math.dist(candidate.pos, fragments[member].pos)
                            <= (
                                candidate.radius
                                + fragments[member].radius
                                + SAME_PLAYER_OVERLAP_EPSILON
                                + attraction_reach
                            )
                            for member in component
                        ):
                            remaining.remove(index)
                            component.add(index)
                            changed = True
                if len(component) <= 1:
                    continue
                group = [fragments[index] for index in component]
                mass = sum(fragment.mass for fragment in group)
                envelopes.append(
                    EnemyBlob(
                        player_id=player_id,
                        blob_id=-1 - min(fragment.blob_id for fragment in group),
                        x=sum(fragment.x * fragment.mass for fragment in group) / mass,
                        y=sum(fragment.y * fragment.mass for fragment in group) / mass,
                        radius=math.sqrt(mass),
                        direction=normalise(
                            (
                                sum(fragment.direction[0] * fragment.mass for fragment in group),
                                sum(fragment.direction[1] * fragment.mass for fragment in group),
                            )
                        ),
                        stale_rounds=max(fragment.stale_rounds for fragment in group),
                        merge_cooldown=max(fragment.merge_cooldown for fragment in group),
                    )
                )
        return tuple(envelopes)

    def _resolve_interactions(
        self,
        own_blobs: list[OwnBlob],
        enemies: tuple[EnemyBlob, ...],
        captured_enemy_ids: set[tuple[int, int]],
        arena_size: float = ARENA_SIZE,
    ) -> tuple[list[OwnBlob], tuple[EnemyBlob, ...], float, int]:
        own_by_id = {blob.blob_id: blob for blob in own_blobs}
        enemy_by_key = {enemy.key: enemy for enemy in enemies}
        score = 0.0
        captures = 0

        # Match the engine's largest-first, one-consumption-at-a-time loop.
        # Rebuilding the order after every eat is essential: the eater's new
        # radius can immediately unlock another capture in the same round.
        while True:
            actors: list[tuple[str, int | tuple[int, int], OwnBlob | EnemyBlob]] = [
                ("own", blob_id, blob)
                for blob_id, blob in own_by_id.items()
            ]
            actors.extend(
                ("enemy", key, enemy)
                for key, enemy in enemy_by_key.items()
            )
            actors.sort(
                key=lambda item: (
                    -item[2].radius,
                    self._own_player_id
                    if item[0] == "own"
                    else item[1][0],  # type: ignore[index]
                    item[1] if isinstance(item[1], int) else item[1][1],
                )
            )

            changed = False
            for eater_kind, eater_key, _ in actors:
                eater = (
                    own_by_id.get(int(eater_key))
                    if eater_kind == "own"
                    else enemy_by_key.get(eater_key)  # type: ignore[arg-type]
                )
                if eater is None:
                    continue

                for target_kind, target_key, _ in actors:
                    if eater_kind == target_kind == "own":
                        continue
                    if eater_kind == target_kind == "enemy":
                        assert isinstance(eater_key, tuple) and isinstance(target_key, tuple)
                        if eater_key[0] == target_key[0]:
                            continue
                    target = (
                        own_by_id.get(int(target_key))
                        if target_kind == "own"
                        else enemy_by_key.get(target_key)  # type: ignore[arg-type]
                    )
                    if target is None:
                        continue
                    if not can_eat_player_blob(eater.radius, target.radius):
                        continue
                    if squared_distance(eater.pos, target.pos) > eater.radius * eater.radius:
                        continue

                    grown_radius = math.sqrt(eater.mass + target.mass)
                    if eater_kind == "own":
                        assert isinstance(eater_key, int) and isinstance(target_key, tuple)
                        own_by_id[eater_key] = _with_grown_radius(
                            eater,
                            grown_radius,
                            arena_size,
                        )
                        del enemy_by_key[target_key]
                        captured_enemy_ids.add(target_key)
                        score += 52.0 * target.mass + 40.0
                        captures += 1
                    else:
                        assert isinstance(eater_key, tuple)
                        enemy_by_key[eater_key] = _with_grown_radius(
                            eater,
                            grown_radius,
                            arena_size,
                        )
                        if target_kind == "own":
                            assert isinstance(target_key, int)
                            del own_by_id[target_key]
                            score -= 520.0 + target.mass * 90.0
                        else:
                            assert isinstance(target_key, tuple)
                            del enemy_by_key[target_key]
                    changed = True
                    break
                if changed:
                    break
            if not changed:
                break

        survivors = [own_by_id[key] for key in sorted(own_by_id)]
        remaining = tuple(enemy_by_key[key] for key in sorted(enemy_by_key))
        return survivors, remaining, score, captures

    def _risk_score(
        self,
        own_blobs: list[OwnBlob],
        enemies: tuple[EnemyBlob, ...],
        safety_weight: float,
        arena_size: float = ARENA_SIZE,
    ) -> tuple[float, float, bool]:
        penalty = 0.0
        min_margin = math.inf
        endangered_blob_ids: set[int] = set()
        total_mass = sum(own.mass for own in own_blobs)
        risk_enemies = (
            *enemies,
            *self._future_enemy_envelopes(enemies),
        )
        for own in own_blobs:
            player_penalties: dict[int, float] = {}
            for enemy in risk_enemies:
                if not can_eat_player_blob(enemy.radius, own.radius):
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
                    if enemy.blob_id >= 0:
                        endangered_blob_ids.add(own.blob_id)
                    enemy_penalty = (
                        440.0
                        + 75.0 * own.mass
                        + min(180.0, -margin * 35.0)
                    ) * uncertainty
                else:
                    # Continuous contact-time pressure replaces the old hard
                    # margin<9 boundary.  A soon-mergeable enemy envelope can
                    # therefore make retreat valuable six to eight turns before
                    # its physical split attack becomes immediate.
                    scaled_pressure = (12.0 - margin) / 2.0
                    if scaled_pressure > 40.0:
                        pressure = scaled_pressure
                    else:
                        pressure = math.log1p(math.exp(scaled_pressure))
                    enemy_penalty = (
                        pressure * (12.0 + 2.0 * own.mass) * uncertainty
                    )
                    wall_trap = self._wall_trap_factor(own, enemy, arena_size)
                    enemy_penalty += (
                        wall_trap
                        * pressure
                        * (8.0 + 2.0 * own.mass)
                        * WALL_TRAP_RISK_SCALE
                        * uncertainty
                    )
                player_penalties[enemy.player_id] = max(
                    player_penalties.get(enemy.player_id, 0.0),
                    enemy_penalty,
                )
            penalty += (
                own.mass / max(total_mass, EPSILON)
            ) * sum(player_penalties.values())
        unavoidable = bool(own_blobs) and len(endangered_blob_ids) == len(own_blobs)
        return penalty * safety_weight, min_margin, unavoidable

    def _wall_trap_factor(
        self,
        own: OwnBlob,
        enemy: EnemyBlob,
        arena_size: float,
    ) -> float:
        """Estimate how much a nearby wall blocks direct retreat from an enemy."""

        away_x, away_y = normalise((own.x - enemy.x, own.y - enemy.y))
        if away_x == 0.0 and away_y == 0.0:
            return 0.0

        # The replay failures were already unrecoverable several turns before
        # the blob physically touched the wall.  Ten world units approximates
        # the distance needed to turn a direct retreat into a viable tangent,
        # while the outward component keeps unrelated nearby walls irrelevant.
        escape_horizon = 10.0
        walls = (
            (own.x - own.radius, max(0.0, -away_x)),
            (arena_size - own.radius - own.x, max(0.0, away_x)),
            (own.y - own.radius, max(0.0, -away_y)),
            (arena_size - own.radius - own.y, max(0.0, away_y)),
        )
        blocked = sum(
            outward_component
            * max(0.0, 1.0 - max(0.0, clearance) / escape_horizon)
            for clearance, outward_component in walls
        )
        return min(1.0, blocked)

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

        available = [food for food in foods if food.food_id not in eaten_food_ids]
        if available:
            distances = sorted(math.dist(primary.pos, food.pos) for food in available)
            value += sum(2.0 / (distance + 1.0) for distance in distances[:6])

        for enemy in enemies:
            if enemy.stale_rounds:
                continue
            distance = math.dist(primary.pos, enemy.pos)
            if can_eat_player_blob(primary.radius, enemy.radius):
                gap = max(0.0, distance - primary.radius)
                value += aggression * min(9.0, enemy.mass * 3.2 / (gap + 1.0))
            elif not can_eat_player_blob(enemy.radius, primary.radius) and distance < 8.0:
                value -= (8.0 - distance) * 0.3
        return value

    def _escape_vector(self, node: SearchNode) -> tuple[float, float]:
        x = 0.0
        y = 0.0
        risk_enemies = (
            *node.enemies,
            *self._future_enemy_envelopes(node.enemies),
        )
        for own in node.own_blobs:
            for enemy in risk_enemies:
                if not can_eat_player_blob(enemy.radius, own.radius):
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
            if not can_eat_player_blob(child_radius, enemy.radius):
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
            can_eat_player_blob(enemy.radius, primary.radius / SQRT2)
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
            control_cost=node.control_cost,
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
        if not self.endgame_adaptation:
            return 1.0
        if progress >= 0.72 and rank_position <= 2:
            return 1.8
        if progress >= 0.88:
            return 1.35
        return 1.0

    def _aggression(self, rank_position: int, progress: float) -> float:
        if not self.endgame_adaptation:
            return 1.0
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
    return engine_movement_speed(
        radius,
        base_speed=BASE_PLAYER_SPEED,
        radius_factor=PLAYER_SPEED_RADIUS_FACTOR,
        minimum_speed=MIN_PLAYER_SPEED,
    )


def _decayed_radius(radius: float) -> float:
    return engine_decayed_radius(
        radius,
        decay_rate=MASS_DECAY_RATE,
        minimum_radius=STARTING_RADIUS,
    )


def _can_consume_virus(blob_radius: float, virus_radius: float) -> bool:
    return engine_can_consume_virus(
        blob_radius,
        virus_radius,
        eat_size_ratio=EAT_SIZE_RATIO,
    )


def _with_grown_radius(
    blob: OwnBlob | EnemyBlob,
    radius: float,
    arena_size: float,
) -> OwnBlob | EnemyBlob:
    """Grow a simulated blob and apply the engine's same-round arena clamp."""

    return replace(
        blob,
        x=_clamp(blob.x, radius, arena_size - radius),
        y=_clamp(blob.y, radius, arena_size - radius),
        radius=radius,
    )


def _can_split_eat(predator_radius: float, prey_radius: float) -> bool:
    return (
        predator_radius * predator_radius >= SPLIT_MIN_MASS
        and can_eat_player_blob(predator_radius / SQRT2, prey_radius)
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


class ReplayDominanceStrategy(ThreatAwareRecedingHorizonStrategy):
    """One search policy for survival, virus snowballing, and rival capture.

    Replay observations enter as value features and semantic candidates rather
    than phase-specific overrides.  Virus risk remains a continuous expected-
    retained-mass term; unsafe-looking harvests are still search candidates.
    """

    name = "replay_dominance"

    _VIRUS_POTENTIAL_HORIZON = 18.0
    # One virus contributes 2.25 mass, the same as one hundred food pellets.
    # Spread that long-term value over the approach horizon so collecting one
    # incidental pellet cannot repeatedly interrupt a safe virus route.
    _VIRUS_POTENTIAL_SLOPE = 12.0
    _CAPTURE_HORIZON = 8.0
    _CAPTURE_CLOSING_TEMPERATURE = 0.05

    def __init__(
        self,
        depth: int | None = None,
        width: int | None = None,
        angular_samples: int | None = None,
    ) -> None:
        super().__init__(
            depth=depth,
            width=width,
            angular_samples=10 if angular_samples is None else angular_samples,
        )
        # A three-millisecond per-turn cap is at most 4.2 seconds over 1,400
        # rounds.  Use a slightly larger accounting bank so small deadline
        # overruns do not switch the final third of a match to the simplistic
        # fallback while remaining well below the engine's eight-second limit.
        self.compute_budget_seconds = float(
            os.environ.get("BOT_REPLAY_TOTAL_BUDGET_SECONDS", "6.0")
        )
        self.survival_midpoint_base = float(
            os.environ.get(
                "BOT_REPLAY_SURVIVAL_MIDPOINT_BASE",
                os.environ.get("BOT_REPLAY_SAFETY_BUFFER_BASE", "5.0"),
            )
        )
        self.survival_midpoint_scale = float(
            os.environ.get(
                "BOT_REPLAY_SURVIVAL_MIDPOINT_SCALE",
                os.environ.get("BOT_REPLAY_SAFETY_BUFFER_SCALE", "2.0"),
            )
        )
        self.survival_temperature = float(
            os.environ.get("BOT_REPLAY_SURVIVAL_TEMPERATURE", "2.0")
        )
        self.recovery_mass = float(
            os.environ.get("BOT_REPLAY_RECOVERY_MASS", "3.0")
        )
        self._rival_values: dict[int, float] = {}
        self._utility_cache: dict[tuple[object, ...], float] = {}
        self._virus_retention_cache: dict[
            tuple[int, int, int], tuple[SearchNode, float]
        ] = {}
        self._risk_envelope_cache: dict[
            int, tuple[tuple[EnemyBlob, ...], tuple[EnemyBlob, ...]]
        ] = {}
        self.transition_budget_scale = int(
            os.environ.get("BOT_REPLAY_TRANSITION_BUDGET_SCALE", "12")
        )
        self.minimum_transition_budget = int(
            os.environ.get("BOT_REPLAY_MIN_TRANSITIONS", "3")
        )
        # An anytime search must compare at least two complete root
        # transitions.  Otherwise the first semantic candidate (often a virus
        # or prey) becomes a forced action when one exact transition consumes
        # the deadline.  Root utility caching below makes this affordable.
        self.minimum_root_actions = 2

    def _uses_compute_time_bank(self) -> bool:
        # Fixed work keeps candidate ordering reproducible, while the measured
        # bank is still required because the competition worker is materially
        # slower than the local runner.  The search loop enforces both limits.
        return True

    def _transition_budget(
        self,
        own_blob_count: int,
        enemy_count: int = 0,
    ) -> int:
        # Exact transitions become roughly proportional to blob count because
        # every fragment participates in movement, stabilisation, and
        # interactions.  An inverse allocation therefore keeps per-turn work
        # approximately constant without changing the state value function.
        work_units = own_blob_count + enemy_count / 3.0
        return max(
            self.minimum_transition_budget,
            round(self.transition_budget_scale / max(1.0, work_units)),
        )

    def _choose(self, context, *, deadline: float, turn_budget: float):
        # Cache the complete value and its expensive virus/future-enemy
        # subproblems across candidate generation and rollout evaluation.
        self._utility_cache.clear()
        self._virus_retention_cache.clear()
        self._risk_envelope_cache.clear()
        state = context.game.state
        rankings = tuple(int(player_id) for player_id in state.rankings)
        try:
            rank_index = rankings.index(int(state.me.player_id))
        except ValueError:
            rank_index = len(rankings)

        # Scoreboard proximity is a continuous feature, not a phase switch.
        # Eating the player beside us in rank is worth more than farming a
        # distant last-place player, whether we are first, third, or seventh.
        self._rival_values = {
            player_id: 1.0 / (1.0 + abs(other_index - rank_index))
            for other_index, player_id in enumerate(rankings)
            if other_index != rank_index
        }
        return super()._choose(
            context,
            deadline=deadline,
            turn_budget=turn_budget,
        )

    def _candidate_actions(
        self,
        *,
        node,
        foods,
        food_targets,
        viruses,
        arena_size: float,
        first_step: bool,
        allow_split: bool = True,
        angle_offset: int = 0,
    ):
        inherited = list(
            super()._candidate_actions(
                node=node,
                foods=foods,
                food_targets=food_targets,
                viruses=viruses,
                arena_size=arena_size,
                first_step=first_step,
                angle_offset=angle_offset,
            )
        )

        virus_actions = self._virus_actions(
            node=node,
            viruses=viruses,
            arena_size=arena_size,
            limit=3 if first_step else 1,
        )

        # A direct flee vector can point into the arena clamp when a predator
        # approaches from the interior.  Wide tangents stay in the same search
        # (they are not forced actions), but make the viable route around the
        # predator available before an anytime root search reaches its limit.
        escape = self._escape_vector(node)
        wide_escape_actions = []
        if escape != (0.0, 0.0):
            wide_escape_actions = [
                Action(_rotate(escape, math.pi / 2), reason="escape_wide_tangent"),
                Action(_rotate(escape, -math.pi / 2), reason="escape_wide_tangent"),
            ]

        rival_actions = []
        if first_step and self._rival_values:
            rivals = [
                enemy
                for enemy in node.enemies
                if enemy.player_id in self._rival_values
                and enemy.stale_rounds == 0
                and any(
                    can_eat_player_blob(own.radius, enemy.radius)
                    for own in node.own_blobs
                )
            ]
            rivals.sort(
                key=lambda enemy: (
                    -self._prey_expected_mass(node, enemy, arena_size),
                    squared_distance(node.center, enemy.pos),
                    enemy.player_id,
                    enemy.blob_id,
                )
            )
            for enemy in rivals[:2]:
                intercept = self._intercept_direction(node.primary, enemy)
                rival_actions.append(Action(intercept, reason="rival_prey"))
                if self._split_can_capture(node, enemy, intercept):
                    rival_actions.append(
                        Action(intercept, split=True, reason="split_rival_prey")
                    )

        # Preserve emergency escape as the first anytime candidate, then make
        # competitive captures available before ordinary resource gathering.
        # The evaluator remains free to reject every rival action.
        escape_prefix = 0
        while (
            escape_prefix < len(inherited)
            and "escape" in inherited[escape_prefix].reason
        ):
            escape_prefix += 1
        return self._dedupe_actions(
            [
                *inherited[:escape_prefix],
                *wide_escape_actions,
                *rival_actions,
                *virus_actions,
                *inherited[escape_prefix:],
            ]
        )

    def _resolve_interactions(
        self,
        own_blobs,
        enemies,
        captured_enemy_ids,
        arena_size: float = ARENA_SIZE,
    ):
        before = {
            enemy.key: enemy
            for enemy in enemies
            if enemy.player_id in self._rival_values
        }
        updated, remaining, score, captures = super()._resolve_interactions(
            own_blobs,
            enemies,
            captured_enemy_ids,
            arena_size,
        )
        remaining_keys = {enemy.key for enemy in remaining}
        competitive_value = sum(
            self._rival_values[enemy.player_id]
            * (30.0 + enemy.mass * 28.0)
            for key, enemy in before.items()
            if key not in remaining_keys
        )
        return updated, remaining, score + competitive_value, captures

    def _step(
        self,
        *,
        node,
        action,
        foods,
        viruses,
        arena_size: float,
        first_step: bool,
        safety_weight: float,
        aggression: float,
    ):
        before = self._cached_search_utility(
            node,
            foods=foods,
            viruses=viruses,
            arena_size=arena_size,
            safety_weight=safety_weight,
        )
        movement_efficiency = self._movement_efficiency(
            node.own_blobs,
            action.direction,
            arena_size,
        )
        result = super()._step(
            node=node,
            action=action,
            foods=foods,
            viruses=viruses,
            arena_size=arena_size,
            first_step=first_step,
            safety_weight=safety_weight,
            aggression=aggression,
        )
        if not result.node.own_blobs:
            dead_node = replace(result.node, score=node.score - 100_000.0)
            return replace(result, node=dead_node)
        after = self._cached_search_utility(
            result.node,
            foods=foods,
            viruses=viruses,
            arena_size=arena_size,
            safety_weight=safety_weight,
        )
        direction = normalise(action.direction)
        shaped_node = replace(
            result.node,
            score=(
                node.score
                + after
                - before
                - (1.0 - movement_efficiency) * 2.0
                - self._turn_cost(node.last_direction, direction) * 0.6
            ),
        )
        # Non-split danger is priced by retained mass rather than removed as a
        # fatal branch.  Only an immediately losing split keeps the parent's
        # physical admissibility rejection.
        return replace(result, node=shaped_node, fatal=result.fatal and action.split)

    def _cached_search_utility(
        self,
        node: SearchNode,
        *,
        foods,
        viruses,
        arena_size: float,
        safety_weight: float,
    ) -> float:
        key = (
            node.own_blobs,
            node.enemies,
            node.eaten_food_ids,
            node.captured_enemy_ids,
            node.consumed_virus_ids,
            arena_size,
            safety_weight,
        )
        cached = self._utility_cache.get(key)
        if cached is not None:
            return cached
        value = self._search_utility(
            node,
            foods=foods,
            viruses=viruses,
            arena_size=arena_size,
            safety_weight=safety_weight,
        )
        self._utility_cache[key] = value
        return value

    def _search_utility(
        self,
        node,
        *,
        foods,
        viruses,
        arena_size: float,
        safety_weight: float,
    ) -> float:
        """Return one mass-normalised utility shared by every action type."""
        if not node.own_blobs:
            return -1_000_000.0

        risk_enemies = self._risk_enemies(node.enemies)
        survival_midpoint = (
            self.survival_midpoint_base
            + self.survival_midpoint_scale * safety_weight
        )
        safe_mass = 0.0
        survival_probabilities: list[float] = []
        for own in node.own_blobs:
            survival = 1.0
            for enemy in risk_enemies:
                if not can_eat_player_blob(enemy.radius, own.radius):
                    continue
                danger_radius = enemy.radius
                if _can_split_eat(enemy.radius, own.radius):
                    danger_radius = max(
                        danger_radius,
                        _split_attack_reach(enemy.radius),
                    )
                margin = (
                    math.dist(own.pos, enemy.pos)
                    - danger_radius
                    - enemy.stale_rounds * 0.35
                    - self._wall_trap_factor(own, enemy, arena_size) * 4.0
                )
                scaled = _clamp(
                    (margin - survival_midpoint)
                    / max(self.survival_temperature, 0.1),
                    -40.0,
                    40.0,
                )
                survival = min(survival, 1.0 / (1.0 + math.exp(-scaled)))
            survival_probabilities.append(survival)
            safe_mass += own.mass * survival

        # Fragment outcomes are correlated: one predator can sweep several
        # pieces in sequence.  An independent-union probability would make a
        # 16-way split artificially look almost certain to survive.
        continuation_probability = max(survival_probabilities, default=0.0)

        opportunities: list[float] = []
        for food in foods:
            if food.food_id in node.eaten_food_ids:
                continue
            gap = min(
                max(0.0, math.dist(own.pos, food.pos) - own.radius)
                for own in node.own_blobs
            )
            opportunities.append(
                FOOD_RADIUS * FOOD_RADIUS * math.exp(-gap / 6.0)
            )

        for virus in viruses:
            if virus.virus_id in node.consumed_virus_ids:
                continue
            candidates = [
                own
                for own in node.own_blobs
                if self._can_still_consume_virus_at_contact(own, virus)
            ]
            virus_values: list[float] = []
            for origin in candidates:
                gap = max(0.0, math.dist(origin.pos, virus.pos) - origin.radius)
                retention = self._virus_retained_mass_fraction(
                    node,
                    origin,
                    virus,
                    arena_size,
                )
                virus_values.append(
                    virus.radius
                    * virus.radius
                    * retention
                    * math.exp(-gap / self._VIRUS_POTENTIAL_HORIZON)
                )
            if virus_values:
                # One virus is one resource even when several of our blobs can
                # reach it.  Count the best acquisition route, not every origin.
                opportunities.append(max(virus_values))

        for enemy in node.enemies:
            if enemy.stale_rounds:
                continue
            edible = [
                own
                for own in node.own_blobs
                if can_eat_player_blob(own.radius, enemy.radius)
            ]
            if not edible:
                continue
            opportunities.append(
                self._prey_expected_mass(node, enemy, arena_size)
            )

        opportunities.sort(reverse=True)
        opportunity_mass = sum(
            value * weight
            for value, weight in zip(opportunities[:3], (1.0, 0.25, 0.1), strict=False)
        )
        competitor_mass_debt = sum(
            self._rival_values.get(enemy.player_id, 0.0) * enemy.mass
            for enemy in node.enemies
        )
        return 100.0 * (
            safe_mass
            + continuation_probability * opportunity_mass
            - competitor_mass_debt
            - self.recovery_mass * (1.0 - continuation_probability)
        )

    def _terminal_score(self, node: SearchNode) -> float:
        # ``_search_utility`` is applied as a potential difference at every
        # transition, so the path score already contains the frontier value.
        return node.score

    def _prey_candidate_priority(
        self,
        node: SearchNode,
        enemy: EnemyBlob,
        arena_size: float,
    ) -> tuple[float, ...]:
        return (
            -self._prey_expected_mass(node, enemy, arena_size),
            squared_distance(node.center, enemy.pos),
            enemy.player_id,
            enemy.blob_id,
        )

    def _order_root_actions(self, actions: tuple[Action, ...]) -> tuple[Action, ...]:
        """Interleave semantic families before exact root evaluation.

        A fixed work budget must compare different hypotheses, rather than
        spending every transition on small variations of one escape or split
        attack.  Scheduling changes here; every action still uses the same
        exact transition and state value.
        """

        family_order = (
            "escape",
            "virus",
            "baseline",
            "prey",
            "resource",
            "position",
            "explore",
        )
        grouped: dict[str, list[Action]] = {family: [] for family in family_order}
        for action in actions:
            reason = action.reason
            if "escape" in reason:
                family = "escape"
            elif reason in {"keep", "continue"}:
                family = "baseline"
            elif "prey" in reason:
                family = "prey"
            elif "virus" in reason:
                family = "virus"
            elif "food" in reason or "farm" in reason:
                family = "resource"
            elif reason in {"center", "wall"} or "wall" in reason:
                family = "position"
            else:
                family = "explore"
            grouped[family].append(action)

        ordered: list[Action] = []
        max_family_size = max((len(group) for group in grouped.values()), default=0)
        for index in range(max_family_size):
            for family in family_order:
                group = grouped[family]
                if index < len(group):
                    ordered.append(group[index])
        return tuple(ordered)

    def _order_deeper_actions(self, actions: tuple[Action, ...]) -> tuple[Action, ...]:
        return self._order_root_actions(actions)

    def _actions_per_node_limit(self, depth_index: int) -> int:
        return 6 if depth_index == 0 else 1

    def _prey_expected_mass(
        self,
        node: SearchNode,
        enemy: EnemyBlob,
        arena_size: float,
    ) -> float:
        capture_probability = max(
            (
                self._prey_capture_probability(own, enemy, arena_size)
                for own in node.own_blobs
                if can_eat_player_blob(own.radius, enemy.radius)
            ),
            default=0.0,
        )
        rival_value = self._rival_values.get(enemy.player_id, 0.0)
        return capture_probability * enemy.mass * (1.0 + rival_value)

    def _prey_capture_probability(
        self,
        own: OwnBlob,
        enemy: EnemyBlob,
        arena_size: float,
    ) -> float:
        """Estimate capture reach from the actual clamped closing speed."""
        delta_x = enemy.x - own.x
        delta_y = enemy.y - own.y
        distance = math.hypot(delta_x, delta_y)
        gap = max(0.0, distance - own.radius)
        if gap <= EPSILON:
            return 1.0
        direction = (delta_x / distance, delta_y / distance)

        moved_own = self._move_own(own, direction, arena_size)
        enemy_direction = normalise(enemy.direction)
        flee_direction = normalise(
            (
                direction[0] * 0.62 + enemy_direction[0] * 0.38,
                direction[1] * 0.62 + enemy_direction[1] * 0.38,
            )
        )
        enemy_speed = _speed(enemy.radius)
        enemy_x = _clamp(
            enemy.x + flee_direction[0] * enemy_speed,
            enemy.radius,
            arena_size - enemy.radius,
        )
        enemy_y = _clamp(
            enemy.y + flee_direction[1] * enemy_speed,
            enemy.radius,
            arena_size - enemy.radius,
        )
        next_gap = max(
            0.0,
            math.hypot(enemy_x - moved_own.x, enemy_y - moved_own.y)
            - own.radius,
        )
        closing = gap - next_gap
        temperature = self._CAPTURE_CLOSING_TEMPERATURE
        scaled = _clamp(closing / temperature, -40.0, 40.0)
        effective_closing = temperature * math.log1p(math.exp(scaled))
        return math.exp(
            -gap
            / max(self._CAPTURE_HORIZON * effective_closing, EPSILON)
        )

    def _movement_efficiency(
        self,
        own_blobs,
        direction: tuple[float, float],
        arena_size: float,
    ) -> float:
        """Return mass-weighted useful speed after arena clamping.

        This is a property of every move, rather than a wall mode or a list of
        forbidden targets.  A diagonal escape that loses half of its velocity
        into a wall therefore receives the same treatment as a blocked food or
        virus approach.
        """
        unit = normalise(direction)
        expected = 0.0
        useful = 0.0
        for blob in own_blobs:
            speed = _speed(blob.radius)
            moved = self._move_own(blob, unit, arena_size)
            expected += blob.mass * speed
            useful += blob.mass * max(
                0.0,
                (moved.x - blob.x) * unit[0]
                + (moved.y - blob.y) * unit[1],
            )
        if expected <= EPSILON:
            return 1.0
        return _clamp(useful / expected, 0.0, 1.0)

    def _resolve_own_viruses(
        self,
        *,
        own_blobs,
        enemies=(),
        viruses,
        consumed_virus_ids,
        arena_size: float,
    ):
        """Mirror the engine's touching-blob virus split exactly."""
        score = 0.0
        penalty = 0.0
        for virus in viruses:
            if virus.virus_id in consumed_virus_ids:
                continue
            before_collision = own_blobs
            collision = self._apply_virus_collision(
                own_blobs=own_blobs,
                enemies=enemies,
                virus=virus,
                arena_size=arena_size,
            )
            if collision is None:
                continue
            consumed_virus_ids.add(virus.virus_id)
            own_blobs, enemies, origin, piece_count = collision
            if origin is None:
                continue
            total_mass = origin.mass + virus.radius * virus.radius
            retained_mass_fraction = self._post_virus_retained_mass_fraction(
                own_blobs=before_collision,
                enemies=enemies,
                origin=origin,
                virus=virus,
                arena_size=arena_size,
            )

            terminal_value = (
                self._VIRUS_POTENTIAL_HORIZON * self._VIRUS_POTENTIAL_SLOPE
            )
            score += retained_mass_fraction * (
                terminal_value + 70.0 + virus.radius * virus.radius * 13.0
            )
            largest_share_drop = max(
                0.0,
                origin.mass / max(total_mass, EPSILON) - 1.0 / piece_count,
            )
            penalty += largest_share_drop * 32.0
            penalty += (1.0 - retained_mass_fraction) * (
                100.0 + origin.mass * 8.0
            )
        return own_blobs, enemies, score, penalty

    def _virus_retained_mass_fraction(
        self,
        node,
        origin,
        virus,
        arena_size: float,
    ) -> float:
        cache_key = (id(node), origin.blob_id, virus.virus_id)
        cached = self._virus_retention_cache.get(cache_key)
        if cached is not None and cached[0] is node:
            return cached[1]
        matching = next(
            (blob for blob in node.own_blobs if blob.blob_id == origin.blob_id),
            None,
        )
        if matching is None:
            return 0.0
        retained = self._post_virus_retained_mass_fraction(
            # Search nodes are authoritative root states or the result of the
            # exact final stabilisation in ``_step``.  Re-stabilising here for
            # every virus candidate was redundant and consumed enough of the
            # three-millisecond turn budget to truncate threat search.
            own_blobs=node.own_blobs,
            enemies=node.enemies,
            origin=matching,
            virus=virus,
            arena_size=arena_size,
        )
        self._virus_retention_cache[cache_key] = (node, retained)
        return retained

    def _risk_enemies(
        self,
        enemies: tuple[EnemyBlob, ...],
    ) -> tuple[EnemyBlob, ...]:
        """Return current and future envelopes once per rollout state."""

        cached = self._risk_envelope_cache.get(id(enemies))
        if cached is not None and cached[0] is enemies:
            return cached[1]
        risk_enemies = (*enemies, *self._future_enemy_envelopes(enemies))
        self._risk_envelope_cache[id(enemies)] = (enemies, risk_enemies)
        return risk_enemies

    def _post_virus_retained_mass_fraction(
        self,
        *,
        own_blobs,
        enemies,
        origin,
        virus,
        arena_size: float,
    ) -> float:
        """Estimate the mass that remains controllable after an exact pop.

        This is deliberately a continuous value rather than a safe/unsafe
        virus rule.  Each generated fragment is weighted by predator reach and
        by whether the wall blocks its direct retreat.  The common beam
        evaluator can therefore trade a partially exposed pop against food,
        prey, or escape without switching modes.
        """
        piece_count = max(1, MAX_BLOB_COUNT - len(own_blobs) + 1)
        total_mass = origin.mass + virus.radius * virus.radius
        piece_radius = math.sqrt(total_mass / piece_count)
        post_radii = [piece_radius]
        post_radii.extend(
            blob.radius
            for blob in own_blobs
            if blob.blob_id != origin.blob_id
        )
        enemy_tuple = enemies if isinstance(enemies, tuple) else tuple(enemies)
        risk_enemies = self._risk_enemies(enemy_tuple)
        if not any(
            can_eat_player_blob(enemy.radius, radius)
            for enemy in risk_enemies
            for radius in post_radii
        ):
            return 1.0
        fragments = self._virus_replacement_fragments(
            origin=origin,
            piece_radius=piece_radius,
            piece_count=piece_count,
            arena_size=arena_size,
            occupied_ids={blob.blob_id for blob in own_blobs},
        )
        post_blobs = [
            blob for blob in own_blobs if blob.blob_id != origin.blob_id
        ] + fragments
        if not post_blobs:
            return 0.0

        retained_mass = 0.0
        for blob in post_blobs:
            retention = 1.0
            for enemy in risk_enemies:
                if not can_eat_player_blob(enemy.radius, blob.radius):
                    continue
                danger_radius = enemy.radius
                if _can_split_eat(enemy.radius, blob.radius):
                    danger_radius = max(
                        danger_radius,
                        _split_attack_reach(enemy.radius),
                    )
                margin = (
                    math.dist(blob.pos, enemy.pos)
                    - danger_radius
                    - enemy.stale_rounds * 0.35
                )
                pressure = _clamp((8.0 - margin) / 8.0, 0.0, 1.0)
                wall_trap = self._wall_trap_factor(blob, enemy, arena_size)
                predator_retention = 1.0 - pressure * (
                    0.55 + 0.45 * wall_trap
                )
                if margin <= 0.0:
                    predator_retention = 0.0
                retention = min(retention, predator_retention)
            retained_mass += blob.mass * retention
        return _clamp(
            retained_mass / max(sum(blob.mass for blob in post_blobs), EPSILON),
            0.0,
            1.0,
        )

    def _virus_potential(self, node, viruses, arena_size: float) -> float:
        approach_values = []
        for virus in viruses:
            if virus.virus_id in node.consumed_virus_ids:
                continue
            candidates = [
                blob
                for blob in node.own_blobs
                if self._can_still_consume_virus_at_contact(blob, virus)
            ]
            for origin in candidates:
                retained_mass_fraction = self._virus_retained_mass_fraction(
                    node, origin, virus, arena_size
                )
                center_gap = max(
                    0.0,
                    math.dist(origin.pos, virus.pos) - origin.radius,
                )
                approach_values.append(
                    max(0.0, self._VIRUS_POTENTIAL_HORIZON - center_gap)
                    * self._VIRUS_POTENTIAL_SLOPE
                    * retained_mass_fraction
                )
        return max(approach_values, default=0.0)

    def _virus_actions(
        self,
        *,
        node,
        viruses,
        arena_size: float,
        limit: int,
    ):
        scored = []
        for virus in viruses:
            if virus.virus_id in node.consumed_virus_ids:
                continue
            candidates = [
                blob
                for blob in node.own_blobs
                if self._can_still_consume_virus_at_contact(blob, virus)
            ]
            if not candidates:
                continue
            origin = min(
                candidates,
                key=lambda blob: squared_distance(blob.pos, virus.pos),
            )
            retained_mass_fraction = self._virus_retained_mass_fraction(
                node, origin, virus, arena_size
            )
            gap = max(0.0, math.dist(origin.pos, virus.pos) - origin.radius)
            direction = normalise(
                (virus.pos[0] - origin.x, virus.pos[1] - origin.y)
            )
            scored.append(
                (
                    -retained_mass_fraction,
                    gap,
                    virus.virus_id,
                    Action(direction, reason="virus_harvest"),
                )
            )
        scored.sort(key=lambda item: (item[0], item[1], item[2]))
        return [action for _, _, _, action in scored[:limit]]

    def _can_still_consume_virus_at_contact(self, blob, virus) -> bool:
        if not _can_consume_virus(blob.radius, virus.radius):
            return False
        center_gap = max(0.0, math.dist(blob.pos, virus.pos) - blob.radius)
        turns_to_contact = math.ceil(center_gap / _speed(blob.radius))
        projected_mass = decayed_mass_after_turns(
            blob.mass,
            turns_to_contact,
            decay_rate=MASS_DECAY_RATE,
            minimum_radius=STARTING_RADIUS,
        )
        return engine_can_consume_virus(
            math.sqrt(projected_mass),
            virus.radius,
            eat_size_ratio=EAT_SIZE_RATIO,
        )

    def _safety_weight(self, rank_position: int, progress: float) -> float:
        rank_strength = max(0.0, min(1.0, (4.0 - rank_position) / 3.0))
        # Replay deaths show that a first elimination usually starts a respawn
        # loop even from sixth or seventh place.  Survival therefore has a
        # meaningful baseline value at every rank; the continuous lead/progress
        # term still makes preserving a winning mass advantage more valuable.
        return 1.3 + rank_strength * (0.35 + 0.85 * progress)

    def _aggression(self, rank_position: int, progress: float) -> float:
        rank_strength = max(0.0, min(1.0, (4.0 - rank_position) / 3.0))
        return 1.0 + 0.4 * progress * (1.0 - rank_strength)

    def _position_value(
        self,
        own_blobs,
        enemies,
        foods,
        eaten_food_ids,
        arena_size: float,
        aggression: float,
    ) -> float:
        value = super()._position_value(
            own_blobs,
            enemies,
            foods,
            eaten_food_ids,
            arena_size,
            aggression,
        )
        if not own_blobs or not enemies:
            return value

        total_mass = sum(blob.mass for blob in own_blobs)
        strongest_trap = 0.0
        for own in own_blobs:
            wall_clearance = min(
                own.x - own.radius,
                own.y - own.radius,
                arena_size - own.radius - own.x,
                arena_size - own.radius - own.y,
            )
            wall_exposure = max(0.0, 14.0 - wall_clearance)
            if wall_exposure <= 0.0:
                continue
            mass_share = own.mass / max(total_mass, EPSILON)
            for enemy in enemies:
                if not (
                    can_eat_player_blob(enemy.radius, own.radius)
                    or _can_split_eat(enemy.radius, own.radius)
                ):
                    continue
                distance = math.dist(own.pos, enemy.pos)
                proximity = max(0.0, 30.0 - distance) / 30.0
                retreat_blocked = self._wall_trap_factor(
                    own,
                    enemy,
                    arena_size,
                )
                strongest_trap = max(
                    strongest_trap,
                    wall_exposure
                    * wall_exposure
                    * proximity
                    * (0.25 + 0.75 * retreat_blocked)
                    * (1.0 + mass_share)
                    * 2.5,
                )
        return value - strongest_trap

    def _direct_virus_decision(
        self,
        *,
        own_blobs,
        enemies,
        viruses,
        arena_size: float,
        rank_position: int,
        progress: float,
    ):
        # Virus collection is represented by the same beam candidates and
        # evaluator as food, prey, escape, and center movement.  Avoid a
        # separate mode that bypasses the search near a virus.
        return None
