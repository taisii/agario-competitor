from __future__ import annotations

"""Robust receding-horizon strategy used by the submission entry point.

The strategy deliberately separates prediction from evaluation:

* opponent movement is represented as a current anonymous set because the
  competition payload reallocates public blob IDs on every query;
* predator futures use short isotropic reachability rather than pretending an
  anonymous blob has a reliable long-lived velocity track;
* own movement, splitting, eject velocity, decay, food, and eating follow the
  engine's update order closely;
* semantic actions are ordered by safety and usefulness, allowing the search
  to return its best completed root candidate when the time budget expires.

Only standard-library code and public agario-kit models are used at runtime.
"""

import math
import json
import os
import sys
from dataclasses import dataclass, field, replace
from functools import cached_property
from time import perf_counter

from lib.config.arena import ARENA_SIZE, MAX_BLOB_COUNT, VIRUS_SIZE
from lib.config.player import (
    EAT_SIZE_RATIO,
    FOOD_RADIUS,
    MASS_DECAY_RATE,
    MERGE_ATTRACTION_SPEED,
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
    can_consume_virus as _replay_can_consume_virus,
    circle_intersects_square,
    decayed_mass_after_turns,
    decayed_radius as _replay_decayed_radius,
    select_largest_first,
    virus_replacement_positions,
)
from strategies.base import StrategyContext, StrategyDecision
from strategies.features import (
    can_eat_player_blob,
    normalise,
    player_speed,
    squared_distance,
)
from strategies.world_transition import CompleteJointCommand, JointPhysicalTransition


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
    eject_vx: float = 0.0
    eject_vy: float = 0.0

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
    consumed_virus_ids: frozenset[int] = field(default_factory=frozenset)
    projected_food: int = 0
    projected_captures: int = 0
    projected_viruses: int = 0
    min_safety_margin: float = math.inf

    @cached_property
    def total_mass(self) -> float:
        return sum(blob.mass for blob in self.own_blobs)

    @cached_property
    def primary(self) -> OwnBlob:
        return max(self.own_blobs, key=lambda blob: blob.radius)

    @cached_property
    def center(self) -> tuple[float, float]:
        return _mass_center(self.own_blobs)


@dataclass(frozen=True)
class StepResult:
    node: SearchNode
    fatal: bool = False
    movement_efficiency: float = 1.0
    hazard_summary: HazardSummary | None = None


@dataclass(frozen=True)
class OwnMovement:
    """One authoritative own-blob movement pass and its clamp metrics."""

    blobs: tuple[OwnBlob, ...]
    blocked_distance: float
    efficiency: float


@dataclass(frozen=True)
class PhysicalStepCoreResult:
    """Policy-free event result plus metrics needed by legacy finalisation."""

    node: SearchNode
    movement_efficiency: float
    blocked_distance: float
    split_lost_fragment: bool
    split_penalty: float
    virus_penalty: float
    food_reward: float
    interaction_score: float


@dataclass(slots=True)
class ProxyMovement:
    """Short-horizon kinematics shared by every approximate value term."""

    efficiency: float
    blobs: list[ProxyBlobMotion] = field(default_factory=list)


@dataclass(slots=True)
class ProxyBlobMotion:
    """One cheap linearised own-blob trajectory."""

    blob_id: int
    source_blob_id: int
    start_x: float
    start_y: float
    x: float
    y: float
    radius: float
    speed: float
    delta_x: float = 0.0
    delta_y: float = 0.0
    length_sq: float = 0.0
    short_path: tuple[tuple[float, float], ...] = ()

    @property
    def mass(self) -> float:
        return self.radius * self.radius


@dataclass(slots=True)
class ProxyEnemyMotion:
    """One cheap adversarial enemy trajectory."""

    enemy: EnemyBlob
    x: float
    y: float
    direction: tuple[float, float]
    speed: float


@dataclass(frozen=True, slots=True)
class ProxyOwnSource:
    blob: OwnBlob
    speed: float
    split_radius: float | None
    split_speed: float | None


@dataclass(frozen=True, slots=True)
class ProxyMotionTemplate:
    """Action-independent coefficients for one projected blob trajectory."""

    blob_id: int
    source_blob_id: int
    base_start_x: float
    base_start_y: float
    directional_start: float
    static_eject_x: float
    static_eject_y: float
    directional_travel: float
    radius: float
    speed: float
    initial_eject_x: float
    initial_eject_y: float
    directional_eject_speed: float


@dataclass(frozen=True, slots=True)
class ProxyValue:
    """Components of the same state value used by exact beam evaluation."""

    safe_mass: float
    continuation_probability: float
    opportunity_mass: float
    cohesion_mass: float
    competitor_mass_debt: float
    recovery_debt: float

    @property
    def total(self) -> float:
        return 100.0 * (
            self.safe_mass
            + self.continuation_probability * self.opportunity_mass
            + self.cohesion_mass
            - self.competitor_mass_debt
            - self.recovery_debt
        )


@dataclass(frozen=True, slots=True)
class ProxyThreat:
    source_blob_id: int
    enemy: EnemyBlob
    source_radius: float
    normal_ordinary_radius: float | None
    normal_split_attack_radius: float | None
    split_ordinary_radius: float | None
    split_split_attack_radius: float | None
    away_x: float
    away_y: float
    initial_margin: float
    motion_index: int = -1


@dataclass(frozen=True, slots=True)
class ProxyFoodTarget:
    source_blob_id: int
    food: FoodModel
    direction: tuple[float, float]
    gap: float
    normal_motion_range: tuple[int, int] = (-1, -1)
    split_motion_range: tuple[int, int] = (-1, -1)


@dataclass(frozen=True, slots=True)
class ProxyVirusTarget:
    source_blob_id: int
    virus: VirusModel
    direction: tuple[float, float]
    gap: float
    normal_motion_range: tuple[int, int] = (-1, -1)
    split_motion_range: tuple[int, int] = (-1, -1)
    threat_motion_indices: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ProxyPreyTarget:
    source_blob_id: int
    enemy: EnemyBlob
    direction: tuple[float, float]
    expected_mass: float
    motion_index: int = -1
    competitor_debt: float = 0.0
    normal_motion_range: tuple[int, int] = (-1, -1)
    split_motion_range: tuple[int, int] = (-1, -1)


@dataclass(frozen=True, slots=True)
class ProxyKinematics:
    normal_speed: float
    normal_eject_x: float
    normal_eject_y: float
    normal_radius: float
    split_speed: float
    split_static_eject_x: float
    split_static_eject_y: float
    split_directional_eject: float
    split_placement: float
    split_radius: float


@dataclass(frozen=True, slots=True)
class ProxyCoarseOpportunity:
    """First-order value of one independent growth opportunity."""

    value: float
    gradient_x: float
    gradient_y: float


@dataclass(frozen=True, slots=True)
class ProxyAnalysis:
    """Action-independent inputs for cheap differential rollouts."""

    node: SearchNode
    foods: tuple[ProxyFoodTarget, ...]
    viruses: tuple[ProxyVirusTarget, ...]
    prey: tuple[ProxyPreyTarget, ...]
    motion_enemies: tuple[EnemyBlob, ...]
    normal_threats_by_blob: tuple[tuple[ProxyThreat, ...], ...]
    split_threats_by_blob: tuple[tuple[ProxyThreat, ...], ...]
    own_sources: tuple[ProxyOwnSource, ...]
    own_source_by_id: dict[int, ProxyOwnSource]
    normal_motion_templates: tuple[ProxyMotionTemplate, ...]
    split_motion_templates: tuple[ProxyMotionTemplate, ...]
    normal_motion_ranges_by_source: dict[int, tuple[int, int]]
    split_motion_ranges_by_source: dict[int, tuple[int, int]]
    enemy_speeds: tuple[float, ...]
    enemy_speed_by_key: dict[tuple[int, int], float]
    observed_enemy_directions: tuple[tuple[float, float], ...]
    observed_enemy_weights: tuple[float, ...]
    normal_hunter_masks: tuple[int, ...]
    split_hunter_masks: tuple[int, ...]
    normal_predator_masks: tuple[int, ...]
    split_predator_masks: tuple[int, ...]
    competitor_mass_debt: float
    escape_vector: tuple[float, float]
    safety_gradient: tuple[float, float]
    coarse_opportunities: tuple[ProxyCoarseOpportunity, ...]
    coarse_opportunity_baseline: float
    kinematics: ProxyKinematics
    split_state_delta: float
    baseline: ProxyValue
    discount_sum: float
    future_weight: float


@dataclass(frozen=True)
class NodeGeometry:
    """Action-independent mass geometry for one search node."""

    total_mass: float
    center: tuple[float, float]
    primary: OwnBlob


@dataclass(frozen=True)
class HazardSummary:
    """One own/enemy pair scan used by admissibility and state value."""

    min_margin: float
    unavoidable: bool
    safe_mass: float
    continuation_probability: float


@dataclass(frozen=True)
class RootAuditRequest:
    node: SearchNode
    actions: tuple[Action, ...]
    foods: tuple[FoodModel, ...]
    viruses: tuple[VirusModel, ...]
    arena_size: float
    safety_weight: float
    aggression: float
    transition_budget: int | None


@dataclass(frozen=True)
class PlanningTurn:
    """Action-independent state prepared once for either search policy."""

    node: SearchNode
    foods: tuple[FoodModel, ...]
    food_targets: tuple[tuple[float, float], ...]
    viruses: tuple[VirusModel, ...]
    arena_size: float
    round_number: int


class ThreatAwareRecedingHorizonStrategy:
    """Threat-first beam search with exact public-rule tactical simulation."""

    name = "threat_aware_receding_horizon"
    _DIAGNOSTIC_ACTION_LIMIT = 5

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
        self.depth = (
            depth
            if depth is not None
            else int(
                _receding_horizon_setting(
                    "BOT_RECEDING_HORIZON_DEPTH", "BOT_CHAMPION_DEPTH", "3"
                )
            )
        )
        self.width = (
            width
            if width is not None
            else int(
                _receding_horizon_setting(
                    "BOT_RECEDING_HORIZON_WIDTH", "BOT_CHAMPION_WIDTH", "4"
                )
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
        self._next_enemy_track_id = 0
        # Stabilisation is a pure function of one immutable enemy layout and
        # the arena.  Expected-response evaluation deliberately reuses the
        # same enemy transition across several own continuations, so retain
        # its exact result for the duration of one observed turn.
        self._enemy_stabilisation_cache: dict[
            tuple[tuple[EnemyBlob, ...], float], tuple[EnemyBlob, ...]
        ] = {}
        self._enemy_group_stabilisation_cache: dict[
            tuple[tuple[EnemyBlob, ...], float], tuple[EnemyBlob, ...]
        ] = {}

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
                decision = self._time_budget_fallback(context)
            else:
                decision = self._choose(
                    context,
                    deadline=started_at + turn_budget,
                    turn_budget=turn_budget,
                )
        finally:
            self.compute_spent_seconds += perf_counter() - started_at
        return replace(
            decision,
            diagnostics={
                **decision.diagnostics,
                "compute_spent_ms": round(
                    self.compute_spent_seconds * 1000.0,
                    3,
                ),
            },
        )

    def _choose(
        self,
        context: StrategyContext,
        *,
        deadline: float,
        turn_budget: float,
    ) -> StrategyDecision:
        state = context.game.state
        turn = self._prepare_turn(context)
        if turn is None:
            return StrategyDecision(
                direction=self.previous_direction,
                reason="dead_fallback",
            )
        start = turn.node
        foods = turn.foods
        viruses = turn.viruses
        food_targets = turn.food_targets
        arena_size = turn.arena_size
        round_number = turn.round_number
        own_blobs = start.own_blobs
        enemies = start.enemies
        rank_position = self._rank_position(state.rankings, state.me.player_id)
        progress = round_number / max(1, int(state.max_rounds))
        safety_weight = self._safety_weight(rank_position, progress)
        aggression = (
            self._aggression(rank_position, progress)
            if self._uses_base_transition_score()
            else 1.0
        )

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
                    audit_seconds = self._audit_root_candidate_ranking(
                        node=node,
                        actions=actions,
                        foods=foods,
                        viruses=viruses,
                        arena_size=arena_size,
                        safety_weight=safety_weight,
                        aggression=aggression,
                        transition_budget=transition_budget,
                    )
                    deadline += audit_seconds
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
                    required_actions = self._required_actions_for_depth(
                        depth_index,
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
                        if (
                            best_rejected is None
                            or result.node.score > best_rejected.score
                        ):
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
        decision_min_safety_margin = self._decision_min_safety_margin(
            best,
        )
        return StrategyDecision(
            direction=direction,
            split=best.first_split,
            target_kind="escape"
            if "escape" in reason
            else ("prey" if "prey" in reason else "beam"),
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
                "min_safety_margin": decision_min_safety_margin,
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

    def _prepare_turn(self, context: StrategyContext) -> PlanningTurn | None:
        """Convert one observation into immutable planning inputs exactly once.

        Both exact search and proxy-only fallback consume this boundary.  It
        keeps enemy-memory updates, visibility limits, and candidate inputs
        consistent while avoiding repeated exposure analysis during sorting.
        """

        self._enemy_stabilisation_cache.clear()
        self._enemy_group_stabilisation_cache.clear()
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
            return None

        arena_size = float(state.map.size or ARENA_SIZE)
        viruses = tuple(state.visible_viruses)
        planning_enemies = self._planning_enemies(
            context,
            own_blobs,
            arena_size,
            viruses,
        )
        center = _mass_center(own_blobs)
        foods = tuple(
            sorted(
                state.visible_food,
                key=lambda food: squared_distance(center, food.pos),
            )[: self.max_food]
        )
        exposed_own_radii = self._exposed_own_radii(own_blobs)
        enemies = tuple(
            sorted(
                planning_enemies,
                key=lambda enemy: self._enemy_priority(
                    enemy,
                    own_blobs,
                    center,
                    exposed_own_radii,
                ),
            )[: self.max_enemies]
        )
        node = SearchNode(
            own_blobs=own_blobs,
            enemies=enemies,
            score=0.0,
            first_direction=self.previous_direction,
            first_split=False,
            first_reason="keep",
            last_direction=self.previous_direction,
        )
        return PlanningTurn(
            node=node,
            foods=foods,
            # Clustering is quadratic, but depends only on this observation.
            food_targets=tuple(self._food_targets(center, foods)),
            viruses=viruses,
            arena_size=arena_size,
            round_number=int(state.round),
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

        if transition_budget is not None and transitions_evaluated >= transition_budget:
            return "transition_budget"
        # Root candidates are needed to produce a decision even if setup used
        # the nominal turn budget. Deeper candidates are optional and can be
        # skipped before their comparatively expensive generation begins.
        if depth_index > 0 and uses_time_bank and perf_counter() >= deadline:
            return "deadline"
        return None

    def _turn_budget_seconds(self, *, round_number: int, max_rounds: int) -> float:
        remaining_budget = max(
            0.0, self.compute_budget_seconds - self.compute_spent_seconds
        )
        remaining_rounds = max(1, max_rounds - round_number)
        return min(self.max_turn_seconds, remaining_budget / remaining_rounds)

    def _uses_compute_time_bank(self) -> bool:
        """Whether wall time, rather than fixed work, limits this search."""

        return True

    def _approximate_value_fallback(
        self,
        context: StrategyContext,
    ) -> StrategyDecision:
        """Extension hook for policies that provide the shared value proxy."""

        turn = self._prepare_turn(context)
        if turn is None:
            return StrategyDecision(
                direction=self.previous_direction,
                reason="time_bank_dead",
            )
        node = turn.node
        own_blobs = node.own_blobs
        enemies = node.enemies
        foods = turn.foods
        viruses = turn.viruses
        actions = self._candidate_actions(
            node=node,
            foods=foods,
            food_targets=turn.food_targets,
            viruses=viruses,
            arena_size=turn.arena_size,
            first_step=True,
            angle_offset=turn.round_number,
        )
        prey = [
            enemy
            for enemy in enemies
            if enemy.stale_rounds == 0
            and any(can_eat_player_blob(own.radius, enemy.radius) for own in own_blobs)
        ]
        scored = self._root_proxy_scores
        if not scored:
            return StrategyDecision(
                direction=self.previous_direction,
                reason="proxy_no_candidate",
            )
        best, best_score = scored[0]
        keep_key = self._action_key(Action(self.previous_direction))
        keep_score = next(
            (score for action, score in scored if self._action_key(action) == keep_key),
            best_score,
        )
        # The proxy already prices turning.  Executing an interpolated heading
        # would be a different, unevaluated action and previously kept the bot
        # moving into threats or walls despite selecting a safe candidate.
        direction = normalise(best.direction)
        self.previous_direction = direction
        return StrategyDecision(
            direction=direction,
            split=best.split,
            target_kind=(
                "escape"
                if "escape" in best.reason
                else "prey"
                if "prey" in best.reason
                else "food"
                if "food" in best.reason
                else "beam"
            ),
            reason=best.reason,
            score=best_score,
            diagnostics={
                "approximate_fallback": True,
                "proxy_only": True,
                "transitions_evaluated": 0,
                "transitions_by_depth": [],
                "fallback_candidates": len(actions),
                "fallback_prey": len(prey),
                "fallback_improvement": best_score - keep_score,
                "candidate_family_counts": dict(self._root_candidate_families),
                "proxy_candidates_refined": self._root_proxy_refined,
                "static_food_sweep_added": getattr(
                    self, "_static_food_sweep_added", False
                ),
                "static_food_sweep_predicted_advantage": getattr(
                    self, "_static_food_sweep_predicted_advantage", 0
                ),
                "proxy_rank_audit_samples": getattr(
                    self, "_proxy_rank_audit_samples", 0
                ),
                "proxy_rank_audit_recall_at_k": (
                    self._proxy_rank_audit_hits / self._proxy_rank_audit_samples
                    if getattr(self, "_proxy_rank_audit_samples", 0)
                    else None
                ),
                "proxy_rank_audit_best_coarse_rank": (
                    getattr(
                        self,
                        "_proxy_rank_audit_last_best_coarse_rank",
                        None,
                    )
                ),
                "proxy_rank_audit_best_reason": getattr(
                    self, "_proxy_rank_audit_last_best_reason", None
                ),
                "proxy_rank_audit_best_split": getattr(
                    self, "_proxy_rank_audit_last_best_split", None
                ),
                "proxy_rank_audit_regret": getattr(
                    self, "_proxy_rank_audit_last_regret", None
                ),
                "proxy_top_actions": [
                    {
                        "reason": action.reason,
                        "split": action.split,
                        "score": round(score, 6),
                    }
                    for action, score in scored[
                        : min(
                            self._root_proxy_refined,
                            self._DIAGNOSTIC_ACTION_LIMIT,
                        )
                    ]
                ],
                "proxy_top_two_gap": (
                    scored[0][1] - scored[1][1] if len(scored) >= 2 else None
                ),
                "compute_spent_ms": round(self.compute_spent_seconds * 1000.0, 3),
                "compute_budget_ms": round(self.compute_budget_seconds * 1000.0, 3),
                "compute_remaining_ms": round(
                    max(
                        0.0,
                        self.compute_budget_seconds - self.compute_spent_seconds,
                    )
                    * 1000.0,
                    3,
                ),
                "competition_compute_budget_ms": round(
                    getattr(self, "competition_compute_budget_seconds", 8.0) * 1000.0,
                    3,
                ),
                "competition_compute_remaining_ms": round(
                    max(
                        0.0,
                        getattr(self, "competition_compute_budget_seconds", 8.0)
                        - self.compute_spent_seconds,
                    )
                    * 1000.0,
                    3,
                ),
            },
        )

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
            return StrategyDecision(
                direction=self.previous_direction, reason="time_bank_dead"
            )

        away_x = 0.0
        away_y = 0.0
        for enemy in state.visible_blobs:
            if not any(
                can_eat_player_blob(enemy.radius, own.radius) for own in own_blobs
            ):
                continue
            nearest = min(
                own_blobs, key=lambda own: squared_distance(own.pos, enemy.pos)
            )
            distance = max(math.dist(nearest.pos, enemy.pos), 0.25)
            away_x += (
                (nearest.pos[0] - enemy.pos[0]) * enemy.radius / (distance * distance)
            )
            away_y += (
                (nearest.pos[1] - enemy.pos[1]) * enemy.radius / (distance * distance)
            )

        direction = normalise((away_x, away_y))
        reason = "time_bank_escape"
        if direction == (0.0, 0.0) and state.visible_food:
            center = (state.me.x, state.me.y)
            food = min(
                state.visible_food, key=lambda item: squared_distance(center, item.pos)
            )
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
        exposed_own_radii: tuple[tuple[OwnBlob, float], ...] | None = None,
    ) -> tuple[int, float, float]:
        """Keep threats to current or post-split fragments before harmless blobs."""

        threat_margins: list[float] = []
        exposed = exposed_own_radii or self._exposed_own_radii(own_blobs)
        for own, candidate_radius in exposed:
            if not can_eat_player_blob(enemy.radius, candidate_radius):
                continue
            danger_radius = enemy.radius
            if _can_split_eat(enemy.radius, candidate_radius):
                danger_radius = max(
                    danger_radius,
                    _split_chain_attack_reach(enemy.radius, candidate_radius),
                )
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
                <= 18.0 * player_speed(own.radius)
                for virus in viruses
            )
            if virus_reachable:
                virus_piece_radius = math.sqrt(
                    (own.mass + VIRUS_SIZE * VIRUS_SIZE) / virus_piece_count
                )
                exposed.append((own, virus_piece_radius))
        return tuple(exposed)

    @staticmethod
    def _visible_enemies(visible_blobs) -> tuple[EnemyBlob, ...]:
        """Canonicalise one observation without trusting per-query blob IDs."""

        ordered = sorted(
            visible_blobs,
            key=lambda blob: (
                int(blob.player_id),
                float(blob.pos[0]),
                float(blob.pos[1]),
                float(blob.radius),
                int(blob.merge_cooldown),
            ),
        )
        next_id_by_player: dict[int, int] = {}
        enemies: list[EnemyBlob] = []
        for blob in ordered:
            player_id = int(blob.player_id)
            ephemeral_id = next_id_by_player.get(player_id, 0)
            next_id_by_player[player_id] = ephemeral_id + 1
            enemies.append(
                EnemyBlob(
                    player_id=player_id,
                    blob_id=ephemeral_id,
                    x=float(blob.pos[0]),
                    y=float(blob.pos[1]),
                    radius=float(blob.radius),
                    merge_cooldown=int(blob.merge_cooldown),
                )
            )
        return tuple(enemies)

    def _planning_enemies(
        self,
        context: StrategyContext,
        own_blobs: tuple[OwnBlob, ...],
        arena_size: float,
        viruses: tuple[VirusModel, ...],
    ) -> tuple[EnemyBlob, ...]:
        """Use only the current anonymous view for the default planner."""

        del own_blobs, arena_size, viruses
        return self._visible_enemies(context.game.state.visible_blobs)

    def _update_enemy_memory(
        self,
        context: StrategyContext,
        own_blobs: tuple[OwnBlob, ...],
        arena_size: float,
        viruses: tuple[VirusModel, ...] = (),
    ) -> tuple[EnemyBlob, ...]:
        state = context.game.state
        round_number = int(state.round)
        visible_blobs = tuple(state.visible_blobs)

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
            speed = player_speed(track.radius)
            radius = _decayed_radius(track.radius)
            advanced[key] = replace(
                track,
                x=_clamp(track.x + direction[0] * speed, radius, arena_size - radius),
                y=_clamp(track.y + direction[1] * speed, radius, arena_size - radius),
                radius=radius,
                direction=direction,
                merge_cooldown=max(0, track.merge_cooldown - 1),
            )

        matched_observations = self._match_enemy_observations(
            advanced,
            visible_blobs,
        )
        retired_merges = self._retired_merged_tracks(
            advanced,
            visible_blobs,
            matched_observations,
        )
        for key in retired_merges:
            advanced.pop(key, None)
        visible_track_keys: set[tuple[int, int]] = set()
        for observation_index, blob in enumerate(visible_blobs):
            key = matched_observations.get(observation_index)
            if key is None:
                used_ids = {
                    track_id
                    for player_id, track_id in advanced
                    if player_id == blob.player_id
                }
                next_id = max(
                    self._next_enemy_track_id,
                    max(used_ids, default=-1) + 1,
                )
                self._next_enemy_track_id = next_id + 1
                key = (blob.player_id, next_id)
            visible_track_keys.add(key)
            direction = self.last_moves.get(blob.player_id, ((0.0, 0.0), False))[0]
            advanced[key] = EnemyTrack(
                player_id=blob.player_id,
                blob_id=key[1],
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
            if key not in visible_track_keys and should_be_visible:
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

    @staticmethod
    def _retired_merged_tracks(
        predicted: dict[tuple[int, int], EnemyTrack],
        observations: tuple,
        matches: dict[int, tuple[int, int]],
    ) -> set[tuple[int, int]]:
        """Retire unmatched tracks whose complete merge is now authoritative.

        A merge observation inherits one internal survivor ID.  Other ready,
        touching tracks in the same physical component are consumed only when
        their combined mass and center explain that observation.  Split
        observations cannot satisfy the mass check, so one child inherits the
        parent ID and every additional child remains a new observation.
        """

        matched_keys = set(matches.values())
        retired: set[tuple[int, int]] = set()
        for observation_index, survivor_key in matches.items():
            survivor = predicted[survivor_key]
            observation = observations[observation_index]
            if survivor.merge_cooldown > 0:
                continue
            component = {survivor_key}
            changed = True
            while changed:
                changed = False
                for key, candidate in predicted.items():
                    if (
                        key in component
                        or key in retired
                        or candidate.player_id != survivor.player_id
                        or candidate.merge_cooldown > 0
                    ):
                        continue
                    if any(
                        math.dist(
                            (candidate.x, candidate.y),
                            (predicted[member].x, predicted[member].y),
                        )
                        <= (
                            candidate.radius
                            + predicted[member].radius
                            + 2.0 * MERGE_ATTRACTION_SPEED
                            + SAME_PLAYER_OVERLAP_EPSILON
                        )
                        for member in component
                    ):
                        component.add(key)
                        changed = True
            if len(component) == 1:
                continue
            tracks = tuple(predicted[key] for key in component)
            total_mass = sum(track.radius * track.radius for track in tracks)
            observed_mass = observation.radius * observation.radius
            center = (
                sum(track.x * track.radius * track.radius for track in tracks)
                / total_mass,
                sum(track.y * track.radius * track.radius for track in tracks)
                / total_mass,
            )
            if math.dist(center, observation.pos) > max(
                0.25,
                2.0 * MERGE_ATTRACTION_SPEED,
            ):
                continue
            derived_observations = []
            for other_index, other in enumerate(observations):
                if other.player_id != survivor.player_id:
                    continue
                assigned_track = matches.get(other_index)
                if assigned_track in component:
                    derived_observations.append(other_index)
                    continue
                if assigned_track is not None:
                    # The observation has independent provenance outside this
                    # ready/touching component, even when it is spatially near.
                    continue
                if (
                    other.radius * other.radius <= total_mass * 1.05
                    and math.dist(center, other.pos)
                    <= math.sqrt(total_mass) + SPLIT_EJECT_SPEED + 2.0
                ):
                    derived_observations.append(other_index)
            conservation_tolerance = max(0.05, total_mass * 0.05)
            if derived_observations != [observation_index]:
                # More than one observation derives from this component, so
                # its mass remains split/crossing instead of having one merge
                # survivor.  Assignment provenance prevents a nearby but
                # independent track from blocking a real merge.
                continue

            minimum_observed_mass = total_mass - conservation_tolerance
            # Engine player-eating repeats until stable, so a freshly merged
            # blob may gain any number of legal prey in this same transition.
            # Mass is therefore one-sided: decay gives the lower bound, while
            # component geometry and the single authoritative survivor prevent
            # an unrelated observation from consuming these tracks.
            if observed_mass < minimum_observed_mass:
                continue
            retired.update(component - matched_keys)
        return retired

    @staticmethod
    def _match_enemy_observations(
        predicted: dict[tuple[int, int], EnemyTrack],
        observations: tuple,
    ) -> dict[int, tuple[int, int]]:
        """Globally match re-indexed public blobs to persistent tracks.

        agario-kit 2026.1.14 numbers the visible subset from zero on every
        query, so the public blob index is explicitly not an identity.  Each
        player's feasible bipartite graph is solved once after every track has
        advanced: cardinality is maximised first, then total geometric cost is
        minimised.  This handles crossing fragments without the order-dependent
        track theft of a greedy matcher.
        """

        result: dict[int, tuple[int, int]] = {}
        player_ids = sorted(
            {
                *(track.player_id for track in predicted.values()),
                *(blob.player_id for blob in observations),
            }
        )
        for player_id in player_ids:
            track_rows = tuple(
                sorted(
                    (
                        (key, track)
                        for key, track in predicted.items()
                        if track.player_id == player_id
                    ),
                    key=lambda row: row[0],
                )
            )
            observation_rows = tuple(
                (index, blob)
                for index, blob in enumerate(observations)
                if blob.player_id == player_id
            )
            if not track_rows or not observation_rows:
                continue

            edge_costs: list[list[float | None]] = []
            maximum_edge_cost = 0.0
            for _, track in track_rows:
                row = []
                for _, blob in observation_rows:
                    distance = math.dist((track.x, track.y), blob.pos)
                    radius_delta = abs(track.radius - blob.radius)
                    maximum_displacement = (
                        SPLIT_EJECT_SPEED + 2.0 + track.radius + blob.radius
                    )
                    cost = (
                        distance + radius_delta
                        if distance <= maximum_displacement
                        else None
                    )
                    row.append(cost)
                    if cost is not None:
                        maximum_edge_cost = max(maximum_edge_cost, cost)
                edge_costs.append(row)

            # One avoided dummy assignment must dominate every possible sum
            # of physical edge costs, making maximum cardinality the primary
            # objective of the ordinary minimum-cost assignment below.
            pair_limit = min(len(track_rows), len(observation_rows))
            unmatched_cost = (maximum_edge_cost + 1.0) * (pair_limit + 1)
            forbidden_cost = unmatched_cost * (len(track_rows) + 2)
            cost_matrix = [
                [forbidden_cost if cost is None else cost for cost in physical_costs]
                + [unmatched_cost] * len(track_rows)
                for physical_costs in edge_costs
            ]
            assignment = ThreatAwareRecedingHorizonStrategy._minimum_cost_assignment(
                cost_matrix
            )
            for track_index, column in enumerate(assignment):
                if column >= len(observation_rows):
                    continue
                if edge_costs[track_index][column] is None:
                    continue
                observation_index = observation_rows[column][0]
                result[observation_index] = track_rows[track_index][0]
        return result

    @staticmethod
    def _minimum_cost_assignment(costs: list[list[float]]) -> tuple[int, ...]:
        """Return the minimum-cost unique column for every row in O(R²C)."""

        row_count = len(costs)
        if row_count == 0:
            return ()
        column_count = len(costs[0])
        if column_count < row_count or any(len(row) != column_count for row in costs):
            raise ValueError(
                "assignment matrix must be rectangular with rows <= columns"
            )

        row_potential = [0.0] * (row_count + 1)
        column_potential = [0.0] * (column_count + 1)
        matched_row = [0] * (column_count + 1)
        predecessor = [0] * (column_count + 1)
        for row in range(1, row_count + 1):
            matched_row[0] = row
            minimum = [math.inf] * (column_count + 1)
            used = [False] * (column_count + 1)
            column = 0
            while True:
                used[column] = True
                active_row = matched_row[column]
                delta = math.inf
                next_column = 0
                for candidate in range(1, column_count + 1):
                    if used[candidate]:
                        continue
                    reduced = (
                        costs[active_row - 1][candidate - 1]
                        - row_potential[active_row]
                        - column_potential[candidate]
                    )
                    if reduced < minimum[candidate]:
                        minimum[candidate] = reduced
                        predecessor[candidate] = column
                    if minimum[candidate] < delta:
                        delta = minimum[candidate]
                        next_column = candidate
                for candidate in range(column_count + 1):
                    if used[candidate]:
                        row_potential[matched_row[candidate]] += delta
                        column_potential[candidate] -= delta
                    else:
                        minimum[candidate] -= delta
                column = next_column
                if matched_row[column] == 0:
                    break
            while True:
                previous = predecessor[column]
                matched_row[column] = matched_row[previous]
                column = previous
                if column == 0:
                    break

        assignment = [-1] * row_count
        for column in range(1, column_count + 1):
            if matched_row[column]:
                assignment[matched_row[column] - 1] = column - 1
        return tuple(assignment)

    def _stale_enemy_can_threaten_transition(
        self,
        track: EnemyTrack,
        own_blobs: tuple[OwnBlob, ...],
        viruses: tuple[VirusModel, ...],
    ) -> bool:
        # Existing fragments are authoritative state: any predator that can
        # eat one remains relevant, even if it cannot eat our largest blob.
        if any(can_eat_player_blob(track.radius, own.radius) for own in own_blobs):
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
                <= 18.0 * player_speed(own.radius)
                for virus in viruses
            )
            if not virus_reachable:
                continue
            virus_piece_radius = math.sqrt(
                (own.mass + VIRUS_SIZE * VIRUS_SIZE) / virus_piece_count
            )
            # A stale small blob may take one piece, but the replay collapse
            # comes from a large enemy splitting through many pieces.  Preserve
            # that tail risk without treating every unseen prey as a sweeper.
            if _can_split_eat(track.radius, virus_piece_radius):
                return True
        return False

    def _raw_candidate_actions(
        self,
        *,
        node: SearchNode,
        foods: tuple[FoodModel, ...],
        food_targets: tuple[tuple[float, float], ...],
        arena_size: float,
        first_step: bool,
        angle_offset: int = 0,
    ) -> list[Action]:
        actions: list[Action] = []

        # This order is part of the anytime-search contract: if the root is
        # cut short, it must already have considered the tactically important
        # options rather than an arbitrary prefix of the angular grid.
        escape = self._escape_vector(node)
        if escape != (0.0, 0.0):
            actions.append(Action(escape, reason="escape"))
            # Tangential options often escape a wall/predator pincer better
            # than a pure potential-field vector.
            actions.append(
                Action(_rotate(escape, math.pi / 8), reason="escape_tangent")
            )
            actions.append(
                Action(_rotate(escape, -math.pi / 8), reason="escape_tangent")
            )

        actions.append(
            Action(node.last_direction, reason="keep" if first_step else "continue")
        )
        if not first_step:
            for angle in (-math.pi / 6, -math.pi / 12, math.pi / 12, math.pi / 6):
                actions.append(
                    Action(_rotate(node.last_direction, angle), reason="steer")
                )

        center = node.center
        available_food = [
            food for food in foods if food.food_id not in node.eaten_food_ids
        ]
        if available_food:
            nearest_food = min(
                available_food,
                key=lambda food: squared_distance(center, food.pos),
            )
            actions.append(
                Action(
                    normalise(
                        (
                            nearest_food.pos[0] - center[0],
                            nearest_food.pos[1] - center[1],
                        )
                    ),
                    reason="nearest_food",
                )
            )
        target_limit = 4 if first_step else 2
        for target in food_targets[:target_limit]:
            actions.append(
                Action(
                    normalise((target[0] - center[0], target[1] - center[1])),
                    reason="food_cluster",
                )
            )

        prey = [
            enemy
            for enemy in node.enemies
            if enemy.stale_rounds == 0
            and any(
                can_eat_player_blob(own.radius, enemy.radius) for own in node.own_blobs
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
        actions.append(
            Action(
                normalise((arena_size / 2.0 - center[0], arena_size / 2.0 - center[1])),
                reason="center",
            )
        )

        if first_step:
            sample_count = max(8, self.angular_samples)
            actions.extend(
                Action(
                    (
                        math.cos(
                            TAU * ((index + angle_offset) % sample_count) / sample_count
                        ),
                        math.sin(
                            TAU * ((index + angle_offset) % sample_count) / sample_count
                        ),
                    ),
                    reason="angle",
                )
                for index in range(sample_count)
            )

        return actions

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
        actions = self._raw_candidate_actions(
            node=node,
            foods=foods,
            food_targets=food_targets,
            arena_size=arena_size,
            first_step=first_step,
            angle_offset=angle_offset,
        )
        return self._dedupe_actions(actions, profile_prefix="base_candidate")

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

    @staticmethod
    def _action_key(action: Action) -> tuple[bool, int, bool]:
        direction = normalise(action.direction)
        is_zero = direction == (0.0, 0.0)
        angle_bin = int(round(math.atan2(direction[1], direction[0]) / TAU * 96)) % 96
        return is_zero, angle_bin, action.split

    def _actions_per_node_limit(self, depth_index: int) -> int | None:
        return None

    def _uses_base_transition_score(self) -> bool:
        return True

    def _decision_min_safety_margin(
        self,
        node: SearchNode,
    ) -> float:
        return node.min_safety_margin

    def _record_profile_count(self, name: str, amount: int = 1) -> None:
        """Optional low-overhead observation hook for specialised policies."""

    def _required_actions_for_depth(
        self,
        depth_index: int,
    ) -> int:
        return self.minimum_root_actions if depth_index == 0 else 1

    def _audit_root_candidate_ranking(
        self,
        *,
        node: SearchNode,
        actions: tuple[Action, ...],
        foods: tuple[FoodModel, ...],
        viruses: tuple[VirusModel, ...],
        arena_size: float,
        safety_weight: float,
        aggression: float,
        transition_budget: int | None,
    ) -> float:
        """Optional offline hook for measuring approximate-ranking recall."""

        return 0.0

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
        """Rank one transition using the legacy opponent policy."""

        return self._physical_step(
            node=node,
            action=action,
            foods=foods,
            viruses=viruses,
            arena_size=arena_size,
            first_step=first_step,
            safety_weight=safety_weight,
            aggression=aggression,
            joint_command=None,
        )

    def _physical_step(
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
        joint_command: CompleteJointCommand | None,
    ) -> StepResult:
        """Run physical events, then apply legacy hazard and utility policy."""

        core = self._physical_step_core(
            node=node,
            action=action,
            foods=foods,
            viruses=viruses,
            arena_size=arena_size,
            first_step=first_step,
            joint_command=joint_command,
            collect_utility_metrics=self._uses_base_transition_score(),
        )
        return self._finalize_physical_step(
            source_node=node,
            action=action,
            foods=foods,
            arena_size=arena_size,
            safety_weight=safety_weight,
            aggression=aggression,
            core=core,
        )

    def _physical_step_core(
        self,
        *,
        node: SearchNode,
        action: Action,
        foods: tuple[FoodModel, ...],
        viruses: tuple[VirusModel, ...],
        arena_size: float,
        first_step: bool,
        joint_command: CompleteJointCommand | None,
        collect_utility_metrics: bool,
    ) -> PhysicalStepCoreResult:
        """Apply only movement and engine-ordered world events.

        ``joint_command=None`` preserves the historical per-fragment opponent
        projection used by search ranking.  Supplying a complete command uses
        one legal command per player while retaining identical event order.
        Risk, fatality, position value, and turn costs are finalized elsewhere.
        """

        direction = normalise(action.direction)
        own_blobs = list(node.own_blobs)
        split_penalty = 0.0
        food_reward = 0.0
        pre_split_radius_by_id = {blob.blob_id: blob.radius for blob in own_blobs}
        if action.split:
            own_blobs = self._apply_split(own_blobs, direction, arena_size)
            if collect_utility_metrics:
                split_penalty = (
                    6.0
                    + max(
                        0,
                        len(own_blobs) - len(node.own_blobs),
                    )
                    * 1.5
                )
        split_blob_ids = (
            {
                blob.blob_id
                for blob in own_blobs
                if blob.blob_id not in pre_split_radius_by_id
                or blob.radius < pre_split_radius_by_id[blob.blob_id]
            }
            if action.split
            else set()
        )

        movement = self._move_own_blobs(
            own_blobs,
            direction,
            arena_size,
            calculate_blocked=collect_utility_metrics,
            calculate_efficiency=not collect_utility_metrics and not action.split,
        )
        own_blobs = list(movement.blobs)
        if joint_command is None:
            enemies = self._move_enemies(node.enemies, own_blobs, arena_size)
        else:
            live_player_ids = {
                self._own_player_id,
                *(enemy.player_id for enemy in node.enemies),
            }
            if joint_command.player_ids != live_player_ids:
                raise ValueError("joint command does not match the live world")
            own_command = joint_command.for_player(self._own_player_id)
            if own_command.unit != direction or own_command.split != action.split:
                raise ValueError("joint own command must match the root action")
            enemies = self._move_enemy_players_with_joint_command(
                node.enemies,
                joint_command,
                arena_size,
            )

        eaten_food_ids = set(node.eaten_food_ids)
        consumed_virus_ids = set(node.consumed_virus_ids)
        projected_food = node.projected_food
        projected_captures = node.projected_captures
        projected_viruses = node.projected_viruses

        own_blobs = [
            replace(blob, radius=_decayed_radius(blob.radius)) for blob in own_blobs
        ]
        own_blobs = self._stabilise_own_blobs(own_blobs, arena_size)
        enemies = self._stabilise_enemy_blobs(enemies, arena_size)

        # Engine order is stabilise -> viruses -> stabilise -> food.  Keeping
        # virus resolution after food can incorrectly push a blob over the
        # consumption threshold or preserve a pre-pop large cell.
        pre_virus_own = own_blobs
        pre_virus_enemies = enemies
        own_blobs, enemies, virus_penalty, own_viruses = self._resolve_own_viruses(
            own_blobs=own_blobs,
            enemies=enemies,
            viruses=viruses,
            consumed_virus_ids=consumed_virus_ids,
            arena_size=arena_size,
            calculate_penalty=collect_utility_metrics,
        )
        projected_viruses += own_viruses
        if own_blobs is not pre_virus_own or enemies is not pre_virus_enemies:
            own_blobs = self._stabilise_own_blobs(own_blobs, arena_size)
            enemies = self._stabilise_enemy_blobs(enemies, arena_size)

        all_blobs: dict[tuple[int, int], OwnBlob | EnemyBlob] = {
            (self._own_player_id, blob.blob_id): blob for blob in own_blobs
        }
        all_blobs.update(((enemy.player_id, enemy.blob_id), enemy) for enemy in enemies)
        consumed_food = False
        for food in foods:
            if food.food_id in eaten_food_ids:
                continue
            winner: tuple[tuple[int, int], OwnBlob | EnemyBlob] | None = None
            winner_order: tuple[float, int, int] | None = None
            for key, blob in all_blobs.items():
                if squared_distance(blob.pos, food.pos) > blob.radius * blob.radius:
                    continue
                order = (-blob.radius, key[0], key[1])
                if winner_order is None or order < winner_order:
                    winner = (key, blob)
                    winner_order = order
            if winner is None:
                continue
            key, eater = winner
            all_blobs[key] = _with_grown_radius(
                eater,
                math.sqrt(eater.mass + FOOD_RADIUS * FOOD_RADIUS),
                arena_size,
            )
            consumed_food = True
            eaten_food_ids.add(food.food_id)
            if key[0] == self._own_player_id:
                projected_food += 1
                if collect_utility_metrics:
                    food_reward += 7.0

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

        pre_interaction_own_count = len(own_blobs)
        pre_interaction_enemy_count = len(enemies)
        own_blobs, enemies, interaction_score, captures = self._resolve_interactions(
            own_blobs,
            enemies,
            arena_size,
            calculate_score=collect_utility_metrics,
        )
        split_lost_fragment = bool(
            split_blob_ids - {blob.blob_id for blob in own_blobs}
        )
        projected_captures += captures
        if (
            consumed_food
            or len(own_blobs) != pre_interaction_own_count
            or len(enemies) != pre_interaction_enemy_count
        ):
            own_blobs = self._stabilise_own_blobs(own_blobs, arena_size)
            enemies = self._stabilise_enemy_blobs(enemies, arena_size)
        if not own_blobs:
            dead = self._replace_node(
                node=node,
                own_blobs=(),
                enemies=enemies,
                score=node.score,
                direction=direction,
                action=action,
                first_step=first_step,
                eaten_food_ids=eaten_food_ids,
                consumed_virus_ids=consumed_virus_ids,
                projected_food=projected_food,
                projected_captures=projected_captures,
                projected_viruses=projected_viruses,
                min_safety_margin=node.min_safety_margin,
            )
            return PhysicalStepCoreResult(
                node=dead,
                movement_efficiency=movement.efficiency,
                blocked_distance=movement.blocked_distance,
                split_lost_fragment=split_lost_fragment,
                split_penalty=split_penalty,
                virus_penalty=virus_penalty,
                food_reward=food_reward,
                interaction_score=interaction_score,
            )

        next_node = self._replace_node(
            node=node,
            own_blobs=tuple(own_blobs),
            enemies=enemies,
            score=node.score,
            direction=direction,
            action=action,
            first_step=first_step,
            eaten_food_ids=eaten_food_ids,
            consumed_virus_ids=consumed_virus_ids,
            projected_food=projected_food,
            projected_captures=projected_captures,
            projected_viruses=projected_viruses,
            min_safety_margin=node.min_safety_margin,
        )
        return PhysicalStepCoreResult(
            node=next_node,
            movement_efficiency=movement.efficiency,
            blocked_distance=movement.blocked_distance,
            split_lost_fragment=split_lost_fragment,
            split_penalty=split_penalty,
            virus_penalty=virus_penalty,
            food_reward=food_reward,
            interaction_score=interaction_score,
        )

    def _finalize_physical_step(
        self,
        *,
        source_node: SearchNode,
        action: Action,
        foods: tuple[FoodModel, ...],
        arena_size: float,
        safety_weight: float,
        aggression: float,
        core: PhysicalStepCoreResult,
    ) -> StepResult:
        """Attach legacy risk and utility semantics to one physical result."""

        score = core.node.score
        uses_base_score = self._uses_base_transition_score()
        if uses_base_score:
            score -= core.split_penalty
            score -= core.virus_penalty * safety_weight
            score += core.food_reward
            score += core.interaction_score * aggression
        if not core.node.own_blobs:
            return StepResult(
                replace(
                    core.node,
                    score=score - 100_000.0,
                    min_safety_margin=-math.inf,
                ),
                fatal=True,
                movement_efficiency=core.movement_efficiency,
            )

        own_blobs = core.node.own_blobs
        enemies = core.node.enemies
        risk_penalty, min_margin, unavoidable, hazard_summary = self._risk_analysis(
            own_blobs,
            enemies,
            safety_weight,
            arena_size,
        )
        if uses_base_score:
            score -= risk_penalty
            score += self._position_value(
                own_blobs,
                enemies,
                foods,
                set(core.node.eaten_food_ids),
                aggression,
            )
            score -= core.blocked_distance * BLOCKED_MOVEMENT_COST
            score -= self._turn_cost(
                source_node.last_direction, core.node.last_direction
            )
            score -= max(0, len(own_blobs) - 1) * 0.65

        next_node = replace(
            core.node,
            score=score,
            min_safety_margin=min(source_node.min_safety_margin, min_margin),
        )
        unsafe_split = action.split and (core.split_lost_fragment or min_margin <= 0.0)
        return StepResult(
            next_node,
            fatal=unavoidable or unsafe_split,
            movement_efficiency=core.movement_efficiency,
            hazard_summary=hazard_summary,
        )

    def _joint_physical_step(
        self,
        *,
        node: SearchNode,
        action: Action,
        foods: tuple[FoodModel, ...],
        viruses: tuple[VirusModel, ...],
        arena_size: float,
        first_step: bool,
        joint_command: CompleteJointCommand,
    ) -> JointPhysicalTransition[SearchNode]:
        """Expose legal-response physics without StepResult policy semantics."""

        result = self._physical_step_core(
            node=node,
            action=action,
            foods=foods,
            viruses=viruses,
            arena_size=arena_size,
            first_step=first_step,
            joint_command=joint_command,
            collect_utility_metrics=False,
        )
        return JointPhysicalTransition(
            state=result.node,
            final_own_mass=result.node.total_mass if result.node.own_blobs else 0.0,
            dead=not result.node.own_blobs,
            movement_efficiency=result.movement_efficiency,
        )

    def _resolve_own_viruses(
        self,
        *,
        own_blobs: list[OwnBlob],
        enemies: tuple[EnemyBlob, ...] = (),
        viruses: tuple[VirusModel, ...],
        consumed_virus_ids: set[int],
        arena_size: float,
        calculate_penalty: bool = True,
    ) -> tuple[list[OwnBlob], tuple[EnemyBlob, ...], float, int]:
        """Apply the public engine's touching-blob virus transition.

        The physical state transition is shared. The returned controllability
        penalty is used only by policies that retain the base transition score.
        """
        penalty = 0.0
        own_consumed = 0
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
            own_blobs, enemies, origin = collision
            if origin is None:
                continue
            own_consumed += 1
            if calculate_penalty:
                penalty += 240.0 + origin.mass * 18.0
        return own_blobs, enemies, penalty, own_consumed

    def _apply_virus_collision(
        self,
        *,
        own_blobs: list[OwnBlob],
        enemies: tuple[EnemyBlob, ...],
        virus: VirusModel,
        arena_size: float,
    ) -> tuple[list[OwnBlob], tuple[EnemyBlob, ...], OwnBlob | None] | None:
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
            return own_blobs, enemies, origin
        enemies = enemies[:index] + tuple(fragments) + enemies[index + 1 :]
        return own_blobs, enemies, None

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
                    eject_vx=0.0,
                    eject_vy=0.0,
                )
            )
        return fragments

    def _apply_split(
        self,
        blobs: list[OwnBlob],
        direction: tuple[float, float],
        arena_size: float,
    ) -> list[OwnBlob]:
        split_direction, split_ejection = _split_placement_and_ejection(direction)
        result = list(blobs)
        starting_ids = [
            blob.blob_id for blob in sorted(blobs, key=lambda blob: blob.blob_id)
        ]
        next_id = max(starting_ids, default=-1) + 1
        by_id = {blob.blob_id: blob for blob in result}
        for blob_id in starting_ids:
            if len(by_id) >= MAX_BLOB_COUNT:
                break
            blob = by_id.get(blob_id)
            if blob is None or blob.mass < SPLIT_MIN_MASS:
                continue
            child_radius = blob.radius / SQRT2
            parent = replace(
                blob, radius=child_radius, merge_cooldown=SPLIT_COOLDOWN_FRAMES
            )
            child = OwnBlob(
                blob_id=next_id,
                x=_clamp(
                    blob.x
                    + split_direction[0]
                    * (2.0 * child_radius + SAME_PLAYER_OVERLAP_EPSILON),
                    child_radius,
                    arena_size - child_radius,
                ),
                y=_clamp(
                    blob.y
                    + split_direction[1]
                    * (2.0 * child_radius + SAME_PLAYER_OVERLAP_EPSILON),
                    child_radius,
                    arena_size - child_radius,
                ),
                radius=child_radius,
                merge_cooldown=SPLIT_COOLDOWN_FRAMES,
                eject_vx=split_ejection[0],
                eject_vy=split_ejection[1],
            )
            by_id[blob_id] = parent
            by_id[next_id] = child
            next_id += 1
        return list(by_id.values())

    def _move_own(
        self,
        blob: OwnBlob,
        direction: tuple[float, float],
        arena_size: float,
        *,
        speed: float | None = None,
    ) -> OwnBlob:
        """Move one blob using a caller-prepared radius-dependent speed."""

        speed = player_speed(blob.radius) if speed is None else speed
        x = blob.x + direction[0] * speed + blob.eject_vx
        y = blob.y + direction[1] * speed + blob.eject_vy
        return OwnBlob(
            blob_id=blob.blob_id,
            x=_clamp(x, blob.radius, arena_size - blob.radius),
            y=_clamp(y, blob.radius, arena_size - blob.radius),
            radius=blob.radius,
            merge_cooldown=max(0, blob.merge_cooldown - 1),
            eject_vx=_damped(blob.eject_vx),
            eject_vy=_damped(blob.eject_vy),
        )

    def _move_own_blobs(
        self,
        blobs: list[OwnBlob] | tuple[OwnBlob, ...],
        direction: tuple[float, float],
        arena_size: float,
        *,
        calculate_blocked: bool = True,
        calculate_efficiency: bool = True,
        profile_counter: str = "physics_own_blob_moves",
        speed_cache: dict[float, float] | None = None,
    ) -> OwnMovement:
        """Move every blob once and derive both movement-cost measures."""

        unit = normalise(direction)
        moved_blobs: list[OwnBlob] = []
        self._record_profile_count(profile_counter, len(blobs))
        total_mass = sum(blob.mass for blob in blobs) if calculate_blocked else 0.0
        lost = 0.0
        expected_useful = 0.0
        actual_useful = 0.0
        for blob in blobs:
            if speed_cache is None:
                speed = player_speed(blob.radius)
            else:
                try:
                    speed = speed_cache[blob.radius]
                except KeyError:
                    speed = player_speed(blob.radius)
                    speed_cache[blob.radius] = speed
            moved = self._move_own(
                blob,
                unit,
                arena_size,
                speed=speed,
            )
            moved_blobs.append(moved)
            if calculate_blocked:
                intended_dx = unit[0] * speed + blob.eject_vx
                intended_dy = unit[1] * speed + blob.eject_vy
                intended_distance = math.hypot(intended_dx, intended_dy)
                actual_distance = math.dist(blob.pos, moved.pos)
                lost += blob.mass * max(0.0, intended_distance - actual_distance)
            if calculate_efficiency:
                expected_useful += blob.mass * speed
                actual_useful += blob.mass * max(
                    0.0,
                    (moved.x - blob.x) * unit[0] + (moved.y - blob.y) * unit[1],
                )
        if not calculate_blocked or total_mass <= EPSILON:
            blocked_distance = 0.0
        else:
            blocked_distance = lost / total_mass
        if not calculate_efficiency or expected_useful <= EPSILON:
            efficiency = 1.0
        else:
            efficiency = _clamp(actual_useful / expected_useful, 0.0, 1.0)
        return OwnMovement(tuple(moved_blobs), blocked_distance, efficiency)

    def _apply_attraction(
        self,
        blobs: list[OwnBlob | EnemyBlob],
        arena_size: float,
    ) -> list[OwnBlob | EnemyBlob]:
        if len(blobs) <= 1:
            return blobs
        total_mass = sum(blob.mass for blob in blobs)
        center_x = sum(blob.x * blob.mass for blob in blobs) / total_mass
        center_y = sum(blob.y * blob.mass for blob in blobs) / total_mass
        result: list[OwnBlob | EnemyBlob] = []
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
                    x=_clamp(
                        blob.x + dx / distance * step,
                        blob.radius,
                        arena_size - blob.radius,
                    ),
                    y=_clamp(
                        blob.y + dy / distance * step,
                        blob.radius,
                        arena_size - blob.radius,
                    ),
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
        center_x = (
            sum(item[1] * item[3] * item[3] for item in work.values()) / total_mass
        )
        center_y = (
            sum(item[2] * item[3] * item[3] for item in work.values()) / total_mass
        )
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
            prey: OwnBlob | None = None
            hunter: OwnBlob | None = None
            prey_distance = math.inf
            hunter_distance = math.inf
            for own in own_blobs:
                distance = squared_distance(enemy.pos, own.pos)
                if (
                    can_eat_player_blob(enemy.radius, own.radius)
                    and distance < prey_distance
                ):
                    prey = own
                    prey_distance = distance
                if (
                    can_eat_player_blob(own.radius, enemy.radius)
                    and distance < hunter_distance
                ):
                    hunter = own
                    hunter_distance = distance
            observed = normalise(enemy.direction)
            if prey is not None:
                adversarial = normalise((prey.x - enemy.x, prey.y - enemy.y))
                direction = normalise(
                    (
                        adversarial[0] * 0.82 + observed[0] * 0.18,
                        adversarial[1] * 0.82 + observed[1] * 0.18,
                    )
                )
            elif hunter is not None:
                flee = normalise((enemy.x - hunter.x, enemy.y - hunter.y))
                direction = normalise(
                    (
                        flee[0] * 0.62 + observed[0] * 0.38,
                        flee[1] * 0.62 + observed[1] * 0.38,
                    )
                )
            else:
                direction = observed
            speed = player_speed(enemy.radius)
            moved.append(
                replace(
                    enemy,
                    x=_clamp(
                        enemy.x + direction[0] * speed + enemy.eject_vx,
                        enemy.radius,
                        arena_size - enemy.radius,
                    ),
                    y=_clamp(
                        enemy.y + direction[1] * speed + enemy.eject_vy,
                        enemy.radius,
                        arena_size - enemy.radius,
                    ),
                    radius=_decayed_radius(enemy.radius),
                    direction=direction,
                    merge_cooldown=max(0, enemy.merge_cooldown - 1),
                    eject_vx=_damped(enemy.eject_vx),
                    eject_vy=_damped(enemy.eject_vy),
                )
            )
        return tuple(moved)

    def _move_enemy_players_with_joint_command(
        self,
        enemies: tuple[EnemyBlob, ...],
        joint_command: CompleteJointCommand,
        arena_size: float,
    ) -> tuple[EnemyBlob, ...]:
        """Advance every enemy fragment with its player's single command."""

        by_player: dict[int, list[EnemyBlob]] = {}
        for enemy in enemies:
            by_player.setdefault(enemy.player_id, []).append(enemy)

        moved: list[EnemyBlob] = []
        for player_id in sorted(by_player):
            command = joint_command.for_player(player_id)
            unit = command.unit
            group = by_player[player_id]
            if command.split:
                group = self._apply_enemy_split(group, unit, arena_size)
            for enemy in group:
                speed = player_speed(enemy.radius)
                moved.append(
                    replace(
                        enemy,
                        x=_clamp(
                            enemy.x + unit[0] * speed + enemy.eject_vx,
                            enemy.radius,
                            arena_size - enemy.radius,
                        ),
                        y=_clamp(
                            enemy.y + unit[1] * speed + enemy.eject_vy,
                            enemy.radius,
                            arena_size - enemy.radius,
                        ),
                        radius=_decayed_radius(enemy.radius),
                        direction=unit,
                        merge_cooldown=max(0, enemy.merge_cooldown - 1),
                        eject_vx=_damped(enemy.eject_vx),
                        eject_vy=_damped(enemy.eject_vy),
                    )
                )
        return tuple(moved)

    @staticmethod
    def _apply_enemy_split(
        enemies: list[EnemyBlob],
        direction: tuple[float, float],
        arena_size: float,
    ) -> list[EnemyBlob]:
        """Apply one legal split command to a single player's fragments."""

        split_direction, split_ejection = _split_placement_and_ejection(direction)
        by_id = {enemy.blob_id: enemy for enemy in enemies}
        starting_ids = sorted(by_id)
        next_id = max(starting_ids, default=-1) + 1
        for blob_id in starting_ids:
            if len(by_id) >= MAX_BLOB_COUNT:
                break
            enemy = by_id.get(blob_id)
            if enemy is None or enemy.mass < SPLIT_MIN_MASS:
                continue
            child_radius = enemy.radius / SQRT2
            by_id[blob_id] = replace(
                enemy,
                radius=child_radius,
                merge_cooldown=SPLIT_COOLDOWN_FRAMES,
            )
            by_id[next_id] = EnemyBlob(
                player_id=enemy.player_id,
                blob_id=next_id,
                x=_clamp(
                    enemy.x
                    + split_direction[0]
                    * (2.0 * child_radius + SAME_PLAYER_OVERLAP_EPSILON),
                    child_radius,
                    arena_size - child_radius,
                ),
                y=_clamp(
                    enemy.y
                    + split_direction[1]
                    * (2.0 * child_radius + SAME_PLAYER_OVERLAP_EPSILON),
                    child_radius,
                    arena_size - child_radius,
                ),
                radius=child_radius,
                direction=direction,
                stale_rounds=enemy.stale_rounds,
                merge_cooldown=SPLIT_COOLDOWN_FRAMES,
                eject_vx=split_ejection[0],
                eject_vy=split_ejection[1],
            )
            next_id += 1
        return [by_id[blob_id] for blob_id in sorted(by_id)]

    def _stabilise_enemy_blobs(
        self,
        enemies: tuple[EnemyBlob, ...],
        arena_size: float,
    ) -> tuple[EnemyBlob, ...]:
        """Return the exact stabilised layout, memoized within one turn."""

        key = (enemies, arena_size)
        cached = self._enemy_stabilisation_cache.get(key)
        if cached is not None:
            return cached
        by_player: dict[int, list[EnemyBlob]] = {}
        for enemy in enemies:
            by_player.setdefault(enemy.player_id, []).append(enemy)
        rows: list[EnemyBlob] = []
        for player_id in sorted(by_player):
            group = tuple(by_player[player_id])
            group_key = (group, arena_size)
            stabilised = self._enemy_group_stabilisation_cache.get(group_key)
            if stabilised is None:
                stabilised = self._stabilise_enemy_group_uncached(group, arena_size)
                self._enemy_group_stabilisation_cache[group_key] = stabilised
            rows.extend(stabilised)
        result = tuple(sorted(rows, key=lambda enemy: enemy.key))
        self._enemy_stabilisation_cache[key] = result
        return result

    def _stabilise_enemy_blobs_uncached(
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
            result.extend(
                self._stabilise_enemy_group_uncached(
                    tuple(by_player[player_id]),
                    arena_size,
                )
            )
        return tuple(sorted(result, key=lambda enemy: enemy.key))

    def _stabilise_enemy_group_uncached(
        self,
        enemies: tuple[EnemyBlob, ...],
        arena_size: float,
    ) -> tuple[EnemyBlob, ...]:
        """Stabilise one player's fragments without consulting memo tables."""

        if len(enemies) <= 1:
            return enemies
        group = list(self._apply_attraction(list(enemies), arena_size))
        if all(enemy.merge_cooldown > 0 for enemy in group):
            # Both merge passes are provable no-ops. The two consecutive
            # four-iteration separation passes are therefore exactly one
            # eight-iteration pass, avoiding an intermediate tuple plus a
            # second set of materialised EnemyBlob replacements.
            return self._separate_enemy_blobs(
                tuple(group),
                arena_size,
                iterations=8,
            )
        group = list(self._merge_enemy_blobs(tuple(group), arena_size))
        group = list(self._separate_enemy_blobs(tuple(group), arena_size))
        group = list(self._merge_enemy_blobs(tuple(group), arena_size))
        return self._separate_enemy_blobs(tuple(group), arena_size)

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
                        first.radius + second.radius + SAME_PLAYER_OVERLAP_EPSILON
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
                        eject_vx=(
                            survivor.eject_vx * survivor.mass
                            + consumed.eject_vx * consumed.mass
                        )
                        / combined_mass,
                        eject_vy=(
                            survivor.eject_vy * survivor.mass
                            + consumed.eject_vy * consumed.mass
                        )
                        / combined_mass,
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
        # Enemy-virus transitions can create four independent 16-fragment
        # groups in one simulated root.  Replacing two frozen dataclasses for
        # every overlapping pair made this otherwise small relaxation allocate
        # thousands of short-lived objects.  Keep the exact key/pair/iteration
        # order and arithmetic, but mutate only turn-local x/y work rows; one
        # final replace materialises each blob after the relaxation converges.
        work = {
            enemy.key: [enemy, enemy.x, enemy.y, enemy.radius, enemy.mass]
            for enemy in enemies
        }
        for _ in range(iterations):
            changed = False
            keys = sorted(work)
            for index, first_key in enumerate(keys):
                for second_key in keys[index + 1 :]:
                    if first_key[0] != second_key[0]:
                        continue
                    first = work[first_key]
                    second = work[second_key]
                    dx = second[1] - first[1]
                    dy = second[2] - first[2]
                    distance = math.hypot(dx, dy)
                    minimum = first[3] + second[3] + SAME_PLAYER_OVERLAP_EPSILON
                    if distance >= minimum:
                        continue
                    if distance <= EPSILON:
                        nx, ny = (1.0, 0.0)
                    else:
                        nx, ny = (dx / distance, dy / distance)
                    overlap = minimum - distance
                    total_mass = first[4] + second[4]
                    first_move = overlap * second[4] / total_mass
                    second_move = overlap * first[4] / total_mass
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
                break
        return tuple(
            row[0]
            if row[1] == row[0].x and row[2] == row[0].y
            else replace(row[0], x=row[1], y=row[2])
            for key in sorted(work)
            for row in (work[key],)
        )

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
                                sum(
                                    fragment.direction[0] * fragment.mass
                                    for fragment in group
                                ),
                                sum(
                                    fragment.direction[1] * fragment.mass
                                    for fragment in group
                                ),
                            )
                        ),
                        stale_rounds=max(fragment.stale_rounds for fragment in group),
                        merge_cooldown=max(
                            fragment.merge_cooldown for fragment in group
                        ),
                    )
                )
        return tuple(envelopes)

    def _risk_enemies(
        self,
        enemies: tuple[EnemyBlob, ...],
    ) -> tuple[EnemyBlob, ...]:
        return (*enemies, *self._future_enemy_envelopes(enemies))

    def _resolve_interactions(
        self,
        own_blobs: list[OwnBlob],
        enemies: tuple[EnemyBlob, ...],
        arena_size: float = ARENA_SIZE,
        *,
        calculate_score: bool = True,
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
                ("own", blob_id, blob) for blob_id, blob in own_by_id.items()
            ]
            actors.extend(("enemy", key, enemy) for key, enemy in enemy_by_key.items())
            actors.sort(
                key=lambda item: (
                    -item[2].radius,
                    self._own_player_id if item[0] == "own" else item[1][0],  # type: ignore[index]
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
                        assert isinstance(eater_key, tuple) and isinstance(
                            target_key, tuple
                        )
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
                    if (
                        squared_distance(eater.pos, target.pos)
                        > eater.radius * eater.radius
                    ):
                        continue

                    grown_radius = math.sqrt(eater.mass + target.mass)
                    if eater_kind == "own":
                        assert isinstance(eater_key, int) and isinstance(
                            target_key, tuple
                        )
                        own_by_id[eater_key] = _with_grown_radius(
                            eater,
                            grown_radius,
                            arena_size,
                        )
                        del enemy_by_key[target_key]
                        if calculate_score:
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
                            if calculate_score:
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

    def _risk_analysis(
        self,
        own_blobs: list[OwnBlob],
        enemies: tuple[EnemyBlob, ...],
        safety_weight: float,
        arena_size: float = ARENA_SIZE,
    ) -> tuple[float, float, bool, HazardSummary | None]:
        penalty, margin, unavoidable = self._risk_score(
            own_blobs,
            enemies,
            safety_weight,
            arena_size,
        )
        return penalty, margin, unavoidable, None

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
        risk_enemies = self._risk_enemies(enemies)
        for own in own_blobs:
            player_penalties: dict[int, float] = {}
            for enemy in risk_enemies:
                if not can_eat_player_blob(enemy.radius, own.radius):
                    continue
                distance = math.dist(own.pos, enemy.pos)
                normal_margin = distance - enemy.radius
                danger_radius = enemy.radius
                if _can_split_eat(enemy.radius, own.radius):
                    danger_radius = max(
                        danger_radius,
                        _split_chain_attack_reach(enemy.radius, own.radius),
                    )
                split_margin = distance - danger_radius
                margin = min(normal_margin, split_margin)
                min_margin = min(min_margin, margin)
                uncertainty = 1.0 + enemy.stale_rounds * 0.12
                if margin <= 0.0:
                    if enemy.blob_id >= 0:
                        endangered_blob_ids.add(own.blob_id)
                    enemy_penalty = (
                        440.0 + 75.0 * own.mass + min(180.0, -margin * 35.0)
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
                    enemy_penalty = pressure * (12.0 + 2.0 * own.mass) * uncertainty
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
            penalty += (own.mass / max(total_mass, EPSILON)) * sum(
                player_penalties.values()
            )
        unavoidable = bool(own_blobs) and len(endangered_blob_ids) == len(own_blobs)
        return penalty * safety_weight, min_margin, unavoidable

    def _wall_trap_factor(
        self,
        own: OwnBlob,
        enemy: EnemyBlob,
        arena_size: float,
    ) -> float:
        """Estimate how much a nearby wall blocks direct retreat from an enemy."""

        return self._wall_trap_factor_at(
            own.x,
            own.y,
            own.radius,
            enemy,
            arena_size,
        )

    def _wall_trap_factor_at(
        self,
        own_x: float,
        own_y: float,
        own_radius: float,
        enemy: EnemyBlob,
        arena_size: float,
    ) -> float:
        """Numeric wall risk used by both model objects and virtual fragments."""

        delta_x = own_x - enemy.x
        delta_y = own_y - enemy.y
        magnitude = math.hypot(delta_x, delta_y)
        if magnitude <= EPSILON:
            return 0.0
        away_x = delta_x / magnitude
        away_y = delta_y / magnitude

        # The replay failures were already unrecoverable several turns before
        # the blob physically touched the wall.  Ten world units approximates
        # the distance needed to turn a direct retreat into a viable tangent,
        # while the outward component keeps unrelated nearby walls irrelevant.
        escape_horizon = 10.0
        left_weight = max(0.0, -away_x)
        right_weight = max(0.0, away_x)
        bottom_weight = max(0.0, -away_y)
        top_weight = max(0.0, away_y)
        blocked = (
            left_weight * max(0.0, 1.0 - max(0.0, own_x - own_radius) / escape_horizon)
            + right_weight
            * max(
                0.0,
                1.0 - max(0.0, arena_size - own_radius - own_x) / escape_horizon,
            )
            + bottom_weight
            * max(0.0, 1.0 - max(0.0, own_y - own_radius) / escape_horizon)
            + top_weight
            * max(
                0.0,
                1.0 - max(0.0, arena_size - own_radius - own_y) / escape_horizon,
            )
        )
        return min(1.0, blocked)

    def _position_value(
        self,
        own_blobs: list[OwnBlob],
        enemies: tuple[EnemyBlob, ...],
        foods: tuple[FoodModel, ...],
        eaten_food_ids: set[int],
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
            elif (
                not can_eat_player_blob(enemy.radius, primary.radius) and distance < 8.0
            ):
                value -= (8.0 - distance) * 0.3
        return value

    def _escape_vector(self, node: SearchNode) -> tuple[float, float]:
        x = 0.0
        y = 0.0
        risk_enemies = self._risk_enemies(node.enemies)
        for own in node.own_blobs:
            for enemy in risk_enemies:
                if not can_eat_player_blob(enemy.radius, own.radius):
                    continue
                danger_radius = enemy.radius
                if _can_split_eat(enemy.radius, own.radius):
                    danger_radius = max(
                        danger_radius,
                        _split_chain_attack_reach(enemy.radius, own.radius),
                    )
                distance = math.dist(own.pos, enemy.pos)
                if distance > danger_radius + 8.0:
                    continue
                away = normalise((own.x - enemy.x, own.y - enemy.y))
                severity = max(0.2, danger_radius + 8.0 - distance) / max(
                    distance, 0.25
                )
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
        foods: tuple[FoodModel, ...],
    ) -> list[tuple[float, float]]:
        if not foods:
            return []
        # The observation is already sorted by center distance in ``_choose``.
        nearest = foods[0]
        neighbour_groups: list[list[FoodModel]] = [[] for _ in foods]
        for index, food in enumerate(foods):
            # Earlier neighbours were appended by their outer iteration. Add
            # self here, then later neighbours, preserving the old sum order.
            neighbour_groups[index].append(food)
            for other_index in range(index + 1, len(foods)):
                other = foods[other_index]
                if squared_distance(food.pos, other.pos) <= 9.0:
                    neighbour_groups[index].append(other)
                    neighbour_groups[other_index].append(food)
        scored: list[tuple[float, tuple[float, float]]] = []
        for index in range(len(foods)):
            neighbours = neighbour_groups[index]
            target = (
                sum(other.pos[0] for other in neighbours) / len(neighbours),
                sum(other.pos[1] for other in neighbours) / len(neighbours),
            )
            distance = math.dist(center, target)
            score = (len(neighbours) + 0.5) / (distance + 1.5)
            scored.append((score, target))
        scored.sort(reverse=True)
        targets: list[tuple[float, float]] = [nearest.pos]
        seen: set[tuple[int, int]] = {
            (round(nearest.pos[0] * 2), round(nearest.pos[1] * 2))
        }
        for _, target in scored:
            key = (round(target[0] * 2), round(target[1] * 2))
            if key in seen:
                continue
            seen.add(key)
            targets.append(target)
        return targets

    def _intercept_direction(
        self, own: OwnBlob, enemy: EnemyBlob
    ) -> tuple[float, float]:
        distance = math.dist(own.pos, enemy.pos)
        lookahead = min(3.0, distance / max(player_speed(own.radius), 0.1) * 0.3)
        target = (
            enemy.x + enemy.direction[0] * player_speed(enemy.radius) * lookahead,
            enemy.y + enemy.direction[1] * player_speed(enemy.radius) * lookahead,
        )
        return normalise((target[0] - own.x, target[1] - own.y))

    def _split_can_capture(
        self,
        node: SearchNode,
        enemy: EnemyBlob,
        direction: tuple[float, float],
        *,
        horizon: int = 1,
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
            reach = (
                2.0 * child_radius
                + SPLIT_EJECT_SPEED
                + player_speed(child_radius)
                + child_radius
            )
            if horizon > 1:
                reach += min(horizon - 1, 1) * (
                    player_speed(child_radius) + SPLIT_EJECT_SPEED * SPLIT_EJECT_DRAG
                )
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
            first_angle = (
                int(
                    round(
                        math.atan2(node.first_direction[1], node.first_direction[0])
                        / TAU
                        * 48
                    )
                )
                % 48
            )
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
        consumed_virus_ids: set[int],
        projected_food: int,
        projected_captures: int,
        projected_viruses: int,
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
            consumed_virus_ids=frozenset(consumed_virus_ids),
            projected_food=projected_food,
            projected_captures=projected_captures,
            projected_viruses=projected_viruses,
            min_safety_margin=min_safety_margin,
        )

    def _dedupe_actions(
        self,
        actions: list[Action],
        *,
        profile_prefix: str | None = None,
    ) -> tuple[Action, ...]:
        result: list[Action] = []
        seen: set[tuple[bool, int, bool]] = set()
        zero_drops = 0
        duplicate_drops = 0
        for action in actions:
            direction = normalise(action.direction)
            if direction == (0.0, 0.0):
                zero_drops += 1
                continue
            key = self._action_key(action)
            if key in seen:
                duplicate_drops += 1
                continue
            seen.add(key)
            result.append(Action(direction, action.split, action.reason))
        if profile_prefix is not None:
            self._record_profile_count(f"{profile_prefix}_raw", len(actions))
            self._record_profile_count(f"{profile_prefix}_unique", len(result))
            self._record_profile_count(f"{profile_prefix}_zero_drops", zero_drops)
            self._record_profile_count(
                f"{profile_prefix}_duplicate_drops",
                duplicate_drops,
            )
        return tuple(result)

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


def _decayed_radius(radius: float) -> float:
    return _replay_decayed_radius(
        radius,
        decay_rate=MASS_DECAY_RATE,
        minimum_radius=STARTING_RADIUS,
    )


def _can_consume_virus(blob_radius: float, virus_radius: float) -> bool:
    return _replay_can_consume_virus(
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


def _mass_center(blobs: tuple[OwnBlob, ...]) -> tuple[float, float]:
    total_mass = sum(blob.mass for blob in blobs)
    if total_mass <= EPSILON:
        return max(blobs, key=lambda blob: blob.radius).pos
    return (
        sum(blob.x * blob.mass for blob in blobs) / total_mass,
        sum(blob.y * blob.mass for blob in blobs) / total_mass,
    )


def _can_split_eat(predator_radius: float, prey_radius: float) -> bool:
    return predator_radius * predator_radius >= SPLIT_MIN_MASS and can_eat_player_blob(
        predator_radius / SQRT2, prey_radius
    )


def _split_attack_reach(predator_radius: float) -> float:
    """One-round center-distance reach of a directly aimed split attack."""

    child_radius = predator_radius / SQRT2
    return 3.0 * child_radius + SPLIT_EJECT_SPEED + player_speed(child_radius)


def _split_chain_attack_reach(
    predator_radius: float,
    prey_radius: float,
    *,
    max_splits: int = MAX_BLOB_COUNT.bit_length() - 1,
) -> float:
    """Return the farthest center reach of one repeatedly splitting lineage."""

    attacker_radius = predator_radius
    forward_center_offset = 0.0
    reach = predator_radius
    for _ in range(max(0, max_splits)):
        if not _can_split_eat(attacker_radius, prey_radius):
            break
        child_radius = attacker_radius / SQRT2
        forward_center_offset += (
            2.0 * child_radius + SPLIT_EJECT_SPEED + player_speed(child_radius)
        )
        reach = max(reach, forward_center_offset + child_radius)
        attacker_radius = child_radius
    return reach


def _split_placement_and_ejection(
    direction: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Resolve the engine's distinct zero-command placement and launch rules."""

    control_unit = normalise(direction)
    if control_unit == (0.0, 0.0):
        return (1.0, 0.0), (0.0, 0.0)
    return control_unit, (
        control_unit[0] * SPLIT_EJECT_SPEED,
        control_unit[1] * SPLIT_EJECT_SPEED,
    )


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
    _CAPTURE_HORIZON = 8.0
    _CAPTURE_CLOSING_TEMPERATURE = 0.05
    _TURN_CACHE_NAMES = (
        "_utility_cache",
        "_utility_identity_cache",
        "_virus_retention_cache",
        "_risk_envelope_cache",
        "_node_geometry_cache",
        "_prey_expected_mass_cache",
        "_virus_expected_mass_cache",
        "_hazard_summary_cache",
    )
    _AUDIT_CACHE_NAMES = (*_TURN_CACHE_NAMES, "_virus_fragment_layout_cache")
    _MOVEMENT_INEFFICIENCY_PENALTY = 2.0
    _VIRUS_FRAGMENT_LAYOUT_CACHE_LIMIT = 128

    def __init__(
        self,
        depth: int | None = None,
        width: int | None = None,
        angular_samples: int | None = None,
    ) -> None:
        resolved_depth = (
            depth
            if depth is not None
            else int(
                _receding_horizon_setting(
                    "BOT_RECEDING_HORIZON_DEPTH",
                    "BOT_CHAMPION_DEPTH",
                    "1",
                )
            )
        )
        resolved_width = (
            width
            if width is not None
            else int(
                _receding_horizon_setting(
                    "BOT_RECEDING_HORIZON_WIDTH",
                    "BOT_CHAMPION_WIDTH",
                    "1",
                )
            )
        )
        super().__init__(
            depth=resolved_depth,
            width=resolved_width,
            angular_samples=10 if angular_samples is None else angular_samples,
        )
        # Full exact rollout is an offline oracle only. The official engine's
        # cumulative timeout includes runtime outside this strategy and proved
        # materially tighter than the local strategy timer.
        self.compute_budget_seconds = float(
            os.environ.get("BOT_REPLAY_TOTAL_BUDGET_SECONDS", "0.0")
        )
        self.max_turn_seconds = float(
            os.environ.get(
                "BOT_REPLAY_MAX_TURN_SECONDS",
                _receding_horizon_setting(
                    "BOT_RECEDING_HORIZON_MAX_TURN_SECONDS",
                    "BOT_CHAMPION_MAX_TURN_SECONDS",
                    "0.003",
                ),
            )
        )
        self.competition_compute_budget_seconds = float(
            os.environ.get("BOT_REPLAY_COMPETITION_BUDGET_SECONDS", "8.0")
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
        self.recovery_mass = float(os.environ.get("BOT_REPLAY_RECOVERY_MASS", "6.0"))
        self._rival_values: dict[int, float] = {}
        self._utility_cache: dict[tuple[object, ...], float] = {}
        # Search nodes are immutable and both caches are turn-local. This
        # identity front cache avoids re-hashing a repeated parent while the
        # structural cache still reuses values for distinct equivalent nodes.
        self._utility_identity_cache: dict[
            int,
            tuple[SearchNode, float, float, float],
        ] = {}
        self._virus_retention_cache: dict[
            tuple[int, int, float], tuple[SearchNode, float]
        ] = {}
        self._risk_envelope_cache: dict[
            int, tuple[tuple[EnemyBlob, ...], tuple[EnemyBlob, ...]]
        ] = {}
        self._node_geometry_cache: dict[
            int,
            tuple[SearchNode, NodeGeometry],
        ] = {}
        self._prey_expected_mass_cache: dict[
            tuple[int, tuple[int, int], float],
            tuple[SearchNode, EnemyBlob, float],
        ] = {}
        self._virus_expected_mass_cache: dict[
            tuple[int, int, float],
            tuple[SearchNode, VirusModel, float | None],
        ] = {}
        self._hazard_summary_cache: dict[
            tuple[
                tuple[OwnBlob, ...],
                tuple[EnemyBlob, ...],
                float,
                float,
            ],
            HazardSummary,
        ] = {}
        self._virus_fragment_layout_cache: dict[
            tuple[float, int],
            tuple[tuple[float, float], ...],
        ] = {}
        self.transition_budget_scale = int(
            os.environ.get("BOT_REPLAY_TRANSITION_BUDGET_SCALE", "6")
        )
        self.exact_root_action_limit = max(
            1,
            int(os.environ.get("BOT_REPLAY_EXACT_ROOT_ACTION_LIMIT", "6")),
        )
        self.exact_deeper_action_limit = max(
            1,
            int(os.environ.get("BOT_REPLAY_EXACT_DEEPER_ACTION_LIMIT", "1")),
        )
        self.exact_max_blobs = max(
            1,
            int(os.environ.get("BOT_REPLAY_EXACT_MAX_BLOBS", str(MAX_BLOB_COUNT))),
        )
        self.exact_min_blobs = max(
            1,
            int(os.environ.get("BOT_REPLAY_EXACT_MIN_BLOBS", "1")),
        )
        self.proxy_horizon = max(
            1,
            int(os.environ.get("BOT_REPLAY_PROXY_HORIZON", "8")),
        )
        # Static resources remain useful over the full potential horizon, but
        # observed opponents change direction too often for an eight-turn
        # straight-line forecast. Dynamic contact risk therefore uses only a
        # short local reachability horizon inside the same projected action.
        self.proxy_dynamic_horizon = max(
            1,
            int(os.environ.get("BOT_REPLAY_DYNAMIC_HORIZON", "1")),
        )
        self.proxy_reachable_threats = (
            os.environ.get("BOT_REPLAY_THREAT_MODEL", "reachable") == "reachable"
        )
        # Static resources keep the full horizon. Opponent motion is activated
        # only when ordinary contact can occur before the next observation;
        # this preserves the strong static-growth policy without branching on
        # unpredictable enemy trajectories.
        self.proxy_enemy_motion = os.environ.get(
            "BOT_REPLAY_ENEMY_MOTION",
            "event_gated",
        )
        self.proxy_coherent_threat_scenarios = (
            os.environ.get("BOT_REPLAY_THREAT_AGGREGATION", "independent") == "coherent"
        )
        self.proxy_static_food_sweep = (
            os.environ.get("BOT_REPLAY_STATIC_FOOD_SWEEP", "0") != "0"
        )
        self.shadow_force_round = int(
            os.environ.get("BOT_REPLAY_SHADOW_FORCE_ROUND", "-1")
        )
        self.shadow_force_rank = int(
            os.environ.get("BOT_REPLAY_SHADOW_FORCE_RANK", "0")
        )
        self.proxy_cohesion_enabled = os.environ.get("BOT_REPLAY_COHESION", "0") != "0"
        self._effective_proxy_horizon = self.proxy_horizon
        self._effective_proxy_dynamic_horizon = min(
            self.proxy_dynamic_horizon,
            self.proxy_horizon,
        )
        self._proxy_has_future_after_horizon = True
        self.proxy_discount = _clamp(
            float(os.environ.get("BOT_REPLAY_PROXY_DISCOUNT", "0.82")),
            0.0,
            1.0,
        )
        self.proxy_food_limit = max(
            1,
            int(os.environ.get("BOT_REPLAY_PROXY_FOOD_LIMIT", "12")),
        )
        self.proxy_virus_limit = max(
            1,
            int(os.environ.get("BOT_REPLAY_PROXY_VIRUS_LIMIT", "4")),
        )
        self.proxy_refine_limit = max(
            1,
            int(os.environ.get("BOT_REPLAY_PROXY_REFINE_LIMIT", "64")),
        )
        self.proxy_refine_blob_work = max(
            1,
            int(os.environ.get("BOT_REPLAY_PROXY_REFINE_BLOB_WORK", "128")),
        )
        self.proxy_min_refine = max(
            1,
            int(os.environ.get("BOT_REPLAY_PROXY_MIN_REFINE", "8")),
        )
        # Split is a discrete, costly event rather than a direction modifier.
        # The experimental switch lets paired benchmarks isolate the effect of
        # generating only splits with an explicit capture/farming purpose.
        self.proxy_semantic_split = (
            os.environ.get("BOT_REPLAY_SEMANTIC_SPLIT", "0") != "0"
        )
        # Coarse scoring is useful for choosing a representative *within* an
        # action family.  Refining one or two representatives per active family
        # preserves safety/opportunity roles without paying for every nearby
        # angular variant.
        self.proxy_family_refine = (
            os.environ.get("BOT_REPLAY_FAMILY_REFINE", "0") != "0"
        )
        self.proxy_family_refine_limit = max(
            1,
            int(os.environ.get("BOT_REPLAY_FAMILY_REFINE_LIMIT", "7")),
        )
        self.proxy_coarse_after_seconds = max(
            0.0,
            float(
                os.environ.get(
                    "BOT_REPLAY_PROXY_COARSE_AFTER_SECONDS",
                    str(self.competition_compute_budget_seconds * 0.7),
                )
            ),
        )
        self._competition_coarse_mode = False
        self._proxy_rank_audit_every_n = max(
            0,
            int(os.environ.get("BOT_REPLAY_PROXY_RANK_AUDIT_EVERY_N", "0")),
        )
        self._proxy_rank_audit_samples = 0
        self._proxy_rank_audit_hits = 0
        self._proxy_rank_audit_last_best_coarse_rank: int | None = None
        self._proxy_rank_audit_last_best_reason: str | None = None
        self._proxy_rank_audit_last_best_split: bool | None = None
        self._proxy_rank_audit_last_regret: float | None = None
        self.proxy_threat_limit = max(
            1,
            int(os.environ.get("BOT_REPLAY_PROXY_THREAT_LIMIT", "4")),
        )
        self.proxy_observed_direction_weight = _clamp(
            float(
                os.environ.get(
                    "BOT_REPLAY_PROXY_OBSERVED_DIRECTION_WEIGHT",
                    "0.18",
                )
            ),
            0.0,
            1.0,
        )
        self._proxy_safety_weight = 1.3
        self.minimum_transition_budget = int(
            os.environ.get("BOT_REPLAY_MIN_TRANSITIONS", "2")
        )
        # An anytime search must compare at least two complete root
        # transitions.  Otherwise the first semantic candidate (often a virus
        # or prey) becomes a forced action when one exact transition consumes
        # the deadline.  Root utility caching below makes this affordable.
        self.minimum_root_actions = 2
        self._root_proxy_scores: tuple[tuple[Action, float], ...] = ()
        self._root_candidate_families: dict[str, int] = {}
        self._root_proxy_refined = 0
        self._static_food_sweep_added = False
        self._static_food_sweep_predicted_advantage = 0
        self._audit_every_n = int(os.environ.get("BOT_REPLAY_AUDIT_EVERY_N", "0"))
        self._audit_samples = 0
        self._audit_hits = 0
        self._audit_last_exact_rank: int | None = None
        self._audit_last_raw_rank: int | None = None
        self._audit_last_fatal_count = 0
        self._audit_spent_seconds = 0.0
        self._pending_audit: RootAuditRequest | None = None
        self._audit_last_exact_reason: str | None = None
        self._audit_last_exact_split: bool | None = None
        self._audit_last_exact_regret: float | None = None
        self._current_round = 0
        self._profile_every_n = int(os.environ.get("BOT_REPLAY_PROFILE_EVERY_N", "0"))
        self._profile_stderr = os.environ.get("BOT_REPLAY_PROFILE_STDERR", "0") != "0"
        self._profile_active = False
        self._profile_seconds: dict[str, float] = {}
        self._profile_calls: dict[str, int] = {}
        self._profile_counts: dict[str, int] = {}
        self._profile_values: dict[str, float] = {}
        self._proxy_blob_motion_scratch: list[ProxyBlobMotion] = []
        self._proxy_enemy_motion_scratch: list[ProxyEnemyMotion] = []
        self._proxy_blob_motion_active: list[ProxyBlobMotion] = []
        self._proxy_enemy_motion_active: list[ProxyEnemyMotion] = []
        self._proxy_movement_scratch = ProxyMovement(
            1.0,
            self._proxy_blob_motion_active,
        )

    def _profile_start(self) -> float | None:
        return perf_counter() if self._profile_active else None

    def _record_profile(self, name: str, started_at: float | None) -> None:
        if started_at is None:
            return
        self._record_profile_elapsed(name, perf_counter() - started_at)

    def _record_profile_elapsed(self, name: str, elapsed: float) -> None:
        if not self._profile_active:
            return
        self._profile_seconds[name] = self._profile_seconds.get(name, 0.0) + elapsed
        self._profile_calls[name] = self._profile_calls.get(name, 0) + 1

    def _record_cache_access(self, name: str, *, hit: bool) -> None:
        if not self._profile_active:
            return
        self._record_profile_count(f"{name}_{'hit' if hit else 'miss'}")
        call_key = f"cache_{name}"
        self._profile_calls[call_key] = self._profile_calls.get(call_key, 0) + 1

    def _clear_turn_caches(self) -> None:
        for name in self._TURN_CACHE_NAMES:
            getattr(self, name).clear()

    def _record_profile_count(self, name: str, amount: int = 1) -> None:
        if not self._profile_active:
            return
        self._profile_counts[name] = self._profile_counts.get(name, 0) + amount

    def _record_profile_value(self, name: str, value: float) -> None:
        if not self._profile_active:
            return
        self._profile_values[name] = self._profile_values.get(name, 0.0) + value
        self._record_profile_count(f"{name}_samples")
        if abs(value) > EPSILON:
            self._record_profile_count(f"{name}_nonzero")

    def _begin_replay_turn(self, state) -> float | None:
        """Reset turn-local analysis state shared by search and fallback."""

        self._current_round = int(state.round)
        self._profile_active = (
            self._profile_every_n > 0
            and self._current_round % self._profile_every_n == 0
        )
        profile_started = self._profile_start()
        self._clear_turn_caches()
        self._root_proxy_scores = ()
        self._root_candidate_families = {}
        self._root_proxy_refined = 0
        self._static_food_sweep_added = False
        self._static_food_sweep_predicted_advantage = 0
        self._pending_audit = None
        self._own_player_id = int(state.me.player_id)
        rankings = tuple(int(player_id) for player_id in state.rankings)
        try:
            rank_index = rankings.index(self._own_player_id)
        except ValueError:
            rank_index = len(rankings)
        self._rival_values = {
            player_id: 1.0 / (1.0 + abs(other_index - rank_index))
            for other_index, player_id in enumerate(rankings)
            if other_index != rank_index
        }
        progress = self._current_round / max(
            1,
            int(getattr(state, "max_rounds", 1400)),
        )
        self._effective_proxy_horizon = min(
            self.proxy_horizon,
            max(1, int(getattr(state, "max_rounds", 1400)) - self._current_round),
        )
        self._effective_proxy_dynamic_horizon = min(
            self.proxy_dynamic_horizon,
            self._effective_proxy_horizon,
        )
        self._proxy_has_future_after_horizon = (
            self._current_round + self._effective_proxy_horizon
            < int(getattr(state, "max_rounds", 1400))
        )
        self._proxy_safety_weight = self._safety_weight(
            rank_index + 1,
            progress,
        )
        return profile_started

    def _emit_profile(self, decision: StrategyDecision) -> StrategyDecision:
        if not self._profile_active:
            return decision
        diagnostics = decision.diagnostics
        seconds = self._profile_seconds
        calls = self._profile_calls
        counts = self._profile_counts
        values = self._profile_values
        inclusive_ms = {
            name: round(elapsed * 1000.0, 6)
            for name, elapsed in sorted(seconds.items())
        }
        candidate_children = sum(
            seconds.get(name, 0.0)
            for name in (
                "proxy",
                "proxy_coarse",
                "proxy_refine",
            )
        )
        step_children = sum(
            seconds.get(name, 0.0)
            for name in (
                "step_parent_utility",
                "step_split_movement_efficiency",
                "physics",
                "step_child_utility",
                "step_shaping",
            )
        )
        candidate_total = seconds.get("candidate", 0.0)
        step_total = seconds.get("step", 0.0)
        audit_total = seconds.get("audit", 0.0)
        choose_total = seconds.get("choose", 0.0)
        fallback_total = seconds.get("fallback", 0.0)
        phase_seconds = {
            "setup_and_search_control": max(
                0.0,
                choose_total - candidate_total - step_total - audit_total,
            ),
            "candidate_shared_analysis": seconds.get("proxy", 0.0),
            "candidate_coarse_rank": seconds.get("proxy_coarse", 0.0),
            "candidate_geometric_refine": seconds.get("proxy_refine", 0.0),
            "candidate_overhead": max(
                0.0,
                seconds.get("candidate", 0.0) - candidate_children,
            ),
            "search_parent_utility": seconds.get("step_parent_utility", 0.0),
            "search_split_movement_efficiency": seconds.get(
                "step_split_movement_efficiency",
                0.0,
            ),
            "search_physics": seconds.get("physics", 0.0),
            "search_child_utility": seconds.get("step_child_utility", 0.0),
            "search_shaping": seconds.get("step_shaping", 0.0),
            "search_step_overhead": max(
                0.0,
                seconds.get("step", 0.0) - step_children,
            ),
            "audit_diagnostic": seconds.get("audit", 0.0),
            # The fallback operation includes its candidate generation.  Only
            # its own control overhead belongs in this additive phase; the
            # candidate phases above already account for the nested work.
            "fallback": max(
                0.0,
                fallback_total - candidate_total - step_total - audit_total,
            ),
        }
        profile = {
            "schema_version": 1,
            "round": self._current_round,
            "sample_every_n": self._profile_every_n,
            "compute_resource_ms": {
                "spent_before_turn": round(
                    self.compute_spent_seconds * 1000.0,
                    3,
                ),
                "competition_budget": round(
                    self.competition_compute_budget_seconds * 1000.0,
                    3,
                ),
                "competition_remaining_before_turn": round(
                    max(
                        0.0,
                        self.competition_compute_budget_seconds
                        - self.compute_spent_seconds,
                    )
                    * 1000.0,
                    3,
                ),
            },
            "phase_ms": {
                name: round(elapsed * 1000.0, 6)
                for name, elapsed in phase_seconds.items()
            },
            "operation_inclusive_ms": inclusive_ms,
            "calls": dict(sorted(calls.items())),
            "counts": dict(sorted(counts.items())),
            "value_sums": {
                name: round(value, 9) for name, value in sorted(values.items())
            },
        }
        if self._profile_stderr:
            print(
                "REPLAY_PROFILE "
                + json.dumps(profile, separators=(",", ":"), sort_keys=True),
                file=sys.stderr,
                flush=True,
            )
        self._profile_seconds.clear()
        self._profile_calls.clear()
        self._profile_counts.clear()
        self._profile_values.clear()
        return replace(
            decision,
            diagnostics={**diagnostics, "replay_profile": profile},
        )

    def _uses_base_transition_score(self) -> bool:
        # This policy replaces the base score with a utility difference.
        return False

    def choose(self, context: StrategyContext) -> StrategyDecision:
        """Run the geometric proxy as the primary competition policy.

        A positive exact-search bank is still supported for offline A/B and
        audit runs.  With the production default of zero, entering the base
        class's time-bank fallback would produce the same action but falsely
        label every healthy turn as degraded execution.
        """

        if (
            self.compute_budget_seconds > EPSILON
            and self.exact_min_blobs
            <= len(context.game.state.me.blobs)
            <= self.exact_max_blobs
        ):
            return super().choose(context)

        if (
            not self._competition_coarse_mode
            and self.proxy_coarse_after_seconds > 0.0
            and self.compute_spent_seconds >= self.proxy_coarse_after_seconds
        ):
            self._competition_coarse_mode = True
        started_at = perf_counter()
        try:
            decision = self._proxy_decision(context)
        finally:
            self.compute_spent_seconds += perf_counter() - started_at
        return replace(
            decision,
            diagnostics={
                **decision.diagnostics,
                "compute_spent_ms": round(
                    self.compute_spent_seconds * 1000.0,
                    3,
                ),
                "competition_compute_budget_ms": round(
                    self.competition_compute_budget_seconds * 1000.0,
                    3,
                ),
                "competition_compute_remaining_ms": round(
                    max(
                        0.0,
                        self.competition_compute_budget_seconds
                        - self.compute_spent_seconds,
                    )
                    * 1000.0,
                    3,
                ),
                "competition_coarse_mode": self._competition_coarse_mode,
            },
        )

    def _proxy_decision(self, context: StrategyContext) -> StrategyDecision:
        profile_started = self._begin_replay_turn(context.game.state)
        decision = super()._approximate_value_fallback(context)
        decision = self._shadow_forced_decision(decision)
        diagnostics = dict(decision.diagnostics)
        candidate_count = int(diagnostics.get("fallback_candidates") or 0)
        diagnostics.update(
            approximate_fallback=False,
            primary_proxy=True,
            search_stop_reason="proxy_complete",
            root_actions_generated=candidate_count,
            root_actions_proxy_evaluated=int(
                diagnostics.get("proxy_candidates_refined") or 0
            ),
            root_actions_evaluated=0,
        )
        self._record_profile("choose", profile_started)
        return self._emit_profile(replace(decision, diagnostics=diagnostics))

    def _shadow_forced_decision(
        self,
        decision: StrategyDecision,
    ) -> StrategyDecision:
        """Force one recorded proxy rank for offline counterfactual rollouts."""

        if (
            self._current_round != self.shadow_force_round
            or self.shadow_force_rank <= 0
            or self.shadow_force_rank > len(self._root_proxy_scores)
        ):
            return decision
        action, score = self._root_proxy_scores[self.shadow_force_rank - 1]
        direction = normalise(action.direction)
        self.previous_direction = direction
        return StrategyDecision(
            direction=direction,
            split=action.split,
            target_kind=(
                "escape"
                if "escape" in action.reason
                else "prey"
                if "prey" in action.reason
                else "food"
                if "food" in action.reason
                else "beam"
            ),
            reason=action.reason,
            score=score,
            diagnostics={
                **decision.diagnostics,
                "shadow_forced": True,
                "shadow_forced_round": self.shadow_force_round,
                "shadow_forced_rank": self.shadow_force_rank,
            },
        )

    def _time_budget_fallback(
        self,
        context: StrategyContext,
    ) -> StrategyDecision:
        profile_started = self._begin_replay_turn(context.game.state)
        decision = super()._approximate_value_fallback(context)
        self._record_profile("fallback", profile_started)
        return self._emit_profile(decision)

    def _turn_budget_seconds(self, *, round_number: int, max_rounds: int) -> float:
        # Offline audit work must not spend the policy's competition search bank.
        effective_spent = max(
            0.0,
            self.compute_spent_seconds - self._audit_spent_seconds,
        )
        remaining_budget = max(0.0, self.compute_budget_seconds - effective_spent)
        remaining_rounds = max(1, max_rounds - round_number)
        return min(self.max_turn_seconds, remaining_budget / remaining_rounds)

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
        budget = max(
            self.minimum_transition_budget,
            round(self.transition_budget_scale / max(1.0, work_units)),
        )
        if own_blob_count <= 2:
            budget = max(2, budget)
        return budget

    def _choose(self, context, *, deadline: float, turn_budget: float):
        state = context.game.state
        profile_started = self._begin_replay_turn(state)
        decision = super()._choose(
            context,
            deadline=deadline,
            turn_budget=turn_budget,
        )
        self._run_pending_audit()
        selected_proxy_rank = None
        selected_key = self._action_key(
            Action(decision.direction, decision.split, decision.reason)
        )
        for index, (action, _) in enumerate(self._root_proxy_scores, start=1):
            if self._action_key(action) == selected_key:
                selected_proxy_rank = index
                break
        proxy_gap = None
        if len(self._root_proxy_scores) >= 2:
            proxy_gap = self._root_proxy_scores[0][1] - self._root_proxy_scores[1][1]
        diagnostics = dict(decision.diagnostics)
        diagnostics.update(
            proxy_only=False,
            compute_remaining_ms=round(
                max(
                    0.0,
                    self.compute_budget_seconds - self.compute_spent_seconds,
                )
                * 1000.0,
                3,
            ),
            competition_compute_budget_ms=round(
                self.competition_compute_budget_seconds * 1000.0,
                3,
            ),
            competition_compute_remaining_ms=round(
                max(
                    0.0,
                    self.competition_compute_budget_seconds
                    - self.compute_spent_seconds,
                )
                * 1000.0,
                3,
            ),
            candidate_family_counts=dict(self._root_candidate_families),
            proxy_candidates_refined=self._root_proxy_refined,
            static_food_sweep_added=self._static_food_sweep_added,
            static_food_sweep_predicted_advantage=(
                self._static_food_sweep_predicted_advantage
            ),
            proxy_top_actions=[
                {
                    "reason": action.reason,
                    "split": action.split,
                    "score": round(score, 6),
                }
                for action, score in self._root_proxy_scores[
                    : min(self._root_proxy_refined, self._DIAGNOSTIC_ACTION_LIMIT)
                ]
            ],
            selected_proxy_rank=selected_proxy_rank,
            proxy_top_two_gap=(round(proxy_gap, 6) if proxy_gap is not None else None),
            candidate_recall_samples=self._audit_samples,
            candidate_recall_at_k=(
                round(self._audit_hits / self._audit_samples, 6)
                if self._audit_samples
                else None
            ),
            exact_best_proxy_rank=self._audit_last_exact_rank,
            exact_best_raw_proxy_rank=self._audit_last_raw_rank,
            audit_fatal_candidates=self._audit_last_fatal_count,
            audit_spent_ms=round(self._audit_spent_seconds * 1000.0, 3),
            exact_best_reason=self._audit_last_exact_reason,
            exact_best_split=self._audit_last_exact_split,
            proxy_exact_regret=(
                round(self._audit_last_exact_regret, 6)
                if self._audit_last_exact_regret is not None
                else None
            ),
        )
        profiled_decision = replace(decision, diagnostics=diagnostics)
        self._record_profile("choose", profile_started)
        return self._emit_profile(profiled_decision)

    def _static_food_sweep_directions(
        self,
        *,
        node: SearchNode,
        foods: tuple[FoodModel, ...],
        horizon: int,
        limit: int = 1,
    ) -> tuple[tuple[float, float], ...]:
        """Find headings with maximum exact free-space pellet coverage.

        For each nearby food and each fragment, the headings whose constant
        direction segment intersects that food form an angular interval. We
        union intervals belonging to the same food so a pellet is counted only
        once, then sweep the interval boundaries. This optimises the actual
        shared direction input, unlike a route through region centroids.

        Walls and dynamic hazards are intentionally left to the shared proxy;
        this routine only adds static candidates and never replaces the
        existing food, escape, or exploration actions.
        """

        if not node.own_blobs or not foods or horizon <= 0 or limit <= 0:
            return ()
        available = [food for food in foods if food.food_id not in node.eaten_food_ids]
        if not available:
            return ()
        available.sort(
            key=lambda food: min(
                max(0.0, math.dist(blob.pos, food.pos) - blob.radius)
                for blob in node.own_blobs
            )
        )
        nearby = available[: self.proxy_food_limit]
        events: dict[float, float] = {}
        base_score = 0.0
        for food in nearby:
            intervals: list[tuple[float, float]] = []
            direction_independent = False
            for blob in node.own_blobs:
                delta_x = food.pos[0] - blob.x
                delta_y = food.pos[1] - blob.y
                distance = math.hypot(delta_x, delta_y)
                if distance <= blob.radius + EPSILON:
                    direction_independent = True
                    break
                travel = player_speed(blob.radius) * horizon
                if travel <= EPSILON or distance > travel + blob.radius:
                    continue
                tangent_distance = math.sqrt(
                    max(0.0, distance * distance - blob.radius * blob.radius)
                )
                if travel + EPSILON >= tangent_distance:
                    half_width = math.asin(_clamp(blob.radius / distance, 0.0, 1.0))
                else:
                    cosine = (
                        distance * distance
                        + travel * travel
                        - blob.radius * blob.radius
                    ) / (2.0 * distance * travel)
                    half_width = math.acos(_clamp(cosine, -1.0, 1.0))
                center = math.atan2(delta_y, delta_x) % TAU
                start = center - half_width
                end = center + half_width
                if start < 0.0:
                    intervals.append((0.0, end))
                    intervals.append((start + TAU, TAU))
                elif end >= TAU:
                    intervals.append((0.0, end - TAU))
                    intervals.append((start, TAU))
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
            if len(merged) == 1 and merged[0][0] <= EPSILON and merged[0][1] >= TAU:
                continue
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
                segments.append((active, (start + end) / 2.0))
        if not segments:
            return ()
        last = normalise(node.last_direction)
        segments.sort(
            key=lambda row: (
                -row[0],
                -(math.cos(row[1]) * last[0] + math.sin(row[1]) * last[1]),
                row[1],
            )
        )
        return tuple(
            (math.cos(angle), math.sin(angle)) for _, angle in segments[:limit]
        )

    def _static_food_capture_count(
        self,
        *,
        node: SearchNode,
        foods: tuple[FoodModel, ...],
        direction: tuple[float, float],
        horizon: int,
    ) -> int:
        """Count distinct pellets captured by one free-space constant heading."""

        unit = normalise(direction)
        if unit == (0.0, 0.0) or horizon <= 0:
            return 0
        count = 0
        for food in foods:
            if food.food_id in node.eaten_food_ids:
                continue
            for blob in node.own_blobs:
                travel = player_speed(blob.radius) * horizon
                if (
                    self._point_segment_distance_sq(
                        food.pos[0],
                        food.pos[1],
                        blob.x,
                        blob.y,
                        blob.x + unit[0] * travel,
                        blob.y + unit[1] * travel,
                    )
                    <= blob.radius * blob.radius
                ):
                    count += 1
                    break
        return count

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
        profile_started = self._profile_start()
        proxy_started = self._profile_start()
        proxy_analysis = self._proxy_analysis(
            node=node,
            foods=foods,
            viruses=viruses,
            arena_size=arena_size,
        )
        self._record_profile("proxy", proxy_started)
        node_geometry = self._node_geometry(node)
        own_by_id = {blob.blob_id: blob for blob in node.own_blobs}
        semantic_split_actions: list[Action] = []
        actions: list[Action] = [
            Action(
                node.last_direction,
                reason="keep" if first_step else "continue",
            )
        ]

        escape = proxy_analysis.escape_vector
        escape_actions: tuple[Action, ...] = ()
        if escape != (0.0, 0.0):
            escape_actions = (
                Action(escape, reason="escape"),
                Action(_rotate(escape, math.pi / 8), reason="escape_tangent"),
                Action(_rotate(escape, -math.pi / 8), reason="escape_tangent"),
                Action(_rotate(escape, math.pi / 2), reason="escape_wide_tangent"),
                Action(_rotate(escape, -math.pi / 2), reason="escape_wide_tangent"),
            )
            actions.extend(escape_actions)
        actions.extend(
            Action(_rotate(node.last_direction, angle), reason="steer")
            for angle in (-math.pi / 6, -math.pi / 12, math.pi / 12, math.pi / 6)
        )

        available_food = [
            food for food in foods if food.food_id not in node.eaten_food_ids
        ]
        if available_food:
            nearest_food = min(
                available_food,
                key=lambda food: squared_distance(node_geometry.center, food.pos),
            )
            actions.append(
                Action(
                    normalise(
                        (
                            nearest_food.pos[0] - node_geometry.center[0],
                            nearest_food.pos[1] - node_geometry.center[1],
                        )
                    ),
                    reason="nearest_food",
                )
            )
        for target in food_targets[: 4 if first_step else 2]:
            actions.append(
                Action(
                    normalise(
                        (
                            target[0] - node_geometry.center[0],
                            target[1] - node_geometry.center[1],
                        )
                    ),
                    reason="food_cluster",
                )
            )
        virus_actions = self._proxy_virus_actions(
            proxy_analysis.viruses,
            limit=3 if first_step else 1,
        )
        actions.extend(virus_actions)
        normal_prey_limit = 3 if first_step else 2
        for target in proxy_analysis.prey[:normal_prey_limit]:
            origin = own_by_id.get(target.source_blob_id)
            if origin is not None:
                intercept = self._intercept_direction(origin, target.enemy)
                actions.append(
                    Action(
                        intercept,
                        reason="prey",
                    )
                )
        # Split feasibility is not ordered by the same quantity as ordinary
        # pursuit. A smaller fourth prey can be capturable when the three
        # higher-value prey are too large for the child fragment, so inspect
        # every bounded proxy prey target before selecting semantic splits.
        for target in proxy_analysis.prey:
            origin = own_by_id.get(target.source_blob_id)
            if origin is None:
                continue
            intercept = self._intercept_direction(origin, target.enemy)
            if self._split_can_capture(
                node,
                target.enemy,
                intercept,
                horizon=min(2, self._effective_proxy_dynamic_horizon),
            ):
                semantic_split_actions.append(
                    Action(intercept, split=True, reason="split_prey")
                )

        wall = self._wall_vector(node_geometry.primary, arena_size)
        if wall != (0.0, 0.0):
            actions.append(Action(wall, reason="wall_escape"))
        actions.append(
            Action(
                normalise(
                    (
                        arena_size / 2.0 - node_geometry.center[0],
                        arena_size / 2.0 - node_geometry.center[1],
                    )
                ),
                reason="center",
            )
        )
        if first_step:
            sample_count = max(8, self.angular_samples)
            for index in range(sample_count):
                angle = TAU * ((index + angle_offset) % sample_count) / sample_count
                direction = (math.cos(angle), math.sin(angle))
                reason = (
                    "escape_angle"
                    if escape != (0.0, 0.0)
                    and direction[0] * escape[0] + direction[1] * escape[1] > 0.5
                    else "angle"
                )
                actions.append(Action(direction, reason=reason))

        if first_step and self.proxy_static_food_sweep:
            sweep_directions = self._static_food_sweep_directions(
                node=node,
                foods=foods,
                horizon=self._effective_proxy_horizon,
            )
            if sweep_directions:
                sweep_direction = sweep_directions[0]
                sweep_count = self._static_food_capture_count(
                    node=node,
                    foods=foods,
                    direction=sweep_direction,
                    horizon=self._effective_proxy_horizon,
                )
                existing_count = max(
                    self._static_food_capture_count(
                        node=node,
                        foods=foods,
                        direction=action.direction,
                        horizon=self._effective_proxy_horizon,
                    )
                    for action in actions
                )
                self._static_food_sweep_predicted_advantage = (
                    sweep_count - existing_count
                )
                if sweep_count > existing_count:
                    actions.append(Action(sweep_direction, reason="food_sweep"))
                    self._static_food_sweep_added = True

        # Equivalent physical actions receive one semantic label for telemetry.
        # This ordering does not allocate search slots: every unique action is
        # still ranked by the shared proxy below.  It only prevents a prey or
        # virus route aligned with "keep" from being reported as passive play.
        actions = list(
            self._dedupe_actions(sorted(actions, key=self._proxy_reason_priority))
        )
        if (
            allow_split
            and len(node.own_blobs) < MAX_BLOB_COUNT
            and any(blob.mass >= SPLIT_MIN_MASS for blob in node.own_blobs)
        ):
            if self.proxy_semantic_split:
                escape_action = self._semantic_escape_split(
                    node=node,
                    escape_actions=escape_actions,
                    arena_size=arena_size,
                    analysis=proxy_analysis,
                    geometry=node_geometry,
                )
                if escape_action is not None:
                    semantic_split_actions.append(escape_action)
                farm_action = self._safe_farm_split(
                    node,
                    available_food,
                    food_targets,
                )
                if farm_action is not None:
                    semantic_split_actions.append(farm_action)
                split_extensions = semantic_split_actions
            else:
                split_extensions = [
                    Action(
                        action.direction,
                        split=True,
                        reason=(
                            action.reason if action.split else f"split_{action.reason}"
                        ),
                    )
                    for action in actions
                ]
            actions = list(
                self._dedupe_actions(
                    [
                        *actions,
                        *split_extensions,
                    ]
                )
            )
        configured_refine_limit = (
            self.proxy_refine_limit
            if first_step
            else max(2, self.proxy_refine_limit // 2)
        )
        refine_limit = min(
            configured_refine_limit,
            max(
                self.proxy_min_refine,
                self.proxy_refine_blob_work // max(1, len(node.own_blobs)),
            ),
        )
        if first_step and self._competition_coarse_mode:
            refine_limit = 1
        use_family_refine = self.proxy_family_refine and first_step and len(actions) > 1
        if refine_limit >= len(actions) and not use_family_refine:
            refine_actions = tuple(actions)
            coarse_remainder: tuple[tuple[Action, float], ...] = ()
        else:
            coarse_started = self._profile_start()
            coarse_scored = tuple(
                sorted(
                    (
                        (
                            action,
                            self._coarse_action_value(
                                node=node,
                                action=action,
                                arena_size=arena_size,
                                proxy_analysis=proxy_analysis,
                                node_geometry=node_geometry,
                            ),
                        )
                        for action in actions
                    ),
                    key=lambda item: (-item[1], self._action_key(item[0])),
                )
            )
            self._record_profile("proxy_coarse", coarse_started)
            if use_family_refine:
                refine_actions, coarse_remainder = self._family_refine_partition(
                    coarse_scored,
                    limit=min(refine_limit, self.proxy_family_refine_limit),
                )
            else:
                refine_actions = tuple(
                    action for action, _ in coarse_scored[:refine_limit]
                )
                coarse_remainder = coarse_scored[refine_limit:]
        refine_count = len(refine_actions)
        refine_started = self._profile_start()
        refined = sorted(
            (
                (
                    action,
                    self._approximate_action_value(
                        node=node,
                        action=action,
                        foods=foods,
                        viruses=viruses,
                        arena_size=arena_size,
                        proxy_analysis=proxy_analysis,
                    ),
                )
                for action in refine_actions
            ),
            key=lambda item: (-item[1], self._action_key(item[0])),
        )
        self._record_profile("proxy_refine", refine_started)
        if (
            first_step
            and coarse_remainder
            and self._proxy_rank_audit_every_n > 0
            and self._current_round % self._proxy_rank_audit_every_n == 0
        ):
            audit_started = perf_counter()
            audit_remainder = [
                (
                    action,
                    self._approximate_action_value(
                        node=node,
                        action=action,
                        foods=foods,
                        viruses=viruses,
                        arena_size=arena_size,
                        proxy_analysis=proxy_analysis,
                    ),
                )
                for action, _ in coarse_remainder
            ]
            coarse_order = tuple(action for action in refine_actions) + tuple(
                action for action, _ in coarse_remainder
            )
            best_action, best_full_score = min(
                (*refined, *audit_remainder),
                key=lambda item: (-item[1], self._action_key(item[0])),
            )
            best_key = self._action_key(best_action)
            best_coarse_rank = next(
                index
                for index, action in enumerate(coarse_order, start=1)
                if self._action_key(action) == best_key
            )
            production_best_score = refined[0][1]
            self._proxy_rank_audit_samples += 1
            self._proxy_rank_audit_hits += best_coarse_rank <= refine_limit
            self._proxy_rank_audit_last_best_coarse_rank = best_coarse_rank
            self._proxy_rank_audit_last_best_reason = best_action.reason
            self._proxy_rank_audit_last_best_split = best_action.split
            self._proxy_rank_audit_last_regret = best_full_score - production_best_score
            self._record_profile_elapsed(
                "proxy_rank_audit",
                perf_counter() - audit_started,
            )
        scored = tuple([*refined, *coarse_remainder])
        if first_step:
            family_counts: dict[str, int] = {}
            for action, _ in scored:
                family = self._action_family(action)
                family_counts[family] = family_counts.get(family, 0) + 1
            # Refined and coarse values are intentionally on different
            # approximations. Only refined scores may drive fallback choice,
            # score gaps, shadow ranks, or diagnostics.
            self._root_proxy_scores = tuple(refined)
            self._root_candidate_families = family_counts
            self._root_proxy_refined = refine_count
        result = tuple(action for action, _ in scored)
        self._record_profile("candidate", profile_started)
        return result

    @staticmethod
    def _action_family(action: Action) -> str:
        reason = action.reason.removeprefix("split_")
        if action.split and "prey" in reason:
            return "prey_split"
        if action.split and ("food" in reason or "farm" in reason):
            return "resource_split"
        if action.split:
            return "split_event"
        if "escape" in reason:
            return "escape"
        if "virus" in reason:
            return "virus"
        if "prey" in reason:
            return "prey"
        if "food" in reason or "farm" in reason:
            return "resource"
        if reason in {"keep", "continue"}:
            return "baseline"
        if "wall" in reason or reason == "center":
            return "position"
        return "explore"

    def _semantic_escape_split(
        self,
        *,
        node: SearchNode,
        escape_actions: tuple[Action, ...],
        arena_size: float,
        analysis: ProxyAnalysis,
        geometry: NodeGeometry,
    ) -> Action | None:
        """Return one recovery split only when no-split cannot retain as much.

        This is a viability recovery option, not a mobility bonus.  The same
        one-round reachable threat model evaluates every ordinary escape and
        its split counterpart.  If any ordinary escape retains all mass, or
        splitting does not improve the best retained-mass lower estimate, no
        escape split enters the candidate set.
        """

        if not escape_actions:
            return None
        total_mass = sum(blob.mass for blob in node.own_blobs)
        if total_mass <= EPSILON:
            return None

        def retained_mass(action: Action) -> float:
            return self._coarse_projected_safety(
                node=node,
                action=action,
                arena_size=arena_size,
                horizon=1,
                analysis=analysis,
                geometry=geometry,
            )[3]

        best_no_split = max(retained_mass(action) for action in escape_actions)
        if best_no_split >= total_mass - EPSILON:
            return None

        split_rows = tuple(
            (
                retained_mass(
                    Action(
                        action.direction,
                        split=True,
                        reason="split_escape",
                    )
                ),
                action.direction,
            )
            for action in escape_actions
        )
        best_split_mass, best_direction = max(split_rows, key=lambda row: row[0])
        if best_split_mass <= best_no_split + EPSILON:
            return None
        return Action(best_direction, split=True, reason="split_escape")

    def _family_refine_partition(
        self,
        coarse_scored: tuple[tuple[Action, float], ...],
        *,
        limit: int,
    ) -> tuple[tuple[Action, ...], tuple[tuple[Action, float], ...]]:
        """Allocate refine work between events and no-split direction coverage.

        Split capture/farming are discontinuous events and receive their own
        representatives.  Two slots are reserved for no-split angular coverage
        so a full set of semantic families can never crowd out the vertical
        escape in a symmetric pincer.  Coverage only proposes directions; each
        selected action still passes through the detailed geometric evaluator.
        """

        if not coarse_scored or limit <= 0:
            return (), coarse_scored

        by_family: dict[str, list[int]] = {}
        for index, (action, _) in enumerate(coarse_scored):
            by_family.setdefault(self._action_family(action), []).append(index)

        target_count = min(limit, len(coarse_scored))
        no_split_indices = tuple(
            index for index, (action, _) in enumerate(coarse_scored) if not action.split
        )
        coverage_slots = min(2, max(0, len(no_split_indices) - 1), target_count)
        semantic_budget = target_count - coverage_slots
        selected: set[int] = set()

        for family in ("baseline", "escape"):
            indices = by_family.get(family, ())
            if indices and len(selected) < semantic_budget:
                selected.add(indices[0])

        event_representatives = sorted(
            by_family[family][0]
            for family in (
                "prey_split",
                "resource_split",
                "prey",
                "virus",
                "resource",
            )
            if by_family.get(family)
        )
        for index in event_representatives:
            if len(selected) >= semantic_budget:
                break
            selected.add(index)

        unit_directions = tuple(
            normalise(action.direction) for action, _ in coarse_scored
        )

        def add_most_separated_no_split() -> bool:
            remaining = [index for index in no_split_indices if index not in selected]
            if not remaining:
                return False
            selected_no_split = [
                index for index in selected if not coarse_scored[index][0].split
            ]
            if not selected_no_split:
                selected.add(remaining[0])
                return True

            def angular_separation(index: int) -> tuple[float, int]:
                direction = unit_directions[index]
                minimum = min(
                    1.0
                    - _clamp(
                        direction[0] * unit_directions[chosen][0]
                        + direction[1] * unit_directions[chosen][1],
                        -1.0,
                        1.0,
                    )
                    for chosen in selected_no_split
                )
                return (minimum, -index)

            selected.add(max(remaining, key=angular_separation))
            return True

        for _ in range(coverage_slots):
            if len(selected) >= target_count or not add_most_separated_no_split():
                break

        # Sparse scenes may expose fewer semantic events than their allocation.
        # Continue widening with physical no-split directions before falling
        # back to coarse order for any final event slots.
        while len(selected) < target_count and add_most_separated_no_split():
            pass
        for index in range(len(coarse_scored)):
            if len(selected) >= target_count:
                break
            selected.add(index)

        selected_indices = tuple(sorted(selected))
        return (
            tuple(coarse_scored[index][0] for index in selected_indices),
            tuple(
                item
                for index, item in enumerate(coarse_scored)
                if index not in selected
            ),
        )

    @staticmethod
    def _proxy_reason_priority(action: Action) -> tuple[int, str]:
        reason = action.reason.removeprefix("split_")
        if "prey" in reason:
            return (0, reason)
        if "virus" in reason:
            return (1, reason)
        if "escape" in reason:
            return (2, reason)
        if "food" in reason or "farm" in reason:
            return (3, reason)
        if "wall" in reason or reason == "center":
            return (4, reason)
        if reason in {"keep", "continue"}:
            return (5, reason)
        return (6, reason)

    @staticmethod
    def _threat_action_radii(
        threat: ProxyThreat,
        projected_radius: float,
    ) -> tuple[float | None, float | None]:
        """Return exclusive ordinary/split-attack envelopes for one shape."""

        if projected_radius < threat.source_radius - EPSILON:
            return (
                threat.split_ordinary_radius,
                threat.split_split_attack_radius,
            )
        return (
            threat.normal_ordinary_radius,
            threat.normal_split_attack_radius,
        )

    @staticmethod
    def _exclusive_threat_margin(
        *,
        distance: float,
        ordinary_radius: float | None,
        split_attack_radius: float | None,
        ordinary_reach: float = 0.0,
    ) -> float:
        """Combine mutually exclusive ordinary and split enemy actions."""

        margins = []
        if ordinary_radius is not None:
            margins.append(distance - ordinary_radius - ordinary_reach)
        if split_attack_radius is not None:
            margins.append(distance - split_attack_radius)
        return min(margins, default=math.inf)

    def _node_geometry(self, node: SearchNode) -> NodeGeometry:
        cached = self._node_geometry_cache.get(id(node))
        if cached is not None and cached[0] is node:
            self._record_cache_access("node", hit=True)
            return cached[1]
        geometry = NodeGeometry(
            total_mass=node.total_mass,
            center=node.center,
            primary=node.primary,
        )
        self._node_geometry_cache[id(node)] = (node, geometry)
        self._record_cache_access("node", hit=False)
        return geometry

    @staticmethod
    def _formation_cohesion_mass(
        blobs: tuple[OwnBlob, ...]
        | tuple[ProxyBlobMotion, ...]
        | list[ProxyBlobMotion],
        cooldowns: tuple[int, ...],
    ) -> float:
        """Return satellite mass recoverable through a compact formation.

        A fragmented player is not just a bag of independent survival
        probabilities: all pieces share one control direction and eventually
        merge.  This term measures how much mass outside the largest piece is
        already close enough to function as one defensive group, discounted by
        the remaining merge cooldown.
        """

        if len(blobs) <= 1:
            return 0.0
        masses = tuple(blob.radius * blob.radius for blob in blobs)
        total_mass = sum(masses)
        if total_mass <= EPSILON:
            return 0.0
        center_x = sum(blob.x * mass for blob, mass in zip(blobs, masses)) / total_mass
        center_y = sum(blob.y * mass for blob, mass in zip(blobs, masses)) / total_mass
        surface_variance = (
            sum(
                mass
                * max(
                    0.0,
                    math.hypot(blob.x - center_x, blob.y - center_y) - blob.radius,
                )
                ** 2
                for blob, mass in zip(blobs, masses)
            )
            / total_mass
        )
        surface_spread = math.sqrt(surface_variance)
        formation_radius = math.sqrt(total_mass)
        compactness = formation_radius / (formation_radius + surface_spread)
        remaining_cooldown = max(cooldowns, default=0)
        merge_readiness = math.exp(-remaining_cooldown / max(1, SPLIT_COOLDOWN_FRAMES))
        satellite_mass = total_mass - max(masses)
        return satellite_mass * compactness * merge_readiness

    def _projected_formation_cohesion_mass(
        self,
        *,
        node: SearchNode,
        blobs: tuple[ProxyBlobMotion, ...] | list[ProxyBlobMotion],
        projection_steps: int,
    ) -> float:
        if not self.proxy_cohesion_enabled:
            return 0.0
        original_by_id = {blob.blob_id: blob for blob in node.own_blobs}
        source_counts: dict[int, int] = {}
        for blob in blobs:
            source_counts[blob.source_blob_id] = (
                source_counts.get(blob.source_blob_id, 0) + 1
            )
        cooldowns = tuple(
            SPLIT_COOLDOWN_FRAMES
            if source_counts.get(blob.source_blob_id, 0) > 1
            else max(
                0,
                original_by_id[blob.source_blob_id].merge_cooldown - projection_steps,
            )
            for blob in blobs
        )
        return self._formation_cohesion_mass(blobs, cooldowns)

    def _virus_expected_mass(
        self,
        node: SearchNode,
        virus: VirusModel,
        arena_size: float,
    ) -> float | None:
        """Return the best expected mass for one virus in exact search."""

        cache_key = (id(node), virus.virus_id, arena_size)
        cached = self._virus_expected_mass_cache.get(cache_key)
        if cached is not None and cached[0] is node and cached[1] is virus:
            self._record_cache_access("virus", hit=True)
            return cached[2]

        best: float | None = None
        for origin in node.own_blobs:
            if not self._can_still_consume_virus_at_contact(origin, virus):
                continue
            gap = max(0.0, math.dist(origin.pos, virus.pos) - origin.radius)
            retention = self._virus_retained_mass_fraction(
                node,
                origin,
                virus,
                arena_size,
            )
            expected_mass = (
                virus.radius
                * virus.radius
                * retention
                * math.exp(-gap / self._VIRUS_POTENTIAL_HORIZON)
            )
            if best is None or expected_mass > best:
                best = expected_mass

        self._virus_expected_mass_cache[cache_key] = (node, virus, best)
        self._record_cache_access("virus", hit=False)
        return best

    def _approximate_action_value(
        self,
        *,
        node: SearchNode,
        action: Action,
        foods: tuple[FoodModel, ...],
        viruses: tuple[VirusModel, ...] = (),
        arena_size: float,
        proxy_analysis: ProxyAnalysis | None = None,
    ) -> float:
        """Rank an action with cheap physics and the shared state value.

        The old proxy multiplied one local gradient by a one-turn displacement.
        That was fast, but it could not express a wall capture, a split speedup,
        or a threat crossed halfway through a move.  This evaluator projects a
        short linear trajectory, measures segment intersections, and compares
        the same safe-mass/opportunity/rival value used by exact beam nodes.
        """

        direction = normalise(action.direction)
        analysis = proxy_analysis or self._proxy_analysis(
            node=node,
            foods=foods,
            viruses=viruses,
            arena_size=arena_size,
        )
        self._record_profile_count("proxy_actions_evaluated")
        project_started = self._profile_start()
        movement = self._proxy_project_action(
            node=node,
            action=action,
            arena_size=arena_size,
            horizon=self._effective_proxy_horizon,
            unit=direction,
            own_sources=analysis.own_sources,
            motion_templates=(
                analysis.split_motion_templates
                if action.split
                else analysis.normal_motion_templates
            ),
        )
        self._record_profile("proxy_project_action", project_started)
        self._record_profile_count("proxy_projected_blobs", len(movement.blobs))
        enemy_started = self._profile_start()
        enemy_motions = self._proxy_enemy_motions(
            analysis.motion_enemies,
            movement.blobs,
            horizon=self._effective_proxy_horizon,
            arena_size=arena_size,
            enemy_speeds=analysis.enemy_speeds,
            observed_directions=analysis.observed_enemy_directions,
            observed_weights=analysis.observed_enemy_weights,
            hunter_masks=(
                analysis.split_hunter_masks
                if action.split
                else analysis.normal_hunter_masks
            ),
            predator_masks=(
                analysis.split_predator_masks
                if action.split
                else analysis.normal_predator_masks
            ),
        )
        self._record_profile("proxy_enemy_motions", enemy_started)
        self._record_profile_count("proxy_projected_enemies", len(enemy_motions))
        value_started = self._profile_start()
        projected = self._proxy_state_value(
            node=node,
            blobs=movement.blobs,
            enemy_motions=enemy_motions,
            foods=analysis.foods,
            viruses=analysis.viruses,
            prey=analysis.prey,
            threats_by_blob=(
                analysis.split_threats_by_blob
                if action.split
                else analysis.normal_threats_by_blob
            ),
            competitor_mass_debt=analysis.competitor_mass_debt,
            split_motion=action.split,
            arena_size=arena_size,
            dynamic_fraction=(
                self._effective_proxy_dynamic_horizon / self._effective_proxy_horizon
            ),
            dynamic_steps=self._effective_proxy_dynamic_horizon,
            projection_steps=self._effective_proxy_horizon,
        )
        self._record_profile("proxy_state_value", value_started)
        return (
            (projected.total - analysis.baseline.total) * analysis.future_weight
            - (1.0 - movement.efficiency)
            * self._MOVEMENT_INEFFICIENCY_PENALTY
            * analysis.discount_sum
            - self._turn_cost(node.last_direction, direction) * 0.6
        )

    def _coarse_action_value(
        self,
        *,
        node: SearchNode,
        action: Action,
        arena_size: float,
        proxy_analysis: ProxyAnalysis,
        node_geometry: NodeGeometry,
    ) -> float:
        """Bounded first-stage score for every generated action."""

        direction = normalise(action.direction)
        threats_by_blob = (
            proxy_analysis.split_threats_by_blob
            if action.split
            else proxy_analysis.normal_threats_by_blob
        )
        if any(threats_by_blob) and (
            len(node.own_blobs) >= 8 or self.proxy_family_refine
        ):
            (
                displacement_x,
                displacement_y,
                efficiency,
                projected_safe_mass,
                projected_cohesion_mass,
            ) = self._coarse_projected_safety(
                node=node,
                action=action,
                arena_size=arena_size,
                horizon=self._effective_proxy_horizon,
                analysis=proxy_analysis,
                geometry=node_geometry,
            )
            safety_delta = 100.0 * (
                projected_safe_mass - proxy_analysis.baseline.safe_mass
            )
            cohesion_delta = 100.0 * (
                projected_cohesion_mass - proxy_analysis.baseline.cohesion_mass
            )
        else:
            displacement_x, displacement_y, efficiency = self._coarse_proxy_movement(
                action=action,
                arena_size=arena_size,
                horizon=self._effective_proxy_horizon,
                geometry=node_geometry,
                kinematics=proxy_analysis.kinematics,
            )
            safety_gradient_x, safety_gradient_y = proxy_analysis.safety_gradient
            safety_delta = (
                safety_gradient_x * displacement_x + safety_gradient_y * displacement_y
            )
            cohesion_delta = 0.0
        displacement_scale = proxy_analysis.future_weight
        if self._proxy_has_future_after_horizon:
            projected_opportunity_values = [
                max(
                    0.0,
                    opportunity.value
                    + opportunity.gradient_x * displacement_x
                    + opportunity.gradient_y * displacement_y,
                )
                for opportunity in proxy_analysis.coarse_opportunities
            ]
            projected_opportunity_values.extend(
                self._coarse_prey_opportunity_values(
                    action=action,
                    arena_size=arena_size,
                    analysis=proxy_analysis,
                )
            )
            projected_opportunity_values.sort(reverse=True)
            projected_opportunity = sum(
                value * weight
                for value, weight in zip(
                    projected_opportunity_values[:3],
                    (1.0, 0.25, 0.1),
                    strict=False,
                )
            )
        else:
            projected_opportunity = 0.0
        coarse_value_delta = (
            safety_delta
            + cohesion_delta
            + projected_opportunity
            - proxy_analysis.coarse_opportunity_baseline
        )
        return (
            coarse_value_delta * displacement_scale
            + (
                proxy_analysis.split_state_delta * displacement_scale
                if action.split
                else 0.0
            )
            - (1.0 - efficiency)
            * self._MOVEMENT_INEFFICIENCY_PENALTY
            * proxy_analysis.discount_sum
            - self._turn_cost(node.last_direction, direction) * 0.6
        )

    def _coarse_projected_safety(
        self,
        *,
        node: SearchNode,
        action: Action,
        arena_size: float,
        horizon: int,
        analysis: ProxyAnalysis,
        geometry: NodeGeometry,
    ) -> tuple[float, float, float, float, float]:
        """Project per-fragment safety without the full opportunity rollout.

        A center-only gradient cannot rank fragmented states near a wall: two
        directions with the same center displacement may save entirely
        different fragments.  This middle stage projects every own fragment
        and its small, preselected threat set.  Enemy motion is approximated
        directly per threatened fragment, avoiding the full enemy/prey/food/
        virus state evaluation while preserving the important nonlinear loss.
        """

        movement = self._proxy_project_action(
            node=node,
            action=action,
            arena_size=arena_size,
            horizon=horizon,
            unit=normalise(action.direction),
            own_sources=analysis.own_sources,
            motion_templates=(
                (
                    analysis.split_motion_templates
                    if action.split
                    else analysis.normal_motion_templates
                )
                if horizon == self._effective_proxy_horizon
                else None
            ),
        )
        blobs = movement.blobs
        total_mass = sum(blob.mass for blob in blobs)
        if total_mass <= EPSILON:
            return (0.0, 0.0, movement.efficiency, 0.0, 0.0)
        projected_center_x = sum(blob.x * blob.mass for blob in blobs) / total_mass
        projected_center_y = sum(blob.y * blob.mass for blob in blobs) / total_mass
        threats_by_blob = (
            analysis.split_threats_by_blob
            if action.split
            else analysis.normal_threats_by_blob
        )
        survival_midpoint = (
            self.survival_midpoint_base
            + self.survival_midpoint_scale * self._proxy_safety_weight
        )
        temperature = max(self.survival_temperature, 0.1)
        dynamic_fraction = min(
            1.0,
            self._effective_proxy_dynamic_horizon / max(1, horizon),
        )
        safe_mass = 0.0
        coherent_survivals = (
            self._coherent_reachable_survivals(
                blobs=blobs,
                threats_by_blob=threats_by_blob,
                enemy_speeds=analysis.enemy_speeds,
                arena_size=arena_size,
                dynamic_fraction=dynamic_fraction,
                dynamic_steps=self._effective_proxy_dynamic_horizon,
            )
            if self.proxy_reachable_threats and self.proxy_coherent_threat_scenarios
            else None
        )
        for blob_index, blob in enumerate(blobs):
            if coherent_survivals is not None:
                safe_mass += blob.mass * coherent_survivals[blob_index]
                continue
            threats = threats_by_blob[blob_index]
            if not threats:
                safe_mass += blob.mass
                continue
            worst_margin = math.inf
            for threat in threats:
                enemy = threat.enemy
                motion_index = threat.motion_index
                ordinary_radius, split_attack_radius = self._threat_action_radii(
                    threat,
                    blob.radius,
                )
                if ordinary_radius is None and split_attack_radius is None:
                    continue
                if self.proxy_reachable_threats:
                    for step in range(self._effective_proxy_dynamic_horizon + 1):
                        own_x, own_y = self._proxy_short_position(
                            blob,
                            step=step,
                            step_count=self._effective_proxy_dynamic_horizon,
                            fallback_fraction=dynamic_fraction,
                        )
                        distance = math.hypot(own_x - enemy.x, own_y - enemy.y)
                        action_margin = self._exclusive_threat_margin(
                            distance=distance,
                            ordinary_radius=ordinary_radius,
                            split_attack_radius=(
                                split_attack_radius if step > 0 else None
                            ),
                            ordinary_reach=(analysis.enemy_speeds[motion_index] * step),
                        )
                        margin = (
                            action_margin
                            - enemy.stale_rounds * 0.35
                            - self._wall_trap_factor_at(
                                own_x,
                                own_y,
                                blob.radius,
                                enemy,
                                arena_size,
                            )
                            * 4.0
                        )
                        worst_margin = min(worst_margin, margin)
                else:
                    own_x, own_y = self._proxy_short_position(
                        blob,
                        fallback_fraction=dynamic_fraction,
                    )
                    observed = analysis.observed_enemy_directions[motion_index]
                    observed_weight = analysis.observed_enemy_weights[motion_index]
                    chase = normalise((own_x - enemy.x, own_y - enemy.y))
                    direction = normalise(
                        (
                            chase[0] * (1.0 - observed_weight)
                            + observed[0] * observed_weight,
                            chase[1] * (1.0 - observed_weight)
                            + observed[1] * observed_weight,
                        )
                    )
                    enemy_distance = (
                        analysis.enemy_speeds[motion_index]
                        * self._effective_proxy_dynamic_horizon
                    )
                    enemy_x = _clamp(
                        enemy.x + direction[0] * enemy_distance,
                        enemy.radius,
                        arena_size - enemy.radius,
                    )
                    enemy_y = _clamp(
                        enemy.y + direction[1] * enemy_distance,
                        enemy.radius,
                        arena_size - enemy.radius,
                    )
                    closest = self._relative_segment_distance(
                        blob.start_x,
                        blob.start_y,
                        own_x,
                        own_y,
                        enemy.x,
                        enemy.y,
                        enemy_x,
                        enemy_y,
                    )
                    action_margin = self._exclusive_threat_margin(
                        distance=closest,
                        ordinary_radius=ordinary_radius,
                        split_attack_radius=None,
                    )
                    if split_attack_radius is not None:
                        split_closest = math.hypot(
                            own_x - enemy.x,
                            own_y - enemy.y,
                        )
                        action_margin = min(
                            action_margin,
                            self._exclusive_threat_margin(
                                distance=split_closest,
                                ordinary_radius=None,
                                split_attack_radius=split_attack_radius,
                            ),
                        )
                    margin = (
                        action_margin
                        - enemy.stale_rounds * 0.35
                        - self._wall_trap_factor_at(
                            own_x,
                            own_y,
                            blob.radius,
                            enemy,
                            arena_size,
                        )
                        * 4.0
                    )
                    worst_margin = min(worst_margin, margin)
            if worst_margin == math.inf:
                survival = 1.0
            else:
                scaled = _clamp(
                    (worst_margin - survival_midpoint) / temperature,
                    -40.0,
                    40.0,
                )
                survival = 1.0 / (1.0 + math.exp(-scaled))
            safe_mass += blob.mass * survival
        return (
            projected_center_x - geometry.center[0],
            projected_center_y - geometry.center[1],
            movement.efficiency,
            safe_mass,
            self._projected_formation_cohesion_mass(
                node=node,
                blobs=blobs,
                projection_steps=horizon,
            ),
        )

    def _coarse_prey_opportunity_values(
        self,
        *,
        action: Action,
        arena_size: float,
        analysis: ProxyAnalysis,
    ) -> tuple[float, ...]:
        """Project sparse prey values without constructing a full proxy state."""

        if not analysis.prey:
            return ()
        projected_values: list[float] = []
        for target in analysis.prey:
            source = analysis.own_source_by_id.get(target.source_blob_id)
            if source is None:
                continue
            enemy = target.enemy
            paths = self._coarse_source_paths(
                source=source,
                action=action,
                arena_size=arena_size,
                horizon=self._effective_proxy_horizon,
            )
            edible_paths = tuple(
                path for path in paths if can_eat_player_blob(path[4], enemy.radius)
            )
            if not edible_paths:
                continue
            hunter = min(
                edible_paths,
                key=lambda path: (enemy.x - path[2]) ** 2 + (enemy.y - path[3]) ** 2,
            )
            observed = normalise(enemy.direction)
            flee = normalise((enemy.x - hunter[2], enemy.y - hunter[3]))
            enemy_direction = normalise(
                (
                    flee[0] * 0.62 + observed[0] * 0.38,
                    flee[1] * 0.62 + observed[1] * 0.38,
                )
            )
            enemy_speed = (
                0.0
                if self.proxy_enemy_motion == "threat_reachable"
                else analysis.enemy_speed_by_key.get(
                    enemy.key,
                    player_speed(enemy.radius),
                )
            )
            enemy_distance = enemy_speed
            enemy_distance *= self._effective_proxy_horizon
            enemy_end_x = _clamp(
                enemy.x + enemy_direction[0] * enemy_distance,
                enemy.radius,
                arena_size - enemy.radius,
            )
            enemy_end_y = _clamp(
                enemy.y + enemy_direction[1] * enemy_distance,
                enemy.radius,
                arena_size - enemy.radius,
            )
            rival_value = self._rival_values.get(enemy.player_id, 0.0)
            best_probability = 0.0
            for path in edible_paths:
                motion = ProxyEnemyMotion(
                    enemy=enemy,
                    x=enemy_end_x,
                    y=enemy_end_y,
                    direction=enemy_direction,
                    speed=enemy_speed,
                )
                blob = ProxyBlobMotion(
                    blob_id=source.blob.blob_id,
                    source_blob_id=source.blob.blob_id,
                    start_x=path[0],
                    start_y=path[1],
                    x=path[2],
                    y=path[3],
                    radius=path[4],
                    speed=path[5],
                )
                if (
                    self._relative_segment_distance(
                        path[0],
                        path[1],
                        path[2],
                        path[3],
                        enemy.x,
                        enemy.y,
                        enemy_end_x,
                        enemy_end_y,
                    )
                    <= path[4]
                ):
                    best_probability = 1.0
                    break
                best_probability = max(
                    best_probability,
                    self._proxy_prey_capture_probability(
                        blob,
                        motion,
                        arena_size,
                    ),
                )
            projected_values.append(
                100.0 * best_probability * enemy.mass * (1.0 + rival_value)
            )
        return tuple(projected_values)

    @staticmethod
    def _coarse_source_paths(
        *,
        source: ProxyOwnSource,
        action: Action,
        arena_size: float,
        horizon: int,
    ) -> tuple[tuple[float, float, float, float, float, float], ...]:
        """Return start/end/radius tuples for one source under cheap physics."""

        direction = normalise(action.direction)
        horizon = max(1, horizon)
        drag_sum = (
            float(horizon)
            if SPLIT_EJECT_DRAG >= 1.0 - EPSILON
            else (1.0 - SPLIT_EJECT_DRAG**horizon)
            / max(1.0 - SPLIT_EJECT_DRAG, EPSILON)
        )
        own = source.blob
        rows: list[tuple[float, float, float, float, float, float]] = []
        if action.split and source.split_radius is not None:
            assert source.split_speed is not None
            radius = source.split_radius
            rows.append(
                (
                    own.x,
                    own.y,
                    _clamp(
                        own.x
                        + direction[0] * source.split_speed * horizon
                        + own.eject_vx * drag_sum,
                        radius,
                        arena_size - radius,
                    ),
                    _clamp(
                        own.y
                        + direction[1] * source.split_speed * horizon
                        + own.eject_vy * drag_sum,
                        radius,
                        arena_size - radius,
                    ),
                    radius,
                    source.split_speed,
                )
            )
            child_x = _clamp(
                own.x + direction[0] * (2.0 * radius + SAME_PLAYER_OVERLAP_EPSILON),
                radius,
                arena_size - radius,
            )
            child_y = _clamp(
                own.y + direction[1] * (2.0 * radius + SAME_PLAYER_OVERLAP_EPSILON),
                radius,
                arena_size - radius,
            )
            rows.append(
                (
                    child_x,
                    child_y,
                    _clamp(
                        child_x
                        + direction[0]
                        * (source.split_speed * horizon + SPLIT_EJECT_SPEED * drag_sum),
                        radius,
                        arena_size - radius,
                    ),
                    _clamp(
                        child_y
                        + direction[1]
                        * (source.split_speed * horizon + SPLIT_EJECT_SPEED * drag_sum),
                        radius,
                        arena_size - radius,
                    ),
                    radius,
                    source.split_speed,
                )
            )
            return tuple(rows)

        radius = own.radius
        rows.append(
            (
                own.x,
                own.y,
                _clamp(
                    own.x
                    + direction[0] * source.speed * horizon
                    + own.eject_vx * drag_sum,
                    radius,
                    arena_size - radius,
                ),
                _clamp(
                    own.y
                    + direction[1] * source.speed * horizon
                    + own.eject_vy * drag_sum,
                    radius,
                    arena_size - radius,
                ),
                radius,
                source.speed,
            )
        )
        return tuple(rows)

    def _proxy_kinematics(
        self,
        node: SearchNode,
        geometry: NodeGeometry,
        *,
        own_sources: tuple[ProxyOwnSource, ...] | None = None,
    ) -> ProxyKinematics:
        """Pre-aggregate constant motion coefficients for O(1) ranking."""

        total_mass = max(geometry.total_mass, EPSILON)
        normal_speed = 0.0
        normal_eject_x = 0.0
        normal_eject_y = 0.0
        split_speed = 0.0
        split_static_eject_x = 0.0
        split_static_eject_y = 0.0
        split_directional_eject = 0.0
        split_placement = 0.0
        split_radius = 0.0
        if own_sources is None:
            remaining_slots = MAX_BLOB_COUNT - len(node.own_blobs)
            generated: list[ProxyOwnSource] = []
            for own in sorted(node.own_blobs, key=lambda blob: blob.blob_id):
                if remaining_slots > 0 and own.mass >= SPLIT_MIN_MASS:
                    source_split_radius = own.radius / SQRT2
                    source_split_speed = player_speed(source_split_radius)
                    remaining_slots -= 1
                else:
                    source_split_radius = None
                    source_split_speed = None
                generated.append(
                    ProxyOwnSource(
                        blob=own,
                        speed=player_speed(own.radius),
                        split_radius=source_split_radius,
                        split_speed=source_split_speed,
                    )
                )
            own_sources = tuple(generated)
        for source in own_sources:
            own = source.blob
            normal_speed += own.mass * source.speed
            normal_eject_x += own.mass * own.eject_vx
            normal_eject_y += own.mass * own.eject_vy
            if source.split_radius is not None:
                child_radius = source.split_radius
                assert source.split_speed is not None
                child_mass = own.mass / 2.0
                split_speed += own.mass * source.split_speed
                split_static_eject_x += child_mass * own.eject_vx
                split_static_eject_y += child_mass * own.eject_vy
                split_directional_eject += child_mass * SPLIT_EJECT_SPEED
                split_placement += child_mass * (
                    2.0 * child_radius + SAME_PLAYER_OVERLAP_EPSILON
                )
                split_radius = max(split_radius, child_radius)
            else:
                split_speed += own.mass * source.speed
                split_static_eject_x += own.mass * own.eject_vx
                split_static_eject_y += own.mass * own.eject_vy
                split_radius = max(split_radius, own.radius)
        return ProxyKinematics(
            normal_speed=normal_speed / total_mass,
            normal_eject_x=normal_eject_x / total_mass,
            normal_eject_y=normal_eject_y / total_mass,
            normal_radius=geometry.primary.radius,
            split_speed=split_speed / total_mass,
            split_static_eject_x=split_static_eject_x / total_mass,
            split_static_eject_y=split_static_eject_y / total_mass,
            split_directional_eject=split_directional_eject / total_mass,
            split_placement=split_placement / total_mass,
            split_radius=max(split_radius, EPSILON),
        )

    def _proxy_split_state_delta(
        self,
        *,
        node: SearchNode,
        threats: tuple[ProxyThreat, ...],
        prey_rows: tuple[tuple[float, ProxyPreyTarget], ...],
        viruses: tuple[ProxyVirusTarget, ...],
        arena_size: float,
    ) -> float:
        """Approximate the action-independent value lost or gained by splitting."""

        threats_by_source: dict[int, list[EnemyBlob]] = {}
        for target in threats:
            threats_by_source.setdefault(target.source_blob_id, []).append(target.enemy)
        prey_by_source: dict[int, list[tuple[float, EnemyBlob]]] = {}
        for expected_mass, target in prey_rows:
            prey_by_source.setdefault(target.source_blob_id, []).append(
                (expected_mass, target.enemy)
            )
        viruses_by_source: dict[int, list[VirusModel]] = {}
        for target in viruses:
            viruses_by_source.setdefault(target.source_blob_id, []).append(target.virus)

        survival_midpoint = (
            self.survival_midpoint_base
            + self.survival_midpoint_scale * self._proxy_safety_weight
        )
        temperature = max(self.survival_temperature, 0.1)

        def survival(own: OwnBlob, radius: float) -> float:
            probability = 1.0
            for enemy in threats_by_source.get(own.blob_id, ()):
                if not can_eat_player_blob(enemy.radius, radius):
                    continue
                danger_radius = enemy.radius
                if _can_split_eat(enemy.radius, radius):
                    danger_radius = max(
                        danger_radius,
                        _split_attack_reach(enemy.radius),
                    )
                margin = (
                    math.dist(own.pos, enemy.pos)
                    - danger_radius
                    - enemy.stale_rounds * 0.35
                    - self._wall_trap_factor_at(
                        own.x,
                        own.y,
                        radius,
                        enemy,
                        arena_size,
                    )
                    * 4.0
                )
                scaled = _clamp(
                    (margin - survival_midpoint) / temperature,
                    -40.0,
                    40.0,
                )
                probability = min(
                    probability,
                    1.0 / (1.0 + math.exp(-scaled)),
                )
            return probability

        safe_mass_delta = 0.0
        opportunity_delta = 0.0
        remaining_slots = MAX_BLOB_COUNT - len(node.own_blobs)
        for own in sorted(node.own_blobs, key=lambda blob: blob.blob_id):
            if remaining_slots <= 0 or own.mass < SPLIT_MIN_MASS:
                continue
            child_radius = own.radius / SQRT2
            safe_mass_delta += own.mass * (
                survival(own, child_radius) - survival(own, own.radius)
            )
            for expected_mass, enemy in prey_by_source.get(own.blob_id, ()):
                if not can_eat_player_blob(child_radius, enemy.radius):
                    opportunity_delta -= expected_mass
            for virus in viruses_by_source.get(own.blob_id, ()):
                if _can_consume_virus(child_radius, virus.radius):
                    continue
                gap = max(0.0, math.dist(own.pos, virus.pos) - own.radius)
                opportunity_delta -= (
                    virus.radius
                    * virus.radius
                    * math.exp(-gap / self._VIRUS_POTENTIAL_HORIZON)
                )
            remaining_slots -= 1
        return 100.0 * (safe_mass_delta + opportunity_delta)

    @staticmethod
    def _coarse_proxy_movement(
        *,
        action: Action,
        arena_size: float,
        horizon: int,
        geometry: NodeGeometry,
        kinematics: ProxyKinematics,
    ) -> tuple[float, float, float]:
        direction = normalise(action.direction)
        horizon = max(1, horizon)
        drag_sum = (
            float(horizon)
            if SPLIT_EJECT_DRAG >= 1.0 - EPSILON
            else (1.0 - SPLIT_EJECT_DRAG**horizon)
            / max(1.0 - SPLIT_EJECT_DRAG, EPSILON)
        )
        if action.split:
            speed = kinematics.split_speed
            static_eject_x = kinematics.split_static_eject_x
            static_eject_y = kinematics.split_static_eject_y
            directional_distance = (
                kinematics.split_placement
                + kinematics.split_directional_eject * drag_sum
            )
            radius = kinematics.split_radius
        else:
            speed = kinematics.normal_speed
            static_eject_x = kinematics.normal_eject_x
            static_eject_y = kinematics.normal_eject_y
            directional_distance = 0.0
            radius = kinematics.normal_radius
        intended_x = (
            geometry.center[0]
            + direction[0] * (speed * horizon + directional_distance)
            + static_eject_x * drag_sum
        )
        intended_y = (
            geometry.center[1]
            + direction[1] * (speed * horizon + directional_distance)
            + static_eject_y * drag_sum
        )
        projected_x = _clamp(intended_x, radius, arena_size - radius)
        projected_y = _clamp(intended_y, radius, arena_size - radius)
        displacement_x = projected_x - geometry.center[0]
        displacement_y = projected_y - geometry.center[1]
        expected_useful = speed * horizon
        actual_useful = max(
            0.0,
            displacement_x * direction[0] + displacement_y * direction[1],
        )
        efficiency = (
            1.0
            if expected_useful <= EPSILON
            else _clamp(actual_useful / expected_useful, 0.0, 1.0)
        )
        return displacement_x, displacement_y, efficiency

    def _proxy_analysis(
        self,
        *,
        node: SearchNode,
        foods: tuple[FoodModel, ...],
        viruses: tuple[VirusModel, ...],
        arena_size: float,
    ) -> ProxyAnalysis:
        geometry = self._node_geometry(node)
        selected_food_models = tuple(
            sorted(
                (food for food in foods if food.food_id not in node.eaten_food_ids),
                key=lambda food: squared_distance(geometry.center, food.pos),
            )[: self.proxy_food_limit]
        )
        selected_virus_models = tuple(
            sorted(
                (
                    virus
                    for virus in viruses
                    if virus.virus_id not in node.consumed_virus_ids
                ),
                key=lambda virus: squared_distance(geometry.center, virus.pos),
            )[: self.proxy_virus_limit]
        )
        risk_enemies = self._risk_enemies(node.enemies)
        own_speeds = {
            blob.blob_id: player_speed(blob.radius) for blob in node.own_blobs
        }
        food_targets: list[ProxyFoodTarget] = []
        for food in selected_food_models:
            source = min(
                node.own_blobs,
                key=lambda blob: (
                    max(0.0, math.dist(blob.pos, food.pos) - blob.radius),
                    -blob.radius,
                    blob.blob_id,
                ),
            )
            gap = max(0.0, math.dist(source.pos, food.pos) - source.radius)
            food_targets.append(
                ProxyFoodTarget(
                    source_blob_id=source.blob_id,
                    food=food,
                    direction=normalise(
                        (food.pos[0] - source.x, food.pos[1] - source.y)
                    ),
                    gap=gap,
                )
            )

        virus_targets: list[ProxyVirusTarget] = []
        for virus in selected_virus_models:
            origins = [
                blob
                for blob in node.own_blobs
                if self._can_still_consume_virus_at_contact(blob, virus)
            ]
            if not origins:
                continue
            source = min(
                origins,
                key=lambda blob: (
                    max(0.0, math.dist(blob.pos, virus.pos) - blob.radius),
                    -blob.radius,
                    blob.blob_id,
                ),
            )
            gap = max(0.0, math.dist(source.pos, virus.pos) - source.radius)
            virus_targets.append(
                ProxyVirusTarget(
                    source_blob_id=source.blob_id,
                    virus=virus,
                    direction=normalise(
                        (virus.pos[0] - source.x, virus.pos[1] - source.y)
                    ),
                    gap=gap,
                )
            )

        prey_rows: list[tuple[float, ProxyPreyTarget]] = []
        for enemy in node.enemies:
            if enemy.stale_rounds:
                continue
            best_origin: OwnBlob | None = None
            best_expected_mass = 0.0
            enemy_speed = player_speed(enemy.radius)
            for own in node.own_blobs:
                if not can_eat_player_blob(own.radius, enemy.radius):
                    continue
                gap = max(0.0, math.dist(own.pos, enemy.pos) - own.radius)
                flee = normalise((enemy.x - own.x, enemy.y - own.y))
                unclamped_x = (
                    enemy.x + flee[0] * enemy_speed * self._effective_proxy_horizon
                )
                unclamped_y = (
                    enemy.y + flee[1] * enemy_speed * self._effective_proxy_horizon
                )
                clamped_x = _clamp(
                    unclamped_x,
                    enemy.radius,
                    arena_size - enemy.radius,
                )
                clamped_y = _clamp(
                    unclamped_y,
                    enemy.radius,
                    arena_size - enemy.radius,
                )
                escape_efficiency = _clamp(
                    math.hypot(clamped_x - enemy.x, clamped_y - enemy.y)
                    / max(enemy_speed * self._effective_proxy_horizon, EPSILON),
                    0.0,
                    1.0,
                )
                closing_speed = max(
                    0.05,
                    own_speeds[own.blob_id] - enemy_speed * escape_efficiency,
                )
                probability = math.exp(
                    -gap / max(self._CAPTURE_HORIZON * closing_speed, EPSILON)
                )
                rival_value = self._rival_values.get(enemy.player_id, 0.0)
                expected_mass = probability * enemy.mass * (1.0 + rival_value)
                if best_origin is None or expected_mass > best_expected_mass:
                    best_origin = own
                    best_expected_mass = expected_mass
            if best_origin is None:
                continue
            prey_rows.append(
                (
                    best_expected_mass,
                    ProxyPreyTarget(
                        source_blob_id=best_origin.blob_id,
                        enemy=enemy,
                        direction=normalise(
                            (enemy.x - best_origin.x, enemy.y - best_origin.y)
                        ),
                        expected_mass=best_expected_mass,
                    ),
                )
            )
        prey_rows.sort(
            key=lambda row: (
                -row[0],
                row[1].enemy.player_id,
                row[1].enemy.blob_id,
            )
        )
        prey_targets = tuple(row[1] for row in prey_rows[:4])

        threat_targets: list[ProxyThreat] = []
        escape_x = 0.0
        escape_y = 0.0
        can_add_fragment = len(node.own_blobs) < MAX_BLOB_COUNT
        for own in node.own_blobs:
            split_radius = (
                own.radius / SQRT2
                if can_add_fragment and own.mass >= SPLIT_MIN_MASS
                else None
            )
            # Enemy ordinary pursuit and split attack are mutually exclusive
            # actions. Keep their envelopes separate for both of our shapes.
            threat_rows: list[
                tuple[
                    tuple[float, int, int],
                    EnemyBlob,
                    float,
                    float,
                    float | None,
                    float | None,
                    float | None,
                    float | None,
                ]
            ] = []
            for enemy in risk_enemies:
                distance = math.dist(own.pos, enemy.pos)
                away_x, away_y = normalise((own.x - enemy.x, own.y - enemy.y))
                normal_ordinary_radius = None
                normal_split_attack_radius = None
                if can_eat_player_blob(enemy.radius, own.radius):
                    normal_ordinary_radius = enemy.radius
                    if _can_split_eat(enemy.radius, own.radius):
                        normal_split_attack_radius = _split_attack_reach(enemy.radius)
                    normal_static_radius = max(
                        normal_ordinary_radius,
                        normal_split_attack_radius or 0.0,
                    )
                    if distance <= normal_static_radius + 8.0:
                        severity = max(
                            0.2,
                            normal_static_radius + 8.0 - distance,
                        ) / max(distance, 0.25)
                        weight = severity * own.mass * (1.0 + enemy.stale_rounds * 0.1)
                        escape_x += away_x * weight
                        escape_y += away_y * weight

                split_ordinary_radius = None
                split_split_attack_radius = None
                if split_radius is not None and can_eat_player_blob(
                    enemy.radius,
                    split_radius,
                ):
                    split_ordinary_radius = enemy.radius
                    if _can_split_eat(enemy.radius, split_radius):
                        split_split_attack_radius = _split_attack_reach(enemy.radius)
                exposed_danger_radius = (
                    max(
                        radius
                        for radius in (
                            (
                                split_ordinary_radius
                                if split_radius is not None
                                else normal_ordinary_radius
                            ),
                            (
                                split_split_attack_radius
                                if split_radius is not None
                                else normal_split_attack_radius
                            ),
                        )
                        if radius is not None
                    )
                    if any(
                        radius is not None
                        for radius in (
                            (
                                split_ordinary_radius
                                if split_radius is not None
                                else normal_ordinary_radius
                            ),
                            (
                                split_split_attack_radius
                                if split_radius is not None
                                else normal_split_attack_radius
                            ),
                        )
                    )
                    else None
                )
                if exposed_danger_radius is None:
                    continue
                threat_rows.append(
                    (
                        (
                            distance - exposed_danger_radius,
                            enemy.player_id,
                            enemy.blob_id,
                        ),
                        enemy,
                        away_x,
                        away_y,
                        normal_ordinary_radius,
                        normal_split_attack_radius,
                        split_ordinary_radius,
                        split_split_attack_radius,
                    )
                )

            threat_rows.sort(key=lambda row: row[0])
            by_sector: dict[int, tuple[object, ...]] = {}
            for row in threat_rows:
                enemy = row[1]
                angle = math.atan2(enemy.y - own.y, enemy.x - own.x)
                sector = int(round(angle / TAU * 8.0)) % 8
                by_sector.setdefault(sector, row)
            selected_rows = sorted(by_sector.values(), key=lambda row: row[0])[
                : self.proxy_threat_limit
            ]
            if len(selected_rows) < self.proxy_threat_limit:
                selected_keys = {row[1].key for row in selected_rows}
                selected_rows.extend(
                    row for row in threat_rows if row[1].key not in selected_keys
                )
                selected_rows = selected_rows[: self.proxy_threat_limit]
            for (
                sort_key,
                enemy,
                away_x,
                away_y,
                normal_ordinary_radius,
                normal_split_attack_radius,
                split_ordinary_radius,
                split_split_attack_radius,
            ) in selected_rows:
                threat_targets.append(
                    ProxyThreat(
                        source_blob_id=own.blob_id,
                        enemy=enemy,
                        source_radius=own.radius,
                        normal_ordinary_radius=normal_ordinary_radius,
                        normal_split_attack_radius=normal_split_attack_radius,
                        split_ordinary_radius=split_ordinary_radius,
                        split_split_attack_radius=split_split_attack_radius,
                        away_x=away_x,
                        away_y=away_y,
                        initial_margin=sort_key[0],
                    )
                )

        motion_by_key: dict[tuple[int, int], EnemyBlob] = {}
        for target in (*threat_targets, *prey_targets):
            motion_by_key.setdefault(target.enemy.key, target.enemy)
        motion_enemies = tuple(motion_by_key.values())
        motion_index_by_key = {
            enemy.key: index for index, enemy in enumerate(motion_enemies)
        }
        prey_targets = tuple(
            replace(
                target,
                motion_index=motion_index_by_key[target.enemy.key],
                competitor_debt=(
                    self._rival_values.get(target.enemy.player_id, 0.0)
                    * target.enemy.mass
                ),
            )
            for target in prey_targets
        )
        threat_targets = [
            replace(
                target,
                motion_index=motion_index_by_key[target.enemy.key],
            )
            for target in threat_targets
        ]
        threat_motion_indices: dict[int, list[int]] = {}
        for target in threat_targets:
            threat_motion_indices.setdefault(target.source_blob_id, []).append(
                target.motion_index
            )
        threat_motion_indices_by_source = {
            source_blob_id: tuple(indices)
            for source_blob_id, indices in threat_motion_indices.items()
        }
        self._record_profile_count("proxy_analysis_nodes")
        self._record_profile_count("proxy_food_targets", len(food_targets))
        self._record_profile_count("proxy_virus_targets", len(virus_targets))
        self._record_profile_count("proxy_prey_targets", len(prey_targets))
        self._record_profile_count("proxy_threat_targets", len(threat_targets))
        self._record_profile_count("proxy_motion_enemies", len(motion_enemies))
        physical_enemy_speeds = {
            enemy.key: player_speed(enemy.radius) for enemy in motion_enemies
        }
        event_enemy_keys: set[tuple[int, int]] = (
            {threat.enemy.key for threat in threat_targets}
            if self.proxy_enemy_motion == "threat_reachable"
            else set()
        )
        if self.proxy_enemy_motion == "event_gated":
            own_by_id = {blob.blob_id: blob for blob in node.own_blobs}
            steps = self._effective_proxy_dynamic_horizon
            drag_sum = (
                float(steps)
                if SPLIT_EJECT_DRAG >= 1.0 - EPSILON
                else (1.0 - SPLIT_EJECT_DRAG**steps)
                / max(1.0 - SPLIT_EJECT_DRAG, EPSILON)
            )
            for threat in threat_targets:
                own = own_by_id[threat.source_blob_id]
                enemy = threat.enemy
                distance = math.dist(own.pos, enemy.pos)
                enemy_reach = physical_enemy_speeds[enemy.key] * steps
                inherited_eject = math.hypot(own.eject_vx, own.eject_vy) * drag_sum
                normal_approach = own_speeds[own.blob_id] * steps + inherited_eject
                normal_event = (
                    threat.normal_ordinary_radius is not None
                    and distance
                    <= threat.normal_ordinary_radius + enemy_reach + normal_approach
                )
                split_event = False
                if threat.split_ordinary_radius is not None:
                    child_radius = own.radius / SQRT2
                    child_speed = player_speed(child_radius)
                    split_approach = (
                        2.0 * child_radius
                        + SAME_PLAYER_OVERLAP_EPSILON
                        + child_speed * steps
                        + SPLIT_EJECT_SPEED * drag_sum
                        + inherited_eject
                    )
                    split_event = (
                        distance
                        <= threat.split_ordinary_radius + enemy_reach + split_approach
                    )
                if normal_event or split_event:
                    event_enemy_keys.add(enemy.key)
        enemy_speeds = tuple(
            (
                0.0
                if self.proxy_enemy_motion == "stationary"
                or (
                    self.proxy_enemy_motion
                    in {
                        "event_gated",
                        "threat_reachable",
                    }
                    and enemy.key not in event_enemy_keys
                )
                else physical_enemy_speeds[enemy.key]
            )
            for enemy in motion_enemies
        )
        observed_enemy_directions = tuple(
            normalise(enemy.direction) for enemy in motion_enemies
        )
        observed_enemy_weights = tuple(
            self.proxy_observed_direction_weight if direction != (0.0, 0.0) else 0.0
            for direction in observed_enemy_directions
        )
        threat_lists: dict[int, list[ProxyThreat]] = {}
        for target in threat_targets:
            threat_lists.setdefault(target.source_blob_id, []).append(target)
        threats_by_source = {
            source_blob_id: tuple(targets)
            for source_blob_id, targets in threat_lists.items()
        }

        own_sources_list: list[ProxyOwnSource] = []
        remaining_slots = MAX_BLOB_COUNT - len(node.own_blobs)
        for blob in sorted(node.own_blobs, key=lambda own: own.blob_id):
            if remaining_slots > 0 and blob.mass >= SPLIT_MIN_MASS:
                split_radius = blob.radius / SQRT2
                split_speed = player_speed(split_radius)
                remaining_slots -= 1
            else:
                split_radius = None
                split_speed = None
            own_sources_list.append(
                ProxyOwnSource(
                    blob=blob,
                    speed=own_speeds[blob.blob_id],
                    split_radius=split_radius,
                    split_speed=split_speed,
                )
            )
        own_sources = tuple(own_sources_list)
        normal_motion_ranges_by_source = {
            source.blob.blob_id: (index, index + 1)
            for index, source in enumerate(own_sources)
        }
        split_motion_ranges_by_source: dict[int, tuple[int, int]] = {}
        split_motion_index = 0
        for source in own_sources:
            motion_count = 2 if source.split_radius is not None else 1
            split_motion_ranges_by_source[source.blob.blob_id] = (
                split_motion_index,
                split_motion_index + motion_count,
            )
            split_motion_index += motion_count
        normal_motion_templates, split_motion_templates = self._proxy_motion_templates(
            node,
            own_sources,
            self._effective_proxy_horizon,
        )
        food_targets = [
            replace(
                target,
                normal_motion_range=normal_motion_ranges_by_source[
                    target.source_blob_id
                ],
                split_motion_range=split_motion_ranges_by_source[target.source_blob_id],
            )
            for target in food_targets
        ]
        virus_targets = [
            replace(
                target,
                normal_motion_range=normal_motion_ranges_by_source[
                    target.source_blob_id
                ],
                split_motion_range=split_motion_ranges_by_source[target.source_blob_id],
                threat_motion_indices=threat_motion_indices_by_source.get(
                    target.source_blob_id,
                    (),
                ),
            )
            for target in virus_targets
        ]
        prey_targets = tuple(
            replace(
                target,
                normal_motion_range=normal_motion_ranges_by_source[
                    target.source_blob_id
                ],
                split_motion_range=split_motion_ranges_by_source[target.source_blob_id],
            )
            for target in prey_targets
        )
        normal_hunter_masks, normal_predator_masks = (
            self._proxy_enemy_eligibility_masks(
                motion_enemies,
                normal_motion_templates,
            )
        )
        if split_motion_templates is normal_motion_templates:
            split_hunter_masks = normal_hunter_masks
            split_predator_masks = normal_predator_masks
        else:
            split_hunter_masks, split_predator_masks = (
                self._proxy_enemy_eligibility_masks(
                    motion_enemies,
                    split_motion_templates,
                )
            )
        normal_threats_by_blob = tuple(
            threats_by_source.get(template.source_blob_id, ())
            for template in normal_motion_templates
        )
        split_threats_by_blob = tuple(
            threats_by_source.get(template.source_blob_id, ())
            for template in split_motion_templates
        )

        own_by_id = {blob.blob_id: blob for blob in node.own_blobs}
        coarse_opportunities: list[ProxyCoarseOpportunity] = []
        for target in food_targets:
            value = 100.0 * FOOD_RADIUS * FOOD_RADIUS * math.exp(-target.gap / 6.0)
            weight = value / 6.0
            coarse_opportunities.append(
                ProxyCoarseOpportunity(
                    value=value,
                    gradient_x=target.direction[0] * weight,
                    gradient_y=target.direction[1] * weight,
                )
            )
        for target in virus_targets:
            value = (
                100.0
                * target.virus.radius
                * target.virus.radius
                * math.exp(-target.gap / self._VIRUS_POTENTIAL_HORIZON)
            )
            weight = value / self._VIRUS_POTENTIAL_HORIZON
            coarse_opportunities.append(
                ProxyCoarseOpportunity(
                    value=value,
                    gradient_x=target.direction[0] * weight,
                    gradient_y=target.direction[1] * weight,
                )
            )
        safety_gradient_x = 0.0
        safety_gradient_y = 0.0
        for target in threat_targets:
            own = own_by_id[target.source_blob_id]
            pressure = math.exp(-max(0.0, target.initial_margin) / 6.0)
            weight = 100.0 * own.mass * (0.2 + pressure) / 6.0
            delta_x = target.away_x * weight
            delta_y = target.away_y * weight
            safety_gradient_x += delta_x
            safety_gradient_y += delta_y

        coarse_values = sorted(
            (
                *(opportunity.value for opportunity in coarse_opportunities),
                *(100.0 * target.expected_mass for target in prey_targets),
            ),
            reverse=True,
        )
        coarse_opportunity_baseline = sum(
            value * weight
            for value, weight in zip(
                coarse_values[:3],
                (1.0, 0.25, 0.1),
                strict=False,
            )
        )

        current_blobs = tuple(
            ProxyBlobMotion(
                blob_id=source.blob.blob_id,
                source_blob_id=source.blob.blob_id,
                start_x=source.blob.x,
                start_y=source.blob.y,
                x=source.blob.x,
                y=source.blob.y,
                radius=source.blob.radius,
                speed=source.speed,
            )
            for source in own_sources
        )
        current_enemies = self._proxy_enemy_motions(
            motion_enemies,
            current_blobs,
            horizon=0,
            arena_size=arena_size,
            enemy_speeds=enemy_speeds,
            observed_directions=observed_enemy_directions,
            observed_weights=observed_enemy_weights,
            hunter_masks=normal_hunter_masks,
            predator_masks=normal_predator_masks,
        )
        competitor_mass_debt = sum(
            self._rival_values.get(enemy.player_id, 0.0) * enemy.mass
            for enemy in node.enemies
        )
        baseline = self._proxy_state_value(
            node=node,
            blobs=current_blobs,
            enemy_motions=current_enemies,
            foods=tuple(food_targets),
            viruses=tuple(virus_targets),
            prey=prey_targets,
            threats_by_blob=normal_threats_by_blob,
            competitor_mass_debt=competitor_mass_debt,
            split_motion=False,
            arena_size=arena_size,
        )
        discount_sum = (
            float(self._effective_proxy_horizon)
            if self.proxy_discount >= 1.0 - EPSILON
            else (1.0 - self.proxy_discount**self._effective_proxy_horizon)
            / max(1.0 - self.proxy_discount, EPSILON)
        )
        analysis = ProxyAnalysis(
            node=node,
            foods=tuple(food_targets),
            viruses=tuple(virus_targets),
            prey=prey_targets,
            motion_enemies=motion_enemies,
            normal_threats_by_blob=normal_threats_by_blob,
            split_threats_by_blob=split_threats_by_blob,
            own_sources=own_sources,
            own_source_by_id={source.blob.blob_id: source for source in own_sources},
            normal_motion_templates=normal_motion_templates,
            split_motion_templates=split_motion_templates,
            normal_motion_ranges_by_source=normal_motion_ranges_by_source,
            split_motion_ranges_by_source=split_motion_ranges_by_source,
            enemy_speeds=enemy_speeds,
            enemy_speed_by_key={
                enemy.key: speed
                for enemy, speed in zip(
                    motion_enemies,
                    enemy_speeds,
                    strict=True,
                )
            },
            observed_enemy_directions=observed_enemy_directions,
            observed_enemy_weights=observed_enemy_weights,
            normal_hunter_masks=normal_hunter_masks,
            split_hunter_masks=split_hunter_masks,
            normal_predator_masks=normal_predator_masks,
            split_predator_masks=split_predator_masks,
            competitor_mass_debt=competitor_mass_debt,
            escape_vector=normalise((escape_x, escape_y)),
            safety_gradient=(safety_gradient_x, safety_gradient_y),
            coarse_opportunities=tuple(coarse_opportunities),
            coarse_opportunity_baseline=coarse_opportunity_baseline,
            kinematics=self._proxy_kinematics(
                node,
                geometry,
                own_sources=own_sources,
            ),
            split_state_delta=self._proxy_split_state_delta(
                node=node,
                threats=tuple(threat_targets),
                prey_rows=tuple(prey_rows[:4]),
                viruses=tuple(virus_targets),
                arena_size=arena_size,
            ),
            baseline=baseline,
            discount_sum=discount_sum,
            future_weight=discount_sum / self._effective_proxy_horizon,
        )
        return analysis

    def _coherent_reachable_survivals(
        self,
        *,
        blobs: tuple[ProxyBlobMotion, ...] | list[ProxyBlobMotion],
        threats_by_blob: tuple[tuple[ProxyThreat, ...], ...],
        enemy_speeds: tuple[float, ...],
        arena_size: float,
        dynamic_fraction: float,
        dynamic_steps: int,
        captured_enemy_mask: int = 0,
    ) -> list[float]:
        """Evaluate one internally consistent short pursuit scenario per enemy.

        The reachable-disk model can place the same enemy at a different disk
        boundary for every one of our fragments. Here a scenario fixes one
        pursued fragment for an enemy, rolls that pursuit forward for the short
        dynamic horizon, and evaluates every fragment against that same path.
        We retain the enemy scenario with the greatest aggregate threatened
        mass, then combine different enemies conservatively.
        """

        survival_midpoint = (
            self.survival_midpoint_base
            + self.survival_midpoint_scale * self._proxy_safety_weight
        )
        temperature = max(self.survival_temperature, 0.1)
        survivals = [1.0] * len(blobs)
        threats_by_enemy: dict[
            int,
            list[tuple[int, ProxyThreat, float | None, float | None]],
        ] = {}
        for blob_index, (blob, threats) in enumerate(
            zip(blobs, threats_by_blob, strict=True)
        ):
            for threat in threats:
                motion_index = threat.motion_index
                if captured_enemy_mask & (1 << motion_index):
                    continue
                ordinary_radius, split_attack_radius = self._threat_action_radii(
                    threat,
                    blob.radius,
                )
                if ordinary_radius is None and split_attack_radius is None:
                    continue
                threats_by_enemy.setdefault(motion_index, []).append(
                    (blob_index, threat, ordinary_radius, split_attack_radius)
                )

        step_count = max(0, dynamic_steps)
        for motion_index, threat_rows in threats_by_enemy.items():
            enemy = threat_rows[0][1].enemy
            speed = enemy_speeds[motion_index]
            most_damaging_survivals: list[float] | None = None
            greatest_mass_loss = -1.0
            radii_by_blob = {
                blob_index: (ordinary_radius, split_attack_radius)
                for blob_index, _, ordinary_radius, split_attack_radius in threat_rows
            }
            target_indices = tuple(radii_by_blob)
            own_paths = {
                blob_index: tuple(
                    self._proxy_short_position(
                        blobs[blob_index],
                        step=step,
                        step_count=step_count,
                        fallback_fraction=dynamic_fraction,
                    )
                    for step in range(step_count + 1)
                )
                for blob_index in target_indices
            }
            contextual_deductions = {
                blob_index: tuple(
                    enemy.stale_rounds * 0.35
                    + self._wall_trap_factor_at(
                        own_x,
                        own_y,
                        blobs[blob_index].radius,
                        enemy,
                        arena_size,
                    )
                    * 4.0
                    for own_x, own_y in own_paths[blob_index]
                )
                for blob_index in target_indices
            }
            initial_margins = {
                blob_index: min(
                    (
                        math.hypot(
                            own_paths[blob_index][0][0] - enemy.x,
                            own_paths[blob_index][0][1] - enemy.y,
                        )
                        - ordinary_radius
                        - contextual_deductions[blob_index][0]
                        for ordinary_radius in (radii_by_blob[blob_index][0],)
                        if ordinary_radius is not None
                    ),
                    default=math.inf,
                )
                for blob_index in target_indices
            }

            # A target identity defines one coherent pursuit policy. The enemy
            # may turn between rounds, but cannot pursue a different fragment
            # in the same round for each risk calculation.
            for target_index in target_indices:
                enemy_x = enemy.x
                enemy_y = enemy.y
                worst_margins = dict(initial_margins)
                for step in range(1, step_count + 1):
                    target_x, target_y = own_paths[target_index][step]
                    chase_x, chase_y = normalise(
                        (target_x - enemy_x, target_y - enemy_y)
                    )
                    enemy_x = _clamp(
                        enemy_x + chase_x * speed,
                        enemy.radius,
                        arena_size - enemy.radius,
                    )
                    enemy_y = _clamp(
                        enemy_y + chase_y * speed,
                        enemy.radius,
                        arena_size - enemy.radius,
                    )
                    for blob_index in target_indices:
                        own_x, own_y = own_paths[blob_index][step]
                        ordinary_radius, split_attack_radius = radii_by_blob[blob_index]
                        action_margins = []
                        if ordinary_radius is not None:
                            action_margins.append(
                                math.hypot(own_x - enemy_x, own_y - enemy_y)
                                - ordinary_radius
                            )
                        if split_attack_radius is not None:
                            action_margins.append(
                                math.hypot(own_x - enemy.x, own_y - enemy.y)
                                - split_attack_radius
                            )
                        margin = (
                            min(action_margins)
                            - contextual_deductions[blob_index][step]
                        )
                        worst_margins[blob_index] = min(
                            worst_margins[blob_index],
                            margin,
                        )

                scenario_survivals = [1.0] * len(blobs)
                mass_loss = 0.0
                for blob_index, margin in worst_margins.items():
                    scaled = _clamp(
                        (margin - survival_midpoint) / temperature,
                        -40.0,
                        40.0,
                    )
                    survival = 1.0 / (1.0 + math.exp(-scaled))
                    scenario_survivals[blob_index] = survival
                    mass_loss += blobs[blob_index].mass * (1.0 - survival)
                if mass_loss > greatest_mass_loss:
                    greatest_mass_loss = mass_loss
                    most_damaging_survivals = scenario_survivals

            if most_damaging_survivals is not None:
                survivals = [
                    min(current, scenario)
                    for current, scenario in zip(
                        survivals,
                        most_damaging_survivals,
                        strict=True,
                    )
                ]
        return survivals

    def _proxy_state_value(
        self,
        *,
        node: SearchNode,
        blobs: tuple[ProxyBlobMotion, ...] | list[ProxyBlobMotion],
        enemy_motions: tuple[ProxyEnemyMotion, ...] | list[ProxyEnemyMotion],
        foods: tuple[ProxyFoodTarget, ...],
        viruses: tuple[ProxyVirusTarget, ...],
        prey: tuple[ProxyPreyTarget, ...],
        threats_by_blob: tuple[tuple[ProxyThreat, ...], ...],
        competitor_mass_debt: float,
        split_motion: bool,
        arena_size: float,
        dynamic_fraction: float = 1.0,
        dynamic_steps: int = 0,
        projection_steps: int = 0,
    ) -> ProxyValue:
        """Evaluate only action-relevant sparse terms of the shared value."""
        prey_started = self._profile_start()
        captured_enemy_mask = 0
        captured_enemy_mass = 0.0
        captured_competitor_debt = 0.0
        prey_opportunities: list[float] = []
        for target in prey:
            enemy = target.enemy
            motion = enemy_motions[target.motion_index]
            growth_motion = (
                ProxyEnemyMotion(
                    enemy=enemy,
                    x=enemy.x,
                    y=enemy.y,
                    direction=(0.0, 0.0),
                    speed=0.0,
                )
                if self.proxy_enemy_motion == "threat_reachable"
                else motion
            )
            if enemy.stale_rounds:
                continue
            captured = False
            best_probability = 0.0
            motion_range = (
                target.split_motion_range
                if split_motion
                else target.normal_motion_range
            )
            for blob_index in range(*motion_range):
                blob = blobs[blob_index]
                if not can_eat_player_blob(blob.radius, enemy.radius):
                    continue
                dynamic_blob_x, dynamic_blob_y = self._proxy_short_position(
                    blob,
                    fallback_fraction=dynamic_fraction,
                )
                dynamic_enemy_x = (
                    enemy.x + (growth_motion.x - enemy.x) * dynamic_fraction
                )
                dynamic_enemy_y = (
                    enemy.y + (growth_motion.y - enemy.y) * dynamic_fraction
                )
                if (
                    self._relative_segment_distance_sq(
                        blob.start_x,
                        blob.start_y,
                        dynamic_blob_x,
                        dynamic_blob_y,
                        enemy.x,
                        enemy.y,
                        dynamic_enemy_x,
                        dynamic_enemy_y,
                    )
                    <= blob.radius * blob.radius
                ):
                    captured_enemy_mask |= 1 << target.motion_index
                    captured_enemy_mass += enemy.mass
                    captured_competitor_debt += target.competitor_debt
                    captured = True
                    break
                best_probability = max(
                    best_probability,
                    self._proxy_prey_capture_probability(
                        blob,
                        growth_motion,
                        arena_size,
                        projection_fraction=dynamic_fraction,
                    ),
                )
            if not captured and best_probability > 0.0:
                rival_value = self._rival_values.get(enemy.player_id, 0.0)
                prey_opportunities.append(
                    best_probability * enemy.mass * (1.0 + rival_value)
                )
        self._record_profile("proxy_value_prey", prey_started)

        survival_started = self._profile_start()
        survival_midpoint = (
            self.survival_midpoint_base
            + self.survival_midpoint_scale * self._proxy_safety_weight
        )
        temperature = max(self.survival_temperature, 0.1)
        safe_mass = 0.0
        continuation_probability = 0.0
        coherent_survivals = (
            self._coherent_reachable_survivals(
                blobs=blobs,
                threats_by_blob=threats_by_blob,
                enemy_speeds=tuple(motion.speed for motion in enemy_motions),
                arena_size=arena_size,
                dynamic_fraction=dynamic_fraction,
                dynamic_steps=dynamic_steps,
                captured_enemy_mask=captured_enemy_mask,
            )
            if self.proxy_reachable_threats and self.proxy_coherent_threat_scenarios
            else None
        )
        for blob_index, blob in enumerate(blobs):
            if coherent_survivals is not None:
                survival = coherent_survivals[blob_index]
                continuation_probability = max(continuation_probability, survival)
                safe_mass += blob.mass * survival
                continue
            threats = threats_by_blob[blob_index]
            if not threats:
                continuation_probability = 1.0
                safe_mass += blob.mass
                continue
            worst_margin = math.inf
            for threat in threats:
                enemy = threat.enemy
                if captured_enemy_mask & (1 << threat.motion_index):
                    continue
                motion = enemy_motions[threat.motion_index]
                ordinary_radius, split_attack_radius = self._threat_action_radii(
                    threat,
                    blob.radius,
                )
                if ordinary_radius is None and split_attack_radius is None:
                    continue
                if self.proxy_reachable_threats:
                    # Opponents can choose a new direction every round.
                    # Evaluate their reachable disk at each short-horizon step
                    # instead of persisting one predicted heading.
                    step_count = max(0, dynamic_steps)
                    for step in range(step_count + 1):
                        own_x, own_y = self._proxy_short_position(
                            blob,
                            step=step,
                            step_count=step_count,
                            fallback_fraction=dynamic_fraction,
                        )
                        distance = math.hypot(own_x - enemy.x, own_y - enemy.y)
                        action_margin = self._exclusive_threat_margin(
                            distance=distance,
                            ordinary_radius=ordinary_radius,
                            split_attack_radius=(
                                split_attack_radius if step > 0 else None
                            ),
                            ordinary_reach=motion.speed * step,
                        )
                        margin = (
                            action_margin
                            - enemy.stale_rounds * 0.35
                            - self._wall_trap_factor_at(
                                own_x,
                                own_y,
                                blob.radius,
                                enemy,
                                arena_size,
                            )
                            * 4.0
                        )
                        worst_margin = min(worst_margin, margin)
                else:
                    own_x, own_y = self._proxy_short_position(
                        blob,
                        fallback_fraction=dynamic_fraction,
                    )
                    enemy_x = enemy.x + (motion.x - enemy.x) * dynamic_fraction
                    enemy_y = enemy.y + (motion.y - enemy.y) * dynamic_fraction
                    closest = self._relative_segment_distance(
                        blob.start_x,
                        blob.start_y,
                        own_x,
                        own_y,
                        enemy.x,
                        enemy.y,
                        enemy_x,
                        enemy_y,
                    )
                    action_margin = self._exclusive_threat_margin(
                        distance=closest,
                        ordinary_radius=ordinary_radius,
                        split_attack_radius=None,
                    )
                    if split_attack_radius is not None:
                        split_closest = math.hypot(
                            own_x - enemy.x,
                            own_y - enemy.y,
                        )
                        action_margin = min(
                            action_margin,
                            self._exclusive_threat_margin(
                                distance=split_closest,
                                ordinary_radius=None,
                                split_attack_radius=split_attack_radius,
                            ),
                        )
                    margin = (
                        action_margin
                        - enemy.stale_rounds * 0.35
                        - self._wall_trap_factor_at(
                            own_x,
                            own_y,
                            blob.radius,
                            enemy,
                            arena_size,
                        )
                        * 4.0
                    )
                    worst_margin = min(worst_margin, margin)
            if worst_margin < math.inf:
                scaled = _clamp(
                    (worst_margin - survival_midpoint) / temperature,
                    -40.0,
                    40.0,
                )
                survival = 1.0 / (1.0 + math.exp(-scaled))
            else:
                survival = 1.0
            continuation_probability = max(continuation_probability, survival)
            safe_mass += blob.mass * survival
        self._record_profile("proxy_value_survival", survival_started)

        realised_mass = captured_enemy_mass
        opportunities = prey_opportunities
        food_started = self._profile_start()
        for target in foods:
            food = target.food
            food_x = food.pos[0]
            food_y = food.pos[1]
            motion_range = (
                target.split_motion_range
                if split_motion
                else target.normal_motion_range
            )
            captured = False
            gap = math.inf
            for blob_index in range(*motion_range):
                blob = blobs[blob_index]
                start_x = blob.start_x
                start_y = blob.start_y
                end_x = blob.x
                end_y = blob.y
                delta_x = blob.delta_x
                delta_y = blob.delta_y
                length_sq = blob.length_sq
                if length_sq <= EPSILON:
                    closest_x = food_x - start_x
                    closest_y = food_y - start_y
                else:
                    fraction = (
                        (food_x - start_x) * delta_x + (food_y - start_y) * delta_y
                    ) / length_sq
                    if fraction < 0.0:
                        fraction = 0.0
                    elif fraction > 1.0:
                        fraction = 1.0
                    closest_x = food_x - (start_x + delta_x * fraction)
                    closest_y = food_y - (start_y + delta_y * fraction)
                if (
                    closest_x * closest_x + closest_y * closest_y
                    <= blob.radius * blob.radius
                ):
                    captured = True
                    break
                blob_gap = math.hypot(end_x - food_x, end_y - food_y) - blob.radius
                if blob_gap < 0.0:
                    blob_gap = 0.0
                if blob_gap < gap:
                    gap = blob_gap
            if captured:
                realised_mass += FOOD_RADIUS * FOOD_RADIUS
            elif gap < math.inf:
                opportunities.append(FOOD_RADIUS * FOOD_RADIUS * math.exp(-gap / 6.0))
        self._record_profile("proxy_value_food", food_started)

        virus_started = self._profile_start()
        for target in viruses:
            virus = target.virus
            best_future = 0.0
            best_capture_delta: float | None = None
            motion_range = (
                target.split_motion_range
                if split_motion
                else target.normal_motion_range
            )
            threat_motion_indices = target.threat_motion_indices
            for blob_index in range(*motion_range):
                blob = blobs[blob_index]
                if not _can_consume_virus(blob.radius, virus.radius):
                    continue
                path_distance_sq = self._point_segment_distance_sq(
                    virus.pos[0],
                    virus.pos[1],
                    blob.start_x,
                    blob.start_y,
                    blob.x,
                    blob.y,
                )
                retention = self._proxy_virus_retention(
                    blob=blob,
                    virus=virus,
                    enemy_motions=enemy_motions,
                    threat_motion_indices=threat_motion_indices,
                    own_blob_count=len(blobs),
                    arena_size=arena_size,
                )
                if path_distance_sq <= blob.radius * blob.radius:
                    capture_delta = (
                        blob.mass + virus.radius * virus.radius
                    ) * retention - blob.mass
                    if best_capture_delta is None or capture_delta > best_capture_delta:
                        best_capture_delta = capture_delta
                    continue
                gap = max(
                    0.0,
                    math.hypot(blob.x - virus.pos[0], blob.y - virus.pos[1])
                    - blob.radius,
                )
                turns_to_contact = math.ceil(gap / blob.speed)
                projected_mass = decayed_mass_after_turns(
                    blob.mass,
                    turns_to_contact,
                    decay_rate=MASS_DECAY_RATE,
                    minimum_radius=STARTING_RADIUS,
                )
                if not _can_consume_virus(math.sqrt(projected_mass), virus.radius):
                    continue
                best_future = max(
                    best_future,
                    virus.radius
                    * virus.radius
                    * retention
                    * math.exp(-gap / self._VIRUS_POTENTIAL_HORIZON),
                )
            if best_capture_delta is not None:
                realised_mass += best_capture_delta
            elif best_future > 0.0:
                opportunities.append(best_future)
        self._record_profile("proxy_value_virus", virus_started)

        aggregate_started = self._profile_start()
        safe_mass += realised_mass * continuation_probability
        first = second = third = 0.0
        for value in opportunities:
            if value >= first:
                first, second, third = value, first, second
            elif value >= second:
                second, third = value, second
            elif value > third:
                third = value
        opportunity_mass = first + 0.25 * second + 0.1 * third
        if not self._proxy_has_future_after_horizon:
            opportunity_mass = 0.0
        self._record_profile("proxy_value_aggregate", aggregate_started)
        return ProxyValue(
            safe_mass=safe_mass,
            continuation_probability=continuation_probability,
            opportunity_mass=opportunity_mass,
            cohesion_mass=self._projected_formation_cohesion_mass(
                node=node,
                blobs=blobs,
                projection_steps=projection_steps,
            ),
            competitor_mass_debt=(competitor_mass_debt - captured_competitor_debt),
            recovery_debt=self.recovery_mass * (1.0 - continuation_probability),
        )

    def _proxy_virus_retention(
        self,
        *,
        blob: ProxyBlobMotion,
        virus: VirusModel,
        enemy_motions: tuple[ProxyEnemyMotion, ...] | list[ProxyEnemyMotion],
        threat_motion_indices: tuple[int, ...],
        own_blob_count: int,
        arena_size: float,
    ) -> float:
        """Cheap fragment-risk surrogate used only for candidate ranking."""

        piece_count = max(1, MAX_BLOB_COUNT - own_blob_count + 1)
        piece_radius = math.sqrt(
            (blob.mass + virus.radius * virus.radius) / piece_count
        )
        retention = 1.0
        temperature = max(self.survival_temperature, 0.1)
        survival_midpoint = self.survival_midpoint_base + 1.3
        for motion_index in threat_motion_indices:
            motion = enemy_motions[motion_index]
            enemy = motion.enemy
            if not can_eat_player_blob(enemy.radius, piece_radius):
                continue
            danger_radius = enemy.radius
            if _can_split_eat(enemy.radius, piece_radius):
                danger_radius = max(
                    danger_radius,
                    _split_attack_reach(enemy.radius),
                )
            margin = (
                math.hypot(blob.x - motion.x, blob.y - motion.y)
                - danger_radius
                - enemy.stale_rounds * 0.35
                - self._wall_trap_factor_at(
                    blob.x,
                    blob.y,
                    piece_radius,
                    enemy,
                    arena_size,
                )
                * 4.0
            )
            scaled = _clamp(
                (margin - survival_midpoint) / temperature,
                -40.0,
                40.0,
            )
            retention = min(retention, 1.0 / (1.0 + math.exp(-scaled)))
        return retention

    def _proxy_prey_capture_probability(
        self,
        blob: ProxyBlobMotion,
        motion: ProxyEnemyMotion,
        arena_size: float,
        *,
        projection_fraction: float = 1.0,
    ) -> float:
        enemy = motion.enemy
        projected_blob_x, projected_blob_y = self._proxy_short_position(
            blob,
            fallback_fraction=projection_fraction,
        )
        projected_enemy_x = enemy.x + (motion.x - enemy.x) * projection_fraction
        projected_enemy_y = enemy.y + (motion.y - enemy.y) * projection_fraction
        delta_x = projected_enemy_x - projected_blob_x
        delta_y = projected_enemy_y - projected_blob_y
        distance = math.hypot(delta_x, delta_y)
        gap = max(0.0, distance - blob.radius)
        if gap <= EPSILON:
            return 1.0
        direction = (delta_x / distance, delta_y / distance)
        own_x = _clamp(
            projected_blob_x + direction[0] * blob.speed,
            blob.radius,
            arena_size - blob.radius,
        )
        own_y = _clamp(
            projected_blob_y + direction[1] * blob.speed,
            blob.radius,
            arena_size - blob.radius,
        )
        flee = normalise(
            (
                projected_enemy_x - projected_blob_x,
                projected_enemy_y - projected_blob_y,
            )
        )
        flee_direction = normalise(
            (
                flee[0] * 0.62 + motion.direction[0] * 0.38,
                flee[1] * 0.62 + motion.direction[1] * 0.38,
            )
        )
        enemy_speed = motion.speed
        enemy_x = _clamp(
            projected_enemy_x + flee_direction[0] * enemy_speed,
            enemy.radius,
            arena_size - enemy.radius,
        )
        enemy_y = _clamp(
            projected_enemy_y + flee_direction[1] * enemy_speed,
            enemy.radius,
            arena_size - enemy.radius,
        )
        next_gap = max(
            0.0,
            math.hypot(enemy_x - own_x, enemy_y - own_y) - blob.radius,
        )
        closing = gap - next_gap
        temperature = self._CAPTURE_CLOSING_TEMPERATURE
        scaled = _clamp(closing / temperature, -40.0, 40.0)
        effective_closing = temperature * math.log1p(math.exp(scaled))
        return math.exp(-gap / max(self._CAPTURE_HORIZON * effective_closing, EPSILON))

    @staticmethod
    def _proxy_enemy_eligibility_masks(
        enemies: tuple[EnemyBlob, ...],
        motion_templates: tuple[ProxyMotionTemplate, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Compile radius-only eat relationships for one projected shape.

        Candidate direction changes positions but not radii or blob ordering, so
        the hot enemy-motion loop can reuse these exact relationships.
        """

        hunter_masks: list[int] = []
        predator_masks: list[int] = []
        for enemy in enemies:
            hunter_mask = 0
            predator_mask = 0
            for blob_index, template in enumerate(motion_templates):
                bit = 1 << blob_index
                if can_eat_player_blob(template.radius, enemy.radius):
                    hunter_mask |= bit
                if can_eat_player_blob(enemy.radius, template.radius):
                    predator_mask |= bit
            hunter_masks.append(hunter_mask)
            predator_masks.append(predator_mask)
        return tuple(hunter_masks), tuple(predator_masks)

    def _proxy_enemy_motions(
        self,
        enemies: tuple[EnemyBlob, ...],
        own_blobs: tuple[ProxyBlobMotion, ...] | list[ProxyBlobMotion],
        *,
        horizon: int,
        arena_size: float,
        enemy_speeds: tuple[float, ...] | None = None,
        observed_directions: tuple[tuple[float, float], ...] | None = None,
        observed_weights: tuple[float, ...] | None = None,
        hunter_masks: tuple[int, ...] | None = None,
        predator_masks: tuple[int, ...] | None = None,
    ) -> list[ProxyEnemyMotion]:
        speeds = (
            enemy_speeds
            if enemy_speeds is not None
            else tuple(player_speed(enemy.radius) for enemy in enemies)
        )
        observed_rows = (
            observed_directions
            if observed_directions is not None
            else tuple(normalise(enemy.direction) for enemy in enemies)
        )
        weight_rows = (
            observed_weights
            if observed_weights is not None
            else tuple(
                self.proxy_observed_direction_weight if observed != (0.0, 0.0) else 0.0
                for observed in observed_rows
            )
        )
        scratch = self._proxy_enemy_motion_scratch
        motions = self._proxy_enemy_motion_active
        motions.clear()
        if not own_blobs:
            for enemy, speed in zip(enemies, speeds, strict=True):
                motion_index = len(motions)
                if motion_index < len(scratch):
                    motion = scratch[motion_index]
                    motion.enemy = enemy
                    motion.x = enemy.x
                    motion.y = enemy.y
                    motion.direction = enemy.direction
                    motion.speed = speed
                else:
                    motion = ProxyEnemyMotion(
                        enemy=enemy,
                        x=enemy.x,
                        y=enemy.y,
                        direction=enemy.direction,
                        speed=speed,
                    )
                    scratch.append(motion)
                motions.append(motion)
            return motions
        for enemy_index, (enemy, base_speed, observed, observed_weight) in enumerate(
            zip(
                enemies,
                speeds,
                observed_rows,
                weight_rows,
                strict=True,
            )
        ):
            adversarial_weight = 1.0 - observed_weight
            prey: ProxyBlobMotion | None = None
            prey_distance = math.inf
            hunter: ProxyBlobMotion | None = None
            hunter_distance = math.inf
            hunter_mask = hunter_masks[enemy_index] if hunter_masks is not None else 0
            predator_mask = (
                predator_masks[enemy_index] if predator_masks is not None else 0
            )
            for blob_index, blob in enumerate(own_blobs):
                distance_sq = (enemy.x - blob.x) ** 2 + (enemy.y - blob.y) ** 2
                if (
                    bool(predator_mask & (1 << blob_index))
                    if predator_masks is not None
                    else can_eat_player_blob(enemy.radius, blob.radius)
                ) and distance_sq < prey_distance:
                    prey = blob
                    prey_distance = distance_sq
                if (
                    bool(hunter_mask & (1 << blob_index))
                    if hunter_masks is not None
                    else can_eat_player_blob(blob.radius, enemy.radius)
                ) and distance_sq < hunter_distance:
                    hunter = blob
                    hunter_distance = distance_sq
            if prey is not None:
                adversarial = normalise((prey.x - enemy.x, prey.y - enemy.y))
                direction = normalise(
                    (
                        adversarial[0] * adversarial_weight
                        + observed[0] * observed_weight,
                        adversarial[1] * adversarial_weight
                        + observed[1] * observed_weight,
                    )
                )
            else:
                if hunter is not None:
                    flee = normalise((enemy.x - hunter.x, enemy.y - hunter.y))
                    direction = normalise(
                        (
                            flee[0] * 0.62 + observed[0] * 0.38,
                            flee[1] * 0.62 + observed[1] * 0.38,
                        )
                    )
                else:
                    direction = observed
            distance = base_speed * max(0, horizon)
            motion_index = len(motions)
            lower = enemy.radius
            upper = arena_size - lower
            projected_x = enemy.x + direction[0] * distance
            if projected_x < lower:
                projected_x = lower
            elif projected_x > upper:
                projected_x = upper
            projected_y = enemy.y + direction[1] * distance
            if projected_y < lower:
                projected_y = lower
            elif projected_y > upper:
                projected_y = upper
            if motion_index < len(scratch):
                motion = scratch[motion_index]
                motion.enemy = enemy
                motion.x = projected_x
                motion.y = projected_y
                motion.direction = direction
                motion.speed = base_speed
            else:
                motion = ProxyEnemyMotion(
                    enemy=enemy,
                    x=projected_x,
                    y=projected_y,
                    direction=direction,
                    speed=base_speed,
                )
                scratch.append(motion)
            motions.append(motion)
        return motions

    @staticmethod
    def _point_segment_distance_sq(
        point_x: float,
        point_y: float,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
    ) -> float:
        delta_x = end_x - start_x
        delta_y = end_y - start_y
        length_sq = delta_x * delta_x + delta_y * delta_y
        if length_sq <= EPSILON:
            point_delta_x = point_x - start_x
            point_delta_y = point_y - start_y
            return point_delta_x * point_delta_x + point_delta_y * point_delta_y
        fraction = _clamp(
            ((point_x - start_x) * delta_x + (point_y - start_y) * delta_y) / length_sq,
            0.0,
            1.0,
        )
        closest_delta_x = point_x - (start_x + delta_x * fraction)
        closest_delta_y = point_y - (start_y + delta_y * fraction)
        return closest_delta_x * closest_delta_x + closest_delta_y * closest_delta_y

    @classmethod
    def _point_segment_distance(
        cls,
        point_x: float,
        point_y: float,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
    ) -> float:
        delta_x = end_x - start_x
        delta_y = end_y - start_y
        length_sq = delta_x * delta_x + delta_y * delta_y
        if length_sq <= EPSILON:
            return math.hypot(point_x - start_x, point_y - start_y)
        fraction = _clamp(
            ((point_x - start_x) * delta_x + (point_y - start_y) * delta_y) / length_sq,
            0.0,
            1.0,
        )
        return math.hypot(
            point_x - (start_x + delta_x * fraction),
            point_y - (start_y + delta_y * fraction),
        )

    @classmethod
    def _relative_segment_distance_sq(
        cls,
        own_start_x: float,
        own_start_y: float,
        own_end_x: float,
        own_end_y: float,
        enemy_start_x: float,
        enemy_start_y: float,
        enemy_end_x: float,
        enemy_end_y: float,
    ) -> float:
        return cls._point_segment_distance_sq(
            0.0,
            0.0,
            own_start_x - enemy_start_x,
            own_start_y - enemy_start_y,
            own_end_x - enemy_end_x,
            own_end_y - enemy_end_y,
        )

    @classmethod
    def _relative_segment_distance(
        cls,
        own_start_x: float,
        own_start_y: float,
        own_end_x: float,
        own_end_y: float,
        enemy_start_x: float,
        enemy_start_y: float,
        enemy_end_x: float,
        enemy_end_y: float,
    ) -> float:
        return cls._point_segment_distance(
            0.0,
            0.0,
            own_start_x - enemy_start_x,
            own_start_y - enemy_start_y,
            own_end_x - enemy_end_x,
            own_end_y - enemy_end_y,
        )

    @staticmethod
    def _proxy_short_position(
        blob: ProxyBlobMotion,
        *,
        step: int | None = None,
        step_count: int = 0,
        fallback_fraction: float = 1.0,
    ) -> tuple[float, float]:
        """Return exact short-horizon own physics when it was projected.

        Long-horizon resource motion is intentionally linearised. Threat and
        immediate-capture terms instead consume the separately integrated
        short path so split eject, inherited eject, and wall clamps occur on
        their real rounds rather than being averaged over the resource path.
        """

        if blob.short_path:
            if step is None:
                return blob.short_path[-1]
            if step_count == len(blob.short_path) - 1:
                return blob.short_path[min(max(0, step), step_count)]
        fraction = fallback_fraction
        if step is not None:
            fraction = 0.0 if step_count == 0 else fallback_fraction * step / step_count
        return (
            blob.start_x + blob.delta_x * fraction,
            blob.start_y + blob.delta_y * fraction,
        )

    @staticmethod
    def _proxy_motion_templates(
        node: SearchNode,
        own_sources: tuple[ProxyOwnSource, ...],
        horizon: int,
    ) -> tuple[tuple[ProxyMotionTemplate, ...], tuple[ProxyMotionTemplate, ...]]:
        """Compile linear trajectory coefficients once for every root action."""

        horizon = max(1, horizon)
        drag_sum = (
            float(horizon)
            if SPLIT_EJECT_DRAG >= 1.0 - EPSILON
            else (1.0 - SPLIT_EJECT_DRAG**horizon)
            / max(1.0 - SPLIT_EJECT_DRAG, EPSILON)
        )
        normal_templates: list[ProxyMotionTemplate] = []
        split_templates: list[ProxyMotionTemplate] = []
        next_proxy_id = max((blob.blob_id for blob in node.own_blobs), default=-1) + 1
        for source in own_sources:
            own = source.blob
            normal = ProxyMotionTemplate(
                blob_id=own.blob_id,
                source_blob_id=own.blob_id,
                base_start_x=own.x,
                base_start_y=own.y,
                directional_start=0.0,
                static_eject_x=own.eject_vx * drag_sum,
                static_eject_y=own.eject_vy * drag_sum,
                directional_travel=source.speed * horizon,
                radius=own.radius,
                speed=source.speed,
                initial_eject_x=own.eject_vx,
                initial_eject_y=own.eject_vy,
                directional_eject_speed=0.0,
            )
            normal_templates.append(normal)
            if source.split_radius is None:
                split_templates.append(normal)
                continue

            assert source.split_speed is not None
            child_radius = source.split_radius
            split_templates.append(
                ProxyMotionTemplate(
                    blob_id=own.blob_id,
                    source_blob_id=own.blob_id,
                    base_start_x=own.x,
                    base_start_y=own.y,
                    directional_start=0.0,
                    static_eject_x=own.eject_vx * drag_sum,
                    static_eject_y=own.eject_vy * drag_sum,
                    directional_travel=source.split_speed * horizon,
                    radius=child_radius,
                    speed=source.split_speed,
                    initial_eject_x=own.eject_vx,
                    initial_eject_y=own.eject_vy,
                    directional_eject_speed=0.0,
                )
            )
            placement = 2.0 * child_radius + SAME_PLAYER_OVERLAP_EPSILON
            split_templates.append(
                ProxyMotionTemplate(
                    blob_id=next_proxy_id,
                    source_blob_id=own.blob_id,
                    base_start_x=own.x,
                    base_start_y=own.y,
                    directional_start=placement,
                    static_eject_x=0.0,
                    static_eject_y=0.0,
                    directional_travel=(
                        source.split_speed * horizon + SPLIT_EJECT_SPEED * drag_sum
                    ),
                    radius=child_radius,
                    speed=source.split_speed,
                    initial_eject_x=0.0,
                    initial_eject_y=0.0,
                    directional_eject_speed=SPLIT_EJECT_SPEED,
                )
            )
            next_proxy_id += 1
        return tuple(normal_templates), tuple(split_templates)

    def _proxy_project_action(
        self,
        *,
        node: SearchNode,
        action: Action,
        arena_size: float,
        horizon: int,
        unit: tuple[float, float] | None = None,
        own_sources: tuple[ProxyOwnSource, ...] | None = None,
        motion_templates: tuple[ProxyMotionTemplate, ...] | None = None,
    ) -> ProxyMovement:
        """Project split placement and repeated movement in O(blob count)."""

        if unit is None:
            unit = normalise(action.direction)
        horizon = max(1, horizon)
        if motion_templates is None:
            if own_sources is None:
                generated_sources: list[ProxyOwnSource] = []
                remaining_slots = MAX_BLOB_COUNT - len(node.own_blobs)
                for blob in sorted(node.own_blobs, key=lambda own: own.blob_id):
                    if remaining_slots > 0 and blob.mass >= SPLIT_MIN_MASS:
                        split_radius = blob.radius / SQRT2
                        split_speed = player_speed(split_radius)
                        remaining_slots -= 1
                    else:
                        split_radius = None
                        split_speed = None
                    generated_sources.append(
                        ProxyOwnSource(
                            blob=blob,
                            speed=player_speed(blob.radius),
                            split_radius=split_radius,
                            split_speed=split_speed,
                        )
                    )
                own_sources = tuple(generated_sources)
            normal_templates, split_templates = self._proxy_motion_templates(
                node,
                own_sources,
                horizon,
            )
            motion_templates = split_templates if action.split else normal_templates

        motions = self._proxy_blob_motion_active
        motions.clear()
        expected_useful = 0.0
        actual_useful = 0.0
        scratch = self._proxy_blob_motion_scratch
        for template in motion_templates:
            radius = template.radius
            upper = arena_size - radius
            if template.directional_start > 0.0:
                start_x = template.base_start_x + unit[0] * template.directional_start
                if start_x < radius:
                    start_x = radius
                elif start_x > upper:
                    start_x = upper
                start_y = template.base_start_y + unit[1] * template.directional_start
                if start_y < radius:
                    start_y = radius
                elif start_y > upper:
                    start_y = upper
            else:
                start_x = template.base_start_x
                start_y = template.base_start_y
            projected_x = (
                start_x
                + unit[0] * template.directional_travel
                + template.static_eject_x
            )
            if projected_x < radius:
                projected_x = radius
            elif projected_x > upper:
                projected_x = upper
            projected_y = (
                start_y
                + unit[1] * template.directional_travel
                + template.static_eject_y
            )
            if projected_y < radius:
                projected_y = radius
            elif projected_y > upper:
                projected_y = upper
            short_x = start_x
            short_y = start_y
            eject_x = template.initial_eject_x
            eject_y = template.initial_eject_y
            directional_eject = template.directional_eject_speed
            short_path = [(short_x, short_y)]
            for _ in range(min(horizon, self._effective_proxy_dynamic_horizon)):
                short_x = _clamp(
                    short_x + unit[0] * (template.speed + directional_eject) + eject_x,
                    radius,
                    upper,
                )
                short_y = _clamp(
                    short_y + unit[1] * (template.speed + directional_eject) + eject_y,
                    radius,
                    upper,
                )
                short_path.append((short_x, short_y))
                eject_x = _damped(eject_x)
                eject_y = _damped(eject_y)
                directional_eject = _damped(directional_eject)
            short_path_tuple = tuple(short_path)
            motion_index = len(motions)
            if motion_index < len(scratch):
                motion = scratch[motion_index]
                motion.blob_id = template.blob_id
                motion.source_blob_id = template.source_blob_id
                motion.start_x = start_x
                motion.start_y = start_y
                motion.x = projected_x
                motion.y = projected_y
                motion.radius = radius
                motion.speed = template.speed
                motion.delta_x = projected_x - start_x
                motion.delta_y = projected_y - start_y
                motion.length_sq = (
                    motion.delta_x * motion.delta_x + motion.delta_y * motion.delta_y
                )
                motion.short_path = short_path_tuple
            else:
                delta_x = projected_x - start_x
                delta_y = projected_y - start_y
                motion = ProxyBlobMotion(
                    blob_id=template.blob_id,
                    source_blob_id=template.source_blob_id,
                    start_x=start_x,
                    start_y=start_y,
                    x=projected_x,
                    y=projected_y,
                    radius=radius,
                    speed=template.speed,
                    delta_x=delta_x,
                    delta_y=delta_y,
                    length_sq=delta_x * delta_x + delta_y * delta_y,
                    short_path=short_path_tuple,
                )
                scratch.append(motion)
            motions.append(motion)
            mass = radius * radius
            expected_useful += mass * template.speed * horizon
            actual_useful += mass * max(
                0.0,
                (projected_x - start_x) * unit[0] + (projected_y - start_y) * unit[1],
            )
        efficiency = (
            1.0
            if expected_useful <= EPSILON
            else _clamp(actual_useful / expected_useful, 0.0, 1.0)
        )
        movement = self._proxy_movement_scratch
        movement.efficiency = efficiency
        return movement

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
        profile_started = self._profile_start()
        parent_utility_started = self._profile_start()
        before = self._cached_search_utility(
            node,
            foods=foods,
            viruses=viruses,
            arena_size=arena_size,
            safety_weight=safety_weight,
        )
        self._record_profile(
            "step_parent_utility",
            parent_utility_started,
        )
        physics_started = self._profile_start()
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
        self._record_profile("physics", physics_started)
        if not result.node.own_blobs:
            dead_node = replace(result.node, score=node.score - 100_000.0)
            shaped_result = replace(result, node=dead_node)
            self._record_profile("step", profile_started)
            return shaped_result
        child_utility_started = self._profile_start()
        after = self._cached_search_utility(
            result.node,
            foods=foods,
            viruses=viruses,
            arena_size=arena_size,
            safety_weight=safety_weight,
            hazard_summary=result.hazard_summary,
        )
        self._record_profile(
            "step_child_utility",
            child_utility_started,
        )
        shaping_started = self._profile_start()
        direction = normalise(action.direction)
        utility_delta = after - before
        movement_efficiency = result.movement_efficiency
        movement_penalty = (
            1.0 - movement_efficiency
        ) * self._MOVEMENT_INEFFICIENCY_PENALTY
        turn_penalty = self._turn_cost(node.last_direction, direction) * 0.6
        self._record_profile_value("step_utility_delta", utility_delta)
        self._record_profile_value("step_movement_penalty", movement_penalty)
        self._record_profile_value("step_turn_penalty", turn_penalty)
        shaped_node = replace(
            result.node,
            score=(node.score + utility_delta - movement_penalty - turn_penalty),
        )
        # Non-split danger is priced by retained mass rather than removed as a
        # fatal branch.  Only an immediately losing split keeps the parent's
        # physical admissibility rejection.
        shaped_result = replace(
            result,
            node=shaped_node,
            fatal=result.fatal and action.split,
        )
        self._record_profile("step_shaping", shaping_started)
        self._record_profile("step", profile_started)
        return shaped_result

    def _cached_search_utility(
        self,
        node: SearchNode,
        *,
        foods,
        viruses,
        arena_size: float,
        safety_weight: float,
        hazard_summary: HazardSummary | None = None,
    ) -> float:
        profile_started = self._profile_start()
        identity_key = id(node)
        identity_cached = self._utility_identity_cache.get(identity_key)
        if (
            identity_cached is not None
            and identity_cached[0] is node
            and identity_cached[1] == arena_size
            and identity_cached[2] == safety_weight
        ):
            self._record_profile_count("utility_hit")
            self._record_profile("utility", profile_started)
            return identity_cached[3]
        key = (
            node.own_blobs,
            node.enemies,
            node.eaten_food_ids,
            node.consumed_virus_ids,
            arena_size,
            safety_weight,
        )
        cached = self._utility_cache.get(key)
        if cached is not None:
            self._utility_identity_cache[identity_key] = (
                node,
                arena_size,
                safety_weight,
                cached,
            )
            self._record_profile_count("utility_hit")
            self._record_profile("utility", profile_started)
            return cached
        self._record_profile_count("utility_miss")
        compute_started = self._profile_start()
        value = self._search_utility(
            node,
            foods=foods,
            viruses=viruses,
            arena_size=arena_size,
            safety_weight=safety_weight,
            hazard_summary=hazard_summary,
        )
        self._record_profile("utility_compute", compute_started)
        self._utility_cache[key] = value
        self._utility_identity_cache[identity_key] = (
            node,
            arena_size,
            safety_weight,
            value,
        )
        self._record_profile("utility", profile_started)
        return value

    def _search_utility(
        self,
        node,
        *,
        foods,
        viruses,
        arena_size: float,
        safety_weight: float,
        hazard_summary: HazardSummary | None = None,
    ) -> float:
        """Return one mass-normalised utility shared by every action type."""
        if not node.own_blobs:
            return -1_000_000.0

        hazard = hazard_summary or self._hazard_summary(
            node.own_blobs,
            node.enemies,
            safety_weight,
            arena_size,
        )
        safe_mass = hazard.safe_mass
        # Fragment outcomes are correlated: one predator can sweep several
        # pieces in sequence.  An independent-union probability would make a
        # 16-way split artificially look almost certain to survive.
        continuation_probability = hazard.continuation_probability

        opportunities: list[float] = []
        for food in foods:
            if food.food_id in node.eaten_food_ids:
                continue
            gap = min(
                max(0.0, math.dist(own.pos, food.pos) - own.radius)
                for own in node.own_blobs
            )
            opportunities.append(FOOD_RADIUS * FOOD_RADIUS * math.exp(-gap / 6.0))

        for virus in viruses:
            if virus.virus_id in node.consumed_virus_ids:
                continue
            expected_mass = self._virus_expected_mass(node, virus, arena_size)
            if expected_mass is not None:
                # One virus is one resource even when several of our blobs can
                # reach it.  Count the best acquisition route, not every origin.
                opportunities.append(expected_mass)

        for enemy in node.enemies:
            if enemy.stale_rounds:
                continue
            expected_mass = self._prey_expected_mass(node, enemy, arena_size)
            if expected_mass > 0.0:
                opportunities.append(expected_mass)

        opportunities.sort(reverse=True)
        opportunity_mass = sum(
            value * weight
            for value, weight in zip(opportunities[:3], (1.0, 0.25, 0.1), strict=False)
        )
        competitor_mass_debt = sum(
            self._rival_values.get(enemy.player_id, 0.0) * enemy.mass
            for enemy in node.enemies
        )
        continuation_opportunity = continuation_probability * opportunity_mass
        cohesion_mass = (
            self._formation_cohesion_mass(
                node.own_blobs,
                tuple(blob.merge_cooldown for blob in node.own_blobs),
            )
            if self.proxy_cohesion_enabled
            else 0.0
        )
        recovery_penalty = self.recovery_mass * (1.0 - continuation_probability)
        self._record_profile_value("utility_safe_mass", safe_mass)
        self._record_profile_value(
            "utility_continuation_opportunity",
            continuation_opportunity,
        )
        self._record_profile_value(
            "utility_competitor_debt",
            competitor_mass_debt,
        )
        self._record_profile_value("utility_cohesion_mass", cohesion_mass)
        self._record_profile_value("utility_recovery_penalty", recovery_penalty)
        return 100.0 * (
            safe_mass
            + continuation_opportunity
            + cohesion_mass
            - competitor_mass_debt
            - recovery_penalty
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
            squared_distance(self._node_geometry(node).center, enemy.pos),
            enemy.player_id,
            enemy.blob_id,
        )

    def _audit_root_candidate_ranking(
        self,
        *,
        node: SearchNode,
        actions: tuple[Action, ...],
        foods: tuple[FoodModel, ...],
        viruses: tuple[VirusModel, ...],
        arena_size: float,
        safety_weight: float,
        aggression: float,
        transition_budget: int | None,
    ) -> float:
        if self._audit_every_n <= 0 or self._current_round % self._audit_every_n:
            return 0.0

        # Schedule offline work after the search has fixed this turn's action.
        # Running exact audit transitions here would consume its deadline.
        self._pending_audit = RootAuditRequest(
            node=node,
            actions=actions,
            foods=foods,
            viruses=viruses,
            arena_size=arena_size,
            safety_weight=safety_weight,
            aggression=aggression,
            transition_budget=transition_budget,
        )
        return 0.0

    def _run_pending_audit(self) -> None:
        if self._pending_audit is None:
            return
        audit_started = perf_counter()
        request = self._pending_audit
        self._pending_audit = None

        # Audit is diagnostic work, not part of the search. Give it isolated
        # caches and counters so it cannot warm or distort the measured search.
        saved_caches = {name: getattr(self, name) for name in self._AUDIT_CACHE_NAMES}
        for name in self._AUDIT_CACHE_NAMES:
            setattr(self, name, {})
        profile_was_active = self._profile_active
        self._profile_active = False
        try:
            audit_node = replace(request.node)
            results = [
                self._step(
                    node=audit_node,
                    action=action,
                    foods=request.foods,
                    viruses=request.viruses,
                    arena_size=request.arena_size,
                    first_step=True,
                    safety_weight=request.safety_weight,
                    aggression=request.aggression,
                )
                for action in request.actions
            ]
        finally:
            for name, cache in saved_caches.items():
                setattr(self, name, cache)
            self._profile_active = profile_was_active

        if results:
            exact_scores = [self._terminal_score(result.node) for result in results]
            raw_best_rank = max(range(len(results)), key=exact_scores.__getitem__) + 1
            admissible_indices = [
                index for index, result in enumerate(results) if not result.fatal
            ]
            exact_best_rank = (
                max(admissible_indices, key=exact_scores.__getitem__) + 1
                if admissible_indices
                else None
            )
            action_limit = self._actions_per_node_limit(0)
            recall_k = len(request.actions) if action_limit is None else action_limit
            if request.transition_budget is not None:
                recall_k = min(recall_k, request.transition_budget)
            self._audit_samples += 1
            self._audit_hits += (
                exact_best_rank is not None and exact_best_rank <= recall_k
            )
            self._audit_last_exact_rank = exact_best_rank
            self._audit_last_raw_rank = raw_best_rank
            if exact_best_rank is not None:
                exact_best_index = exact_best_rank - 1
                exact_best_action = request.actions[exact_best_index]
                self._audit_last_exact_reason = exact_best_action.reason
                self._audit_last_exact_split = exact_best_action.split
                self._audit_last_exact_regret = (
                    exact_scores[exact_best_index] - exact_scores[0]
                )
        self._audit_last_fatal_count = sum(result.fatal for result in results)
        elapsed = perf_counter() - audit_started
        self._audit_spent_seconds += elapsed
        self._record_profile_count("audit_transitions", len(results))
        self._record_profile_count(
            "audit_fatal_candidates", self._audit_last_fatal_count
        )
        self._record_profile_elapsed("audit", elapsed)
        self._record_profile_value("audit_elapsed_ms", elapsed * 1000.0)

    def _actions_per_node_limit(self, depth_index: int) -> int:
        return (
            self.exact_root_action_limit
            if depth_index == 0
            else self.exact_deeper_action_limit
        )

    def _prey_expected_mass(
        self,
        node: SearchNode,
        enemy: EnemyBlob,
        arena_size: float,
    ) -> float:
        cache_key = (id(node), enemy.key, arena_size)
        cached = self._prey_expected_mass_cache.get(cache_key)
        if cached is not None and cached[0] is node and cached[1] is enemy:
            self._record_cache_access("prey", hit=True)
            return cached[2]

        capture_probability = 0.0
        for own in node.own_blobs:
            if not can_eat_player_blob(own.radius, enemy.radius):
                continue
            probability = self._prey_capture_probability(own, enemy, arena_size)
            capture_probability = max(capture_probability, probability)
        rival_value = self._rival_values.get(enemy.player_id, 0.0)
        expected_mass = capture_probability * enemy.mass * (1.0 + rival_value)
        self._prey_expected_mass_cache[cache_key] = (node, enemy, expected_mass)
        self._record_cache_access("prey", hit=False)
        return expected_mass

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
        self._record_profile_count("prey_projection_moves")
        enemy_direction = normalise(enemy.direction)
        flee_direction = normalise(
            (
                direction[0] * 0.62 + enemy_direction[0] * 0.38,
                direction[1] * 0.62 + enemy_direction[1] * 0.38,
            )
        )
        enemy_speed = player_speed(enemy.radius)
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
            math.hypot(enemy_x - moved_own.x, enemy_y - moved_own.y) - own.radius,
        )
        closing = gap - next_gap
        temperature = self._CAPTURE_CLOSING_TEMPERATURE
        scaled = _clamp(closing / temperature, -40.0, 40.0)
        effective_closing = temperature * math.log1p(math.exp(scaled))
        return math.exp(-gap / max(self._CAPTURE_HORIZON * effective_closing, EPSILON))

    def _virus_retained_mass_fraction(
        self,
        node,
        origin,
        virus,
        arena_size: float,
    ) -> float:
        profile_started = self._profile_start()
        # Contact position and travel time change which predators can sweep the
        # fragments, so equal-radius viruses are not interchangeable.
        cache_key = (id(node), origin.blob_id, virus.virus_id)
        cached = self._virus_retention_cache.get(cache_key)
        if cached is not None and cached[0] is node:
            self._record_profile_count("virus_retention_hit")
            self._record_profile(
                "virus_retention",
                profile_started,
            )
            return cached[1]
        self._record_profile_count("virus_retention_miss")
        matching = next(
            (blob for blob in node.own_blobs if blob.blob_id == origin.blob_id),
            None,
        )
        if matching is None:
            self._record_profile(
                "virus_retention",
                profile_started,
            )
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
        self._record_profile(
            "virus_retention",
            profile_started,
        )
        return retained

    def _risk_enemies(
        self,
        enemies: tuple[EnemyBlob, ...],
    ) -> tuple[EnemyBlob, ...]:
        """Return current and future envelopes once per rollout state."""

        profile_started = self._profile_start()
        cached = self._risk_envelope_cache.get(id(enemies))
        if cached is not None and cached[0] is enemies:
            self._record_profile_count("risk_envelope_hit")
            self._record_profile(
                "risk_envelope",
                profile_started,
            )
            return cached[1]
        self._record_profile_count("risk_envelope_miss")
        risk_enemies = (*enemies, *self._future_enemy_envelopes(enemies))
        self._risk_envelope_cache[id(enemies)] = (enemies, risk_enemies)
        self._record_profile(
            "risk_envelope",
            profile_started,
        )
        return risk_enemies

    def _hazard_summary(
        self,
        own_blobs: tuple[OwnBlob, ...] | list[OwnBlob],
        enemies: tuple[EnemyBlob, ...],
        safety_weight: float,
        arena_size: float,
    ) -> HazardSummary:
        """Evaluate each own/enemy geometry once for one physical state."""

        own_tuple = own_blobs if isinstance(own_blobs, tuple) else tuple(own_blobs)
        cache_key = (own_tuple, enemies, arena_size, safety_weight)
        cached = self._hazard_summary_cache.get(cache_key)
        if cached is not None:
            self._record_cache_access("hazard", hit=True)
            return cached

        survival_midpoint = (
            self.survival_midpoint_base + self.survival_midpoint_scale * safety_weight
        )
        temperature = max(self.survival_temperature, 0.1)
        min_margin = math.inf
        endangered_blob_ids: set[int] = set()
        safe_mass = 0.0
        survival_probabilities: list[float] = []
        risk_enemies = self._risk_enemies(enemies)
        for own in own_tuple:
            survival = 1.0
            for enemy in risk_enemies:
                if not can_eat_player_blob(enemy.radius, own.radius):
                    continue
                danger_radius = enemy.radius
                if _can_split_eat(enemy.radius, own.radius):
                    danger_radius = max(
                        danger_radius,
                        _split_chain_attack_reach(enemy.radius, own.radius),
                    )
                raw_margin = math.dist(own.pos, enemy.pos) - danger_radius
                min_margin = min(min_margin, raw_margin)
                if raw_margin <= 0.0 and enemy.blob_id >= 0:
                    endangered_blob_ids.add(own.blob_id)
                utility_margin = (
                    raw_margin
                    - enemy.stale_rounds * 0.35
                    - self._wall_trap_factor(own, enemy, arena_size) * 4.0
                )
                scaled = _clamp(
                    (utility_margin - survival_midpoint) / temperature,
                    -40.0,
                    40.0,
                )
                survival = min(survival, 1.0 / (1.0 + math.exp(-scaled)))
            survival_probabilities.append(survival)
            safe_mass += own.mass * survival

        summary = HazardSummary(
            min_margin=min_margin,
            unavoidable=(
                bool(own_tuple) and len(endangered_blob_ids) == len(own_tuple)
            ),
            safe_mass=safe_mass,
            continuation_probability=max(survival_probabilities, default=0.0),
        )
        self._hazard_summary_cache[cache_key] = summary
        self._record_cache_access("hazard", hit=False)
        return summary

    def _risk_analysis(
        self,
        own_blobs: list[OwnBlob],
        enemies: tuple[EnemyBlob, ...],
        safety_weight: float,
        arena_size: float = ARENA_SIZE,
    ) -> tuple[float, float, bool, HazardSummary]:
        summary = self._hazard_summary(
            own_blobs,
            enemies,
            safety_weight,
            arena_size,
        )
        return 0.0, summary.min_margin, summary.unavoidable, summary

    def _risk_score(
        self,
        own_blobs: list[OwnBlob],
        enemies: tuple[EnemyBlob, ...],
        safety_weight: float,
        arena_size: float = ARENA_SIZE,
    ) -> tuple[float, float, bool]:
        """Compute only split admissibility; utility prices continuous risk."""

        summary = self._hazard_summary(
            own_blobs,
            enemies,
            safety_weight,
            arena_size,
        )
        return 0.0, summary.min_margin, summary.unavoidable

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
        center_distance = math.dist(origin.pos, virus.pos)
        contact_gap = max(0.0, center_distance - origin.radius)
        turns_to_contact = math.ceil(
            contact_gap / max(player_speed(origin.radius), EPSILON)
        )
        contact_direction = normalise(
            (virus.pos[0] - origin.x, virus.pos[1] - origin.y)
        )
        contact_x = _clamp(
            origin.x + contact_direction[0] * contact_gap,
            origin.radius,
            arena_size - origin.radius,
        )
        contact_y = _clamp(
            origin.y + contact_direction[1] * contact_gap,
            origin.radius,
            arena_size - origin.radius,
        )
        projected_origin_mass = decayed_mass_after_turns(
            origin.mass,
            turns_to_contact,
            decay_rate=MASS_DECAY_RATE,
            minimum_radius=STARTING_RADIUS,
        )
        total_mass = projected_origin_mass + virus.radius * virus.radius
        piece_radius = math.sqrt(total_mass / piece_count)
        post_radii = [piece_radius]
        post_radii.extend(
            blob.radius for blob in own_blobs if blob.blob_id != origin.blob_id
        )
        enemy_tuple = enemies if isinstance(enemies, tuple) else tuple(enemies)
        merge_horizon = max(8, min(turns_to_contact, 16))
        risk_enemies = tuple(
            enemy
            for enemy in (
                *enemy_tuple,
                *self._future_enemy_envelopes(
                    enemy_tuple,
                    horizon=merge_horizon,
                ),
            )
            if any(can_eat_player_blob(enemy.radius, radius) for radius in post_radii)
        )
        if not risk_enemies:
            return 1.0
        pursuit_turns = min(turns_to_contact, 12)
        projected_risk_enemies: list[EnemyBlob] = []
        for enemy in risk_enemies:
            pursuit = normalise((contact_x - enemy.x, contact_y - enemy.y))
            travel = pursuit_turns * player_speed(enemy.radius)
            projected_risk_enemies.append(
                replace(
                    enemy,
                    x=_clamp(
                        enemy.x + pursuit[0] * travel,
                        enemy.radius,
                        arena_size - enemy.radius,
                    ),
                    y=_clamp(
                        enemy.y + pursuit[1] * travel,
                        enemy.radius,
                        arena_size - enemy.radius,
                    ),
                )
            )
        offsets = self._virus_fragment_layout(piece_radius, piece_count)
        post_blobs = [
            (blob.x, blob.y, blob.radius, blob.mass)
            for blob in own_blobs
            if blob.blob_id != origin.blob_id
        ]
        post_blobs.extend(
            (
                _clamp(
                    contact_x + offset_x,
                    piece_radius,
                    arena_size - piece_radius,
                ),
                _clamp(
                    contact_y + offset_y,
                    piece_radius,
                    arena_size - piece_radius,
                ),
                piece_radius,
                piece_radius * piece_radius,
            )
            for offset_x, offset_y in offsets
        )
        if not post_blobs:
            return 0.0

        retained_mass = 0.0
        for blob_x, blob_y, blob_radius, blob_mass in post_blobs:
            retention = 1.0
            for enemy in projected_risk_enemies:
                if not can_eat_player_blob(enemy.radius, blob_radius):
                    continue
                center_distance = math.hypot(blob_x - enemy.x, blob_y - enemy.y)
                immediate_radius = enemy.radius
                chain_radius = enemy.radius
                if _can_split_eat(enemy.radius, blob_radius):
                    immediate_radius = max(
                        immediate_radius,
                        _split_attack_reach(enemy.radius),
                    )
                    chain_radius = max(
                        chain_radius,
                        _split_chain_attack_reach(enemy.radius, blob_radius),
                    )
                uncertainty_distance = enemy.stale_rounds * 0.35
                immediate_margin = (
                    center_distance - immediate_radius - uncertainty_distance
                )
                if immediate_margin <= 0.0:
                    retention = 0.0
                    break
                wall_trap = self._wall_trap_factor_at(
                    blob_x,
                    blob_y,
                    blob_radius,
                    enemy,
                    arena_size,
                )
                chain_margin = center_distance - chain_radius - uncertainty_distance
                if chain_margin <= 0.0 and chain_radius > immediate_radius:
                    chain_pressure = _clamp(
                        -chain_margin / (chain_radius - immediate_radius),
                        0.0,
                        1.0,
                    )
                    predator_retention = 1.0 - chain_pressure * (
                        0.65 + 0.35 * wall_trap
                    )
                    retention = min(retention, predator_retention)
                    continue
                pressure = _clamp((8.0 - chain_margin) / 8.0, 0.0, 1.0)
                if pressure <= 0.0:
                    continue
                predator_retention = 1.0 - pressure * (0.55 + 0.45 * wall_trap)
                retention = min(retention, predator_retention)
            retained_mass += blob_mass * retention
        return _clamp(
            retained_mass / max(sum(blob[3] for blob in post_blobs), EPSILON),
            0.0,
            1.0,
        )

    def _virus_fragment_layout(
        self,
        piece_radius: float,
        piece_count: int,
    ) -> tuple[tuple[float, float], ...]:
        """Return deterministic offsets without retaining unbounded radii."""

        layout_key = (piece_radius, piece_count)
        offsets = self._virus_fragment_layout_cache.get(layout_key)
        self._record_cache_access("virus_layout", hit=offsets is not None)
        if offsets is not None:
            return offsets
        offsets = virus_replacement_positions(
            center_x=0.0,
            center_y=0.0,
            piece_radius=piece_radius,
            piece_count=piece_count,
            overlap_epsilon=SAME_PLAYER_OVERLAP_EPSILON,
        )
        if len(self._virus_fragment_layout_cache) >= (
            self._VIRUS_FRAGMENT_LAYOUT_CACHE_LIMIT
        ):
            self._virus_fragment_layout_cache.pop(
                next(iter(self._virus_fragment_layout_cache))
            )
        self._virus_fragment_layout_cache[layout_key] = offsets
        return offsets

    def _proxy_virus_actions(
        self,
        targets: tuple[ProxyVirusTarget, ...],
        *,
        limit: int,
    ) -> list[Action]:
        """Turn the shared eligible-virus analysis into semantic actions.

        Eligibility, source selection, distance, and direction were already
        computed for proxy scoring. Candidate enumeration must not repeat that
        same geometry pass.
        """

        scored = [
            (
                target.gap,
                target.virus.virus_id,
                Action(target.direction, reason="virus_harvest"),
            )
            for target in targets
        ]
        scored.sort(key=lambda item: (item[0], item[1]))
        return [action for _, _, action in scored[:limit]]

    def _can_still_consume_virus_at_contact(self, blob, virus) -> bool:
        self._record_profile_count("virus_consumability_calls")
        if not _can_consume_virus(blob.radius, virus.radius):
            return False
        center_gap = max(0.0, math.dist(blob.pos, virus.pos) - blob.radius)
        turns_to_contact = math.ceil(center_gap / player_speed(blob.radius))
        projected_mass = decayed_mass_after_turns(
            blob.mass,
            turns_to_contact,
            decay_rate=MASS_DECAY_RATE,
            minimum_radius=STARTING_RADIUS,
        )
        return _can_consume_virus(math.sqrt(projected_mass), virus.radius)

    def _safety_weight(self, rank_position: int, progress: float) -> float:
        rank_strength = max(0.0, min(1.0, (4.0 - rank_position) / 3.0))
        # Replay deaths show that a first elimination usually starts a respawn
        # loop even from sixth or seventh place.  Survival therefore has a
        # meaningful baseline value at every rank; the continuous lead/progress
        # term still makes preserving a winning mass advantage more valuable.
        return 1.3 + rank_strength * (0.35 + 0.85 * progress)
