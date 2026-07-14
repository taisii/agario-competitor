from __future__ import annotations

"""Shallow, wide local game search for deliberate, aggressive movement.

The exact simulator remains responsible for the selected first move.  This
module adds a broad two-step local search in front of it.  Every root action is
compared against many immediate continuations, then the whole plan is rebuilt
next turn.  It deliberately avoids a third step because opponent motion and
the visible state change too quickly for that prediction to remain reliable.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
import heapq
import math
import os
from time import perf_counter

from lib.config.arena import ARENA_SIZE
from lib.config.player import (
    EAT_SIZE_RATIO,
    FOOD_RADIUS,
    MASS_DECAY_RATE,
    STARTING_RADIUS,
)
from lib.models.food_model import FoodModel
from lib.models.virus_model import VirusModel
from strategies.base import StrategyContext, StrategyDecision
from strategies.expected_final_mass import ExpectedFinalMassStrategy
from strategies.features import can_eat_player_blob, normalise, player_speed
from strategies.receding_horizon import (
    Action,
    EnemyBlob,
    OwnBlob,
    PlanningTurn,
    ProxyAnalysis,
    SearchNode,
    StepResult,
    TAU,
    ThreatAwareRecedingHorizonStrategy,
    _can_consume_virus,
    _rotate,
    _split_chain_attack_reach,
)


@dataclass(frozen=True)
class LocalTarget:
    """One nearby resource shared by every root rollout."""

    kind: str
    pos: tuple[float, float]
    value: float
    identity: tuple[int, int]


@dataclass(frozen=True, slots=True)
class LocalEnemyResponse:
    """Geometry that is constant across every local rollout state."""

    enemy: EnemyBlob
    speed: float


@dataclass(frozen=True, slots=True)
class AggregateSafety:
    """Cheap fragment-level safety result used by the dense planner tier.

    ``fatal`` is deliberately separate from the continuous value.  A proxy
    opportunity score must never make a fatal split outrank an unsplit escape
    merely because every root happens to be threatened.
    """

    value: float
    fatal: bool
    threatened_mass: float
    worst_margin: float


@dataclass(frozen=True, slots=True)
class LocalRolloutState:
    """Physical state carried through the two-step local rollout.

    Keeping every fragment is essential: reducing a split to its leading child
    loses half the player's mass and can make both safety and option checks
    choose the opposite action from the exact first-step simulator.
    """

    own_blobs: tuple[OwnBlob, ...]
    eaten_food_ids: frozenset[int]
    split_performed: bool = False
    _mass: float = field(init=False, repr=False, compare=False)
    _primary: OwnBlob = field(init=False, repr=False, compare=False)
    _pos: tuple[float, float] = field(init=False, repr=False, compare=False)
    _hash: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.own_blobs:
            raise ValueError("a local rollout state must contain an own blob")
        mass = sum(blob.mass for blob in self.own_blobs)
        object.__setattr__(self, "_mass", mass)
        object.__setattr__(
            self,
            "_primary",
            max(self.own_blobs, key=lambda blob: blob.radius),
        )
        object.__setattr__(
            self,
            "_pos",
            (
                sum(blob.x * blob.mass for blob in self.own_blobs) / mass,
                sum(blob.y * blob.mass for blob in self.own_blobs) / mass,
            ),
        )
        # A rollout state is used repeatedly by the transition, safety,
        # terminal-option, and projected-node caches.  Dataclass' generated
        # hash would rescan every fragment on each lookup, which is especially
        # expensive after a 16-way virus split.
        object.__setattr__(
            self,
            "_hash",
            hash((self.own_blobs, self.eaten_food_ids, self.split_performed)),
        )

    def __hash__(self) -> int:
        return self._hash

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def primary(self) -> OwnBlob:
        return self._primary

    @property
    def radius(self) -> float:
        return self.primary.radius

    @property
    def pos(self) -> tuple[float, float]:
        return self._pos


@dataclass(frozen=True, slots=True)
class LocalCaptureOption:
    """Static geometry for one prey option and its possible virus follow-up."""

    enemy: EnemyBlob
    chain_viruses: tuple[tuple[VirusModel, float], ...]


@dataclass(slots=True)
class LocalEvaluationContext:
    """Turn-local immutable inputs and exact-state caches for local rollouts.

    Cache keys only contain rollout state because all other dependencies live
    in this context, whose lifetime is one root-ranking call.  This prevents a
    cache entry from leaking across turns while removing long argument lists.
    """

    node: SearchNode
    foods: tuple[FoodModel, ...]
    viruses: tuple[VirusModel, ...]
    arena_size: float
    responses: tuple[LocalEnemyResponse, ...]
    captures: tuple[LocalCaptureOption, ...]
    food_by_id: dict[int, FoodModel]
    food_cells: dict[tuple[int, int], tuple[FoodModel, ...]]
    transition_cache: dict[
        tuple[LocalRolloutState, float, float, bool],
        tuple[LocalRolloutState, float],
    ] = field(default_factory=dict)
    safety_cache: dict[tuple[LocalRolloutState, int], float | None] = field(
        default_factory=dict
    )
    option_cache: dict[LocalRolloutState, float] = field(default_factory=dict)
    projected_node_cache: dict[LocalRolloutState, SearchNode] = field(
        default_factory=dict
    )
    predator_reach_cache: dict[tuple[tuple[int, int], float], float] = field(
        default_factory=dict
    )
    contact_checks: int = 0
    transition_seconds: float = 0.0
    contact_seconds: float = 0.0
    response_seconds: float = 0.0
    terminal_seconds: float = 0.0


@dataclass(slots=True)
class LocalBlobPath:
    """The remaining piece of one blob's path after its latest growth event."""

    start_time: float
    start: tuple[float, float]
    end: tuple[float, float]
    version: int = 0


class LocalTacticalSearchStrategy(ExpectedFinalMassStrategy):
    """Compare many two-step routes against locally rational responses."""

    name = "local_tactical_search"

    _FOOD_RANGE = 11.0
    _TACTICAL_RANGE = 30.0
    # Food, prey, virus, escape, and neutral exploration must all survive the
    # first-stage ranking.  Twelve roots by five continuations is still a
    # two-step search; it widens the comparison without pretending that an
    # opponent trajectory remains predictable for a third step.
    _LOCAL_FOOD_LIMIT = 10
    _LOCAL_ROOT_LIMIT = 12
    _TARGET_DIRECTION_LIMIT = 4
    _DEEP_DIRECTION_LIMIT = 5
    # Exact proxy refinement is substantially more expensive than the local
    # position DP.  Keep its measured safe width independent of the broader
    # local root set.
    _PROXY_REFINE_LIMIT = 6
    _OPTION_ROUTE_RANGE = 14.0
    _LOCAL_DISCOUNT = 0.72
    _LOCAL_FOOD_CELL_SIZE = 4.0

    def __init__(
        self,
        depth: int | None = None,
        width: int | None = None,
        angular_samples: int | None = None,
    ) -> None:
        # One exact transition validates a wide root set; the local proxy adds
        # the second move.  Replanning next turn is more reliable than a third
        # prediction through rapidly changing opponents.
        super().__init__(
            depth=1 if depth is None else depth,
            width=6 if width is None else width,
            angular_samples=6 if angular_samples is None else angular_samples,
        )
        self.max_food = min(self.max_food, 16)
        self.max_enemies = min(self.max_enemies, 6)
        # The local planner is the expensive fine ranking for this strategy.
        # Keep the broad coarse candidate set, but avoid refining more roots
        # than the local planner can consume.
        self.proxy_refine_limit = min(
            self.proxy_refine_limit,
            self._PROXY_REFINE_LIMIT,
        )
        self.proxy_min_refine = min(
            self.proxy_min_refine,
            self._PROXY_REFINE_LIMIT,
        )
        # Official workers are materially slower than the local simulator.
        # Once half of the cumulative competition budget has been spent, keep
        # evaluating every semantic candidate with the cheap proxy and stop
        # adding the second local rollout. This is an execution safety valve,
        # not a game-phase policy switch.
        if "BOT_REPLAY_PROXY_COARSE_AFTER_SECONDS" not in os.environ:
            self.proxy_coarse_after_seconds = 4.0
        # The DP is the broad lookahead.  Preserve the submitted strategy's
        # exact-transition bank so it remains available during the endgame.
        self.transition_budget_scale = 2
        self.minimum_transition_budget = 1
        self.local_continuation_prior = float(
            os.environ.get("BOT_LOCAL_CONTINUATION_PRIOR", "0")
        )
        self._local_root_scores: tuple[tuple[Action, float], ...] = ()
        self._local_target_counts: dict[str, int] = {}
        self._local_response_evaluations = 0
        self._local_contact_checks = 0
        self._local_phase_ms: dict[str, float] = {}
        self.use_aggregate_local_dp = False
        self._local_planner_tier = "exact"
        self._local_aggregate_continuations = 0
        self._local_aggregate_safety_checks = 0
        self._local_aggregate_safety_certificates = 0
        self.required_semantic_root: Action | None = None
        self.required_semantic_transition: dict[str, object] | None = None
        self.root_transition_summaries: dict[
            tuple[bool, int, bool], dict[str, object]
        ] = {}
        self.root_transition_results: dict[tuple[bool, int, bool], StepResult] = {}
        self._advisor_planning_turn: PlanningTurn | None = None
        self._local_candidate_proxy_analysis: ProxyAnalysis | None = None
        self._prepared_tracked_enemies: tuple[int, tuple[EnemyBlob, ...]] | None = None

    def prepare_enemy_memory_for_external_gate(
        self,
        context: StrategyContext,
    ) -> tuple[EnemyBlob, ...]:
        """Update enemy memory once for a cheap caller-owned relevance gate.

        A subsequent ``choose`` in the same round consumes the prepared result
        instead of advancing hidden tracks twice.  If search is skipped, the
        tracker still remains current for the next observation.
        """

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
            self._prepared_tracked_enemies = None
            return ()
        self._own_player_id = int(state.me.player_id)
        self._read_public_moves(context)
        tracked = super()._update_enemy_memory(
            context,
            own_blobs,
            float(state.map.size or ARENA_SIZE),
            viruses=tuple(state.visible_viruses),
        )
        self._prepared_tracked_enemies = (int(state.round), tracked)
        return tracked

    def _update_enemy_memory(
        self,
        context: StrategyContext,
        own_blobs: tuple[OwnBlob, ...],
        arena_size: float,
        viruses: tuple[VirusModel, ...] = (),
    ) -> tuple[EnemyBlob, ...]:
        prepared = self._prepared_tracked_enemies
        if prepared is not None and prepared[0] == int(context.game.state.round):
            self._prepared_tracked_enemies = None
            return prepared[1]
        self._prepared_tracked_enemies = None
        return super()._update_enemy_memory(
            context,
            own_blobs,
            arena_size,
            viruses=viruses,
        )

    def _planning_enemies(
        self,
        context: StrategyContext,
        own_blobs: tuple[OwnBlob, ...],
        arena_size: float,
        viruses: tuple[VirusModel, ...],
    ) -> tuple[EnemyBlob, ...]:
        """Use geometric belief memory only for the local tactical planner."""

        self._read_public_moves(context)
        return self._update_enemy_memory(
            context,
            own_blobs,
            arena_size,
            viruses=viruses,
        )

    def _prepare_turn(self, context: StrategyContext) -> PlanningTurn | None:
        turn = super()._prepare_turn(context)
        self._advisor_planning_turn = turn
        return turn

    def ensure_root_transition_summary(
        self,
        action: Action,
    ) -> dict[str, object] | None:
        """Evaluate an advisor root exactly without rebuilding turn memory."""

        key = self._action_key(action)
        summary = self.root_transition_summaries.get(key)
        if summary is not None:
            return summary
        turn = self._advisor_planning_turn
        if turn is None:
            return None
        self._advisor_physical_root_step(
            node=turn.node,
            action=action,
            foods=turn.foods,
            viruses=turn.viruses,
            arena_size=turn.arena_size,
            first_step=True,
            safety_weight=1.0,
            aggression=1.0,
        )
        return self.root_transition_summaries.get(key)

    def choose(self, context: StrategyContext) -> StrategyDecision:
        self._local_root_scores = ()
        self._local_target_counts = {}
        self._local_response_evaluations = 0
        self._local_contact_checks = 0
        self._local_phase_ms = {}
        self._local_planner_tier = (
            "aggregate" if self.use_aggregate_local_dp else "exact"
        )
        self._local_aggregate_continuations = 0
        self._local_aggregate_safety_checks = 0
        self._local_aggregate_safety_certificates = 0
        self.required_semantic_transition = None
        self.root_transition_summaries = {}
        self.root_transition_results = {}
        self._advisor_planning_turn = None
        decision = super().choose(context)
        diagnostics = dict(decision.diagnostics)
        diagnostics.update(
            local_planner_horizon="2",
            local_target_counts=dict(self._local_target_counts),
            local_roots_ranked=len(self._local_root_scores),
            local_selected_rank=self._local_selected_rank(decision),
            local_top_two_gap=(
                round(
                    self._local_root_scores[0][1] - self._local_root_scores[1][1],
                    6,
                )
                if len(self._local_root_scores) >= 2
                else None
            ),
            local_response_evaluations=self._local_response_evaluations,
            local_contact_checks=self._local_contact_checks,
            local_planner_tier=self._local_planner_tier,
            local_aggregate_continuations=self._local_aggregate_continuations,
            local_aggregate_safety_checks=self._local_aggregate_safety_checks,
            local_aggregate_safety_certificates=(
                self._local_aggregate_safety_certificates
            ),
            local_phase_ms=dict(self._local_phase_ms),
        )
        return replace(decision, diagnostics=diagnostics)

    def _local_selected_rank(self, decision: StrategyDecision) -> int | None:
        selected_key = self._action_key(
            Action(decision.direction, decision.split, decision.reason)
        )
        return next(
            (
                rank
                for rank, (action, _) in enumerate(self._local_root_scores, start=1)
                if self._action_key(action) == selected_key
            ),
            None,
        )

    def _turn_cost(
        self,
        previous: tuple[float, float],
        current: tuple[float, float],
    ) -> float:
        """Penalise reversal, not ordinary steering.

        A ninety-degree turn is free.  The cost starts only after movement has
        a component back toward the previous position and rises smoothly to a
        full U-turn.  Escape and capture can still win when reversing is the
        physically best move.
        """

        before = normalise(previous)
        after = normalise(current)
        dot = max(-1.0, min(1.0, before[0] * after[0] + before[1] * after[1]))
        return 2.0 * max(0.0, -dot) ** 2

    def _actions_per_node_limit(self, depth_index: int) -> int:
        return 2 if depth_index == 0 else 1

    def _uses_compute_time_bank(self) -> bool:
        """Use deterministic fixed work rather than wall-clock action choices."""

        return False

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
        self._local_candidate_proxy_analysis = None
        actions = super()._candidate_actions(
            node=node,
            foods=foods,
            food_targets=food_targets,
            viruses=viruses,
            arena_size=arena_size,
            first_step=first_step,
            allow_split=allow_split,
            angle_offset=angle_offset,
        )
        if not first_step:
            return tuple(
                sorted(
                    actions,
                    key=lambda action: (
                        self._turn_cost(node.last_direction, action.direction),
                        self._action_key(action),
                    ),
                )
            )

        geometry = self._node_geometry(node)
        center = geometry.center
        local_viruses = self._relevant_local_viruses(node, viruses)
        local_enemies = self._local_enemies(node)
        future_escape = self._near_future_predator_escape(node, local_enemies)
        if future_escape != (0.0, 0.0):
            actions = (
                Action(future_escape, reason="future_predator_escape"),
                *actions,
            )

        # Replay evidence supports the density field specifically during the
        # fragile small phase.  Once established, retain the submitted policy's
        # prey/virus value ordering instead of continuing to chase pellets.
        use_small_phase_field = node.total_mass <= 4.0
        field_direction, local_food_targets = self._root_food_field_and_targets(
            center,
            foods,
            node.eaten_food_ids,
        )
        if field_direction != (0.0, 0.0):
            actions = (*actions, Action(field_direction, reason="local_food_field"))

        targets = self._local_targets(
            node,
            foods,
            local_viruses,
            arena_size,
            center=center,
            food_targets=local_food_targets,
        )
        tactical_actions = self._tactical_target_actions(center, targets)
        actions = self._dedupe_actions((*tactical_actions, *actions))
        if self.required_semantic_root is not None:
            actions = self._dedupe_actions((self.required_semantic_root, *actions))
        target_counts: dict[str, int] = {}
        for target in targets:
            target_counts[target.kind] = target_counts.get(target.kind, 0) + 1
        self._local_target_counts = target_counts

        if self._competition_coarse_mode:
            ranked = self._rank_local_coarse_actions(
                node=node,
                foods=foods,
                viruses=viruses,
                actions=actions,
                arena_size=arena_size,
            )
        else:
            local_dp_started = self._profile_start()
            ranked = self._rank_roots_by_local_dp(
                node=node,
                actions=actions,
                foods=foods,
                viruses=local_viruses,
                arena_size=arena_size,
                local_enemies=local_enemies,
                targets=targets,
                use_food_field=use_small_phase_field,
            )
            self._record_profile("local_dp", local_dp_started)

        # Collision and wall/merge escapes are based on exact imminent
        # geometry.  Keep the two escape alternatives ahead of the approximate
        # DP so an anytime cutoff cannot turn a safety fix into a regression.
        mandatory = tuple(
            action
            for action in actions
            if action.reason.startswith(
                (
                    "dangerous_virus_escape",
                    "urgent_wall",
                    "leader_merge_escape",
                    "future_predator_escape",
                )
            )
        )[:2]
        return self._dedupe_actions((*mandatory, *ranked))

    def _proxy_analysis(
        self,
        *,
        node: SearchNode,
        foods: tuple[FoodModel, ...],
        viruses: tuple[VirusModel, ...],
        arena_size: float,
    ) -> ProxyAnalysis:
        """Retain the analysis already built by the base candidate generator."""

        analysis = super()._proxy_analysis(
            node=node,
            foods=foods,
            viruses=viruses,
            arena_size=arena_size,
        )
        self._local_candidate_proxy_analysis = analysis
        return analysis

    def _rank_local_coarse_actions(
        self,
        *,
        node: SearchNode,
        actions: tuple[Action, ...],
        foods: tuple[FoodModel, ...],
        viruses: tuple[VirusModel, ...],
        arena_size: float,
    ) -> tuple[Action, ...]:
        """Score locally added semantic actions on the base coarse scale."""

        analysis = self._local_candidate_proxy_analysis
        if analysis is None:
            return actions
        base_scores = {
            self._action_key(action): score for action, score in self._root_proxy_scores
        }
        geometry = self._node_geometry(node)
        scored = []
        for action in actions:
            action_key = self._action_key(action)
            score = base_scores.get(action_key)
            if score is None:
                score = self._coarse_action_value(
                    node=node,
                    action=action,
                    arena_size=arena_size,
                    proxy_analysis=analysis,
                    node_geometry=geometry,
                )
            scored.append((action, score))
        return tuple(
            action
            for action, _ in sorted(
                scored,
                key=lambda item: (-item[1], self._action_key(item[0])),
            )
        )

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
        return self._finalise_local_step(
            result=result,
            node=node,
            action=action,
            arena_size=arena_size,
            first_step=first_step,
            safety_weight=safety_weight,
        )

    def _advisor_physical_root_step(
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
    ) -> StepResult:
        """Run one advisor root without replay utility shaping.

        The advisor consumes physical mass, event counts, hazards, and the
        resulting geometry only.  ReplayDominance's parent/child utility is a
        ranking score for the ordinary search and is both unused here and
        quadratic in dense post-virus enemy fragments.  Preserve its branch
        admissibility convention so the existing local summary policy remains
        unchanged: a live non-split danger is priced, while only a split may be
        fatal before the local partial-loss reclassification below.
        """

        result = ThreatAwareRecedingHorizonStrategy._step(
            self,
            node=node,
            action=action,
            foods=foods,
            viruses=viruses,
            arena_size=arena_size,
            first_step=first_step,
            safety_weight=safety_weight,
            aggression=aggression,
        )
        if result.node.own_blobs:
            result = replace(result, fatal=result.fatal and action.split)
        return self._finalise_local_step(
            result=result,
            node=node,
            action=action,
            arena_size=arena_size,
            first_step=first_step,
            safety_weight=safety_weight,
        )

    def _finalise_local_step(
        self,
        *,
        result: StepResult,
        node: SearchNode,
        action: Action,
        arena_size: float,
        first_step: bool,
        safety_weight: float,
    ) -> StepResult:
        """Apply the local fatal policy and publish first-root evidence."""

        # Exact search used to reject a split when any one resulting fragment
        # was endangered or already eaten. Preserve the actual lost-mass score,
        # but reserve fatal for the lower bound where every surviving fragment
        # is still unavoidable. This allows a secured capture/virus gain to be
        # compared against a partial loss instead of discarding the action.
        if action.split and result.fatal and result.node.own_blobs:
            if result.hazard_summary is not None:
                unavoidable = result.hazard_summary.unavoidable
            else:
                _, _, unavoidable = self._risk_score(
                    list(result.node.own_blobs),
                    result.node.enemies,
                    safety_weight,
                    arena_size,
                )
            if not unavoidable:
                result = replace(result, fatal=False)

        if first_step:
            physical_mass = result.node.total_mass
            safe_mass = (
                result.hazard_summary.safe_mass
                if result.hazard_summary is not None
                else physical_mass
            )
            summary: dict[str, object] = {
                "fatal": result.fatal,
                "immediate_dead": not result.node.own_blobs,
                "surviving_mass": safe_mass,
                "physical_mass": physical_mass,
                "mass_delta": physical_mass - node.total_mass,
                "min_safety_margin": result.node.min_safety_margin,
                "food_gain": result.node.projected_food - node.projected_food,
                "capture_gain": (
                    result.node.projected_captures - node.projected_captures
                ),
                "virus_gain": result.node.projected_viruses - node.projected_viruses,
            }
            self.root_transition_summaries[self._action_key(action)] = summary
            self.root_transition_results[self._action_key(action)] = result
        if (
            first_step
            and self.required_semantic_root is not None
            and self._action_key(action)
            == self._action_key(self.required_semantic_root)
        ):
            if not result.fatal:
                outcome = "safe"
            elif not result.node.own_blobs:
                outcome = "all_mass_lost"
            elif action.split and result.node.min_safety_margin <= 0.0:
                outcome = "unsafe_split_margin_or_fragment_loss"
            else:
                outcome = "unavoidable_predator"
            self.required_semantic_transition = {
                "outcome": outcome,
                **summary,
            }
        # Fatal transitions intentionally have no surviving own blob.  They
        # are filtered by the caller and must never enter value features whose
        # domain requires ``SearchNode.primary``.
        if result.fatal or not result.node.own_blobs:
            return result
        if first_step:
            future_pressure = self._near_future_predator_pressure(result.node)
            if future_pressure > 0.0:
                result = replace(
                    result,
                    node=replace(
                        result.node,
                        score=result.node.score - future_pressure,
                    ),
                )
        if not first_step or not self._local_root_scores:
            return result

        action_key = self._action_key(action)
        rank = next(
            (
                index
                for index, (candidate, _) in enumerate(self._local_root_scores)
                if self._action_key(candidate) == action_key
            ),
            len(self._local_root_scores) - 1,
        )
        # Rank normalisation is robust to the very different raw scales of a
        # food field, a virus pop, and a split capture.  Exact first-step
        # physics can still overturn the DP, but evaluating more roots no
        # longer erases the continuation value that put a root near the top.
        denominator = max(1, len(self._local_root_scores) - 1)
        continuation_prior = self.local_continuation_prior * (1.0 - rank / denominator)
        return replace(
            result,
            node=replace(
                result.node,
                score=result.node.score + continuation_prior,
            ),
        )

    def _rank_roots_by_local_dp(
        self,
        *,
        node: SearchNode,
        actions: tuple[Action, ...],
        foods,
        viruses,
        arena_size: float,
        local_enemies: tuple[EnemyBlob, ...] | None = None,
        targets: tuple[LocalTarget, ...],
        use_food_field: bool = False,
    ) -> tuple[Action, ...]:
        ranking_started = perf_counter()
        if local_enemies is None:
            local_enemies = self._local_enemies(node)

        split_probe = self._apply_split(
            list(node.own_blobs),
            (1.0, 0.0),
            arena_size,
        )
        split_can_create_child = len(split_probe) > len(node.own_blobs)

        def canonical_action(action: Action) -> Action:
            if action.split and not split_can_create_child:
                return replace(action, split=False)
            return action

        candidate_actions = tuple(canonical_action(action) for action in actions)
        base_scores: dict[tuple[bool, int, bool], float] = {}
        for action, score in self._root_proxy_scores:
            key = self._action_key(canonical_action(action))
            base_scores[key] = max(base_scores.get(key, -math.inf), score)

        def local_root_family(action: Action) -> int:
            reason = action.reason
            if reason.startswith(
                (
                    "dangerous_virus_escape",
                    "urgent_wall",
                    "leader_merge_escape",
                    "future_predator_escape",
                    "leader_imitation_9",
                )
            ):
                family = 0
            elif action.split or "prey" in reason or reason == "leader_imitation_49":
                family = 1
            elif "virus" in reason or reason == "leader_imitation_24":
                family = 2
            elif reason in ("local_food_field", "leader_imitation_59"):
                family = 3
            else:
                family = 4
            return family

        # Canonicalise no-op split commands before any selection or safety
        # logic. This keeps the selector, aggregate shield, and published
        # action in one physical transition regime at the 16-blob and
        # below-minimum-mass boundaries.
        def physical_action_key(action: Action) -> tuple[bool, float, float, bool]:
            unit = normalise(action.direction)
            return (
                unit == (0.0, 0.0),
                unit[0],
                unit[1],
                action.split,
            )

        if not any(not action.split for action in candidate_actions):
            # An unsplit continuation is always a legal control alternative
            # and is the safety baseline for the aggregate split shield. Keep
            # it in the action space even when an attack-heavy proxy emits
            # only split-labelled roots.
            base = next(
                (action for action in candidate_actions if action.reason == "hybrid_base"),
                self.required_semantic_root,
            )
            candidate_actions = (
                *candidate_actions,
                Action(
                    base.direction if base is not None else node.last_direction,
                    reason="aggregate_unsplit_baseline",
                ),
            )
        ranked_rows = sorted(
            (
                (
                    action,
                    self._action_key(action),
                    local_root_family(action),
                )
                for action in candidate_actions
            ),
            key=lambda row: (-base_scores.get(row[1], 0.0), row[1]),
        )

        # Select a constrained physical action set, not merely the first 12
        # semantic labels. The required baseline and a non-split comparator
        # are reserved before family coverage; actions with the same resolved
        # direction/split key cannot consume several rollout slots.
        selected: list[tuple[Action, tuple[bool, int, bool], int]] = []
        selected_physical_keys: set[tuple[bool, float, float, bool]] = set()

        def reserve(row: tuple[Action, tuple[bool, int, bool], int] | None) -> None:
            if row is None:
                return
            physical_key = physical_action_key(row[0])
            if physical_key in selected_physical_keys:
                return
            selected.append(row)
            selected_physical_keys.add(physical_key)

        reserve(next((row for row in ranked_rows if row[0].reason == "hybrid_base"), None))
        reserve(next((row for row in ranked_rows if not row[0].split), None))
        for family in range(4):
            reserve(
                next(
                    (row for row in ranked_rows if row[2] == family),
                    None,
                )
            )
        for row in ranked_rows:
            reserve(row)
            if len(selected) >= self._LOCAL_ROOT_LIMIT:
                break
        selected_rows = tuple(selected[: self._LOCAL_ROOT_LIMIT])
        selected_physical_keys = {
            physical_action_key(row[0]) for row in selected_rows
        }
        unselected_actions_by_key: dict[
            tuple[bool, float, float, bool], Action
        ] = {}
        for action, _, _ in ranked_rows:
            physical_key = physical_action_key(action)
            if physical_key not in selected_physical_keys:
                unselected_actions_by_key.setdefault(physical_key, action)
        unselected_actions = tuple(unselected_actions_by_key.values())
        if self.use_aggregate_local_dp:
            return self._rank_roots_by_aggregate_dp(
                node=node,
                selected_rows=selected_rows,
                unselected_actions=unselected_actions,
                base_scores=base_scores,
                foods=foods,
                viruses=viruses,
                arena_size=arena_size,
                local_enemies=local_enemies,
                targets=targets,
                use_food_field=use_food_field,
                ranking_started=ranking_started,
            )
        evaluation = self._local_evaluation_context(
            node=node,
            foods=foods,
            viruses=viruses,
            arena_size=arena_size,
            local_enemies=local_enemies,
        )

        scored: list[tuple[Action, float]] = []
        rollout_cache: dict[tuple[float, float, bool], float] = {}
        last_direction = normalise(node.last_direction)
        initial_state = LocalRolloutState(
            own_blobs=node.own_blobs,
            eaten_food_ids=node.eaten_food_ids,
        )
        for action, action_key, _ in selected_rows:
            unit = normalise(action.direction)
            rollout_key = (unit[0], unit[1], action.split)
            if rollout_key in rollout_cache:
                cached_rollout = rollout_cache[rollout_key]
                base_value = base_scores.get(action_key, 0.0)
                scored.append((action, base_value + 100.0 * cached_rollout))
                continue
            first_state, first_reward = self._advance_local_state(
                evaluation,
                initial_state,
                unit,
                split=action.split,
            )
            first_safety = self._local_safety_value(
                evaluation,
                state=first_state,
                depth=1,
            )
            if first_safety is None:
                rollout_value = -1_000_000.0
                rollout_cache[rollout_key] = rollout_value
                base_value = base_scores.get(action_key, 0.0)
                scored.append((action, base_value + 100.0 * rollout_value))
                continue
            first_value = (
                first_reward
                + first_safety
                - self._turn_cost_between_units(last_direction, unit)
            )
            rollout_value = first_value + self._LOCAL_DISCOUNT * (
                self._terminal_option_value(
                    evaluation,
                    state=first_state,
                )
            )
            for direction in self._local_deeper_directions(
                first_state.pos,
                unit,
                targets,
                use_food_field=use_food_field,
            ):
                second_state, second_reward = self._advance_local_state(
                    evaluation,
                    first_state,
                    direction,
                )
                second_safety = self._local_safety_value(
                    evaluation,
                    state=second_state,
                    depth=2,
                )
                if second_safety is None:
                    continue
                value = (
                    first_value
                    + self._LOCAL_DISCOUNT
                    * (
                        second_reward
                        + second_safety
                        - self._turn_cost_between_units(unit, direction)
                    )
                    + self._LOCAL_DISCOUNT**2
                    * self._terminal_option_value(
                        evaluation,
                        state=second_state,
                    )
                )
                rollout_value = max(rollout_value, value)
            rollout_cache[rollout_key] = rollout_value
            base_value = base_scores.get(action_key, 0.0)
            scored.append((action, base_value + 100.0 * rollout_value))

        scored.sort(key=lambda item: (-item[1], self._action_key(item[0])))
        self._local_root_scores = tuple(scored)
        self._publish_local_root_scores()
        self._local_contact_checks = evaluation.contact_checks
        measured = (
            evaluation.transition_seconds
            + evaluation.response_seconds
            + evaluation.terminal_seconds
        )
        total = perf_counter() - ranking_started
        self._local_phase_ms = {
            "root": max(0.0, total - measured) * 1000.0,
            "transition": max(
                0.0,
                evaluation.transition_seconds - evaluation.contact_seconds,
            )
            * 1000.0,
            "contact": evaluation.contact_seconds * 1000.0,
            "response": evaluation.response_seconds * 1000.0,
            "terminal": evaluation.terminal_seconds * 1000.0,
        }
        return (*tuple(action for action, _ in scored), *unselected_actions)

    def _rank_roots_by_aggregate_dp(
        self,
        *,
        node: SearchNode,
        selected_rows,
        unselected_actions: tuple[Action, ...],
        base_scores: dict[tuple[bool, int, bool], float],
        foods: tuple[FoodModel, ...],
        viruses: tuple[VirusModel, ...],
        arena_size: float,
        local_enemies: tuple[EnemyBlob, ...],
        targets: tuple[LocalTarget, ...],
        use_food_field: bool,
        ranking_started: float,
    ) -> tuple[Action, ...]:
        """Score the same 12×5 tree with aggregate opportunity features.

        Only resource value is aggregated. Every own fragment remains present
        in movement and predator checks, so a small endangered fragment cannot
        disappear behind a safe center-of-mass estimate.
        """

        if not any(not action.split for action, _, _ in selected_rows):
            raise RuntimeError("aggregate root set must contain an unsplit action")

        scored: list[tuple[Action, float, AggregateSafety]] = []
        risk_enemies = self._risk_enemies(local_enemies)
        danger_reach_cache: dict[tuple[float, float, int], float | None] = {}
        movement_speed_cache: dict[float, float] = {}
        last_direction = normalise(node.last_direction)
        for action, action_key, _ in selected_rows:
            unit = normalise(action.direction)
            first_blobs = self._advance_aggregate_fragments(
                node.own_blobs,
                unit,
                arena_size,
                split=action.split,
                speed_cache=movement_speed_cache,
            )
            first_safety = self._aggregate_fragment_safety(
                first_blobs,
                risk_enemies,
                depth=1,
                danger_reach_cache=danger_reach_cache,
            )
            first_center = self._aggregate_center(first_blobs)
            continuations = self._aggregate_continuations(
                first_center,
                unit,
                targets,
                use_food_field=use_food_field,
            )
            self._local_aggregate_continuations += len(continuations)
            if first_safety.fatal:
                rollout_value = first_safety.value
            else:
                first_value = (
                    self._aggregate_opportunity_value(
                        first_blobs,
                        targets,
                        center=first_center,
                    )
                    + first_safety.value
                    - self._turn_cost_between_units(last_direction, unit)
                )
                rollout_value = first_value
                for direction in continuations:
                    second_blobs = self._advance_aggregate_fragments(
                        first_blobs,
                        direction,
                        arena_size,
                        speed_cache=movement_speed_cache,
                    )
                    second_safety = self._aggregate_fragment_safety(
                        second_blobs,
                        risk_enemies,
                        depth=2,
                        danger_reach_cache=danger_reach_cache,
                    )
                    if second_safety.fatal:
                        continue
                    second_value = (
                        self._aggregate_opportunity_value(second_blobs, targets)
                        + second_safety.value
                        - self._turn_cost_between_units(unit, direction)
                    )
                    rollout_value = max(
                        rollout_value,
                        first_value + self._LOCAL_DISCOUNT * second_value,
                    )
            scored.append(
                (
                    action,
                    base_scores.get(action_key, 0.0) + 100.0 * rollout_value,
                    first_safety,
                )
            )

        # Safety bands are lexicographic, not an arbitrary additive penalty:
        # no finite proxy/opportunity value can lift a fatal split over an
        # unsplit least-bad escape. Within a fatal band, preserve mass first and
        # then choose the route with the least-deep predator overlap.
        scored.sort(key=self._aggregate_root_rank_key)
        scored = self._shield_aggregate_roots(
            node=node,
            scored=scored,
            foods=foods,
            viruses=viruses,
            arena_size=arena_size,
        )
        # The surrounding proxy contract expects descending numeric scores.
        # Publish ordinal scores that exactly preserve the lexicographic order;
        # raw aggregate values remain the final tie-breaker above.
        root_count = len(scored)
        self._local_root_scores = tuple(
            (action, float(root_count - rank))
            for rank, (action, _, _) in enumerate(scored)
        )
        self._publish_local_root_scores()
        self._local_phase_ms = {
            "root": (perf_counter() - ranking_started) * 1000.0,
            "transition": 0.0,
            "contact": 0.0,
            "response": 0.0,
            "terminal": 0.0,
        }
        return (
            *tuple(action for action, _ in self._local_root_scores),
            *unselected_actions,
        )

    def _shield_aggregate_roots(
        self,
        *,
        node: SearchNode,
        scored: list[tuple[Action, float, AggregateSafety]],
        foods: tuple[FoodModel, ...],
        viruses: tuple[VirusModel, ...],
        arena_size: float,
    ) -> list[tuple[Action, float, AggregateSafety]]:
        """Exact-check ranked split roots until a survivable root is found.

        Aggregate opportunity evaluation intentionally omits interaction event
        order. Virus contact and adversarial movement can therefore invalidate
        predator-only geometry. Unsplit roots retain the conservative aggregate
        certificate; split roots use authoritative interaction physics.
        """

        fatal_rows: list[tuple[tuple[Action, float, AggregateSafety], SearchNode]] = []
        for index, row in enumerate(scored):
            action = row[0]
            if not action.split:
                # The aggregate reach is conservative for an unsplit root: it
                # advances every own fragment and gives each predator its full
                # walking/split-chain envelope. Exact validation is retained for
                # every split because fragment loss and virus-pop event order are
                # intentionally outside that model.
                self._local_aggregate_safety_certificates += 1
                return [
                    row,
                    *scored[index + 1 :],
                    *(fatal_row for fatal_row, _ in fatal_rows),
                ]
            result = self._step(
                node=node,
                action=action,
                foods=foods,
                viruses=viruses,
                arena_size=arena_size,
                first_step=True,
                safety_weight=1.0,
                aggression=1.0,
            )
            self._local_aggregate_safety_checks += 1
            if result.fatal:
                fatal_rows.append((row, result.node))
                continue
            return [
                row,
                *scored[index + 1 :],
                *(fatal_row for fatal_row, _ in fatal_rows),
            ]

        # If every root is fatal, retain as much mass and margin as possible.
        # Unsafe split status still places split actions below every unsplit
        # least-bad route.
        fatal_rows.sort(
            key=lambda item: (
                item[0][0].split,
                -item[1].total_mass,
                -item[1].min_safety_margin,
                -item[1].score,
                self._action_key(item[0][0]),
            )
        )
        return [row for row, _ in fatal_rows]

    def _aggregate_root_rank_key(
        self,
        row: tuple[Action, float, AggregateSafety],
    ) -> tuple:
        action, raw_score, safety = row
        if not safety.fatal:
            return (0, 0.0, 0.0, -raw_score, self._action_key(action))
        safety_band = 2 if action.split else 1
        return (
            safety_band,
            safety.threatened_mass,
            -safety.worst_margin,
            -raw_score,
            self._action_key(action),
        )

    def _publish_local_root_scores(self) -> None:
        """Make the local DP, rather than its input proxy, select the move."""

        self._root_proxy_scores = self._local_root_scores
        family_counts: dict[str, int] = {}
        for action, _ in self._local_root_scores:
            family = self._action_family(action)
            family_counts[family] = family_counts.get(family, 0) + 1
        self._root_candidate_families = family_counts
        self._root_proxy_refined = len(self._local_root_scores)

    def _advance_aggregate_fragments(
        self,
        blobs: tuple[OwnBlob, ...],
        direction: tuple[float, float],
        arena_size: float,
        *,
        split: bool = False,
        speed_cache: dict[float, float] | None = None,
    ) -> tuple[OwnBlob, ...]:
        unit = normalise(direction)
        before_move = list(blobs)
        if split:
            before_move = self._apply_split(before_move, unit, arena_size)
        moved = self._move_own_blobs(
            before_move,
            unit,
            arena_size,
            calculate_blocked=False,
            calculate_efficiency=False,
            speed_cache=speed_cache,
        ).blobs
        return moved

    @staticmethod
    def _aggregate_center(
        blobs: tuple[OwnBlob, ...],
    ) -> tuple[float, float]:
        total_mass = sum(blob.mass for blob in blobs)
        return (
            sum(blob.x * blob.mass for blob in blobs) / total_mass,
            sum(blob.y * blob.mass for blob in blobs) / total_mass,
        )

    def _aggregate_opportunity_value(
        self,
        blobs: tuple[OwnBlob, ...],
        targets: tuple[LocalTarget, ...],
        *,
        center: tuple[float, float] | None = None,
    ) -> float:
        if center is None:
            center = self._aggregate_center(blobs)
        largest = max(blobs, key=lambda blob: blob.radius)
        travel = 2.0 * player_speed(largest.radius)
        weights = {"food": 0.35, "virus": 1.35, "prey": 1.55}
        reachable_values = []
        for target in targets:
            gap = max(0.0, math.dist(center, target.pos) - largest.radius)
            if gap > travel:
                continue
            reachable_values.append(
                weights[target.kind] * target.value * math.exp(-gap / max(travel, 1.0))
            )
        reachable_values.sort(reverse=True)
        return sum(reachable_values[:3])

    def _aggregate_fragment_safety(
        self,
        own_blobs: tuple[OwnBlob, ...],
        enemies: tuple[EnemyBlob, ...],
        *,
        depth: int,
        danger_reach_cache: dict[tuple[float, float, int], float | None],
    ) -> AggregateSafety:
        total_mass = sum(blob.mass for blob in own_blobs)
        threatened_mass = 0.0
        value = 0.0
        comparisons = 0
        global_worst_margin = math.inf
        for own in own_blobs:
            worst_margin = math.inf
            for enemy in enemies:
                comparisons += 1
                cache_key = (enemy.radius, own.radius, depth)
                try:
                    danger_reach = danger_reach_cache[cache_key]
                except KeyError:
                    if can_eat_player_blob(enemy.radius, own.radius):
                        speed = player_speed(enemy.radius)
                        danger_reach = max(
                            enemy.radius + speed * depth,
                            _split_chain_attack_reach(enemy.radius, own.radius)
                            + speed * max(0, depth - 1),
                        )
                    else:
                        danger_reach = None
                    danger_reach_cache[cache_key] = danger_reach
                if danger_reach is None:
                    continue
                worst_margin = min(
                    worst_margin,
                    math.dist(own.pos, enemy.pos) - danger_reach,
                )
            global_worst_margin = min(global_worst_margin, worst_margin)
            if worst_margin <= 0.0:
                threatened_mass += own.mass
            elif worst_margin < 4.0:
                value -= own.mass * (4.0 - worst_margin) / 4.0
        self._local_response_evaluations += comparisons
        fatal = threatened_mass >= total_mass
        return AggregateSafety(
            value=value - threatened_mass,
            fatal=fatal,
            threatened_mass=threatened_mass,
            worst_margin=global_worst_margin,
        )

    def _aggregate_continuations(
        self,
        pos: tuple[float, float],
        previous: tuple[float, float],
        targets: tuple[LocalTarget, ...],
        *,
        use_food_field: bool,
    ) -> tuple[tuple[float, float], ...]:
        directions = list(
            self._local_deeper_directions(
                pos,
                previous,
                targets,
                use_food_field=use_food_field,
            )
        )
        seen = {self._direction_sector(direction) for direction in directions}
        for offset in (
            math.pi / 3,
            -math.pi / 3,
            2.0 * math.pi / 3,
            -2.0 * math.pi / 3,
            math.pi,
        ):
            direction = _rotate(previous, offset)
            sector = self._direction_sector(direction)
            if sector in seen:
                continue
            seen.add(sector)
            directions.append(direction)
            if len(directions) >= self._DEEP_DIRECTION_LIMIT:
                break
        return tuple(directions[: self._DEEP_DIRECTION_LIMIT])

    def _local_targets(
        self,
        node: SearchNode,
        foods,
        viruses,
        arena_size: float,
        *,
        center: tuple[float, float] | None = None,
        food_targets: tuple[LocalTarget, ...] | None = None,
    ) -> tuple[LocalTarget, ...]:
        center = node.center if center is None else center
        if food_targets is None:
            _, food_targets = self._root_food_field_and_targets(
                center,
                foods,
                node.eaten_food_ids,
            )
        targets = list(food_targets)

        primary = node.primary
        travel_budget = 2.0 * player_speed(primary.radius)
        contact_budget = primary.radius + travel_budget
        minimum_mass = STARTING_RADIUS * STARTING_RADIUS
        projected_mass = (
            primary.mass
            if primary.mass <= minimum_mass
            else max(
                minimum_mass,
                primary.mass * (1.0 - MASS_DECAY_RATE) ** 2,
            )
        )
        active_foods = tuple(
            food for food in foods if food.food_id not in node.eaten_food_ids
        )

        def setup_value(target_pos: tuple[float, float], target_mass: float) -> float:
            distance = math.dist(center, target_pos)
            if distance > contact_budget:
                return 0.0
            feasible_food = self._two_step_setup_food_mass(
                center=center,
                target=target_pos,
                radius=primary.radius,
                step_distance=player_speed(primary.radius),
                foods=active_foods,
            )
            deficit = EAT_SIZE_RATIO * target_mass - projected_mass
            if deficit < 0.0 or feasible_food <= deficit:
                return 0.0
            unlock_fraction = 1.0 - max(0.0, deficit) / feasible_food
            return (
                target_mass
                * unlock_fraction
                * math.exp(-distance / max(contact_budget, 1.0))
            )

        for virus in viruses:
            if virus.virus_id in node.consumed_virus_ids:
                continue
            distance = math.dist(center, virus.pos)
            if distance > contact_budget:
                continue
            expected_mass = self._virus_expected_mass(node, virus, arena_size)
            value = max(
                0.0,
                expected_mass or 0.0,
                setup_value(virus.pos, virus.radius * virus.radius),
            )
            if value <= 0.0:
                continue
            targets.append(
                LocalTarget(
                    kind="virus",
                    pos=virus.pos,
                    value=value,
                    identity=(1, int(virus.virus_id)),
                )
            )

        for enemy in node.enemies:
            if enemy.stale_rounds:
                continue
            if math.dist(center, enemy.pos) > contact_budget:
                continue
            expected_mass = self._prey_expected_mass(node, enemy, arena_size)
            value = max(
                0.0,
                expected_mass,
                setup_value(enemy.pos, enemy.mass),
            )
            if value <= 0.0:
                continue
            targets.append(
                LocalTarget(
                    kind="prey",
                    pos=enemy.pos,
                    value=value,
                    identity=(2 + enemy.player_id, enemy.blob_id),
                )
            )

        kind_priority = {"prey": 0, "virus": 1, "food": 2}
        return tuple(
            sorted(
                targets,
                key=lambda target: (
                    kind_priority[target.kind],
                    -target.value,
                    math.dist(center, target.pos),
                    target.identity,
                ),
            )
        )

    def _two_step_setup_food_mass(
        self,
        *,
        center: tuple[float, float],
        target: tuple[float, float],
        radius: float,
        step_distance: float,
        foods: tuple[FoodModel, ...],
    ) -> float:
        """Prove the largest food gain on one feasible two-segment route.

        A setup probe is only useful if the mass and target contact happen on
        the same two-step trajectory.  Considering each nearby food as the one
        legal turn point covers direct routes and food-directed first moves;
        food from incompatible directions can never be pooled.
        """

        routes: list[tuple[tuple[float, float], ...]] = []
        direct_contact = self._target_contact_point(center, target, radius)
        if math.dist(center, direct_contact) <= 2.0 * step_distance:
            routes.append((center, direct_contact))
        for waypoint in foods:
            target_contact = self._target_contact_point(
                waypoint.pos,
                target,
                radius,
            )
            if (
                math.dist(center, waypoint.pos) <= step_distance
                and math.dist(waypoint.pos, target_contact) <= step_distance
            ):
                routes.append((center, waypoint.pos, target_contact))

        best_mass = 0.0
        for route in routes:
            target_contact_distance = sum(
                math.dist(start, end)
                for start, end in zip(route, route[1:], strict=False)
            )
            if target_contact_distance <= 1e-12:
                continue
            swept = 0
            for food in foods:
                food_contact_distance = self._route_contact_distance(
                    route,
                    food.pos,
                    radius,
                )
                if (
                    food_contact_distance is not None
                    and food_contact_distance < target_contact_distance - 1e-9
                ):
                    swept += 1
            best_mass = max(best_mass, swept * FOOD_RADIUS * FOOD_RADIUS)
        return best_mass

    @staticmethod
    def _target_contact_point(
        start: tuple[float, float],
        target: tuple[float, float],
        radius: float,
    ) -> tuple[float, float]:
        distance = math.dist(start, target)
        if distance <= radius:
            return start
        travel = distance - radius
        return (
            start[0] + (target[0] - start[0]) * travel / distance,
            start[1] + (target[1] - start[1]) * travel / distance,
        )

    @classmethod
    def _route_contact_distance(
        cls,
        route: tuple[tuple[float, float], ...],
        point: tuple[float, float],
        radius: float,
    ) -> float | None:
        travelled = 0.0
        for start, end in zip(route, route[1:], strict=False):
            segment_length = math.dist(start, end)
            contact = cls._moving_circle_contact_time(
                point=point,
                start=start,
                end=end,
                radius=radius,
                not_before=0.0,
            )
            if contact is not None:
                return travelled + contact * segment_length
            travelled += segment_length
        return None

    def _relevant_local_viruses(
        self,
        node: SearchNode,
        viruses,
    ) -> tuple[VirusModel, ...]:
        """Keep viruses that can affect a two-step route or its terminal option."""

        center = node.center
        primary = node.primary
        relevant_range = (
            primary.radius
            + 2.0 * player_speed(primary.radius)
            + self._OPTION_ROUTE_RANGE
        )
        return tuple(
            virus
            for virus in viruses
            if virus.virus_id not in node.consumed_virus_ids
            and math.dist(center, virus.pos) <= relevant_range
        )

    def _tactical_target_actions(
        self,
        center: tuple[float, float],
        targets: tuple[LocalTarget, ...],
    ) -> tuple[Action, ...]:
        ranked = sorted(
            (target for target in targets if target.kind != "food"),
            key=lambda target: (
                0 if target.kind == "prey" else 1,
                -target.value,
                math.dist(center, target.pos),
                target.identity,
            ),
        )
        selected: list[LocalTarget] = []
        for kind in ("prey", "virus"):
            selected.extend(
                tuple(target for target in ranked if target.kind == kind)[:2]
            )
        return tuple(
            Action(
                normalise((target.pos[0] - center[0], target.pos[1] - center[1])),
                reason=f"local_{target.kind}_probe",
            )
            for target in selected
        )

    def _root_food_field_and_targets(
        self,
        center: tuple[float, float],
        foods,
        eaten_food_ids,
    ) -> tuple[tuple[float, float], tuple[LocalTarget, ...]]:
        """Compute the full food field and retained DP points in one scan."""

        field_x = 0.0
        field_y = 0.0
        food_rows: list[tuple[float, LocalTarget]] = []
        range_squared = self._FOOD_RANGE * self._FOOD_RANGE
        for food in foods:
            if food.food_id in eaten_food_ids:
                continue
            dx = food.pos[0] - center[0]
            dy = food.pos[1] - center[1]
            distance_squared = dx * dx + dy * dy
            if distance_squared > range_squared:
                continue
            distance = max(0.25, math.sqrt(distance_squared))
            weight = 1.0 / (distance * distance)
            field_x += dx / distance * weight
            field_y += dy / distance * weight
            food_rows.append(
                (
                    distance_squared,
                    LocalTarget(
                        kind="food",
                        pos=food.pos,
                        value=FOOD_RADIUS * FOOD_RADIUS,
                        identity=(0, int(food.food_id)),
                    ),
                )
            )
        food_rows.sort(key=lambda row: row[0])
        return (
            normalise((field_x, field_y)),
            tuple(target for _, target in food_rows[: self._LOCAL_FOOD_LIMIT]),
        )

    def _local_enemies(self, node: SearchNode) -> tuple[EnemyBlob, ...]:
        center = node.center
        nearby = sorted(node.enemies, key=lambda enemy: math.dist(center, enemy.pos))[
            :3
        ]
        relevant = list(nearby)
        for enemy in node.enemies:
            if enemy in relevant:
                continue
            threatens = any(
                can_eat_player_blob(enemy.radius, own.radius)
                and math.dist(enemy.pos, own.pos)
                <= _split_chain_attack_reach(enemy.radius, own.radius)
                + 3.0 * player_speed(enemy.radius)
                + 4.0
                for own in node.own_blobs
            )
            if threatens:
                relevant.append(enemy)
        return tuple(
            sorted(relevant, key=lambda enemy: (enemy.player_id, enemy.blob_id))[:6]
        )

    def _near_future_predator_escape(
        self,
        node: SearchNode,
        local_enemies: tuple[EnemyBlob, ...] | None = None,
    ) -> tuple[float, float]:
        """Escape larger neighbours that are only a few pellets from lethal."""

        own = node.primary
        escape_x = 0.0
        escape_y = 0.0
        enemies = (
            local_enemies if local_enemies is not None else self._local_enemies(node)
        )
        for enemy in enemies:
            if not self._is_near_future_predator(own, enemy):
                continue
            dx = own.x - enemy.x
            dy = own.y - enemy.y
            distance = max(0.25, math.hypot(dx, dy))
            weight = 1.0 / (distance * distance)
            escape_x += dx / distance * weight
            escape_y += dy / distance * weight
        return normalise((escape_x, escape_y))

    @staticmethod
    def _is_near_future_predator(own, enemy: EnemyBlob) -> bool:
        if enemy.radius <= own.radius:
            return False
        threshold_mass = EAT_SIZE_RATIO * own.mass
        food_to_lethal = max(0.0, threshold_mass - enemy.mass) / (
            FOOD_RADIUS * FOOD_RADIUS
        )
        return food_to_lethal <= 6.0 and math.dist(own.pos, enemy.pos) <= 6.0

    def _near_future_predator_pressure(self, node: SearchNode) -> float:
        own = node.primary
        pressure = 0.0
        for enemy in self._local_enemies(node):
            if not self._is_near_future_predator(own, enemy):
                continue
            distance = math.dist(own.pos, enemy.pos)
            threshold_mass = EAT_SIZE_RATIO * own.mass
            food_to_lethal = max(0.0, threshold_mass - enemy.mass) / (
                FOOD_RADIUS * FOOD_RADIUS
            )
            urgency = max(0.0, 1.0 - food_to_lethal / 6.0)
            proximity = max(0.0, 1.0 - distance / 6.0)
            containment = max(0.0, (enemy.radius - distance) / enemy.radius)
            pressure += (
                4.0 * node.total_mass * urgency * proximity * (1.0 + 2.0 * containment)
            )
        return pressure

    def _local_deeper_directions(
        self,
        pos: tuple[float, float],
        previous: tuple[float, float],
        targets: tuple[LocalTarget, ...],
        *,
        use_food_field: bool,
    ) -> tuple[tuple[float, float], ...]:
        """Build the next actions from the current DP state.

        Official winners repeatedly bend toward the next food rather than
        preserving a route chosen at the root.  Recomputing target directions
        here makes those turns purposeful while retaining the reversal-only
        control cost.
        """

        field_x = 0.0
        field_y = 0.0
        ranked_targets: list[
            tuple[
                int,
                float,
                float,
                tuple[int, int],
                tuple[float, float],
            ]
        ] = []
        kind_priority = {"prey": 0, "virus": 1, "food": 2}
        for target in targets:
            dx = target.pos[0] - pos[0]
            dy = target.pos[1] - pos[1]
            distance_squared = dx * dx + dy * dy
            raw_distance = math.sqrt(distance_squared)
            distance = max(0.25, raw_distance)
            target_direction = (
                (dx / raw_distance, dy / raw_distance)
                if raw_distance > 0.0
                else (0.0, 0.0)
            )
            ranked_targets.append(
                (
                    kind_priority[target.kind],
                    distance_squared,
                    -target.value,
                    target.identity,
                    target_direction,
                )
            )
            if target.kind == "food":
                weight = 1.0 / (distance * distance)
                field_x += dx / distance * weight
                field_y += dy / distance * weight
        food_field = normalise((field_x, field_y)) if use_food_field else (0.0, 0.0)
        ranked_targets.sort(key=lambda row: row[:4])
        directions = [previous]
        directions.extend(
            row[4] for row in ranked_targets[: self._TARGET_DIRECTION_LIMIT]
        )
        directions.extend(
            (
                food_field,
                _rotate(previous, math.pi / 2),
                _rotate(previous, -math.pi / 2),
                _rotate(previous, math.pi / 6),
                _rotate(previous, -math.pi / 6),
            )
        )
        result: list[tuple[float, float]] = []
        seen: set[int] = set()
        for direction in directions:
            if direction == (0.0, 0.0):
                continue
            sector = self._direction_sector(direction)
            if sector in seen:
                continue
            seen.add(sector)
            result.append(direction)
        return tuple(result[: self._DEEP_DIRECTION_LIMIT])

    def _local_evaluation_context(
        self,
        *,
        node: SearchNode,
        foods,
        viruses,
        arena_size: float,
        local_enemies: tuple[EnemyBlob, ...],
    ) -> LocalEvaluationContext:
        """Prepare all action-independent local facts once for this turn."""

        active_viruses = tuple(
            virus for virus in viruses if virus.virus_id not in node.consumed_virus_ids
        )
        captures = []
        for enemy in node.enemies:
            if enemy.stale_rounds:
                continue
            chain_viruses = []
            for virus in active_viruses:
                route_distance = math.dist(enemy.pos, virus.pos)
                if route_distance <= self._OPTION_ROUTE_RANGE:
                    chain_viruses.append(
                        (
                            virus,
                            math.exp(-route_distance / self._OPTION_ROUTE_RANGE),
                        )
                    )
            captures.append(LocalCaptureOption(enemy, tuple(chain_viruses)))
        active_foods = tuple(
            food for food in foods if food.food_id not in node.eaten_food_ids
        )
        food_cell_rows: dict[tuple[int, int], list[FoodModel]] = {}
        cell_size = self._LOCAL_FOOD_CELL_SIZE
        for food in active_foods:
            cell = (
                math.floor(food.pos[0] / cell_size),
                math.floor(food.pos[1] / cell_size),
            )
            food_cell_rows.setdefault(cell, []).append(food)
        return LocalEvaluationContext(
            node=node,
            foods=active_foods,
            viruses=active_viruses,
            arena_size=arena_size,
            responses=self._local_enemy_responses(local_enemies),
            captures=tuple(captures),
            food_by_id={food.food_id: food for food in active_foods},
            food_cells={
                cell: tuple(cell_foods) for cell, cell_foods in food_cell_rows.items()
            },
        )

    def _local_path_foods(
        self,
        evaluation: LocalEvaluationContext,
        path: LocalBlobPath,
        radius: float,
    ) -> Iterator[FoodModel]:
        """Yield a conservative spatial superset for one swept capsule.

        Foods are indexed by their centre in exactly one cell.  Expanding the
        segment AABB by the current blob radius guarantees that every possible
        contact reaches the unchanged circle/segment quadratic below.  The
        index therefore removes impossible checks without approximating event
        time or changing tie resolution.
        """

        cell_size = self._LOCAL_FOOD_CELL_SIZE
        min_cell_x = math.floor((min(path.start[0], path.end[0]) - radius) / cell_size)
        max_cell_x = math.floor((max(path.start[0], path.end[0]) + radius) / cell_size)
        min_cell_y = math.floor((min(path.start[1], path.end[1]) - radius) / cell_size)
        max_cell_y = math.floor((max(path.start[1], path.end[1]) + radius) / cell_size)
        for cell_x in range(min_cell_x, max_cell_x + 1):
            for cell_y in range(min_cell_y, max_cell_y + 1):
                yield from evaluation.food_cells.get((cell_x, cell_y), ())

    def _advance_local_state(
        self,
        evaluation: LocalEvaluationContext,
        state: LocalRolloutState,
        direction: tuple[float, float],
        *,
        split: bool = False,
    ) -> tuple[LocalRolloutState, float]:
        """Advance exact fragment kinematics and credit only swept food."""

        unit = normalise(direction)
        cache_key = (state, unit[0], unit[1], split)
        cached = evaluation.transition_cache.get(cache_key)
        if cached is not None:
            return cached
        transition_started = perf_counter()

        before_move = list(state.own_blobs)
        if split:
            before_move = self._apply_split(
                before_move,
                unit,
                evaluation.arena_size,
            )
        split_performed = len(before_move) > len(state.own_blobs)
        start_by_id = {blob.blob_id: blob for blob in before_move}
        moved = self._move_own_blobs(
            before_move,
            unit,
            evaluation.arena_size,
            calculate_blocked=False,
            calculate_efficiency=False,
        ).blobs
        minimum_mass = STARTING_RADIUS * STARTING_RADIUS
        blobs = [
            replace(
                blob,
                radius=math.sqrt(
                    blob.mass
                    if blob.mass <= minimum_mass
                    else max(minimum_mass, blob.mass * (1.0 - MASS_DECAY_RATE))
                ),
            )
            for blob in moved
        ]
        blobs = self._stabilise_own_blobs(blobs, evaluation.arena_size)
        paths = {
            index: LocalBlobPath(0.0, start_by_id[blob.blob_id].pos, blob.pos)
            for index, blob in enumerate(blobs)
            if blob.blob_id in start_by_id
        }

        contact_phase_started = perf_counter()
        eaten = set(state.eaten_food_ids)
        food_gain = 0.0
        remaining_food_ids = set(evaluation.food_by_id).difference(eaten)
        events: list[tuple[float, int, float, int, int, int]] = []

        def push_contact(food: FoodModel, index: int, not_before: float) -> None:
            path = paths.get(index)
            if path is None:
                return
            blob = blobs[index]
            evaluation.contact_checks += 1
            contact_time = self._moving_circle_contact_time(
                point=food.pos,
                start=path.start,
                end=path.end,
                radius=blob.radius,
                not_before=not_before,
                segment_start_time=path.start_time,
            )
            if contact_time is not None:
                heapq.heappush(
                    events,
                    (
                        contact_time,
                        food.food_id,
                        -blob.radius,
                        blob.blob_id,
                        index,
                        path.version,
                    ),
                )

        for index, path in paths.items():
            for food in self._local_path_foods(
                evaluation,
                path,
                blobs[index].radius,
            ):
                if food.food_id not in remaining_food_ids:
                    continue
                push_contact(food, index, 0.0)

        while events:
            current_time, food_id, _, _, winner, version = heapq.heappop(events)
            food = (
                evaluation.food_by_id.get(food_id)
                if food_id in remaining_food_ids
                else None
            )
            path = paths.get(winner)
            if food is None or path is None or path.version != version:
                continue
            remaining_food_ids.remove(food_id)
            eater = blobs[winner]
            event_pos = self._path_position(path, current_time)
            grown_radius = math.sqrt(eater.mass + FOOD_RADIUS * FOOD_RADIUS)
            grown_start = (
                max(
                    grown_radius,
                    min(evaluation.arena_size - grown_radius, event_pos[0]),
                ),
                max(
                    grown_radius,
                    min(evaluation.arena_size - grown_radius, event_pos[1]),
                ),
            )
            grown_end = (
                max(
                    grown_radius,
                    min(evaluation.arena_size - grown_radius, eater.x),
                ),
                max(
                    grown_radius,
                    min(evaluation.arena_size - grown_radius, eater.y),
                ),
            )
            blobs[winner] = replace(
                eater,
                x=grown_end[0],
                y=grown_end[1],
                radius=grown_radius,
            )
            paths[winner] = LocalBlobPath(
                current_time,
                grown_start,
                grown_end,
                path.version + 1,
            )
            eaten.add(food_id)
            food_gain += FOOD_RADIUS * FOOD_RADIUS
            for remaining in self._local_path_foods(
                evaluation,
                paths[winner],
                grown_radius,
            ):
                if remaining.food_id in remaining_food_ids:
                    push_contact(remaining, winner, current_time)
        evaluation.contact_seconds += perf_counter() - contact_phase_started
        if food_gain > 0.0:
            blobs = self._stabilise_own_blobs(blobs, evaluation.arena_size)

        result = (
            LocalRolloutState(
                own_blobs=tuple(blobs),
                eaten_food_ids=frozenset(eaten),
                split_performed=state.split_performed or split_performed,
            ),
            food_gain,
        )
        evaluation.transition_cache[cache_key] = result
        evaluation.transition_seconds += perf_counter() - transition_started
        return result

    @staticmethod
    def _path_position(path: LocalBlobPath, time: float) -> tuple[float, float]:
        remaining = 1.0 - path.start_time
        if remaining <= 1e-12:
            return path.end
        fraction = max(0.0, min(1.0, (time - path.start_time) / remaining))
        return (
            path.start[0] + (path.end[0] - path.start[0]) * fraction,
            path.start[1] + (path.end[1] - path.start[1]) * fraction,
        )

    @staticmethod
    def _moving_circle_contact_time(
        *,
        point: tuple[float, float],
        start: tuple[float, float],
        end: tuple[float, float],
        radius: float,
        not_before: float,
        segment_start_time: float = 0.0,
    ) -> float | None:
        """Return the first path parameter where a moving circle reaches a point."""

        segment_duration = 1.0 - segment_start_time
        not_before = max(not_before, segment_start_time)
        if segment_duration <= 1e-12:
            return (
                not_before
                if math.dist(point, end) <= radius and not_before <= 1.0
                else None
            )
        local_not_before = (not_before - segment_start_time) / segment_duration
        delta_x = end[0] - start[0]
        delta_y = end[1] - start[1]
        offset_x = start[0] - point[0]
        offset_y = start[1] - point[1]
        radius_squared = radius * radius
        at_time_x = offset_x + delta_x * local_not_before
        at_time_y = offset_y + delta_y * local_not_before
        if at_time_x * at_time_x + at_time_y * at_time_y <= radius_squared:
            return not_before
        a = delta_x * delta_x + delta_y * delta_y
        if a <= 1e-12:
            return None
        b = 2.0 * (offset_x * delta_x + offset_y * delta_y)
        c = offset_x * offset_x + offset_y * offset_y - radius_squared
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return None
        root = (-b - math.sqrt(discriminant)) / (2.0 * a)
        if root < local_not_before:
            root = (-b + math.sqrt(discriminant)) / (2.0 * a)
        if local_not_before <= root <= 1.0:
            return segment_start_time + root * segment_duration
        return None

    @staticmethod
    def _turn_cost_between_units(
        previous: tuple[float, float],
        current: tuple[float, float],
    ) -> float:
        dot = max(
            -1.0,
            min(1.0, previous[0] * current[0] + previous[1] * current[1]),
        )
        return 2.0 * max(0.0, -dot) ** 2

    def _terminal_option_value(
        self,
        evaluation: LocalEvaluationContext,
        *,
        state: LocalRolloutState,
    ) -> float:
        """Return the best mutually exclusive option available after two moves."""

        if state in evaluation.option_cache:
            return evaluation.option_cache[state]
        terminal_started = perf_counter()

        forage_values = heapq.nlargest(
            3,
            (
                FOOD_RADIUS
                * FOOD_RADIUS
                * math.exp(
                    -min(
                        max(0.0, math.dist(blob.pos, food.pos) - blob.radius)
                        for blob in state.own_blobs
                    )
                    / 2.8
                )
                for food in evaluation.foods
                if food.food_id not in state.eaten_food_ids
            ),
        )
        forage = sum(
            value * weight
            for value, weight in zip(
                forage_values,
                (1.0, 0.5, 0.25),
                strict=False,
            )
        )

        projected_node = self._projected_local_node(evaluation, state)
        best_virus = 0.0
        for virus in evaluation.viruses:
            if not any(
                _can_consume_virus(blob.radius, virus.radius)
                for blob in state.own_blobs
            ):
                continue
            expected_mass = self._virus_expected_mass(
                projected_node,
                virus,
                evaluation.arena_size,
            )
            if expected_mass is not None:
                best_virus = max(best_virus, expected_mass)

        best_prey = 0.0
        best_post_capture_virus = 0.0
        for capture in evaluation.captures:
            enemy = capture.enemy
            source, capture_probability = self._best_local_capture(
                state,
                enemy,
                evaluation.arena_size,
            )
            if source is None:
                continue
            rival_value = self._rival_values.get(enemy.player_id, 0.0)
            prey_value = capture_probability * enemy.mass * (1.0 + rival_value)
            best_prey = max(best_prey, prey_value)
            # The projected capture geometry below is used only to value a
            # follow-up virus.  Most turns have no such route, so avoid an
            # otherwise unused full fragment stabilisation for every prey.
            if not capture.chain_viruses:
                continue
            grown_radius = math.sqrt(source.mass + enemy.mass)
            grown = replace(
                source,
                x=max(
                    grown_radius,
                    min(evaluation.arena_size - grown_radius, enemy.x),
                ),
                y=max(
                    grown_radius,
                    min(evaluation.arena_size - grown_radius, enemy.y),
                ),
                radius=grown_radius,
            )
            post_capture_blobs = tuple(
                self._stabilise_own_blobs(
                    [
                        grown if blob.blob_id == source.blob_id else blob
                        for blob in state.own_blobs
                    ],
                    evaluation.arena_size,
                )
            )
            post_capture_node = replace(
                projected_node,
                own_blobs=post_capture_blobs,
                enemies=tuple(
                    other for other in projected_node.enemies if other.key != enemy.key
                ),
            )
            next_virus = 0.0
            for virus, route_value in capture.chain_viruses:
                if not any(
                    _can_consume_virus(blob.radius, virus.radius)
                    for blob in post_capture_blobs
                ):
                    continue
                expected_mass = self._virus_expected_mass(
                    post_capture_node,
                    virus,
                    evaluation.arena_size,
                )
                if expected_mass is None:
                    continue
                next_virus = max(
                    next_virus,
                    expected_mass * route_value,
                )
            best_post_capture_virus = max(
                best_post_capture_virus,
                capture_probability * self._LOCAL_DISCOUNT * next_virus,
            )

        value = max(forage, best_virus, best_prey, best_post_capture_virus)
        evaluation.option_cache[state] = value
        evaluation.terminal_seconds += perf_counter() - terminal_started
        return value

    @staticmethod
    def _projected_local_node(
        evaluation: LocalEvaluationContext,
        state: LocalRolloutState,
    ) -> SearchNode:
        cached = evaluation.projected_node_cache.get(state)
        if cached is not None:
            return cached
        projected = replace(
            evaluation.node,
            own_blobs=state.own_blobs,
            eaten_food_ids=state.eaten_food_ids,
        )
        evaluation.projected_node_cache[state] = projected
        return projected

    def _best_local_capture(
        self,
        state: LocalRolloutState,
        enemy: EnemyBlob,
        arena_size: float,
    ) -> tuple[OwnBlob | None, float]:
        best_source = None
        best_probability = 0.0
        for own in state.own_blobs:
            if not can_eat_player_blob(own.radius, enemy.radius):
                continue
            probability = self._prey_capture_probability(own, enemy, arena_size)
            if probability > best_probability:
                best_source = own
                best_probability = probability
        return best_source, best_probability

    def _local_enemy_responses(
        self,
        enemies: tuple[EnemyBlob, ...],
    ) -> tuple[LocalEnemyResponse, ...]:
        """Hoist opponent facts shared by all Bellman-style local states."""

        return tuple(
            LocalEnemyResponse(
                enemy=enemy,
                speed=player_speed(enemy.radius),
            )
            for enemy in enemies
        )

    def _local_safety_value(
        self,
        evaluation: LocalEvaluationContext,
        *,
        state: LocalRolloutState,
        depth: int,
    ) -> float | None:
        """Apply worst-case nearby-predator safety separately from option value."""

        key = (state, depth)
        if key in evaluation.safety_cache:
            return evaluation.safety_cache[key]
        response_started = perf_counter()

        threatened_mass = 0.0
        value = 0.0
        comparisons = 0
        for own in state.own_blobs:
            worst_margin = math.inf
            for fact in evaluation.responses:
                enemy = fact.enemy
                comparisons += 1
                if not can_eat_player_blob(enemy.radius, own.radius):
                    continue
                current_distance = math.hypot(
                    own.x - enemy.x,
                    own.y - enemy.y,
                )
                reach_key = (enemy.key, own.radius)
                if reach_key not in evaluation.predator_reach_cache:
                    evaluation.predator_reach_cache[reach_key] = (
                        _split_chain_attack_reach(enemy.radius, own.radius)
                    )
                # A predator can use this tick for normal movement or for the
                # first split in its chain, not both.  Only earlier whole ticks
                # may contribute ordinary travel before a later split.
                normal_reach = enemy.radius + fact.speed * depth
                split_reach = evaluation.predator_reach_cache[
                    reach_key
                ] + fact.speed * max(0, depth - 1)
                danger_reach = max(normal_reach, split_reach)
                worst_margin = min(
                    worst_margin,
                    current_distance - danger_reach,
                )
            if worst_margin <= 0.0:
                threatened_mass += own.mass
            elif worst_margin < 4.0:
                value -= own.mass * (4.0 - worst_margin) / 4.0
        self._local_response_evaluations += comparisons
        if threatened_mass >= state.mass:
            evaluation.safety_cache[key] = None
            evaluation.response_seconds += perf_counter() - response_started
            return None
        value -= threatened_mass
        evaluation.safety_cache[key] = value
        evaluation.response_seconds += perf_counter() - response_started
        return value

    @staticmethod
    def _direction_sector(direction: tuple[float, float]) -> int:
        return int(round(math.atan2(direction[1], direction[0]) / TAU * 16.0)) % 16


class LocalTacticalSearchReferenceStrategy(LocalTacticalSearchStrategy):
    """Correctness-first planner used as the optimisation oracle.

    This version deliberately retains the widest two-step action set and root
    validation.  It is not submission-safe: benchmark it only with the local
    engine timeout explicitly raised.  The production strategy must reproduce
    its important choices before any search-width reduction is accepted.
    """

    name = "local_tactical_search_reference"
    _LOCAL_ROOT_LIMIT = 32
    _PROXY_REFINE_LIMIT = 32
    _TARGET_DIRECTION_LIMIT = 12
    _DEEP_DIRECTION_LIMIT = 24

    def __init__(
        self,
        depth: int | None = None,
        width: int | None = None,
        angular_samples: int | None = None,
    ) -> None:
        super().__init__(
            depth=1 if depth is None else depth,
            width=32 if width is None else width,
            angular_samples=18 if angular_samples is None else angular_samples,
        )
        self.max_food = 24
        self.max_enemies = 10
        self.proxy_refine_limit = 32
        self.proxy_min_refine = 32
        self.proxy_refine_blob_work = max(self.proxy_refine_blob_work, 512)
        self.proxy_coarse_after_seconds = math.inf
        self._competition_coarse_mode = False
        if "BOT_LOCAL_CONTINUATION_PRIOR" not in os.environ:
            self.local_continuation_prior = 4.0

    def _uses_compute_time_bank(self) -> bool:
        return False

    def _transition_budget(
        self,
        own_blob_count: int,
        enemy_count: int = 0,
    ) -> int:
        return 64

    def _actions_per_node_limit(self, depth_index: int) -> int:
        return 64
