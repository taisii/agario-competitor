"""Potential-field baseline with a conservative two-step tactical gate.

The lightweight policy is always evaluated first and remains the exact result
when no nearby interaction can matter. The existing local tactical planner is
invoked for reachable prey, viruses, and predator uncertainty; an
environment-controlled always-full mode keeps the ungated oracle available
for parity and regret measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import os
from time import perf_counter

from lib.config.player import SPLIT_MIN_MASS
from strategies.base import StrategyContext, StrategyDecision
from strategies.features import (
    can_consume_virus,
    can_eat_player_blob,
    normalise,
    player_speed,
)
from strategies.local_tactical_search import LocalTacticalSearchStrategy
from strategies.potential_field import PotentialFieldHunterStrategy
from strategies.receding_horizon import (
    Action,
    EnemyBlob,
    OwnBlob,
    SearchNode,
    StepResult,
    _clamp,
    _decayed_radius,
    _split_chain_attack_reach,
)
from strategies.world_transition import (
    CompleteJointCommand,
    ExpectedEvidence,
    ExpectedOutcomeStats,
    JointPhysicalTransition,
    PlayerCommand,
)


_KEY_SCALE = 1_000.0


@dataclass(frozen=True, slots=True)
class HybridEntity:
    """One public object keyed without its per-query public index."""

    kind: str
    key: tuple[str, int, int, int, int]
    pos: tuple[float, float]
    radius: float
    source: object = field(compare=False, hash=False, repr=False)

    @property
    def mass(self) -> float:
        return self.radius * self.radius


@dataclass(frozen=True, slots=True)
class HybridStaticScene:
    """Turn-local geometry for objects that can change tactical execution."""

    entities: tuple[HybridEntity, ...]

    @property
    def signature(self) -> tuple[tuple[str, int, int, int, int], ...]:
        """Materialise the deterministic test/debug identity only on demand."""

        return tuple(entity.key for entity in self.entities)


@dataclass(frozen=True, slots=True)
class HybridGateResult:
    triggered: bool
    reasons: tuple[str, ...]
    base_lower_bound: float
    tactical_upper_bound: float

    @property
    def estimated_regret(self) -> float:
        return max(0.0, self.tactical_upper_bound - self.base_lower_bound)


class PotentialTacticalHybridStrategy(PotentialFieldHunterStrategy):
    """Keep the proven #36 policy unless a local tactic can plausibly win."""

    name = "potential_tactical_hybrid"
    _EXPECTED_RESPONSE_WEIGHTS = (
        0.5665416775,
        0.0940827335,
        0.1065148082,
        0.0591125103,
        0.0552826292,
        0.1184656414,
    )
    _EXPECTED_SCENARIO_IDS = tuple(range(8))
    _EXPECTED_OFFLINE_SCENARIO_COUNT = 12
    _EXPECTED_CONTINUATION_LIMIT = 5
    _EXPECTED_GAIN_PROBABILITY_THRESHOLD = 0.5

    def __init__(
        self,
        *,
        always_full: bool | None = None,
    ) -> None:
        super().__init__()
        self.always_full = (
            os.environ.get("BOT_POTENTIAL_TACTICAL_ALWAYS_FULL", "0") != "0"
            if always_full is None
            else always_full
        )
        raw_model_error = os.environ.get("BOT_EXPECTED_MODEL_ERROR_BOUND")
        self._expected_model_error_bound = (
            float(raw_model_error) if raw_model_error is not None else math.inf
        )
        if self._expected_model_error_bound < 0.0 or math.isnan(
            self._expected_model_error_bound
        ):
            raise ValueError("BOT_EXPECTED_MODEL_ERROR_BOUND must be non-negative")
        self._tactical = LocalTacticalSearchStrategy()
        # Both tiers retain one deterministic 12×5 width. Production ranks it
        # with aggregate opportunity; always-full is the offline exact oracle.
        self._tactical.proxy_coarse_after_seconds = math.inf
        self._tactical._competition_coarse_mode = False
        self._previous_direction: tuple[float, float] = (1.0, 0.0)
        self.last_hybrid_diagnostics: dict[str, object] = {}
        self._last_expected_evidence_diagnostics: dict[str, object] = {}

    def choose(self, context: StrategyContext) -> StrategyDecision:
        self._last_expected_evidence_diagnostics = {}
        total_started = perf_counter()
        input_started = perf_counter()
        base_decision = super().choose(context)
        state = context.game.state
        own_blobs = tuple(state.me.blobs.values())
        input_ms = (perf_counter() - input_started) * 1000.0
        if not own_blobs:
            diagnostics = self._hybrid_diagnostics(
                HybridGateResult(False, (), 0.0, 0.0),
                base_decision=base_decision,
                tactical_decision=None,
                profile=self._empty_profile(input_ms=input_ms),
                planner_tier="base",
                complexity=0,
            )
            self._complete_profile(diagnostics, total_started)
            self.last_hybrid_diagnostics = diagnostics
            return base_decision

        static_started = perf_counter()
        tracked_enemies = self._tactical.prepare_enemy_memory_for_external_gate(context)
        scene = self._build_static_scene(
            state,
            tracked_enemies=tracked_enemies,
        )
        static_ms = (perf_counter() - static_started) * 1000.0

        root_started = perf_counter()
        gate = self._gate_local_oracle(
            own_blobs=own_blobs,
            base_decision=base_decision,
            scene=scene,
            tracked_enemies=tracked_enemies,
        )
        root_ms = (perf_counter() - root_started) * 1000.0
        complexity = self._primitive_complexity(
            own_count=len(own_blobs),
            food_count=min(len(state.visible_food), self._tactical.max_food),
            virus_count=len(state.visible_viruses),
            enemy_count=min(len(tracked_enemies), self._tactical.max_enemies),
        )
        run_full = self.always_full or gate.triggered
        # In production the local policy is an advisor, not the final arbiter.
        # Rank the unchanged 12×5 tree with bounded aggregate opportunity, then
        # validate the base and proposed tactical roots with exact first-step
        # physics in `_select_advisor_decision`. The full exact tree remains an
        # explicit offline oracle only; partial split states can otherwise make
        # its contact/terminal expansion exceed the cumulative submission bank.
        planner_tier = "exact" if self.always_full else "aggregate"
        if not run_full:
            profile = self._empty_profile(
                input_ms=input_ms,
                static_ms=static_ms,
                root_ms=root_ms,
                output_ms=(perf_counter() - total_started) * 1000.0
                - input_ms
                - static_ms
                - root_ms,
            )
            diagnostics = self._hybrid_diagnostics(
                gate,
                base_decision=base_decision,
                tactical_decision=None,
                profile=profile,
                planner_tier="base",
                complexity=complexity,
            )
            self._complete_profile(diagnostics, total_started)
            self.last_hybrid_diagnostics = diagnostics
            self._previous_direction = base_decision.direction
            # Exact #36 object and diagnostics are the safety contract.
            return base_decision

        self._tactical.previous_direction = self._previous_direction
        self._tactical.use_aggregate_local_dp = planner_tier == "aggregate"
        self._tactical.required_semantic_root = Action(
            base_decision.direction,
            split=base_decision.split,
            reason="hybrid_base",
        )
        tactical_started = perf_counter()
        try:
            tactical_decision = self._tactical.choose(context)
        finally:
            self._tactical.required_semantic_root = None
        tactical_ms = (perf_counter() - tactical_started) * 1000.0

        advisor_started = perf_counter()
        selected_decision, advisor_action = self._select_advisor_decision(
            base_decision,
            tactical_decision,
        )
        advisor_ms = (perf_counter() - advisor_started) * 1000.0

        local_profile = {
            name: float(value)
            for name, value in tactical_decision.diagnostics.get(
                "local_phase_ms",
                {},
            ).items()
        }
        local_total_ms = sum(local_profile.values())
        profile = {
            "input": input_ms,
            "static": static_ms,
            "root": root_ms + local_profile.get("root", 0.0),
            "tactical_setup": max(0.0, tactical_ms - local_total_ms),
            "transition": local_profile.get("transition", 0.0),
            "contact": local_profile.get("contact", 0.0),
            "response": local_profile.get("response", 0.0),
            "terminal": local_profile.get("terminal", 0.0),
            "advisor": advisor_ms,
            "output": 0.0,
        }
        diagnostics = self._hybrid_diagnostics(
            gate,
            base_decision=base_decision,
            tactical_decision=tactical_decision,
            profile=profile,
            planner_tier=planner_tier,
            complexity=complexity,
        )
        diagnostics["hybrid_advisor_action"] = advisor_action
        diagnostics["hybrid_selected_matches_base"] = advisor_action == "base"
        result = replace(
            selected_decision,
            diagnostics={
                **selected_decision.diagnostics,
                **tactical_decision.diagnostics,
                **diagnostics,
            },
        )
        self.last_hybrid_diagnostics = diagnostics
        self._previous_direction = result.direction
        self._last_direction = result.direction
        self._complete_profile(diagnostics, total_started)
        return result

    def _select_advisor_decision(
        self,
        base_decision: StrategyDecision,
        tactical_decision: StrategyDecision,
    ) -> tuple[StrategyDecision, str]:
        """Keep #36 unless exact physics proves danger or secured gain."""

        base_key = self._tactical._action_key(
            Action(base_decision.direction, split=base_decision.split)
        )
        summaries = self._tactical.root_transition_summaries
        base = summaries.get(base_key) or self._tactical.ensure_root_transition_summary(
            Action(
                base_decision.direction,
                split=base_decision.split,
                reason="hybrid_base",
            )
        )
        if base is None:
            return base_decision, "base"

        if not bool(base["fatal"]):
            outcome = "safe"
        elif float(base["physical_mass"]) <= 0.0:
            outcome = "all_mass_lost"
        elif base_decision.split:
            outcome = "unavoidable_split"
        else:
            outcome = "unavoidable_predator"
        self._tactical.required_semantic_transition = {
            "outcome": outcome,
            **base,
        }

        structural_offense = self._is_structural_offense(tactical_decision)
        self._last_expected_evidence_diagnostics[
            "expected_evidence_structural_offense"
        ] = structural_offense
        if not structural_offense:
            self._last_expected_evidence_diagnostics[
                "expected_evidence_skipped_reason"
            ] = "non_offensive_root"
            return base_decision, "base"

        base_resolved_split = self._resolved_split_performed(base_decision)
        tactical_resolved_split = self._resolved_split_performed(tactical_decision)
        self._last_expected_evidence_diagnostics.update(
            expected_evidence_base_resolved_split=base_resolved_split,
            expected_evidence_tactical_resolved_split=tactical_resolved_split,
        )
        if base_resolved_split or tactical_resolved_split:
            # Expected evidence compares a *difference* between two roots, so
            # both roots need held-out support. A real split changes fragment
            # count, cooldown, speed, contact geometry, and the next public
            # information state. The official replay cohort does not contain
            # enough independent split matches to calibrate that regime.
            # A requested split that cannot create a child remains eligible:
            # its resolved transition is still in the non-split regime.
            self._last_expected_evidence_diagnostics[
                "expected_evidence_skipped_reason"
            ] = "resolved_split_root_out_of_distribution"
            return base_decision, "base"
        if not math.isfinite(self._expected_model_error_bound):
            # A fail-closed model cannot possibly authorize a move. Do not pay
            # for 8 sampled joint worlds plus stress rollouts in production
            # merely to reconstruct that foregone conclusion.
            self._last_expected_evidence_diagnostics[
                "expected_evidence_skipped_reason"
            ] = "uncalibrated_model"
            return base_decision, "base"

        evidence = self._expected_evidence(base_decision, tactical_decision)
        stress_soft_veto = bool(
            self._last_expected_evidence_diagnostics.get(
                "expected_evidence_stress_soft_veto",
                False,
            )
        )
        if (
            evidence is not None
            and evidence.supports_override
            and not stress_soft_veto
            and (
                evidence.paired_survival_improvement
                or evidence.tactical_gain_positive_probability
                >= self._EXPECTED_GAIN_PROBABILITY_THRESHOLD
            )
        ):
            return tactical_decision, "tactical_expected_evidence"

        immediate_all_lost = bool(base["immediate_dead"])
        if immediate_all_lost:
            # Immediate death comes from the ordinary planner response, which
            # is candidate-dependent and can steer same-player fragments
            # independently. Keep it as a diagnostic, never a hard veto.
            return base_decision, "base"
        # Capture/virus outcomes and future-loss states are conditional on one
        # candidate-dependent enemy response. They remain diagnostics until a
        # joint current-step response kernel can prove a safe alternative.
        return base_decision, "base"

    def _resolved_split_performed(self, decision: StrategyDecision) -> bool:
        """Whether the exact root kernel would create at least one child.

        The support gate is defined on the resolved transition regime, not the
        raw command bit. When no planning turn is available, a requested split
        fails closed because its legality cannot be established.
        """

        if not decision.split:
            return False
        turn = self._tactical._advisor_planning_turn
        if turn is None:
            return True
        before = turn.node.own_blobs
        after = self._tactical._apply_split(
            list(before),
            normalise(decision.direction),
            turn.arena_size,
        )
        return len(after) > len(before)

    @staticmethod
    def _is_structural_offense(decision: StrategyDecision) -> bool:
        reason = decision.reason.lower()
        return decision.target_kind in {"prey", "virus"} or any(
            token in reason
            for token in ("prey", "virus_harvest", "local_virus", "split_farm")
        )

    def _expected_evidence(
        self,
        base_decision: StrategyDecision,
        tactical_decision: StrategyDecision,
    ) -> ExpectedEvidence | None:
        """Compare only the base and proposed root under legal joint policies.

        The evidence is intentionally uncalibrated in production.  It is
        published for paired replay validation, but cannot authorize an
        override until a finite held-out model error is configured in code.
        """

        turn = self._tactical._advisor_planning_turn
        if turn is None:
            return None
        started = perf_counter()
        transition_cache: dict[
            tuple[object, ...], JointPhysicalTransition
        ] = {}
        command_cache: dict[
            tuple[object, ...],
            dict[int, PlayerCommand],
        ] = {}
        cache_hits = 0
        cache_misses = 0
        command_hits = 0
        command_misses = 0
        response_table = self._expected_response_table(
            {
                enemy.player_id
                for enemy in turn.node.enemies
            },
            sample_count=len(self._EXPECTED_SCENARIO_IDS),
        )

        def advance(
            node: SearchNode,
            action: Action,
            scenario_id: int,
            *,
            first_step: bool,
            response_types: dict[int, int] | None = None,
            cache_namespace: str = "expected",
        ) -> JointPhysicalTransition:
            nonlocal cache_hits, cache_misses, command_hits, command_misses
            command_key = (cache_namespace, node, scenario_id)
            enemy_commands = command_cache.get(command_key)
            if enemy_commands is None:
                command_misses += 1
                enemy_commands = self._expected_enemy_commands(
                    node=node,
                    response_types=(
                        response_table[scenario_id]
                        if response_types is None
                        else response_types
                    ),
                    foods=turn.foods,
                    viruses=turn.viruses,
                )
                command_cache[command_key] = enemy_commands
            else:
                command_hits += 1
            joint = self._expected_joint_command(
                node=node,
                own_action=action,
                enemy_commands=enemy_commands,
            )
            transition_key = self._expected_transition_cache_key(
                node=node,
                action=action,
                joint=joint,
                first_step=first_step,
            )
            cached = transition_cache.get(transition_key)
            if cached is not None:
                cache_hits += 1
                return cached
            cache_misses += 1
            result = self._tactical._joint_physical_step(
                node=node,
                action=action,
                foods=turn.foods,
                viruses=turn.viruses,
                arena_size=turn.arena_size,
                first_step=first_step,
                joint_command=joint,
            )
            transition_cache[transition_key] = result
            return result

        roots = {
            "base": Action(
                base_decision.direction,
                split=base_decision.split,
                reason="expected_base",
            ),
            "tactical": Action(
                tactical_decision.direction,
                split=tactical_decision.split,
                reason="expected_tactical",
            ),
        }
        scenario_weights = (1.0 / len(self._EXPECTED_SCENARIO_IDS),) * len(
            self._EXPECTED_SCENARIO_IDS
        )
        weight_by_scenario = dict(
            zip(self._EXPECTED_SCENARIO_IDS, scenario_weights, strict=True)
        )
        samples_by_label: dict[str, tuple[float, ...]] = {}
        gains_by_label: dict[str, tuple[bool, ...]] = {}
        selected_continuations: dict[
            str, tuple[tuple[int, tuple[float, float] | None], ...]
        ] = {}
        oracle_mean_by_label: dict[str, float] = {}
        for label, root in roots.items():
            mass_by_scenario: dict[int, float] = {}
            gain_by_scenario: dict[int, bool] = {}
            continuation_by_scenario: dict[
                int, tuple[float, float] | None
            ] = {}
            oracle_mass_by_scenario: dict[int, float] = {}
            information_clusters: dict[tuple[object, ...], list[tuple[int, SearchNode]]] = {}
            for scenario_id in self._EXPECTED_SCENARIO_IDS:
                first = advance(
                    turn.node,
                    root,
                    scenario_id,
                    first_step=True,
                )
                if first.dead:
                    mass_by_scenario[scenario_id] = 0.0
                    oracle_mass_by_scenario[scenario_id] = 0.0
                    continuation_by_scenario[scenario_id] = None
                    gain_by_scenario[scenario_id] = (
                        first.state.projected_captures
                        > turn.node.projected_captures
                        or first.state.projected_viruses
                        > turn.node.projected_viruses
                    )
                    continue
                information_clusters.setdefault(
                    self._expected_information_key(first.state),
                    [],
                ).append((scenario_id, first.state))

            continuations = self._expected_continuations(root.direction)
            for cluster in information_clusters.values():
                best_cluster_score = -math.inf
                best_direction: tuple[float, float] | None = None
                best_results: dict[int, JointPhysicalTransition] = {}
                for direction in continuations:
                    candidate_results = {
                        scenario_id: advance(
                            first_state,
                            Action(direction, reason="expected_continuation"),
                            scenario_id,
                            first_step=False,
                        )
                        for scenario_id, first_state in cluster
                    }
                    cluster_score = sum(
                        weight_by_scenario[scenario_id]
                        * result.final_own_mass
                        for scenario_id, result in candidate_results.items()
                    )
                    for scenario_id, result in candidate_results.items():
                        oracle_mass_by_scenario[scenario_id] = max(
                            oracle_mass_by_scenario.get(scenario_id, 0.0),
                            result.final_own_mass,
                        )
                    if cluster_score > best_cluster_score:
                        best_cluster_score = cluster_score
                        best_direction = direction
                        best_results = candidate_results
                assert best_direction is not None
                for scenario_id, result in best_results.items():
                    mass_by_scenario[scenario_id] = result.final_own_mass
                    gain_by_scenario[scenario_id] = (
                        result.state.projected_captures
                        > turn.node.projected_captures
                        or result.state.projected_viruses
                        > turn.node.projected_viruses
                    )
                    continuation_by_scenario[scenario_id] = best_direction

            samples_by_label[label] = tuple(
                mass_by_scenario[scenario_id]
                for scenario_id in self._EXPECTED_SCENARIO_IDS
            )
            gains_by_label[label] = tuple(
                gain_by_scenario[scenario_id]
                for scenario_id in self._EXPECTED_SCENARIO_IDS
            )
            selected_continuations[label] = tuple(
                (scenario_id, continuation_by_scenario[scenario_id])
                for scenario_id in self._EXPECTED_SCENARIO_IDS
            )
            oracle_mean_by_label[label] = sum(
                weight_by_scenario[scenario_id]
                * oracle_mass_by_scenario[scenario_id]
                for scenario_id in self._EXPECTED_SCENARIO_IDS
            )

        stress_modes = (("all_aggressive", 6), ("all_evasive", 7))
        stress_samples_by_label: dict[str, tuple[float, ...]] = {}
        stress_gains_by_label: dict[str, tuple[bool, ...]] = {}
        for label, root in roots.items():
            mass_by_stress: dict[int, float] = {}
            gain_by_stress: dict[int, bool] = {}
            clusters: dict[tuple[object, ...], list[tuple[int, SearchNode]]] = {}
            for stress_id, (stress_name, response_type) in enumerate(stress_modes):
                response_types = {
                    player_id: response_type
                    for player_id in response_table[0]
                }
                first = advance(
                    turn.node,
                    root,
                    stress_id,
                    first_step=True,
                    response_types=response_types,
                    cache_namespace=stress_name,
                )
                if first.dead:
                    mass_by_stress[stress_id] = 0.0
                    gain_by_stress[stress_id] = (
                        first.state.projected_captures
                        > turn.node.projected_captures
                        or first.state.projected_viruses
                        > turn.node.projected_viruses
                    )
                    continue
                clusters.setdefault(
                    self._expected_information_key(first.state),
                    [],
                ).append((stress_id, first.state))

            for cluster in clusters.values():
                best_score = -math.inf
                best_results: dict[int, JointPhysicalTransition] = {}
                for direction in self._expected_continuations(root.direction):
                    candidate_results = {}
                    for stress_id, first_state in cluster:
                        stress_name, response_type = stress_modes[stress_id]
                        response_types = {
                            player_id: response_type
                            for player_id in response_table[0]
                        }
                        candidate_results[stress_id] = advance(
                            first_state,
                            Action(direction, reason="expected_stress_continuation"),
                            stress_id,
                            first_step=False,
                            response_types=response_types,
                            cache_namespace=stress_name,
                        )
                    score = sum(
                        result.final_own_mass
                        for result in candidate_results.values()
                    )
                    if score > best_score:
                        best_score = score
                        best_results = candidate_results
                for stress_id, result in best_results.items():
                    mass_by_stress[stress_id] = result.final_own_mass
                    gain_by_stress[stress_id] = (
                        result.state.projected_captures
                        > turn.node.projected_captures
                        or result.state.projected_viruses
                        > turn.node.projected_viruses
                    )
            stress_samples_by_label[label] = tuple(
                mass_by_stress[stress_id]
                for stress_id in range(len(stress_modes))
            )
            stress_gains_by_label[label] = tuple(
                gain_by_stress[stress_id]
                for stress_id in range(len(stress_modes))
            )

        evidence = ExpectedEvidence(
            scenario_ids=self._EXPECTED_SCENARIO_IDS,
            scenario_weights=scenario_weights,
            base=ExpectedOutcomeStats.from_samples(
                samples_by_label["base"], scenario_weights
            ),
            tactical=ExpectedOutcomeStats.from_samples(
                samples_by_label["tactical"], scenario_weights
            ),
            base_gain_positive_probability=sum(
                weight
                for gained, weight in zip(
                    gains_by_label["base"], scenario_weights, strict=True
                )
                if gained
            ),
            tactical_gain_positive_probability=sum(
                weight
                for gained, weight in zip(
                    gains_by_label["tactical"], scenario_weights, strict=True
                )
                if gained
            ),
            # Production remains fail-closed unless a validated bound is
            # explicitly baked in. The environment hook is for paired local
            # experiments and is absent from official workers.
            heldout_model_error=self._expected_model_error_bound,
        )
        stress_weights = (0.5, 0.5)
        stress_base = ExpectedOutcomeStats.from_samples(
            stress_samples_by_label["base"], stress_weights
        )
        stress_tactical = ExpectedOutcomeStats.from_samples(
            stress_samples_by_label["tactical"], stress_weights
        )
        stress_base_gain = sum(stress_gains_by_label["base"]) / len(stress_modes)
        stress_tactical_gain = (
            sum(stress_gains_by_label["tactical"]) / len(stress_modes)
        )
        stress_gain_optimism = (
            evidence.tactical_gain_positive_probability
            > evidence.base_gain_positive_probability
            and stress_tactical_gain <= stress_base_gain
        )
        stress_soft_veto = (
            stress_tactical.death_rate > stress_base.death_rate
            or stress_tactical.cvar20_mass < stress_base.cvar20_mass
            or stress_gain_optimism
        )
        self._last_expected_evidence_diagnostics = {
            **self._last_expected_evidence_diagnostics,
            "expected_evidence_calibrated": evidence.calibrated,
            "expected_evidence_model_error_bound": evidence.heldout_model_error,
            "expected_evidence_gain_probability_threshold": (
                self._EXPECTED_GAIN_PROBABILITY_THRESHOLD
            ),
            "expected_evidence_supports_override": evidence.supports_override,
            "expected_evidence_scenario_ids": evidence.scenario_ids,
            "expected_evidence_scenario_weights": evidence.scenario_weights,
            "expected_evidence_scenario_count": len(evidence.scenario_ids),
            "expected_evidence_response_table": tuple(
                (
                    scenario_id,
                    tuple(sorted(response_table[scenario_id].items())),
                )
                for scenario_id in self._EXPECTED_SCENARIO_IDS
            ),
            "expected_evidence_base_mean_mass": evidence.base.mean_mass,
            "expected_evidence_tactical_mean_mass": evidence.tactical.mean_mass,
            "expected_evidence_mean_delta": evidence.mean_delta,
            "expected_evidence_base_death_rate": evidence.base.death_rate,
            "expected_evidence_tactical_death_rate": evidence.tactical.death_rate,
            "expected_evidence_paired_death_nonworse": (
                evidence.paired_death_nonworse
            ),
            "expected_evidence_paired_survival_improvement": (
                evidence.paired_survival_improvement
            ),
            "expected_evidence_base_cvar20": evidence.base.cvar20_mass,
            "expected_evidence_tactical_cvar20": evidence.tactical.cvar20_mass,
            "expected_evidence_paired_delta_cvar20": (
                evidence.paired_delta_cvar20
            ),
            "expected_evidence_base_gain_positive_probability": (
                evidence.base_gain_positive_probability
            ),
            "expected_evidence_tactical_gain_positive_probability": (
                evidence.tactical_gain_positive_probability
            ),
            "expected_evidence_selected_continuations": selected_continuations,
            "expected_evidence_oracle_mean_mass": oracle_mean_by_label,
            "expected_evidence_stress_modes": stress_modes,
            "expected_evidence_stress_base_death_rate": stress_base.death_rate,
            "expected_evidence_stress_tactical_death_rate": (
                stress_tactical.death_rate
            ),
            "expected_evidence_stress_base_cvar20": stress_base.cvar20_mass,
            "expected_evidence_stress_tactical_cvar20": stress_tactical.cvar20_mass,
            "expected_evidence_stress_base_gain_probability": stress_base_gain,
            "expected_evidence_stress_tactical_gain_probability": (
                stress_tactical_gain
            ),
            "expected_evidence_stress_gain_optimism": stress_gain_optimism,
            "expected_evidence_stress_soft_veto": stress_soft_veto,
            "expected_evidence_transition_cache_hits": cache_hits,
            "expected_evidence_transition_cache_misses": cache_misses,
            "expected_evidence_command_cache_hits": command_hits,
            "expected_evidence_command_cache_misses": command_misses,
            "expected_evidence_ms": (perf_counter() - started) * 1000.0,
        }
        return evidence

    @classmethod
    def _expected_response_table(
        cls,
        player_ids: set[int] | frozenset[int],
        *,
        sample_count: int,
    ) -> tuple[dict[int, int], ...]:
        """Build one deterministic stratified response table per turn.

        Every player traverses each quantile stratum exactly once.  Stable
        player-key permutations decorrelate their joint responses without
        making the table depend on observation or root-candidate order.
        """

        if sample_count <= 0:
            raise ValueError("expected response table requires positive samples")
        cumulative = []
        running = 0.0
        for weight in cls._EXPECTED_RESPONSE_WEIGHTS:
            running += weight
            cumulative.append(running)
        if not math.isclose(running, 1.0, abs_tol=1e-9):
            raise ValueError("expected response weights must sum to one")

        table = [dict() for _ in range(sample_count)]
        coprime_steps = tuple(
            step
            for step in range(1, sample_count)
            if math.gcd(step, sample_count) == 1
        ) or (1,)
        for player_id in sorted(player_ids):
            mixed = cls._stable_player_mix(player_id)
            offset = mixed % sample_count
            step = coprime_steps[(mixed >> 8) % len(coprime_steps)]
            jitter = (((mixed >> 16) & 0xFFFFFFFF) + 0.5) / (1 << 32)
            for sample_id in range(sample_count):
                stratum = (offset + step * sample_id) % sample_count
                quantile = (stratum + jitter) / sample_count
                response_type = next(
                    index
                    for index, threshold in enumerate(cumulative)
                    if quantile < threshold
                )
                table[sample_id][player_id] = response_type
        return tuple(table)

    @staticmethod
    def _stable_player_mix(player_id: int) -> int:
        value = (player_id & 0xFFFFFFFFFFFFFFFF) + 0x9E3779B97F4A7C15
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9
        value &= 0xFFFFFFFFFFFFFFFF
        value = (value ^ (value >> 27)) * 0x94D049BB133111EB
        value &= 0xFFFFFFFFFFFFFFFF
        return value ^ (value >> 31)

    @staticmethod
    def _expected_information_key(node: SearchNode) -> tuple[object, ...]:
        """Return next-turn information without latent response-mode state."""

        own = tuple(
            sorted(
                (
                    blob.x,
                    blob.y,
                    blob.radius,
                    blob.merge_cooldown,
                    blob.eject_vx,
                    blob.eject_vy,
                )
                for blob in node.own_blobs
            )
        )
        enemies = tuple(
            sorted(
                (
                    enemy.player_id,
                    enemy.x,
                    enemy.y,
                    enemy.radius,
                    enemy.stale_rounds,
                    enemy.merge_cooldown,
                    enemy.direction,
                    enemy.eject_vx,
                    enemy.eject_vy,
                )
                for enemy in node.enemies
            )
        )
        return (
            own,
            enemies,
            node.last_direction,
            node.eaten_food_ids,
            node.consumed_virus_ids,
        )

    @staticmethod
    def _expected_transition_cache_key(
        *,
        node: SearchNode,
        action: Action,
        joint: CompleteJointCommand,
        first_step: bool,
    ) -> tuple[object, ...]:
        return (
            node,
            normalise(action.direction),
            action.split,
            joint.commands,
            first_step,
        )

    def _expected_joint_command(
        self,
        *,
        node: SearchNode,
        own_action: Action,
        enemy_commands: dict[int, PlayerCommand],
    ) -> CompleteJointCommand:
        commands = {
            self._tactical._own_player_id: PlayerCommand(
                own_action.direction,
                split=own_action.split,
            ),
            **enemy_commands,
        }
        return CompleteJointCommand.build(
            live_player_ids={
                self._tactical._own_player_id,
                *(enemy.player_id for enemy in node.enemies),
            },
            commands=commands,
        )

    def _expected_enemy_commands(
        self,
        *,
        node: SearchNode,
        response_types: dict[int, int],
        foods,
        viruses,
    ) -> dict[int, PlayerCommand]:
        """Batch one state/scenario response for every enemy player."""

        by_player: dict[int, list[EnemyBlob]] = {}
        for enemy in node.enemies:
            by_player.setdefault(enemy.player_id, []).append(enemy)
        return {
            player_id: self._expected_enemy_command(
                group,
                node=node,
                response_type=response_types[player_id],
                foods=foods,
                viruses=viruses,
            )
            for player_id, group in by_player.items()
        }

    @staticmethod
    def _expected_enemy_command(
        group: list[EnemyBlob],
        *,
        node: SearchNode,
        response_type: int,
        foods,
        viruses,
    ) -> PlayerCommand:
        mass = sum(enemy.mass for enemy in group)
        center = (
            sum(enemy.x * enemy.mass for enemy in group) / mass,
            sum(enemy.y * enemy.mass for enemy in group) / mass,
        )
        observed = normalise(
            (
                sum(enemy.direction[0] * enemy.mass for enemy in group),
                sum(enemy.direction[1] * enemy.mass for enemy in group),
            )
        )
        nearest_own = min(
            node.own_blobs,
            key=lambda own: math.dist(center, own.pos),
        )
        chase = normalise((nearest_own.x - center[0], nearest_own.y - center[1]))
        flee = (-chase[0], -chase[1])
        observed_or_chase = observed if observed != (0.0, 0.0) else chase
        if response_type == 0:
            return PlayerCommand(observed_or_chase)
        if response_type == 1:
            intercept = normalise(
                (
                    nearest_own.x
                    + node.last_direction[0] * player_speed(nearest_own.radius)
                    - center[0],
                    nearest_own.y
                    + node.last_direction[1] * player_speed(nearest_own.radius)
                    - center[1],
                )
            )
            return PlayerCommand(intercept if intercept != (0.0, 0.0) else chase)
        if response_type == 2:
            return PlayerCommand(flee)
        if response_type == 3:
            return PlayerCommand((-chase[1], chase[0]))
        if response_type == 4:
            return PlayerCommand((chase[1], -chase[0]))
        if response_type not in (5, 6, 7):
            raise ValueError(f"unknown expected response type: {response_type}")

        predator_split = len(group) < 16 and any(
            enemy.mass >= SPLIT_MIN_MASS
            and can_eat_player_blob(enemy.radius / math.sqrt(2.0), nearest_own.radius)
            and math.dist(enemy.pos, nearest_own.pos)
            <= _split_chain_attack_reach(enemy.radius, nearest_own.radius)
            for enemy in group
        )
        prey_split = len(group) < 16 and any(
            enemy.mass >= SPLIT_MIN_MASS
            and any(
                can_eat_player_blob(own.radius, enemy.radius) for own in node.own_blobs
            )
            for enemy in group
        )
        if response_type == 6:
            return PlayerCommand(chase, split=predator_split)
        if response_type == 7:
            return PlayerCommand(flee, split=prey_split)
        if predator_split:
            return PlayerCommand(chase, split=True)
        if prey_split:
            return PlayerCommand(flee, split=True)

        eligible_viruses = tuple(
            virus
            for virus in viruses
            if virus.virus_id not in node.consumed_virus_ids
            and any(can_consume_virus(enemy.radius, virus.radius) for enemy in group)
        )
        resource = min(
            eligible_viruses,
            key=lambda virus: math.dist(center, virus.pos),
            default=None,
        )
        if resource is None:
            resource = min(
                (
                    food
                    for food in foods
                    if food.food_id not in node.eaten_food_ids
                ),
                key=lambda food: math.dist(center, food.pos),
                default=None,
            )
        if resource is None:
            return PlayerCommand(observed_or_chase)
        return PlayerCommand(
            normalise(
                (resource.pos[0] - center[0], resource.pos[1] - center[1])
            )
        )

    def _expected_continuations(
        self,
        direction: tuple[float, float],
    ) -> tuple[tuple[float, float], ...]:
        unit = normalise(direction) or self._previous_direction
        result = []
        seen = set()
        for angle in (0.0, -math.pi / 6, math.pi / 6, -math.pi / 2, math.pi / 2):
            cosine = math.cos(angle)
            sine = math.sin(angle)
            candidate = (
                unit[0] * cosine - unit[1] * sine,
                unit[0] * sine + unit[1] * cosine,
            )
            key = self._direction_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            result.append(candidate)
        return tuple(result[: self._EXPECTED_CONTINUATION_LIMIT])

    def _real_predators(self, result: StepResult) -> tuple[EnemyBlob, ...]:
        return tuple(
            enemy
            for enemy in result.node.enemies
            if enemy.blob_id >= 0 and enemy.stale_rounds == 0
        )

    def _certified_next_step_all_lost(self, result: StepResult) -> bool:
        """Prove one real player can cover all escape disks with one command."""

        own_blobs = result.node.own_blobs
        predators = self._real_predators(result)
        turn = self._tactical._advisor_planning_turn
        if not own_blobs or not predators or turn is None:
            return not own_blobs
        if not self._certificate_own_domain_is_event_free(result, turn):
            return False

        def physically_contains(own, enemy: EnemyBlob) -> bool:
            if not can_eat_player_blob(enemy.radius, own.radius):
                return False
            own_displacement = player_speed(own.radius) + math.hypot(
                own.eject_vx,
                own.eject_vy,
            )
            return math.dist(own.pos, enemy.pos) + own_displacement <= enemy.radius

        by_player: dict[int, list[EnemyBlob]] = {}
        for enemy in predators:
            by_player.setdefault(enemy.player_id, []).append(enemy)
        for group in by_player.values():
            if not self._certificate_player_group_is_complete(result, group):
                continue
            cannot_merge_this_step = len(group) == 1 or all(
                enemy.merge_cooldown > 1 for enemy in group
            )
            if cannot_merge_this_step and any(
                not any(
                    can_eat_player_blob(enemy.radius, own.radius) for enemy in group
                )
                for own in own_blobs
            ):
                # A common command and same-player separation cannot change
                # radius. If even one escape disk has no size-compatible eater
                # and cooldown prevents a merge after decrement, this player
                # cannot possibly certify full cover under any direction.
                continue
            for direction in self._shared_enemy_command_candidates(group, own_blobs):
                moved = self._move_enemy_group_with_shared_command(
                    group,
                    direction,
                    turn.arena_size,
                )
                if not self._certificate_witness_is_event_free(
                    result,
                    moved,
                    turn,
                ):
                    continue
                # Joint containment is valid here: every resultant fragment was
                # advanced by the same physically compatible player command.
                if all(
                    any(physically_contains(own, enemy) for enemy in moved)
                    for own in own_blobs
                ):
                    return True
        return False

    def _certificate_own_domain_is_event_free(
        self,
        result: StepResult,
        turn,
    ) -> bool:
        """Reject certificates whose escape disk contains another event.

        The containment proof models only normal movement. Food growth, virus
        ejection, or an own capture can change radius or place fragments outside
        that disk before player interaction. Returning unknown/False is the
        sound result whenever any such event is reachable.
        """

        foods = tuple(
            food
            for food in getattr(turn, "foods", ())
            if food.food_id not in result.node.eaten_food_ids
        )
        viruses = tuple(
            virus
            for virus in getattr(turn, "viruses", ())
            if virus.virus_id not in result.node.consumed_virus_ids
        )
        for own in result.node.own_blobs:
            displacement = player_speed(own.radius) + math.hypot(
                own.eject_vx,
                own.eject_vy,
            )
            contact_reach = own.radius + displacement
            if any(math.dist(own.pos, food.pos) <= contact_reach for food in foods):
                return False
            if any(
                can_consume_virus(own.radius, virus.radius)
                and math.dist(own.pos, virus.pos) <= contact_reach
                for virus in viruses
            ):
                return False
            if any(
                can_eat_player_blob(own.radius, enemy.radius)
                and math.dist(own.pos, enemy.pos)
                <= contact_reach + player_speed(enemy.radius)
                for enemy in result.node.enemies
            ):
                return False
        return True

    @staticmethod
    def _certificate_player_group_is_complete(
        result: StepResult,
        group: list[EnemyBlob],
    ) -> bool:
        """Require every tracked fragment of the witness player to be visible."""

        player_id = group[0].player_id
        tracked_group = tuple(
            enemy for enemy in result.node.enemies if enemy.player_id == player_id
        )
        return bool(tracked_group) and all(
            enemy.blob_id >= 0 and enemy.stale_rounds == 0 for enemy in tracked_group
        )

    def _certificate_witness_is_event_free(
        self,
        result: StepResult,
        moved: tuple[EnemyBlob, ...],
        turn,
    ) -> bool:
        """Validate the projected witness before applying containment."""

        foods = tuple(
            food
            for food in getattr(turn, "foods", ())
            if food.food_id not in result.node.eaten_food_ids
        )
        viruses = tuple(
            virus
            for virus in getattr(turn, "viruses", ())
            if virus.virus_id not in result.node.consumed_virus_ids
        )
        if any(
            math.dist(enemy.pos, food.pos) <= enemy.radius
            for enemy in moved
            for food in foods
        ):
            return False
        if any(
            can_consume_virus(enemy.radius, virus.radius)
            and math.dist(enemy.pos, virus.pos) <= enemy.radius
            for enemy in moved
            for virus in viruses
        ):
            return False

        player_id = moved[0].player_id
        third_players = tuple(
            enemy for enemy in result.node.enemies if enemy.player_id != player_id
        )
        third_groups: dict[int, list[EnemyBlob]] = {}
        for third in third_players:
            if third.blob_id < 0 or third.stale_rounds > 0:
                return False
            third_groups.setdefault(third.player_id, []).append(third)
        if any(
            sum(enemy.merge_cooldown <= 1 for enemy in group) >= 2
            for group in third_groups.values()
        ):
            # A third player can change size before player interaction. The
            # pairwise check below is sound only while its blobs cannot merge.
            return False
        for witness in moved:
            for third in third_players:
                third_radius = _decayed_radius(third.radius)
                # The witness has already moved; the extra witness-speed term
                # deliberately broadens the guard for third-player attraction
                # and tracking uncertainty. It can only suppress a certificate.
                closure = player_speed(third.radius) + player_speed(witness.radius)
                distance = math.dist(witness.pos, third.pos)
                if can_eat_player_blob(witness.radius, third_radius) and (
                    distance <= witness.radius + closure
                ):
                    return False
                if can_eat_player_blob(third_radius, witness.radius) and (
                    distance <= third_radius + closure
                ):
                    return False
        return True

    def _shared_enemy_command_candidates(
        self,
        enemies: list[EnemyBlob],
        own_blobs: tuple[OwnBlob, ...],
    ) -> tuple[tuple[float, float], ...]:
        """Return a finite normal-move witness set, never an exhaustive model.

        Omitted headings and enemy split commands can only make the all-loss
        certificate return unknown/False: every enumerated heading is still
        projected and checked exactly. Per player this is bounded by E+O+1;
        across P players the bound is E+P(O+1), rather than the former E*O
        cross product for each fragmented group.
        """

        enemy_mass = sum(enemy.mass for enemy in enemies)
        enemy_center = (
            sum(enemy.x * enemy.mass for enemy in enemies) / enemy_mass,
            sum(enemy.y * enemy.mass for enemy in enemies) / enemy_mass,
        )
        own_mass = sum(own.mass for own in own_blobs)
        own_center = (
            sum(own.x * own.mass for own in own_blobs) / own_mass,
            sum(own.y * own.mass for own in own_blobs) / own_mass,
        )
        raw = [enemy.direction for enemy in enemies]
        raw.extend(
            (own.x - enemy_center[0], own.y - enemy_center[1]) for own in own_blobs
        )
        raw.append(
            (
                own_center[0] - enemy_center[0],
                own_center[1] - enemy_center[1],
            )
        )
        candidates: dict[tuple[int, int], tuple[float, float]] = {}
        for direction in raw:
            unit = normalise(direction)
            if unit == (0.0, 0.0):
                continue
            key = (round(unit[0] * _KEY_SCALE), round(unit[1] * _KEY_SCALE))
            candidates.setdefault(key, unit)
        # Insertion order tests the public observed commands first, allowing a
        # concrete witness to stop the player scan before bearing alternatives.
        return tuple(candidates.values())

    def _move_enemy_group_with_shared_command(
        self,
        enemies: list[EnemyBlob],
        direction: tuple[float, float],
        arena_size: float,
    ) -> tuple[EnemyBlob, ...]:
        unit = normalise(direction)
        moved = tuple(
            replace(
                enemy,
                x=_clamp(
                    enemy.x + unit[0] * player_speed(enemy.radius),
                    enemy.radius,
                    arena_size - enemy.radius,
                ),
                y=_clamp(
                    enemy.y + unit[1] * player_speed(enemy.radius),
                    enemy.radius,
                    arena_size - enemy.radius,
                ),
                radius=_decayed_radius(enemy.radius),
                direction=unit,
                merge_cooldown=max(0, enemy.merge_cooldown - 1),
            )
            for enemy in enemies
        )
        return self._tactical._stabilise_enemy_blobs(moved, arena_size)

    def _certified_next_step_survival(self, result: StepResult) -> bool:
        """Prove at least one fragment clears every real predator reach disk."""

        if not result.node.own_blobs:
            return False
        turn = self._tactical._advisor_planning_turn
        if turn is None or not self._certificate_own_domain_is_event_free(result, turn):
            return False
        if not self._certificate_survival_domain_is_event_free(result, turn):
            return False
        predators = self._real_predators(result)
        if not predators:
            return True
        for own in result.node.own_blobs:
            own_displacement = player_speed(own.radius) + math.hypot(
                own.eject_vx,
                own.eject_vy,
            )
            if all(
                not can_eat_player_blob(enemy.radius, own.radius)
                or math.dist(own.pos, enemy.pos)
                - max(
                    enemy.radius + player_speed(enemy.radius),
                    _split_chain_attack_reach(enemy.radius, own.radius),
                )
                - own_displacement
                > 0.0
                for enemy in predators
            ):
                return True
        return False

    def _certificate_survival_domain_is_event_free(
        self,
        result: StepResult,
        turn,
    ) -> bool:
        """Restrict survival proofs to visible singleton enemy players."""

        enemies = result.node.enemies
        counts: dict[int, int] = {}
        for enemy in enemies:
            if enemy.blob_id < 0 or enemy.stale_rounds > 0:
                return False
            counts[enemy.player_id] = counts.get(enemy.player_id, 0) + 1
        if any(count != 1 for count in counts.values()):
            return False

        foods = tuple(
            food
            for food in getattr(turn, "foods", ())
            if food.food_id not in result.node.eaten_food_ids
        )
        viruses = tuple(
            virus
            for virus in getattr(turn, "viruses", ())
            if virus.virus_id not in result.node.consumed_virus_ids
        )
        for enemy in enemies:
            contact_reach = enemy.radius + player_speed(enemy.radius)
            if any(math.dist(enemy.pos, food.pos) <= contact_reach for food in foods):
                return False
            if any(
                can_consume_virus(enemy.radius, virus.radius)
                and math.dist(enemy.pos, virus.pos) <= contact_reach
                for virus in viruses
            ):
                return False

        for index, first in enumerate(enemies):
            for second in enemies[index + 1 :]:
                if first.player_id == second.player_id:
                    continue
                closure = player_speed(first.radius) + player_speed(second.radius)
                distance = math.dist(first.pos, second.pos)
                if can_eat_player_blob(first.radius, second.radius) and (
                    distance <= first.radius + closure
                ):
                    return False
                if can_eat_player_blob(second.radius, first.radius) and (
                    distance <= second.radius + closure
                ):
                    return False
        return True

    @staticmethod
    def _build_static_scene(
        state,
        *,
        tracked_enemies: tuple[EnemyBlob, ...] | None = None,
    ) -> HybridStaticScene:
        raw = []
        raw.extend(
            ("virus", virus.pos, virus.radius, virus) for virus in state.visible_viruses
        )
        enemies = state.visible_blobs if tracked_enemies is None else tracked_enemies
        raw.extend(("enemy", enemy.pos, enemy.radius, enemy) for enemy in enemies)
        grouped: dict[tuple[str, int, int, int], list[tuple]] = {}
        for kind, pos, radius, source in raw:
            geometry = (
                kind,
                round(pos[0] * _KEY_SCALE),
                round(pos[1] * _KEY_SCALE),
                round(radius * _KEY_SCALE),
            )
            grouped.setdefault(geometry, []).append((pos, radius, source))

        entities = []
        for geometry in sorted(grouped):
            kind = geometry[0]
            rows = grouped[geometry]
            for ordinal, (pos, radius, source) in enumerate(rows):
                entities.append(
                    HybridEntity(
                        kind=kind,
                        key=(*geometry, ordinal),
                        pos=(float(pos[0]), float(pos[1])),
                        radius=float(radius),
                        source=source,
                    )
                )
        entities_tuple = tuple(entities)
        return HybridStaticScene(entities=entities_tuple)

    def _gate_local_oracle(
        self,
        *,
        own_blobs: tuple,
        base_decision: StrategyDecision,
        scene: HybridStaticScene,
        tracked_enemies: tuple[EnemyBlob, ...] | None = None,
    ) -> HybridGateResult:
        # ``base_decision`` is intentionally not treated as a proved lower
        # bound. Directional alignment cannot prove equivalent split, contact,
        # or future-safety outcomes over a multi-step interaction.
        _ = base_decision
        horizon = 3.0
        base_lower = 0.0
        tactical_upper = 0.0
        reasons: set[str] = set()

        if tracked_enemies is None:
            tracked_enemies = tuple(
                EnemyBlob(
                    player_id=int(entity.source.player_id),
                    blob_id=index,
                    x=entity.pos[0],
                    y=entity.pos[1],
                    radius=entity.radius,
                    direction=(0.0, 0.0),
                    stale_rounds=int(getattr(entity.source, "stale_rounds", 0)),
                    merge_cooldown=int(entity.source.merge_cooldown),
                )
                for index, entity in enumerate(scene.entities)
                if entity.kind == "enemy"
            )
        merged_enemies = self._tactical._future_enemy_envelopes(
            tracked_enemies,
            horizon=int(horizon),
        )
        threat_envelopes = (
            *((enemy, False) for enemy in tracked_enemies),
            *((enemy, True) for enemy in merged_enemies),
        )
        for envelope, merged in threat_envelopes:
            for own in own_blobs:
                if not can_eat_player_blob(envelope.radius, own.radius):
                    continue
                normal_reach = envelope.radius + horizon * player_speed(envelope.radius)
                split_reach = _split_chain_attack_reach(
                    envelope.radius,
                    own.radius,
                ) + (horizon - 1.0) * player_speed(envelope.radius)
                danger_reach = max(normal_reach, split_reach)
                hidden_uncertainty = envelope.stale_rounds > 0
                if (
                    not hidden_uncertainty
                    and math.dist(own.pos, envelope.pos) > danger_reach
                ):
                    continue
                tactical_upper = max(tactical_upper, own.radius * own.radius)
                reasons.add("predator_safety")
                if hidden_uncertainty:
                    reasons.add("hidden_predator_uncertainty")
                if merged:
                    reasons.add("merged_predator_safety")

        for entity in scene.entities:
            if entity.kind == "virus":
                for own in own_blobs:
                    if not can_consume_virus(own.radius, entity.radius):
                        continue
                    distance = math.dist(own.pos, entity.pos)
                    reach = own.radius + horizon * player_speed(own.radius)
                    if distance > reach:
                        continue
                    upper = entity.mass * math.exp(-distance / max(reach, 1.0))
                    tactical_upper = max(tactical_upper, upper)
                    reasons.add("reachable_virus")
                continue

            for own in own_blobs:
                if not can_eat_player_blob(own.radius, entity.radius):
                    continue
                distance = math.dist(own.pos, entity.pos)
                walking_reach = own.radius + horizon * player_speed(own.radius)
                split_reach = _split_chain_attack_reach(
                    own.radius,
                    entity.radius,
                )
                reach = max(walking_reach, split_reach)
                if distance > reach:
                    continue
                upper = entity.mass * math.exp(-distance / max(reach, 1.0))
                walking = distance <= walking_reach
                tactical_upper = max(tactical_upper, upper)
                reasons.add("reachable_prey" if walking else "split_capture")

        ordered_reasons = tuple(sorted(reasons))
        return HybridGateResult(
            triggered=bool(ordered_reasons),
            reasons=ordered_reasons,
            base_lower_bound=base_lower,
            tactical_upper_bound=tactical_upper,
        )

    @staticmethod
    def _primitive_complexity(
        *,
        own_count: int,
        food_count: int,
        virus_count: int,
        enemy_count: int,
    ) -> int:
        roots = LocalTacticalSearchStrategy._LOCAL_ROOT_LIMIT
        # 1 root stage + 5 continuation stages + a worst-case exact first-step
        # safety check for every aggregate root.
        stages = 2 + LocalTacticalSearchStrategy._DEEP_DIRECTION_LIMIT
        primitive_work = (
            own_count * max(food_count, 1)
            + own_count * own_count
            + own_count * max(enemy_count, 1)
            + virus_count
        )
        return roots * stages * primitive_work

    @staticmethod
    def _empty_profile(
        *,
        input_ms: float = 0.0,
        static_ms: float = 0.0,
        root_ms: float = 0.0,
        output_ms: float = 0.0,
    ) -> dict[str, float]:
        return {
            "input": input_ms,
            "static": static_ms,
            "root": root_ms,
            "tactical_setup": 0.0,
            "transition": 0.0,
            "contact": 0.0,
            "response": 0.0,
            "terminal": 0.0,
            "advisor": 0.0,
            "output": max(0.0, output_ms),
        }

    @staticmethod
    def _complete_profile(
        diagnostics: dict[str, object],
        total_started: float,
    ) -> None:
        """Close the additive profile with all uninstrumented output work."""

        profile = diagnostics.get("hybrid_profile_ms")
        if not isinstance(profile, dict):
            return
        accounted_ms = sum(
            float(value) for name, value in profile.items() if name != "output"
        )
        profile["output"] = max(
            0.0,
            (perf_counter() - total_started) * 1000.0 - accounted_ms,
        )

    def _hybrid_diagnostics(
        self,
        gate: HybridGateResult,
        *,
        base_decision: StrategyDecision,
        tactical_decision: StrategyDecision | None,
        profile: dict[str, float],
        planner_tier: str,
        complexity: int,
    ) -> dict[str, object]:
        action_matches = bool(
            tactical_decision is not None
            and tactical_decision.split == base_decision.split
            and self._direction_key(tactical_decision.direction)
            == self._direction_key(base_decision.direction)
        )
        base_key = self._tactical._action_key(
            Action(base_decision.direction, split=base_decision.split)
        )
        base_local = (
            next(
                (
                    (rank, score)
                    for rank, (action, score) in enumerate(
                        self._tactical._local_root_scores,
                        start=1,
                    )
                    if self._tactical._action_key(action) == base_key
                ),
                (None, None),
            )
            if tactical_decision is not None
            else (None, None)
        )
        return {
            "hybrid_triggered": gate.triggered,
            "hybrid_full_executed": tactical_decision is not None,
            "hybrid_always_full": self.always_full,
            "hybrid_trigger_reasons": gate.reasons,
            "hybrid_base_lower_bound": gate.base_lower_bound,
            "hybrid_tactical_upper_bound": gate.tactical_upper_bound,
            "hybrid_estimated_regret": gate.estimated_regret,
            "hybrid_action_matches_base": action_matches,
            "hybrid_base_reason": base_decision.reason,
            "hybrid_base_split": base_decision.split,
            "hybrid_tactical_reason": (
                tactical_decision.reason if tactical_decision is not None else None
            ),
            "hybrid_tactical_split": (
                tactical_decision.split if tactical_decision is not None else None
            ),
            "hybrid_base_local_rank": base_local[0],
            "hybrid_base_local_score": base_local[1],
            "hybrid_base_transition": (
                self._tactical.required_semantic_transition
                if tactical_decision is not None
                else None
            ),
            "hybrid_planner_tier": planner_tier,
            "hybrid_complexity": complexity,
            "hybrid_profile_ms": dict(profile),
            **self._last_expected_evidence_diagnostics,
        }

    @staticmethod
    def _direction_key(direction: tuple[float, float]) -> tuple[int, int]:
        unit = normalise(direction)
        return (round(unit[0] * 1_000_000), round(unit[1] * 1_000_000))
