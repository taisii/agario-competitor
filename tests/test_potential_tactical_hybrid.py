from __future__ import annotations

from collections import Counter
from dataclasses import replace
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from lib.models.blob_model import BlobModel, VisibleBlobModel  # noqa: E402
from lib.models.food_model import FoodModel  # noqa: E402
from lib.models.virus_model import VirusModel  # noqa: E402
from strategies.base import StrategyContext, StrategyDecision  # noqa: E402
from strategies.features import normalise  # noqa: E402
from strategies.local_tactical_search import LocalTacticalSearchStrategy  # noqa: E402
from strategies.potential_field import PotentialFieldHunterStrategy  # noqa: E402
from strategies.potential_tactical_hybrid import (  # noqa: E402
    PotentialTacticalHybridStrategy,
)
from strategies.receding_horizon import (  # noqa: E402
    Action,
    EnemyBlob,
    OwnBlob,
    SearchNode,
    StepResult,
)
from strategies.world_transition import (  # noqa: E402
    CompleteJointCommand,
    ExpectedEvidence,
    ExpectedOutcomeStats,
    PlayerCommand,
)


def _context(
    *,
    own: tuple[BlobModel, ...] | None = None,
    enemies: tuple[VisibleBlobModel, ...] = (),
    foods: tuple[FoodModel, ...] = (),
    viruses: tuple[VirusModel, ...] = (),
    round_number: int = 100,
    view_center: tuple[float, float] | None = None,
    vision_size: float = 20.0,
) -> StrategyContext:
    own = own or (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=1.0),)
    total_mass = sum(blob.radius * blob.radius for blob in own)
    center = (
        sum(blob.pos[0] * blob.radius * blob.radius for blob in own) / total_mass,
        sum(blob.pos[1] * blob.radius * blob.radius for blob in own) / total_mass,
    )
    state = SimpleNamespace(
        me=SimpleNamespace(
            player_id=0,
            x=center[0],
            y=center[1],
            radius=math.sqrt(total_mass),
            alive=True,
            blobs={blob.blob_id: blob for blob in own},
        ),
        visible_blobs=list(enemies),
        visible_food=list(foods),
        visible_viruses=list(viruses),
        map=SimpleNamespace(size=60.0),
        round=round_number,
        max_rounds=1400,
        rankings=list(range(8)),
        view_center=center if view_center is None else view_center,
        vision_size=vision_size,
    )
    return StrategyContext(
        game=SimpleNamespace(state=state),
        query=SimpleNamespace(update={}),
    )


def _enemy(
    *,
    blob_id: int,
    pos: tuple[float, float],
    radius: float,
    player_id: int = 1,
    merge_cooldown: int = 0,
) -> VisibleBlobModel:
    return VisibleBlobModel(
        player_id=player_id,
        team_id=player_id,
        blob_id=blob_id,
        pos=pos,
        radius=radius,
        merge_cooldown=merge_cooldown,
    )


def _exact_first_step(
    context: StrategyContext,
    decision: StrategyDecision,
) -> StepResult:
    checker = LocalTacticalSearchStrategy()
    turn = checker._prepare_turn(context)
    assert turn is not None
    return checker._step(
        node=turn.node,
        action=Action(
            decision.direction,
            split=decision.split,
            reason=decision.reason,
        ),
        foods=turn.foods,
        viruses=turn.viruses,
        arena_size=turn.arena_size,
        first_step=True,
        safety_weight=1.0,
        aggression=1.0,
    )


def test_no_tactical_trigger_returns_the_exact_potential_decision() -> None:
    context = _context()
    expected = PotentialFieldHunterStrategy().choose(context)
    hybrid = PotentialTacticalHybridStrategy()

    actual = hybrid.choose(context)

    assert actual == expected
    assert not hybrid.last_hybrid_diagnostics["hybrid_triggered"]
    assert not hybrid.last_hybrid_diagnostics["hybrid_full_executed"]


def test_predator_uncertainty_triggers_full_local_search() -> None:
    context = _context(enemies=(_enemy(blob_id=7, pos=(35.0, 30.0), radius=2.0),))
    hybrid = PotentialTacticalHybridStrategy()

    decision = hybrid.choose(context)

    assert decision.diagnostics["hybrid_full_executed"]
    assert "predator_safety" in decision.diagnostics["hybrid_trigger_reasons"]
    assert decision.diagnostics["local_roots_ranked"] == 12


def test_expected_outcome_uses_scenario_weights_and_uncalibrated_evidence_is_inert() -> None:
    base = ExpectedOutcomeStats.from_samples(
        (0.0, 10.0, 20.0, 30.0),
        (0.5, 0.2, 0.2, 0.1),
    )
    tactical = ExpectedOutcomeStats.from_samples(
        (5.0, 15.0, 25.0, 35.0),
        (0.5, 0.2, 0.2, 0.1),
    )
    evidence = ExpectedEvidence(
        scenario_ids=(0, 1, 2, 3),
        scenario_weights=(0.5, 0.2, 0.2, 0.1),
        base=base,
        tactical=tactical,
    )

    assert base.mean_mass == 9.0
    assert base.death_rate == 0.5
    assert base.cvar20_mass == 0.0
    assert evidence.mean_delta == 5.0
    assert not evidence.calibrated
    assert not evidence.supports_override


def test_expected_evidence_fails_closed_on_misaligned_scenarios() -> None:
    stats = ExpectedOutcomeStats.from_samples((1.0, 2.0), (0.5, 0.5))

    with pytest.raises(ValueError, match="unique"):
        ExpectedEvidence(
            scenario_ids=(0, 0),
            scenario_weights=(0.5, 0.5),
            base=stats,
            tactical=stats,
        )
    with pytest.raises(ValueError, match="align"):
        ExpectedEvidence(
            scenario_ids=(0,),
            scenario_weights=(1.0,),
            base=stats,
            tactical=stats,
        )
    with pytest.raises(ValueError, match="finite"):
        ExpectedEvidence(
            scenario_ids=(0, 1),
            scenario_weights=(math.nan, 0.5),
            base=stats,
            tactical=stats,
        )


def test_expected_override_rejects_negative_paired_delta_tail() -> None:
    weights = (0.2,) * 5
    base = ExpectedOutcomeStats.from_samples(
        (0.0, 100.0, 100.0, 100.0, 100.0),
        weights,
    )
    tactical = ExpectedOutcomeStats.from_samples(
        (1.0, 99.0, 102.0, 102.0, 102.0),
        weights,
    )
    evidence = ExpectedEvidence(
        scenario_ids=tuple(range(5)),
        scenario_weights=weights,
        base=base,
        tactical=tactical,
        heldout_model_error=0.0,
    )

    assert evidence.mean_delta > 0.0
    assert tactical.death_rate <= base.death_rate
    assert tactical.cvar20_mass >= base.cvar20_mass
    assert evidence.paired_delta_cvar20 == -1.0
    assert not evidence.supports_override


def test_expected_evidence_does_not_cancel_deaths_between_scenarios() -> None:
    weights = (0.5, 0.5)
    evidence = ExpectedEvidence(
        scenario_ids=(0, 1),
        scenario_weights=weights,
        base=ExpectedOutcomeStats.from_samples((0.0, 10.0), weights),
        tactical=ExpectedOutcomeStats.from_samples((5.0, 0.0), weights),
        heldout_model_error=0.0,
    )

    assert evidence.tactical.death_rate == evidence.base.death_rate
    assert not evidence.paired_death_nonworse
    assert not evidence.paired_survival_improvement
    assert not evidence.supports_override


def test_calibrated_expected_gate_requires_nonsplit_structural_offense(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BOT_EXPECTED_MODEL_ERROR_BOUND", "0.8")
    strategy = PotentialTacticalHybridStrategy()
    base = StrategyDecision(direction=(1.0, 0.0), reason="potential_mix")
    tactical = StrategyDecision(direction=(0.0, 1.0), reason="virus_harvest")
    weights = (0.125,) * 8
    evidence = ExpectedEvidence(
        scenario_ids=tuple(range(8)),
        scenario_weights=weights,
        base=ExpectedOutcomeStats.from_samples((1.0,) * 8, weights),
        tactical=ExpectedOutcomeStats.from_samples((3.0,) * 8, weights),
        tactical_gain_positive_probability=1.0,
        heldout_model_error=strategy._expected_model_error_bound,
    )
    base_key = strategy._tactical._action_key(Action(base.direction))
    strategy._tactical.root_transition_summaries[base_key] = {
        "fatal": False,
        "immediate_dead": False,
    }
    monkeypatch.setattr(strategy, "_expected_evidence", lambda *_: evidence)

    selected, reason = strategy._select_advisor_decision(base, tactical)
    assert selected == tactical
    assert reason == "tactical_expected_evidence"

    selected, _ = strategy._select_advisor_decision(
        base,
        replace(tactical, split=True),
    )
    assert selected == base
    selected, _ = strategy._select_advisor_decision(
        base,
        replace(tactical, reason="local_food_field"),
    )
    assert selected == base


def test_uncalibrated_production_gate_skips_expected_world_rollouts(
    monkeypatch,
) -> None:
    strategy = PotentialTacticalHybridStrategy()
    base = StrategyDecision(direction=(1.0, 0.0), reason="potential_mix")
    tactical = StrategyDecision(direction=(0.0, 1.0), reason="virus_harvest")
    base_key = strategy._tactical._action_key(Action(base.direction))
    strategy._tactical.root_transition_summaries[base_key] = {
        "fatal": False,
        "immediate_dead": False,
    }

    def unexpected_evidence(*_args):
        raise AssertionError("fail-closed production must not simulate Expected worlds")

    monkeypatch.setattr(strategy, "_expected_evidence", unexpected_evidence)

    selected, reason = strategy._select_advisor_decision(base, tactical)

    assert selected is base
    assert reason == "base"
    assert strategy._last_expected_evidence_diagnostics == {
        "expected_evidence_structural_offense": True,
        "expected_evidence_base_resolved_split": False,
        "expected_evidence_tactical_resolved_split": False,
        "expected_evidence_skipped_reason": "uncalibrated_model",
    }


def test_expected_gate_fails_closed_when_base_resolves_to_real_split(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BOT_EXPECTED_MODEL_ERROR_BOUND", "0.8")
    strategy = PotentialTacticalHybridStrategy()
    base = StrategyDecision(
        direction=(1.0, 0.0),
        split=True,
        reason="split_prey",
    )
    tactical = StrategyDecision(
        direction=(0.0, 1.0),
        reason="virus_harvest",
    )
    node = SearchNode(
        own_blobs=(OwnBlob(0, 30.0, 30.0, 3.0),),
        enemies=(),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="base",
        last_direction=(1.0, 0.0),
    )
    strategy._tactical._advisor_planning_turn = SimpleNamespace(
        node=node,
        foods=(),
        viruses=(),
        arena_size=60.0,
    )
    base_key = strategy._tactical._action_key(
        Action(base.direction, split=base.split)
    )
    strategy._tactical.root_transition_summaries[base_key] = {
        "fatal": False,
        "immediate_dead": False,
        "physical_mass": node.total_mass,
    }
    called = False

    def unexpected_evidence(*_args):
        nonlocal called
        called = True
        raise AssertionError("split-OOD root must not enter Expected evaluation")

    monkeypatch.setattr(strategy, "_expected_evidence", unexpected_evidence)

    selected, reason = strategy._select_advisor_decision(base, tactical)

    assert selected is base
    assert reason == "base"
    assert called is False
    assert strategy._last_expected_evidence_diagnostics == {
        "expected_evidence_structural_offense": True,
        "expected_evidence_base_resolved_split": True,
        "expected_evidence_tactical_resolved_split": False,
        "expected_evidence_skipped_reason": (
            "resolved_split_root_out_of_distribution"
        ),
    }


def test_expected_gate_allows_split_command_that_resolves_without_child(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BOT_EXPECTED_MODEL_ERROR_BOUND", "0.8")
    strategy = PotentialTacticalHybridStrategy()
    base = StrategyDecision(
        direction=(1.0, 0.0),
        split=True,
        reason="split_prey",
    )
    tactical = StrategyDecision(
        direction=(0.0, 1.0),
        reason="virus_harvest",
    )
    node = SearchNode(
        own_blobs=(OwnBlob(0, 30.0, 30.0, 1.0),),
        enemies=(),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="base",
        last_direction=(1.0, 0.0),
    )
    strategy._tactical._advisor_planning_turn = SimpleNamespace(
        node=node,
        foods=(),
        viruses=(),
        arena_size=60.0,
    )
    weights = (0.125,) * 8
    evidence = ExpectedEvidence(
        scenario_ids=tuple(range(8)),
        scenario_weights=weights,
        base=ExpectedOutcomeStats.from_samples((1.0,) * 8, weights),
        tactical=ExpectedOutcomeStats.from_samples((3.0,) * 8, weights),
        tactical_gain_positive_probability=1.0,
        heldout_model_error=strategy._expected_model_error_bound,
    )
    base_key = strategy._tactical._action_key(
        Action(base.direction, split=base.split)
    )
    strategy._tactical.root_transition_summaries[base_key] = {
        "fatal": False,
        "immediate_dead": False,
        "physical_mass": node.total_mass,
    }
    monkeypatch.setattr(strategy, "_expected_evidence", lambda *_: evidence)

    selected, reason = strategy._select_advisor_decision(base, tactical)

    assert selected is tactical
    assert reason == "tactical_expected_evidence"
    assert strategy._last_expected_evidence_diagnostics[
        "expected_evidence_base_resolved_split"
    ] is False


def test_expected_evidence_uses_eight_joint_samples_not_player_cartesian_product(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BOT_EXPECTED_MODEL_ERROR_BOUND", "1000000000")
    context = _context(
        own=(BlobModel(blob_id=0, pos=(30.0, 30.0), radius=2.0),),
        enemies=(
            _enemy(blob_id=1, player_id=1, pos=(33.0, 30.0), radius=1.0),
            _enemy(blob_id=2, player_id=1, pos=(33.0, 31.0), radius=1.0),
            _enemy(blob_id=3, player_id=2, pos=(37.0, 30.0), radius=3.0),
        ),
    )
    strategy = PotentialTacticalHybridStrategy()
    monkeypatch.setattr(strategy, "_is_structural_offense", lambda _: True)

    decision = strategy.choose(context)

    assert decision.diagnostics["expected_evidence_scenario_ids"] == tuple(range(8))
    assert decision.diagnostics["expected_evidence_scenario_count"] == 8
    assert math.isclose(
        sum(decision.diagnostics["expected_evidence_scenario_weights"]),
        1.0,
    )
    assert decision.diagnostics["expected_evidence_calibrated"]
    assert not decision.diagnostics["expected_evidence_supports_override"]
    assert decision.diagnostics["hybrid_advisor_action"] == "base"
    assert 0 < decision.diagnostics["expected_evidence_transition_cache_misses"] <= 120
    assert decision.diagnostics["expected_evidence_transition_cache_hits"] >= 0
    assert decision.diagnostics["expected_evidence_ms"] >= 0.0
    assert decision.diagnostics["expected_evidence_stress_modes"] == (
        ("all_aggressive", 6),
        ("all_evasive", 7),
    )
    assert (
        decision.diagnostics["expected_evidence_oracle_mean_mass"]["base"]
        >= decision.diagnostics["expected_evidence_base_mean_mass"]
    )
    assert (
        decision.diagnostics["expected_evidence_oracle_mean_mass"]["tactical"]
        >= decision.diagnostics["expected_evidence_tactical_mean_mass"]
    )


def test_expected_response_table_is_stratified_stable_and_decorrelated() -> None:
    strategy = PotentialTacticalHybridStrategy()

    for sample_count in (8, strategy._EXPECTED_OFFLINE_SCENARIO_COUNT):
        table = strategy._expected_response_table(
            {7, 1, 2},
            sample_count=sample_count,
        )
        reordered = strategy._expected_response_table(
            {2, 7, 1},
            sample_count=sample_count,
        )

        assert table == reordered
        assert len(table) == sample_count
        for player_id in (1, 2, 7):
            counts = Counter(row[player_id] for row in table)
            for response_type, weight in enumerate(
                strategy._EXPECTED_RESPONSE_WEIGHTS
            ):
                expected = sample_count * weight
                assert counts[response_type] in {
                    math.floor(expected),
                    math.ceil(expected),
                }

        first = tuple(row[1] for row in table)
        second = tuple(row[2] for row in table)
        assert first != second
        assert sum(left == right for left, right in zip(first, second, strict=True)) < (
            sample_count
        )


def test_expected_information_key_ignores_reindexed_ids_and_diagnostic_counters() -> None:
    strategy = PotentialTacticalHybridStrategy()
    node = SearchNode(
        own_blobs=(OwnBlob(1, 20.0, 20.0, 2.0),),
        enemies=(EnemyBlob(7, 3, 24.0, 20.0, 1.0),),
        score=4.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="first",
        last_direction=(1.0, 0.0),
    )
    relabelled = replace(
        node,
        own_blobs=(replace(node.own_blobs[0], blob_id=99),),
        enemies=(replace(node.enemies[0], blob_id=42),),
        score=-100.0,
        projected_food=9,
        projected_captures=8,
        projected_viruses=7,
        min_safety_margin=-12.0,
    )

    assert strategy._expected_information_key(node) == (
        strategy._expected_information_key(relabelled)
    )


def test_expected_transition_cache_distinguishes_zero_and_east_split() -> None:
    strategy = PotentialTacticalHybridStrategy()
    node = SearchNode(
        own_blobs=(OwnBlob(0, 20.0, 20.0, 2.0),),
        enemies=(),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="first",
        last_direction=(1.0, 0.0),
    )
    zero = Action((0.0, 0.0), split=True)
    east = Action((1.0, 0.0), split=True)
    zero_joint = CompleteJointCommand.build(
        live_player_ids={0},
        commands={0: PlayerCommand(zero.direction, split=True)},
    )
    east_joint = CompleteJointCommand.build(
        live_player_ids={0},
        commands={0: PlayerCommand(east.direction, split=True)},
    )

    assert strategy._expected_transition_cache_key(
        node=node,
        action=zero,
        joint=zero_joint,
        first_step=True,
    ) != strategy._expected_transition_cache_key(
        node=node,
        action=east,
        joint=east_joint,
        first_step=True,
    )


def test_identical_next_information_uses_one_continuation_across_samples(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BOT_EXPECTED_MODEL_ERROR_BOUND", "1000000000")
    context = _context(
        own=(BlobModel(blob_id=0, pos=(30.0, 30.0), radius=2.0),),
        viruses=(VirusModel(virus_id=3, pos=(32.0, 30.0), radius=1.5),),
    )
    strategy = PotentialTacticalHybridStrategy()
    monkeypatch.setattr(strategy, "_is_structural_offense", lambda _: True)

    decision = strategy.choose(context)

    selected = decision.diagnostics["expected_evidence_selected_continuations"]
    for label in ("base", "tactical"):
        live_directions = {
            direction
            for _, direction in selected[label]
            if direction is not None
        }
        assert len(live_directions) <= 1


def test_expected_outcome_is_invariant_to_sample_iteration_order(monkeypatch) -> None:
    monkeypatch.setenv("BOT_EXPECTED_MODEL_ERROR_BOUND", "1000000000")
    context = _context(
        own=(BlobModel(blob_id=0, pos=(30.0, 30.0), radius=2.0),),
        enemies=(
            _enemy(blob_id=1, player_id=1, pos=(34.0, 30.0), radius=1.2),
            _enemy(blob_id=2, player_id=2, pos=(27.0, 34.0), radius=2.4),
        ),
    )
    forward = PotentialTacticalHybridStrategy()
    reverse = PotentialTacticalHybridStrategy()
    monkeypatch.setattr(forward, "_is_structural_offense", lambda _: True)
    monkeypatch.setattr(reverse, "_is_structural_offense", lambda _: True)
    reverse._EXPECTED_SCENARIO_IDS = tuple(reversed(reverse._EXPECTED_SCENARIO_IDS))

    forward_decision = forward.choose(context)
    reverse_decision = reverse.choose(context)

    for key in (
        "expected_evidence_base_mean_mass",
        "expected_evidence_tactical_mean_mass",
        "expected_evidence_base_death_rate",
        "expected_evidence_tactical_death_rate",
        "expected_evidence_base_cvar20",
        "expected_evidence_tactical_cvar20",
    ):
        assert math.isclose(
            forward_decision.diagnostics[key],
            reverse_decision.diagnostics[key],
            abs_tol=1e-12,
        )


def test_adaptive_joint_scenario_splits_fleeing_prey_with_one_player_command() -> None:
    strategy = PotentialTacticalHybridStrategy()
    node = SearchNode(
        own_blobs=(OwnBlob(0, 30.0, 30.0, 4.0),),
        enemies=(EnemyBlob(7, 1, 35.0, 30.0, 2.0),),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )

    command = strategy._expected_enemy_command(
        list(node.enemies),
        node=node,
        response_type=5,
        foods=(),
        viruses=(),
    )

    assert command.split
    assert command.unit == (1.0, 0.0)


def test_base_direction_is_always_one_of_the_locally_scored_roots() -> None:
    context = _context(enemies=(_enemy(blob_id=7, pos=(35.0, 30.0), radius=2.0),))
    hybrid = PotentialTacticalHybridStrategy()
    base = PotentialFieldHunterStrategy().choose(context)

    decision = hybrid.choose(context)

    base_key = hybrid._tactical._action_key(Action(base.direction, split=base.split))
    assert any(
        hybrid._tactical._action_key(action) == base_key
        and action.reason == "hybrid_base"
        for action, _ in hybrid._tactical._local_root_scores
    )
    top_action, top_score = hybrid._tactical._local_root_scores[0]
    assert decision.diagnostics["local_selected_rank"] == 1
    assert decision.diagnostics["hybrid_advisor_action"] == "base"
    assert hybrid._direction_key(decision.direction) == hybrid._direction_key(
        base.direction
    )
    assert top_action
    assert top_score is not None


def test_lowest_proxy_base_root_replaces_the_twelfth_local_slot() -> None:
    strategy = LocalTacticalSearchStrategy()
    ordinary = tuple(
        Action(
            (math.cos(index * math.tau / 13), math.sin(index * math.tau / 13)),
            reason=f"angle_{index}",
        )
        for index in range(12)
    )
    base = Action(
        (math.cos(12 * math.tau / 13), math.sin(12 * math.tau / 13)),
        reason="hybrid_base",
    )
    strategy._root_proxy_scores = tuple(
        (action, 100.0 - index) for index, action in enumerate(ordinary)
    ) + ((base, -1_000.0),)
    node = SearchNode(
        own_blobs=(OwnBlob(0, 30.0, 30.0, 4.0),),
        enemies=(),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )

    strategy._rank_roots_by_local_dp(
        node=node,
        actions=(*ordinary, base),
        foods=(),
        viruses=(),
        arena_size=60.0,
        targets=(),
    )

    assert len(strategy._local_root_scores) == 12
    assert any(
        action.reason == "hybrid_base" for action, _ in strategy._local_root_scores
    )


def test_attack_heavy_root_selection_reserves_physical_unsplit_comparator() -> None:
    strategy = LocalTacticalSearchStrategy()
    strategy.use_aggregate_local_dp = True
    split_actions = tuple(
        Action(
            (math.cos(index * math.tau / 13), math.sin(index * math.tau / 13)),
            split=True,
            reason="hybrid_base" if index == 0 else f"split_prey_{index}",
        )
        for index in range(12)
    )
    unsplit = Action((-1.0, 0.0), reason="low_proxy_escape")
    # This semantic duplicate must not consume a second rollout slot.
    duplicate = replace(split_actions[1], reason="split_alias")
    actions = (*split_actions, duplicate, unsplit)
    strategy._root_proxy_scores = tuple(
        (action, 100.0 - index) for index, action in enumerate(actions)
    )
    node = SearchNode(
        own_blobs=(OwnBlob(0, 30.0, 30.0, 4.0),),
        enemies=(),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )

    ranked = strategy._rank_roots_by_local_dp(
        node=node,
        actions=actions,
        foods=(),
        viruses=(),
        arena_size=60.0,
        targets=(),
    )

    scored_actions = tuple(action for action, _ in strategy._local_root_scores)
    scored_keys = tuple(
        (*normalise(action.direction), action.split) for action in scored_actions
    )
    assert len(scored_actions) == strategy._LOCAL_ROOT_LIMIT
    assert len(set(scored_keys)) == len(scored_keys)
    assert any(action.reason == "hybrid_base" for action in scored_actions)
    assert any(not action.split for action in scored_actions)
    assert (*normalise(unsplit.direction), unsplit.split) in scored_keys
    assert len(
        {(*normalise(action.direction), action.split) for action in ranked}
    ) == len(ranked)


def test_all_split_candidate_space_adds_legal_unsplit_baseline() -> None:
    strategy = LocalTacticalSearchStrategy()
    strategy.use_aggregate_local_dp = True
    actions = tuple(
        Action(
            (math.cos(index * math.tau / 12), math.sin(index * math.tau / 12)),
            split=True,
            reason=f"split_prey_{index}",
        )
        for index in range(12)
    )
    strategy._root_proxy_scores = tuple(
        (action, 12.0 - index) for index, action in enumerate(actions)
    )
    node = SearchNode(
        own_blobs=(OwnBlob(0, 30.0, 30.0, 4.0),),
        enemies=(),
        score=0.0,
        first_direction=(0.0, 1.0),
        first_split=False,
        first_reason="keep",
        last_direction=(0.0, 1.0),
    )

    strategy._rank_roots_by_local_dp(
        node=node,
        actions=actions,
        foods=(),
        viruses=(),
        arena_size=60.0,
        targets=(),
    )

    assert any(
        not action.split and action.reason == "aggregate_unsplit_baseline"
        for action, _ in strategy._local_root_scores
    )


def test_noop_split_candidates_are_canonicalised_before_aggregate_shield() -> None:
    strategy = LocalTacticalSearchStrategy()
    strategy.use_aggregate_local_dp = True
    actions = tuple(
        Action(
            (math.cos(index * math.tau / 12), math.sin(index * math.tau / 12)),
            split=True,
            reason=f"split_prey_{index}",
        )
        for index in range(12)
    )
    strategy._root_proxy_scores = tuple(
        (action, 12.0 - index) for index, action in enumerate(actions)
    )
    node = SearchNode(
        own_blobs=(OwnBlob(0, 30.0, 30.0, 1.0),),
        enemies=(),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )

    ranked = strategy._rank_roots_by_local_dp(
        node=node,
        actions=actions,
        foods=(),
        viruses=(),
        arena_size=60.0,
        targets=(),
    )

    assert ranked
    assert all(not action.split for action in ranked)
    assert all(not action.split for action, _ in strategy._local_root_scores)


def test_fatal_proxy_top_cannot_bypass_the_local_dp_ranking() -> None:
    context = _context(
        own=(BlobModel(blob_id=0, pos=(30.0, 30.0), radius=1.5),),
        enemies=(_enemy(blob_id=1, pos=(30.5, 30.0), radius=3.0),),
    )
    strategy = LocalTacticalSearchStrategy()
    original_rank = strategy._rank_roots_by_local_dp
    incoming_proxy_top: list[Action] = []

    def capture_proxy_top(**kwargs):
        incoming_proxy_top.append(strategy._root_proxy_scores[0][0])
        return original_rank(**kwargs)

    strategy._rank_roots_by_local_dp = capture_proxy_top  # type: ignore[method-assign]
    decision = strategy.choose(context)

    assert incoming_proxy_top[0].split
    assert decision.split is False
    assert decision.diagnostics["local_selected_rank"] == 1
    selected_key = strategy._action_key(
        Action(decision.direction, split=decision.split)
    )
    assert selected_key == strategy._action_key(strategy._local_root_scores[0][0])
    assert decision.score == strategy._local_root_scores[0][1]
    assert strategy._root_proxy_scores == strategy._local_root_scores


def test_reachable_prey_virus_and_safety_each_open_the_gate() -> None:
    strategy = PotentialTacticalHybridStrategy()
    base = StrategyDecision(direction=(-1.0, 0.0), reason="test_base")
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=2.0),)

    cases = (
        (
            _context(
                own=own, enemies=(_enemy(blob_id=1, pos=(33.0, 30.0), radius=1.0),)
            ),
            "reachable_prey",
        ),
        (
            _context(
                own=own,
                viruses=(VirusModel(virus_id=1, pos=(33.0, 30.0), radius=1.5),),
            ),
            "reachable_virus",
        ),
        (
            _context(
                own=(BlobModel(blob_id=0, pos=(30.0, 30.0), radius=1.0),),
                enemies=(_enemy(blob_id=1, pos=(35.0, 30.0), radius=2.0),),
            ),
            "predator_safety",
        ),
    )

    for context, reason in cases:
        scene = strategy._build_static_scene(context.game.state)
        gate = strategy._gate_local_oracle(
            own_blobs=tuple(context.game.state.me.blobs.values()),
            base_decision=base,
            scene=scene,
        )
        assert gate.triggered
        assert reason in gate.reasons


def test_aligned_base_split_still_opens_gate_and_matches_oracle() -> None:
    context = _context(
        own=(BlobModel(blob_id=0, pos=(30.0, 30.0), radius=1.5),),
        enemies=(_enemy(blob_id=1, pos=(31.5, 30.0), radius=0.5),),
    )
    base = PotentialFieldHunterStrategy().choose(context)
    gated = PotentialTacticalHybridStrategy(always_full=False)
    oracle = PotentialTacticalHybridStrategy(always_full=True)

    gated_decision = gated.choose(context)
    oracle_decision = oracle.choose(context)

    assert base.split
    assert gated.last_hybrid_diagnostics["hybrid_triggered"]
    assert gated_decision.split == oracle_decision.split == base.split
    assert gated._direction_key(gated_decision.direction) == gated._direction_key(
        base.direction
    )
    assert oracle._direction_key(oracle_decision.direction) == oracle._direction_key(
        base.direction
    )


def test_future_same_player_merge_is_a_predator_gate_upper_bound() -> None:
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=1.3),)
    context = _context(
        own=own,
        enemies=(
            _enemy(
                blob_id=1,
                player_id=1,
                pos=(32.5, 30.0),
                radius=1.1,
                merge_cooldown=0,
            ),
            _enemy(
                blob_id=2,
                player_id=1,
                pos=(34.5, 30.0),
                radius=1.1,
                merge_cooldown=0,
            ),
        ),
    )
    strategy = PotentialTacticalHybridStrategy()
    tracked = strategy._tactical.prepare_enemy_memory_for_external_gate(context)
    scene = strategy._build_static_scene(
        context.game.state,
        tracked_enemies=tracked,
    )

    gate = strategy._gate_local_oracle(
        own_blobs=own,
        base_decision=StrategyDecision(direction=(-1.0, 0.0)),
        scene=scene,
        tracked_enemies=tracked,
    )

    assert all(enemy.radius < own[0].radius for enemy in tracked)
    assert gate.triggered
    assert "merged_predator_safety" in gate.reasons


def test_hidden_predator_tracker_is_prepared_once_and_matches_direct_oracle() -> None:
    first = _context(
        own=(BlobModel(blob_id=0, pos=(30.0, 30.0), radius=1.0),),
        enemies=(_enemy(blob_id=1, pos=(35.0, 30.0), radius=2.0),),
        round_number=100,
        view_center=(30.0, 30.0),
        vision_size=20.0,
    )
    hidden = _context(
        own=(BlobModel(blob_id=0, pos=(20.0, 30.0), radius=1.0),),
        round_number=101,
        view_center=(20.0, 30.0),
        vision_size=10.0,
    )
    direct = LocalTacticalSearchStrategy()
    oracle = PotentialTacticalHybridStrategy(always_full=True)
    gated = PotentialTacticalHybridStrategy(always_full=False)

    direct.choose(first)
    oracle.choose(first)
    gated.choose(first)
    direct_hidden = direct.choose(hidden)
    oracle_hidden = oracle.choose(hidden)
    gated_hidden = gated.choose(hidden)

    assert gated.last_hybrid_diagnostics["hybrid_triggered"]
    assert (
        "hidden_predator_uncertainty"
        in gated.last_hybrid_diagnostics["hybrid_trigger_reasons"]
    )
    assert gated._direction_key(gated_hidden.direction) == gated._direction_key(
        oracle_hidden.direction
    )
    assert oracle_hidden.split == gated_hidden.split
    assert oracle_hidden.diagnostics["hybrid_advisor_action"] == "base"
    assert direct_hidden.reason


def test_non_primary_fragment_can_open_the_prey_gate() -> None:
    strategy = PotentialTacticalHybridStrategy()
    own = (
        BlobModel(blob_id=0, pos=(5.0, 5.0), radius=3.0),
        BlobModel(blob_id=1, pos=(30.0, 30.0), radius=2.0),
    )
    context = _context(
        own=own,
        enemies=(_enemy(blob_id=1, pos=(33.0, 30.0), radius=1.0),),
    )

    gate = strategy._gate_local_oracle(
        own_blobs=own,
        base_decision=StrategyDecision(direction=(-1.0, 0.0)),
        scene=strategy._build_static_scene(context.game.state),
    )

    assert gate.triggered
    assert "reachable_prey" in gate.reasons


def test_far_tactical_objects_have_zero_upper_bound_and_do_not_trigger() -> None:
    strategy = PotentialTacticalHybridStrategy()
    own = (BlobModel(blob_id=0, pos=(5.0, 5.0), radius=2.0),)
    context = _context(
        own=own,
        enemies=(_enemy(blob_id=1, pos=(55.0, 55.0), radius=1.0),),
        viruses=(VirusModel(virus_id=1, pos=(50.0, 50.0), radius=1.5),),
    )

    gate = strategy._gate_local_oracle(
        own_blobs=own,
        base_decision=StrategyDecision(direction=(1.0, 0.0)),
        scene=strategy._build_static_scene(context.game.state),
    )

    assert not gate.triggered
    assert gate.tactical_upper_bound == 0.0


def test_geometry_signature_ignores_public_ids_and_handles_duplicates() -> None:
    first = _context(
        enemies=(
            _enemy(blob_id=8, pos=(33.0, 30.0), radius=1.0),
            _enemy(blob_id=9, pos=(33.0, 30.0), radius=1.0),
        ),
        foods=(FoodModel(food_id=7, pos=(29.0, 30.0)),),
    )
    reindexed = _context(
        enemies=(
            _enemy(blob_id=0, pos=(33.0, 30.0), radius=1.0),
            _enemy(blob_id=1, pos=(33.0, 30.0), radius=1.0),
        ),
        foods=(FoodModel(food_id=0, pos=(29.0, 30.0)),),
    )

    first_scene = PotentialTacticalHybridStrategy._build_static_scene(first.game.state)
    second_scene = PotentialTacticalHybridStrategy._build_static_scene(
        reindexed.game.state
    )

    assert first_scene.signature == second_scene.signature
    duplicate_keys = [key for key in first_scene.signature if key[0] == "enemy"]
    assert [key[-1] for key in duplicate_keys] == [0, 1]
    assert all(key[0] != "food" for key in first_scene.signature)


def test_geometry_keys_survive_unrelated_add_remove() -> None:
    base = _context(enemies=(_enemy(blob_id=1, pos=(33.0, 30.0), radius=1.0),))
    added = _context(
        enemies=(
            _enemy(blob_id=0, pos=(50.0, 50.0), radius=1.2),
            _enemy(blob_id=1, pos=(33.0, 30.0), radius=1.0),
        )
    )
    base_keys = set(
        PotentialTacticalHybridStrategy._build_static_scene(base.game.state).signature
    )
    added_keys = set(
        PotentialTacticalHybridStrategy._build_static_scene(added.game.state).signature
    )

    assert base_keys < added_keys


def test_always_full_matches_gated_action_when_gate_opens() -> None:
    context = _context(enemies=(_enemy(blob_id=7, pos=(35.0, 30.0), radius=2.0),))
    gated = PotentialTacticalHybridStrategy(always_full=False)
    oracle = PotentialTacticalHybridStrategy(always_full=True)

    gated_decision = gated.choose(context)
    oracle_decision = oracle.choose(context)

    assert gated._direction_key(gated_decision.direction) == oracle._direction_key(
        oracle_decision.direction
    )
    assert gated_decision.split == oracle_decision.split
    assert oracle_decision.diagnostics["hybrid_always_full"]


def test_always_full_evaluates_local_dp_but_keeps_safe_base_by_default() -> None:
    context = _context(enemies=(_enemy(blob_id=7, pos=(35.0, 30.0), radius=2.0),))
    base = PotentialFieldHunterStrategy().choose(context)
    direct = LocalTacticalSearchStrategy()
    direct.proxy_coarse_after_seconds = math.inf
    direct.required_semantic_root = Action(
        base.direction,
        split=base.split,
        reason="hybrid_base",
    )
    direct_decision = direct.choose(context)
    oracle = PotentialTacticalHybridStrategy(always_full=True)

    oracle_decision = oracle.choose(context)

    assert oracle._direction_key(oracle_decision.direction) == oracle._direction_key(
        base.direction
    )
    assert oracle_decision.split == base.split
    assert oracle_decision.diagnostics["hybrid_advisor_action"] == "base"
    assert direct_decision.reason


def test_dense_gate_and_always_full_profiles_without_reducing_roots() -> None:
    blobs = tuple(
        BlobModel(
            blob_id=index,
            pos=(20.0 + index % 4 * 3.0, 20.0 + index // 4 * 3.0),
            radius=0.9,
            merge_cooldown=10,
        )
        for index in range(16)
    )
    foods = tuple(FoodModel(food_id=index, pos=(20.0, 20.0)) for index in range(16))
    context = _context(own=blobs, foods=foods)
    gated = PotentialTacticalHybridStrategy(always_full=False)
    oracle = PotentialTacticalHybridStrategy(always_full=True)

    gated_decision = gated.choose(context)
    oracle_decision = oracle.choose(context)

    assert gated_decision.reason == "potential_mix"
    assert oracle_decision.diagnostics["local_roots_ranked"] == 12
    assert oracle_decision.diagnostics["local_contact_checks"] > 0
    assert set(oracle_decision.diagnostics["hybrid_profile_ms"]) == {
        "input",
        "static",
        "root",
        "tactical_setup",
        "transition",
        "contact",
        "response",
        "terminal",
        "advisor",
        "output",
    }


def test_dense_complexity_uses_aggregate_12_by_5_without_wall_clock_switch() -> None:
    blobs = tuple(
        BlobModel(
            blob_id=index,
            pos=(20.0 + index % 4 * 3.0, 20.0 + index // 4 * 3.0),
            radius=0.9,
            merge_cooldown=10,
        )
        for index in range(16)
    )
    context = _context(
        own=blobs,
        foods=tuple(FoodModel(food_id=index, pos=(20.0, 20.0)) for index in range(16)),
        enemies=(_enemy(blob_id=7, pos=(30.0, 30.0), radius=2.0),),
    )
    exact = PotentialTacticalHybridStrategy(always_full=True)
    aggregate = PotentialTacticalHybridStrategy(always_full=False)

    exact_decision = exact.choose(context)
    aggregate_decision = aggregate.choose(context)

    assert exact_decision.diagnostics["hybrid_planner_tier"] == "exact"
    assert aggregate_decision.diagnostics["hybrid_planner_tier"] == "aggregate"
    assert exact_decision.diagnostics["local_roots_ranked"] == 12
    assert aggregate_decision.diagnostics["local_roots_ranked"] == 12
    assert aggregate_decision.diagnostics["local_aggregate_continuations"] == 60
    assert aggregate_decision.diagnostics["local_contact_checks"] == 0
    assert aggregate_decision.diagnostics["local_response_evaluations"] > 0
    assert aggregate_decision.diagnostics["local_aggregate_safety_checks"] == 0
    assert aggregate_decision.diagnostics["local_aggregate_safety_certificates"] == 1
    assert aggregate_decision.diagnostics["hybrid_complexity"] > 0
    assert aggregate_decision.split is False
    assert aggregate._direction_key(
        aggregate_decision.direction
    ) == exact._direction_key(exact_decision.direction)
    assert exact_decision.diagnostics["local_selected_rank"] == 1
    assert aggregate_decision.diagnostics["local_selected_rank"] == 1
    assert exact_decision.diagnostics["hybrid_advisor_action"] == "base"
    assert aggregate_decision.diagnostics["hybrid_advisor_action"] == "base"


def test_aggregate_all_threatened_prefers_unsplit_least_bad_escape() -> None:
    """A fatal split proxy must not win when every aggregate root is unsafe."""

    context = _context(
        own=(
            BlobModel(
                blob_id=0,
                pos=(33.36589874628311, 28.70262341124038),
                radius=1.6340308344053363,
            ),
            BlobModel(
                blob_id=1,
                pos=(31.702198048120643, 28.876965055356123),
                radius=1.2323433684625844,
                merge_cooldown=8,
            ),
            BlobModel(
                blob_id=2,
                pos=(30.804764818511728, 28.705046899503646),
                radius=0.8595979147864754,
                merge_cooldown=18,
            ),
        ),
        enemies=(
            _enemy(
                blob_id=10,
                player_id=1,
                pos=(28.567299290877827, 26.652843336941547),
                radius=1.7411697167705071,
                merge_cooldown=8,
            ),
            _enemy(
                blob_id=20,
                player_id=2,
                pos=(29.413267310352623, 27.82889410578619),
                radius=0.8922482153697655,
            ),
            _enemy(
                blob_id=30,
                player_id=3,
                pos=(27.751762345998646, 31.126156825012874),
                radius=2.967171464957575,
                merge_cooldown=8,
            ),
        ),
        foods=(
            FoodModel(food_id=1, pos=(31.47236682025763, 28.259424464874)),
            FoodModel(food_id=2, pos=(28.923125169821706, 33.27636971791047)),
            FoodModel(food_id=3, pos=(34.80239775777902, 27.885209152553195)),
        ),
    )
    aggregate = PotentialTacticalHybridStrategy(always_full=False)

    aggregate_decision = aggregate.choose(context)

    assert aggregate_decision.diagnostics["hybrid_planner_tier"] == "aggregate"
    assert len(aggregate._tactical._local_root_scores) == 12
    assert any(not action.split for action, _ in aggregate._tactical._local_root_scores)
    assert aggregate_decision.split is False
    assert aggregate_decision.diagnostics["local_selected_rank"] == 1

    # Validate the selected aggregate root against the exact first-step
    # simulator without adding that duplicate work to the production hot path.
    exact_result = _exact_first_step(context, aggregate_decision)
    assert exact_result.fatal is False


def test_aggregate_exact_shield_rejects_virus_fragmentation_death() -> None:
    context = _context(
        own=(
            BlobModel(
                blob_id=0,
                pos=(37.07544545753194, 25.556889959029455),
                radius=2.9191370946640824,
                merge_cooldown=5,
            ),
            BlobModel(
                blob_id=1,
                pos=(41.90102160449368, 27.306277018657852),
                radius=1.392640970802977,
                merge_cooldown=12,
            ),
            BlobModel(
                blob_id=2,
                pos=(33.1729401263613, 32.00475539706025),
                radius=2.509291636269347,
            ),
            BlobModel(
                blob_id=3,
                pos=(39.97261766975169, 28.64691213538144),
                radius=1.3011331792061789,
                merge_cooldown=5,
            ),
        ),
        enemies=(
            _enemy(
                blob_id=100,
                player_id=1,
                pos=(40.21554630678927, 25.65301015743068),
                radius=1.4455756280670446,
            ),
        ),
        foods=tuple(
            FoodModel(food_id=index, pos=pos)
            for index, pos in enumerate(
                (
                    (41.92074517582053, 33.78395665006494),
                    (40.5874048464527, 24.859135445979852),
                    (42.89609626432976, 35.17003529527081),
                    (38.484342393013804, 21.77872256633291),
                    (28.504185908032813, 36.04076980115167),
                    (46.21377681754876, 19.698667186274676),
                    (39.27290392625406, 37.954050044826474),
                    (46.182122090689944, 38.150712309393384),
                    (37.475772459465276, 26.97732471419697),
                    (31.188850079159934, 31.352333077123433),
                    (43.28165904371073, 20.066298122961015),
                )
            )
        ),
        viruses=(
            VirusModel(
                virus_id=1,
                pos=(38.26918587814793, 30.104828459729937),
                radius=1.7812603024597595,
            ),
            VirusModel(
                virus_id=2,
                pos=(43.596133029449724, 31.053281978604538),
                radius=1.5915613825100166,
            ),
            VirusModel(
                virus_id=3,
                pos=(46.58177366575312, 19.638705509663414),
                radius=1.6578830440211154,
            ),
        ),
        vision_size=50.0,
    )
    aggregate = PotentialTacticalHybridStrategy(always_full=False)

    decision = aggregate.choose(context)

    assert decision.diagnostics["hybrid_planner_tier"] == "aggregate"
    assert decision.diagnostics["local_aggregate_safety_checks"] > 0
    assert decision.diagnostics["hybrid_advisor_action"] == "base"
    assert _exact_first_step(context, decision).fatal is False


def test_advisor_ensures_missing_base_and_tactical_exact_summaries() -> None:
    context = _context(enemies=(_enemy(blob_id=7, pos=(35.0, 30.0), radius=2.0),))
    strategy = PotentialTacticalHybridStrategy(always_full=True)
    original_choose = strategy._tactical.choose

    def choose_and_clear(context):
        decision = original_choose(context)
        strategy._tactical.root_transition_summaries.clear()
        return decision

    strategy._tactical.choose = choose_and_clear  # type: ignore[method-assign]

    decision = strategy.choose(context)

    # Only the base root is authoritative until a target-specific secured-event
    # certificate exists for the tactical proposal.
    assert len(strategy._tactical.root_transition_summaries) == 1
    assert decision.diagnostics["hybrid_advisor_action"] == "base"


def test_advisor_does_not_recover_from_multiblob_immediate_seed_90670() -> None:
    strategy = PotentialTacticalHybridStrategy()
    strategy._tactical._own_player_id = 0
    node = SearchNode(
        own_blobs=(
            OwnBlob(1, 31.76198475946812, 6.060691759065144, 3.6212189793068106),
            OwnBlob(
                2,
                31.317893013980328,
                8.787803862410723,
                1.68818168186491,
                merge_cooldown=15,
            ),
            OwnBlob(3, 27.857422235143613, 10.276833041186094, 1.4638971928343656),
            OwnBlob(
                4,
                27.693017938929874,
                11.931532871704533,
                3.3292042238326682,
                merge_cooldown=15,
            ),
        ),
        enemies=(
            EnemyBlob(
                1,
                10,
                39.08063234838366,
                13.964157354555102,
                6.652167252938334,
                merge_cooldown=6,
            ),
            EnemyBlob(
                1,
                11,
                36.380840912364995,
                12.471995004883848,
                3.9939168763568027,
            ),
            EnemyBlob(
                2,
                20,
                27.926002112667792,
                21.343561573193742,
                1.2856343919875355,
                merge_cooldown=6,
            ),
            EnemyBlob(
                2,
                21,
                36.93598926706481,
                16.32205691104211,
                6.324416323370187,
            ),
        ),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="seed_90670",
        last_direction=(1.0, 0.0),
    )
    foods = tuple(
        FoodModel(food_id=index, pos=pos)
        for index, pos in enumerate(
            (
                (25.958574316086562, 7.733347184571528),
                (26.407665850686616, 6.014909777715513),
                (28.920026132803837, 0.28960871326618864),
                (34.85130636313746, 8.286576216308347),
                (35.91433200668448, 8.78115973747444),
                (27.727252335247968, 16.052008261443948),
            )
        )
    )
    viruses = tuple(
        VirusModel(virus_id=index, pos=pos, radius=radius)
        for index, (pos, radius) in enumerate(
            (
                ((28.29439892790053, 15.735242354843107), 1.1249818720389948),
                ((36.1861693824776, 1.0), 2.1897710761765206),
                ((37.38666799010159, 6.3262862459440985), 1.1658340775139076),
            )
        )
    )
    strategy._tactical._advisor_planning_turn = SimpleNamespace(
        node=node,
        foods=foods,
        viruses=viruses,
        arena_size=60.0,
    )

    def direction(angle_bin: int) -> tuple[float, float]:
        angle = math.tau * angle_bin / 96
        return math.cos(angle), math.sin(angle)

    base = StrategyDecision(direction=direction(60), reason="base")
    top = Action(direction(50), reason="aggregate_top")
    lower_safe = Action(direction(84), reason="virus_harvest")
    strategy._tactical._local_root_scores = ((top, 2.0), (lower_safe, 1.0))

    selected, reason = strategy._select_advisor_decision(
        base,
        StrategyDecision(direction=top.direction, reason=top.reason),
    )

    assert selected is base
    assert reason == "base"
    results = strategy._tactical.root_transition_results
    assert not results[
        strategy._tactical._action_key(Action(base.direction))
    ].node.own_blobs
    assert strategy._tactical._action_key(top) not in results
    assert strategy._tactical._action_key(lower_safe) not in results


def test_advisor_keeps_base_for_singleton_planner_immediate_loss() -> None:
    strategy = PotentialTacticalHybridStrategy()
    strategy._tactical._own_player_id = 0
    node = SearchNode(
        own_blobs=(OwnBlob(0, 30.0, 30.0, 0.5),),
        enemies=(EnemyBlob(1, 10, 30.0, 31.51, 1.5),),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="singleton_immediate",
        last_direction=(1.0, 0.0),
    )
    strategy._tactical._advisor_planning_turn = SimpleNamespace(
        node=node,
        foods=(),
        viruses=(),
        arena_size=60.0,
    )
    base_action = Action((1.0, 0.0), reason="hybrid_base")
    safe_action = Action((0.0, -1.0), reason="escape")
    strategy._tactical._local_root_scores = (
        (base_action, 2.0),
        (safe_action, 1.0),
    )
    base = StrategyDecision(direction=base_action.direction, reason="base")

    selected, reason = strategy._select_advisor_decision(
        base,
        StrategyDecision(direction=base_action.direction, reason="aggregate_top"),
    )

    assert selected is base
    assert reason == "base"
    results = strategy._tactical.root_transition_results
    assert not results[strategy._tactical._action_key(base_action)].node.own_blobs
    assert strategy._tactical._action_key(safe_action) not in results


def test_immediate_alive_escape_is_not_proof_against_enemy_split() -> None:
    strategy = PotentialTacticalHybridStrategy()
    strategy._tactical._own_player_id = 0
    node = SearchNode(
        own_blobs=(OwnBlob(0, 30.0, 30.0, 1.0),),
        enemies=(EnemyBlob(1, 10, 34.2, 30.0, 3.0, direction=(-1.0, 0.0)),),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="split_counterexample",
        last_direction=(1.0, 0.0),
    )
    strategy._tactical._advisor_planning_turn = SimpleNamespace(
        node=node,
        foods=(),
        viruses=(),
        arena_size=60.0,
    )
    base_action = Action((1.0, 0.0), reason="hybrid_base")
    walking_escape = Action((-1.0, 0.0), reason="walking_escape")
    strategy._tactical._local_root_scores = (
        (base_action, 2.0),
        (walking_escape, 1.0),
    )
    base = StrategyDecision(direction=base_action.direction, reason="base")

    selected, reason = strategy._select_advisor_decision(
        base,
        StrategyDecision(
            direction=walking_escape.direction,
            reason=walking_escape.reason,
        ),
    )

    assert selected is base
    assert reason == "base"
    results = strategy._tactical.root_transition_results
    assert not results[strategy._tactical._action_key(base_action)].node.own_blobs
    # Recovery is not entered, because the walking-only survivor would remain
    # inside the singleton predator's legal split-chain reach.
    assert strategy._tactical._action_key(walking_escape) not in results


def test_multiblob_false_immediate_does_not_trigger_least_loss() -> None:
    strategy = PotentialTacticalHybridStrategy()
    strategy._tactical._own_player_id = 0
    node = SearchNode(
        own_blobs=(
            OwnBlob(0, 20.0, 30.0, 0.5),
            OwnBlob(1, 40.0, 30.0, 0.5),
        ),
        enemies=(
            EnemyBlob(1, 10, 20.0, 31.51, 1.5),
            EnemyBlob(1, 11, 40.0, 28.49, 1.5),
        ),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="incompatible_immediate",
        last_direction=(1.0, 0.0),
    )
    strategy._tactical._advisor_planning_turn = SimpleNamespace(
        node=node,
        foods=(),
        viruses=(),
        arena_size=60.0,
    )
    base_action = Action((1.0, 0.0), reason="hybrid_base")
    alternate = Action((0.0, 1.0), reason="least_loss_candidate")
    strategy._tactical._local_root_scores = (
        (base_action, 2.0),
        (alternate, 1.0),
    )
    base = StrategyDecision(direction=base_action.direction, reason="base")

    selected, reason = strategy._select_advisor_decision(
        base,
        StrategyDecision(direction=alternate.direction, reason=alternate.reason),
    )

    assert selected is base
    assert reason == "base"
    assert strategy._tactical._action_key(alternate) not in (
        strategy._tactical.root_transition_results
    )


def test_next_step_certificate_rejects_illegal_current_multiblob_response() -> None:
    strategy = PotentialTacticalHybridStrategy()
    base = StrategyDecision(direction=(1.0, 0.0), reason="base")
    tactical = StrategyDecision(direction=(0.0, 1.0), reason="local")
    base_key = strategy._tactical._action_key(Action(base.direction))
    tactical_key = strategy._tactical._action_key(Action(tactical.direction))
    current_node = SearchNode(
        own_blobs=(OwnBlob(0, 20.0, 20.0, 1.0),),
        enemies=(
            EnemyBlob(1, 10, 40.0, 40.0, 2.0),
            EnemyBlob(1, 11, 45.0, 40.0, 2.0),
        ),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="current_multiblob",
        last_direction=(1.0, 0.0),
    )
    strategy._tactical._advisor_planning_turn = SimpleNamespace(
        node=current_node,
        foods=(),
        viruses=(),
        arena_size=60.0,
    )
    base_result = _next_step_merging_predator_result()
    strategy._tactical.root_transition_summaries = {
        base_key: {
            "fatal": False,
            "immediate_dead": False,
            "surviving_mass": 1.0,
            "physical_mass": 1.0,
            "food_gain": 0,
            "virus_gain": 0,
            "capture_gain": 0,
        },
        tactical_key: {
            "fatal": False,
            "immediate_dead": False,
            "surviving_mass": 1.0,
            "physical_mass": 1.0,
            "food_gain": 0,
            "virus_gain": 0,
            "capture_gain": 0,
        },
    }
    strategy._tactical.root_transition_results = {base_key: base_result}

    selected, reason = strategy._select_advisor_decision(base, tactical)

    assert selected is base
    assert reason == "base"


def test_advisor_requires_secured_gain_and_safe_mass_lower_bound() -> None:
    strategy = PotentialTacticalHybridStrategy()
    base = StrategyDecision(direction=(1.0, 0.0), reason="base")
    tactical = StrategyDecision(direction=(0.0, 1.0), reason="local")
    base_key = strategy._tactical._action_key(Action(base.direction))
    tactical_key = strategy._tactical._action_key(Action(tactical.direction))

    def summary(
        *,
        fatal=False,
        safe_mass=4.0,
        physical_mass=4.0,
        food=0,
        virus=0,
        capture=0,
    ):
        return {
            "fatal": fatal,
            "immediate_dead": fatal,
            "surviving_mass": safe_mass,
            "physical_mass": physical_mass,
            "food_gain": food,
            "virus_gain": virus,
            "capture_gain": capture,
        }

    strategy._tactical.root_transition_summaries = {
        base_key: summary(),
        tactical_key: summary(food=1, safe_mass=3.9),
    }
    selected, reason = strategy._select_advisor_decision(base, tactical)
    assert selected is base
    assert reason == "base"

    strategy._tactical.root_transition_summaries[tactical_key] = summary(
        virus=1,
    )
    selected, reason = strategy._select_advisor_decision(base, tactical)
    assert selected is base
    assert reason == "base"

    strategy._tactical.root_transition_summaries[tactical_key] = summary(
        food=1,
        safe_mass=4.0,
    )
    selected, reason = strategy._select_advisor_decision(base, tactical)
    assert selected is base
    assert reason == "base"

    strategy._tactical.root_transition_summaries[tactical_key] = summary(
        capture=1,
        safe_mass=4.0,
    )
    selected, reason = strategy._select_advisor_decision(base, tactical)
    assert selected is base
    assert reason == "base"

    strategy._tactical.root_transition_summaries[base_key] = summary(fatal=True)
    strategy._tactical.root_transition_summaries[tactical_key] = summary()
    selected, reason = strategy._select_advisor_decision(base, tactical)
    # Summary-only danger is insufficient for a binary veto: recovery roots
    # must have authoritative StepResults from the exact transition kernel.
    assert selected is base
    assert reason == "base"

    uncertain = summary(fatal=True)
    uncertain["immediate_dead"] = False
    strategy._tactical.root_transition_summaries[base_key] = uncertain
    selected, reason = strategy._select_advisor_decision(base, tactical)
    assert selected is base
    assert reason == "base"


def test_food_only_local_gain_does_not_replace_safe_base_split_prey() -> None:
    angle = 0.3
    context = _context(
        own=(BlobModel(blob_id=0, pos=(30.0, 30.0), radius=2.0),),
        enemies=(
            _enemy(
                blob_id=7,
                pos=(30.0 + 2.0 * math.cos(angle), 30.0 + 2.0 * math.sin(angle)),
                radius=0.45,
            ),
        ),
        foods=(FoodModel(food_id=1, pos=(29.0, 29.0)),),
    )
    base = PotentialFieldHunterStrategy().choose(context)
    strategy = PotentialTacticalHybridStrategy()

    decision = strategy.choose(context)

    assert base.split is True
    assert decision.diagnostics["hybrid_advisor_action"] == "base"
    assert decision.split == base.split
    assert strategy._direction_key(decision.direction) == strategy._direction_key(
        base.direction
    )


def _certificate_result(
    *,
    own: tuple[OwnBlob, ...],
    enemies: tuple[EnemyBlob, ...],
) -> StepResult:
    return StepResult(
        SearchNode(
            own_blobs=own,
            enemies=enemies,
            score=0.0,
            first_direction=(1.0, 0.0),
            first_split=False,
            first_reason="test",
            last_direction=(1.0, 0.0),
        )
    )


def test_fatal_certificate_allows_joint_cover_under_one_shared_command() -> None:
    strategy = PotentialTacticalHybridStrategy()
    strategy._tactical._advisor_planning_turn = SimpleNamespace(arena_size=60.0)
    own = (
        OwnBlob(0, 12.0, 30.0, 1.0),
        OwnBlob(1, 48.0, 30.0, 1.0),
    )

    # The two fragments belong to one player and already jointly cover every
    # escape disk. Joint cover is compatible because both receive the same
    # enumerated command; no impossible per-blob steering is required.
    result = _certificate_result(
        own=own,
        enemies=(
            EnemyBlob(1, 10, 12.0, 30.0, 5.0),
            EnemyBlob(1, 11, 48.0, 30.0, 5.0),
        ),
    )

    assert strategy._certified_next_step_all_lost(result) is True


def test_shared_command_candidates_scale_by_group_plus_own_not_cross_product() -> None:
    strategy = PotentialTacticalHybridStrategy()
    enemies = [
        EnemyBlob(
            1,
            index,
            20.0 + index * 0.1,
            20.0,
            1.0,
            direction=(math.cos(index), math.sin(index)),
        )
        for index in range(16)
    ]
    own = tuple(OwnBlob(index, 30.0, 10.0 + index, 1.0) for index in range(16))

    candidates = strategy._shared_enemy_command_candidates(enemies, own)

    assert len(candidates) <= len(enemies) + len(own) + 1


def test_fatal_certificate_skips_nonmerging_size_incompatible_group() -> None:
    strategy = PotentialTacticalHybridStrategy()
    strategy._tactical._advisor_planning_turn = SimpleNamespace(arena_size=60.0)
    result = _certificate_result(
        own=(OwnBlob(0, 30.0, 30.0, 1.0),),
        enemies=tuple(
            EnemyBlob(
                1,
                index,
                20.0 + index * 0.2,
                20.0,
                0.6,
                merge_cooldown=18,
            )
            for index in range(16)
        ),
    )

    def unexpected_projection(*args, **kwargs):
        raise AssertionError("size-incompatible group should be rejected statically")

    strategy._move_enemy_group_with_shared_command = unexpected_projection

    assert strategy._certified_next_step_all_lost(result) is False


def test_fatal_certificate_rejects_incompatible_same_player_merge() -> None:
    strategy = PotentialTacticalHybridStrategy()
    strategy._tactical._advisor_planning_turn = SimpleNamespace(arena_size=60.0)
    result = _certificate_result(
        own=(
            OwnBlob(0, 41.5572, 17.1995, 0.7924),
            OwnBlob(1, 43.5648, 16.9979, 0.9352),
        ),
        enemies=(
            EnemyBlob(
                1,
                10,
                39.3569,
                22.5419,
                3.5614,
                direction=(-0.4486, 0.8937),
            ),
            EnemyBlob(
                1,
                11,
                47.0892,
                17.6480,
                5.2532,
                direction=(0.0517, 0.9987),
            ),
        ),
    )

    # Per-blob nearest-target steering moves these fragments in incompatible
    # directions and then merges them into a false covering predator. No shared
    # observed/bearing command in the finite certificate set proves all-loss.
    assert strategy._certified_next_step_all_lost(result) is False


def _next_step_merging_predator_result(
    *,
    extra_enemies: tuple[EnemyBlob, ...] = (),
) -> StepResult:
    return _certificate_result(
        own=(OwnBlob(0, 30.0, 30.0, 1.0),),
        enemies=(
            EnemyBlob(
                1,
                10,
                25.999,
                30.0,
                4.0,
                direction=(0.0, 1.0),
                merge_cooldown=1,
            ),
            EnemyBlob(
                1,
                11,
                34.001,
                30.0,
                4.0,
                direction=(0.0, 1.0),
                merge_cooldown=1,
            ),
            *extra_enemies,
        ),
    )


def test_fatal_certificate_accepts_event_free_next_step_merge() -> None:
    strategy = PotentialTacticalHybridStrategy()
    strategy._tactical._advisor_planning_turn = SimpleNamespace(
        arena_size=60.0,
        foods=(),
        viruses=(),
    )

    assert (
        strategy._certified_next_step_all_lost(_next_step_merging_predator_result())
        is True
    )


def test_fatal_certificate_rejects_enemy_virus_event_before_capture() -> None:
    strategy = PotentialTacticalHybridStrategy()
    strategy._tactical._advisor_planning_turn = SimpleNamespace(
        arena_size=60.0,
        foods=(),
        viruses=(VirusModel(virus_id=7, pos=(30.0, 30.0), radius=1.5),),
    )
    result = _next_step_merging_predator_result()

    # The shared upward command merges the two r=4 fragments into an r=5.65
    # covering predator. Engine order then pops that predator on the virus into
    # 16 small fragments before player capture, so the normal-move containment
    # model is outside its proof domain.
    assert strategy._certified_next_step_all_lost(result) is False


def test_fatal_certificate_rejects_reachable_food_threshold_event() -> None:
    strategy = PotentialTacticalHybridStrategy()
    strategy._tactical._advisor_planning_turn = SimpleNamespace(
        arena_size=60.0,
        foods=(FoodModel(food_id=9, pos=(30.0, 30.0)),),
        viruses=(),
    )
    result = _next_step_merging_predator_result()

    # Food growth happens before player interaction and can change the eat-size
    # threshold. The certificate intentionally does not model that event.
    assert strategy._certified_next_step_all_lost(result) is False


def test_fatal_certificate_rejects_possible_third_player_capture() -> None:
    strategy = PotentialTacticalHybridStrategy()
    strategy._tactical._advisor_planning_turn = SimpleNamespace(
        arena_size=60.0,
        foods=(),
        viruses=(),
    )
    result = _next_step_merging_predator_result(
        extra_enemies=(
            EnemyBlob(
                2,
                20,
                30.0,
                38.0,
                6.5,
                direction=(0.0, -1.0),
            ),
        )
    )

    assert strategy._certified_next_step_all_lost(result) is False


def test_fatal_certificate_rejects_incomplete_visible_player_group() -> None:
    strategy = PotentialTacticalHybridStrategy()
    strategy._tactical._advisor_planning_turn = SimpleNamespace(arena_size=60.0)
    result = _certificate_result(
        own=(OwnBlob(0, 30.0, 30.0, 1.0),),
        enemies=(
            EnemyBlob(1, 10, 30.0, 30.0, 6.0),
            EnemyBlob(1, 11, 40.0, 30.0, 1.0, stale_rounds=1),
        ),
    )

    assert strategy._certified_next_step_all_lost(result) is False


def test_fatal_certificate_rejects_possible_contact_with_escape_room() -> None:
    strategy = PotentialTacticalHybridStrategy()
    strategy._tactical._advisor_planning_turn = SimpleNamespace(arena_size=60.0)
    result = _certificate_result(
        own=(OwnBlob(0, 34.8, 30.0, 1.0),),
        enemies=(EnemyBlob(1, 10, 30.0, 30.0, 5.0),),
    )

    assert strategy._certified_next_step_all_lost(result) is False


def test_fatal_certificate_rejects_partial_or_stale_containment() -> None:
    strategy = PotentialTacticalHybridStrategy()
    strategy._tactical._advisor_planning_turn = SimpleNamespace(arena_size=60.0)
    own = (
        OwnBlob(0, 30.0, 30.0, 1.0),
        OwnBlob(1, 45.0, 30.0, 1.0),
    )

    partial = _certificate_result(
        own=own,
        enemies=(EnemyBlob(1, 10, 30.0, 30.0, 5.0),),
    )
    stale = _certificate_result(
        own=(own[0],),
        enemies=(EnemyBlob(1, 10, 30.0, 30.0, 5.0, stale_rounds=1),),
    )

    assert strategy._certified_next_step_all_lost(partial) is False
    assert strategy._certified_next_step_all_lost(stale) is False


def test_fatal_certificate_proves_full_escape_disk_containment() -> None:
    strategy = PotentialTacticalHybridStrategy()
    strategy._tactical._advisor_planning_turn = SimpleNamespace(arena_size=60.0)
    result = _certificate_result(
        own=(
            OwnBlob(0, 29.9, 30.0, 1.0),
            OwnBlob(1, 30.1, 30.0, 1.0),
        ),
        enemies=(EnemyBlob(1, 10, 30.0, 30.0, 6.0),),
    )

    assert strategy._certified_next_step_all_lost(result) is True


def test_survival_certificate_accepts_event_free_distant_singleton() -> None:
    strategy = PotentialTacticalHybridStrategy()
    strategy._tactical._advisor_planning_turn = SimpleNamespace(
        arena_size=60.0,
        foods=(),
        viruses=(),
    )
    result = _certificate_result(
        own=(OwnBlob(0, 10.0, 10.0, 1.0),),
        enemies=(EnemyBlob(1, 10, 50.0, 50.0, 3.0),),
    )

    assert strategy._certified_next_step_survival(result) is True


def test_survival_certificate_rejects_next_step_enemy_merge() -> None:
    strategy = PotentialTacticalHybridStrategy()
    strategy._tactical._advisor_planning_turn = SimpleNamespace(
        arena_size=60.0,
        foods=(),
        viruses=(),
    )
    own = (OwnBlob(0, 30.0, 30.0, 1.0),)
    enemies = tuple(
        EnemyBlob(
            1,
            10 + index,
            30.0 + 1.5 * math.cos(math.tau * index / 4),
            30.0 + 1.5 * math.sin(math.tau * index / 4),
            1.05,
            direction=(1.0, 0.0),
            merge_cooldown=1,
        )
        for index in range(4)
    )
    result = _certificate_result(own=own, enemies=enemies)

    moved = strategy._move_enemy_group_with_shared_command(
        list(enemies),
        (1.0, 0.0),
        60.0,
    )
    assert len(moved) == 1
    assert moved[0].radius > 2.0
    assert strategy._certified_next_step_survival(result) is False


def test_advisor_override_becomes_next_turn_potential_history() -> None:
    first_context = _context(
        enemies=(_enemy(blob_id=7, pos=(35.0, 30.0), radius=2.0),),
        round_number=100,
    )
    second_context = _context(round_number=101)
    strategy = PotentialTacticalHybridStrategy(always_full=True)

    def force_advisor(_, tactical):
        return tactical, "food_proof"

    strategy._select_advisor_decision = force_advisor  # type: ignore[method-assign]
    first = strategy.choose(first_context)
    assert strategy._last_direction == first.direction
    assert strategy._previous_direction == first.direction

    expected = PotentialFieldHunterStrategy()
    expected._last_direction = first.direction
    expected_second = expected.choose(second_context)
    strategy.always_full = False
    actual_second = strategy.choose(second_context)

    assert strategy._direction_key(actual_second.direction) == strategy._direction_key(
        expected_second.direction
    )


def test_split_partial_loss_is_scored_but_not_globally_fatal() -> None:
    strategy = LocalTacticalSearchStrategy()
    strategy._own_player_id = 0
    node = SearchNode(
        own_blobs=(
            OwnBlob(blob_id=0, x=20.0, y=30.0, radius=3.0),
            OwnBlob(blob_id=1, x=40.0, y=30.0, radius=1.0),
        ),
        enemies=(
            EnemyBlob(
                player_id=1,
                blob_id=10,
                x=40.0,
                y=30.0,
                radius=2.0,
            ),
        ),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )

    result = strategy._step(
        node=node,
        action=Action((1.0, 0.0), split=True, reason="split"),
        foods=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
        safety_weight=1.0,
        aggression=1.0,
    )

    assert result.fatal is False
    assert 0.0 < result.node.total_mass < node.total_mass


def test_split_all_fragments_unavoidable_remains_fatal() -> None:
    strategy = LocalTacticalSearchStrategy()
    strategy._own_player_id = 0
    node = SearchNode(
        own_blobs=(OwnBlob(blob_id=0, x=30.0, y=30.0, radius=3.0),),
        enemies=(EnemyBlob(1, 10, 30.0, 30.0, 7.0),),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )

    result = strategy._step(
        node=node,
        action=Action((1.0, 0.0), split=True, reason="split"),
        foods=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
        safety_weight=1.0,
        aggression=1.0,
    )

    assert result.fatal is True
    assert result.node.total_mass == 0.0


def test_dead_fallback_uses_the_normal_hybrid_diagnostic_schema() -> None:
    context = _context()
    context.game.state.me.blobs = {}
    context.game.state.me.alive = False
    strategy = PotentialTacticalHybridStrategy()

    decision = strategy.choose(context)

    assert decision.reason == "dead_fallback"
    diagnostics = strategy.last_hybrid_diagnostics
    assert diagnostics["hybrid_triggered"] is False
    assert diagnostics["hybrid_full_executed"] is False
    assert diagnostics["hybrid_base_lower_bound"] == 0.0
    assert diagnostics["hybrid_tactical_upper_bound"] == 0.0
    assert set(diagnostics["hybrid_profile_ms"]) == {
        "input",
        "static",
        "root",
        "tactical_setup",
        "transition",
        "contact",
        "response",
        "terminal",
        "advisor",
        "output",
    }
