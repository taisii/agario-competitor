from __future__ import annotations

import sys
import math
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from lib.models.virus_model import VirusModel  # noqa: E402
from lib.models.food_model import FoodModel  # noqa: E402
from lib.interface.events.moves.move_player import MovePlayer  # noqa: E402
from engine.state.blob_state import BlobState  # noqa: E402
from engine.state.player_state import PlayerState  # noqa: E402
from engine.state.state_mutator import StateMutator  # noqa: E402
from strategies.receding_horizon import (  # noqa: E402
    Action,
    EnemyBlob,
    EnemyTrack,
    OwnBlob,
    ProxyBlobMotion,
    ProxyThreat,
    ReplayDominanceStrategy,
    SearchNode,
    StepResult,
    _split_attack_reach,
)
from strategies.base import StrategyDecision  # noqa: E402
from strategies.registry import (  # noqa: E402
    available_strategy_names,
    create_strategy,
)
from strategies.features import (  # noqa: E402
    can_eat_player_blob,
    player_speed,
    squared_distance,
)
from strategies.world_transition import (  # noqa: E402
    CompleteJointCommand,
    PlayerCommand,
)
from lib.config.player import MASS_DECAY_RATE, SAME_PLAYER_OVERLAP_EPSILON  # noqa: E402


def test_legacy_receding_horizon_names_resolve_without_duplicate_list_entries() -> None:
    assert create_strategy("champion").name == "threat_aware_receding_horizon"
    assert "champion" not in available_strategy_names()


def test_replay_dominance_is_a_distinct_registered_strategy() -> None:
    strategy = create_strategy("replay_dominance")

    assert isinstance(strategy, ReplayDominanceStrategy)
    assert strategy.name == "replay_dominance"
    assert "replay_dominance" in available_strategy_names()


def test_complete_joint_command_requires_every_live_player_once() -> None:
    command = CompleteJointCommand.build(
        live_player_ids={0, 7},
        commands={
            0: PlayerCommand((1.0, 0.0)),
            7: PlayerCommand((0.0, 1.0), split=True),
        },
    )

    assert command.for_player(7).split
    assert command.player_ids == frozenset({0, 7})

    try:
        CompleteJointCommand.build(
            live_player_ids={0, 7},
            commands={0: PlayerCommand((1.0, 0.0))},
        )
    except ValueError as error:
        assert "missing=(7,)" in str(error)
    else:
        raise AssertionError("incomplete joint command must be rejected")


def test_complete_joint_command_rejects_duplicate_player() -> None:
    try:
        CompleteJointCommand(
            (
                (0, PlayerCommand((1.0, 0.0))),
                (7, PlayerCommand((0.0, 1.0))),
                (7, PlayerCommand((0.0, -1.0))),
            )
        )
    except ValueError as error:
        assert "duplicate players" in str(error)
    else:
        raise AssertionError("duplicate player commands must be rejected")


def test_complete_joint_command_canonicalises_direct_construction() -> None:
    command = CompleteJointCommand(
        (
            (7, PlayerCommand((0.0, 1.0))),
            (0, PlayerCommand((1.0, 0.0))),
        )
    )

    assert tuple(player_id for player_id, _ in command.commands) == (0, 7)


def test_joint_enemy_split_uses_one_command_and_carries_ejection_two_steps() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    enemy = EnemyBlob(7, 3, 20.0, 20.0, 2.0, direction=(-1.0, 0.0))
    first_command = CompleteJointCommand.build(
        live_player_ids={0, 7},
        commands={
            0: PlayerCommand((1.0, 0.0)),
            7: PlayerCommand((0.0, 1.0), split=True),
        },
    )

    first = strategy._move_enemy_players_with_joint_command(
        (enemy,),
        first_command,
        60.0,
    )

    assert len(first) == 2
    assert {blob.direction for blob in first} == {(0.0, 1.0)}
    assert math.isclose(
        sum(blob.mass for blob in first),
        enemy.mass * (1.0 - MASS_DECAY_RATE),
    )
    assert first[1].eject_vy > 0.0

    second_command = CompleteJointCommand.build(
        live_player_ids={0, 7},
        commands={
            0: PlayerCommand((1.0, 0.0)),
            7: PlayerCommand((1.0, 0.0)),
        },
    )
    before_y = first[1].y
    second = strategy._move_enemy_players_with_joint_command(
        first,
        second_command,
        60.0,
    )

    assert second[1].direction == (1.0, 0.0)
    assert second[1].y > before_y
    assert 0.0 < second[1].eject_vy < first[1].eject_vy


def test_zero_direction_split_matches_engine_for_own_and_enemy_geometry() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    own = OwnBlob(blob_id=3, x=20.0, y=20.0, radius=2.0)
    modelled_own = sorted(
        strategy._apply_split([own], (0.0, 0.0), 60.0),
        key=lambda blob: blob.blob_id,
    )

    player = PlayerState(player_id=0, team_id=0)
    player.blobs = {
        own.blob_id: BlobState(
            blob_id=own.blob_id,
            x=own.x,
            y=own.y,
            radius=own.radius,
        )
    }
    player._next_blob_id = own.blob_id + 1
    state = SimpleNamespace(
        players={0: player},
        map=SimpleNamespace(size=60.0),
    )
    StateMutator(state)._apply_split(
        MovePlayer(
            player_id=0,
            direction={"x": 0.0, "y": 0.0},
            split=True,
        )
    )
    authoritative = player.sorted_blobs()

    assert len(modelled_own) == len(authoritative) == 2
    for modelled, expected in zip(modelled_own, authoritative, strict=True):
        assert math.isclose(modelled.x, expected.x, abs_tol=1e-12)
        assert math.isclose(modelled.y, expected.y, abs_tol=1e-12)
        assert math.isclose(modelled.radius, expected.radius, abs_tol=1e-12)
        assert modelled.merge_cooldown == expected.merge_cooldown
        assert modelled.eject_vx == expected.eject_vx == 0.0
        assert modelled.eject_vy == expected.eject_vy == 0.0

    enemy = EnemyBlob(
        player_id=7,
        blob_id=own.blob_id,
        x=own.x,
        y=own.y,
        radius=own.radius,
    )
    modelled_enemy = strategy._apply_enemy_split([enemy], (0.0, 0.0), 60.0)
    for modelled, expected in zip(modelled_enemy, authoritative, strict=True):
        assert math.isclose(modelled.x, expected.x, abs_tol=1e-12)
        assert math.isclose(modelled.y, expected.y, abs_tol=1e-12)
        assert math.isclose(modelled.radius, expected.radius, abs_tol=1e-12)
        assert modelled.merge_cooldown == expected.merge_cooldown
        assert modelled.eject_vx == expected.eject_vx == 0.0
        assert modelled.eject_vy == expected.eject_vy == 0.0


def test_joint_physical_step_does_not_invoke_policy_risk_analysis() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    node = SearchNode(
        own_blobs=(OwnBlob(0, 20.0, 20.0, 2.0),),
        enemies=(EnemyBlob(7, 0, 30.0, 20.0, 1.0),),
        score=17.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="probe",
        last_direction=(1.0, 0.0),
    )
    joint = CompleteJointCommand.build(
        live_player_ids={0, 7},
        commands={
            0: PlayerCommand((1.0, 0.0)),
            7: PlayerCommand((-1.0, 0.0)),
        },
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("joint physical kernel must not evaluate policy risk")

    strategy._risk_analysis = fail_if_called
    result = strategy._joint_physical_step(
        node=node,
        action=Action((1.0, 0.0)),
        foods=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
        joint_command=joint,
    )

    assert not result.dead
    assert result.state.score == node.score
    assert result.state.min_safety_margin == node.min_safety_margin


def test_zero_and_east_actions_have_distinct_physics_cache_keys() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)

    assert strategy._action_key(Action((0.0, 0.0))) != strategy._action_key(
        Action((1.0, 0.0))
    )
    assert strategy._action_key(Action((0.0, 0.0), split=True)) != strategy._action_key(
        Action((1.0, 0.0), split=True)
    )


def test_replay_dominance_compares_two_roots_before_deadline() -> None:
    strategy = ReplayDominanceStrategy()

    assert strategy.minimum_root_actions == 2
    assert strategy._required_actions_for_depth(0) == 2
    assert strategy._required_actions_for_depth(1) == 1


def test_replay_dominance_reserves_competition_time_for_runtime_overhead() -> None:
    strategy = ReplayDominanceStrategy()

    assert math.isclose(strategy.proxy_coarse_after_seconds, 5.6)


def test_replay_dominance_defaults_to_event_gated_one_step_enemy_motion() -> None:
    strategy = ReplayDominanceStrategy()

    assert strategy.proxy_enemy_motion == "event_gated"
    assert strategy.proxy_dynamic_horizon == 1


def test_replay_dominance_terminal_proxy_has_no_unreachable_opportunity() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    strategy._begin_replay_turn(
        SimpleNamespace(
            round=1399,
            max_rounds=1400,
            me=SimpleNamespace(player_id=0),
            rankings=[0],
        )
    )
    own = OwnBlob(blob_id=0, x=30.0, y=30.0, radius=2.0)
    node = SearchNode(
        own_blobs=(own,),
        enemies=(),
        score=0.0,
        first_direction=(0.0, 1.0),
        first_split=False,
        first_reason="keep",
        last_direction=(0.0, 1.0),
    )
    food = FoodModel(food_id=1, pos=(45.0, 30.0))
    analysis = strategy._proxy_analysis(
        node=node,
        foods=(food,),
        viruses=(),
        arena_size=60.0,
    )

    toward = strategy._approximate_action_value(
        node=node,
        action=Action((1.0, 0.0), reason="food"),
        foods=(food,),
        arena_size=60.0,
        proxy_analysis=analysis,
    )
    away = strategy._approximate_action_value(
        node=node,
        action=Action((-1.0, 0.0), reason="away"),
        foods=(food,),
        arena_size=60.0,
        proxy_analysis=analysis,
    )

    assert strategy._effective_proxy_horizon == 1
    assert strategy._proxy_has_future_after_horizon is False
    assert math.isclose(toward, away, abs_tol=1e-12)


def test_dynamic_threat_horizon_is_separate_from_static_resource_horizon(
    monkeypatch,
) -> None:
    own = OwnBlob(blob_id=0, x=30.0, y=30.0, radius=2.0)
    predator = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=42.0,
        y=30.0,
        radius=4.0,
        direction=(-1.0, 0.0),
    )
    threatened = SearchNode(
        own_blobs=(own,),
        enemies=(predator,),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )
    food = FoodModel(food_id=1, pos=(42.0, 30.0))
    safe = replace(threatened, enemies=())

    scores: dict[int, tuple[float, float]] = {}
    for dynamic_horizon in (2, 8):
        monkeypatch.setenv("BOT_REPLAY_DYNAMIC_HORIZON", str(dynamic_horizon))
        strategy = ReplayDominanceStrategy()
        strategy._rival_values = {1: 0.0}
        threat_analysis = strategy._proxy_analysis(
            node=threatened,
            foods=(),
            viruses=(),
            arena_size=60.0,
        )
        threat_score = strategy._approximate_action_value(
            node=threatened,
            action=Action((1.0, 0.0), reason="toward_threat"),
            foods=(),
            arena_size=60.0,
            proxy_analysis=threat_analysis,
        )
        resource_analysis = strategy._proxy_analysis(
            node=safe,
            foods=(food,),
            viruses=(),
            arena_size=60.0,
        )
        resource_score = strategy._approximate_action_value(
            node=safe,
            action=Action((1.0, 0.0), reason="toward_food"),
            foods=(food,),
            arena_size=60.0,
            proxy_analysis=resource_analysis,
        )
        scores[dynamic_horizon] = (threat_score, resource_score)

    assert scores[2][0] != scores[8][0]
    assert math.isclose(scores[2][1], scores[8][1], abs_tol=1e-12)


def test_stationary_enemy_ablation_freezes_enemy_but_not_own_motion(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BOT_REPLAY_ENEMY_MOTION", "stationary")
    strategy = ReplayDominanceStrategy()
    own = OwnBlob(blob_id=0, x=30.0, y=30.0, radius=2.0)
    predator = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=42.0,
        y=30.0,
        radius=4.0,
        direction=(-1.0, 0.0),
    )
    node = SearchNode(
        own_blobs=(own,),
        enemies=(predator,),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )
    strategy._rival_values = {1: 0.0}
    analysis = strategy._proxy_analysis(
        node=node,
        foods=(),
        viruses=(),
        arena_size=60.0,
    )
    movement = strategy._proxy_project_action(
        node=node,
        action=Action((1.0, 0.0), reason="toward_threat"),
        arena_size=60.0,
        horizon=1,
        unit=(1.0, 0.0),
        own_sources=analysis.own_sources,
    )
    enemy_motions = strategy._proxy_enemy_motions(
        analysis.motion_enemies,
        movement.blobs,
        horizon=1,
        arena_size=60.0,
        enemy_speeds=analysis.enemy_speeds,
        observed_directions=analysis.observed_enemy_directions,
        observed_weights=analysis.observed_enemy_weights,
        hunter_masks=analysis.normal_hunter_masks,
        predator_masks=analysis.normal_predator_masks,
    )

    assert movement.blobs[0].x > own.x
    assert analysis.enemy_speeds == (0.0,)
    assert enemy_motions[0].x == predator.x
    assert enemy_motions[0].y == predator.y


def test_event_gated_enemy_motion_only_moves_near_boundary_threats(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BOT_REPLAY_ENEMY_MOTION", "event_gated")
    strategy = ReplayDominanceStrategy()
    own = OwnBlob(blob_id=0, x=30.0, y=30.0, radius=2.0)
    close_predator = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=40.0,
        y=30.0,
        radius=4.0,
    )
    far_predator = EnemyBlob(
        player_id=2,
        blob_id=0,
        x=50.0,
        y=30.0,
        radius=4.0,
    )
    prey = EnemyBlob(
        player_id=3,
        blob_id=0,
        x=30.0,
        y=36.0,
        radius=1.0,
    )
    node = SearchNode(
        own_blobs=(own,),
        enemies=(close_predator, far_predator, prey),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )
    strategy._rival_values = {1: 0.0, 2: 0.0, 3: 0.0}

    analysis = strategy._proxy_analysis(
        node=node,
        foods=(),
        viruses=(),
        arena_size=60.0,
    )

    assert math.isclose(
        analysis.enemy_speed_by_key[close_predator.key],
        player_speed(close_predator.radius),
    )
    assert analysis.enemy_speed_by_key[far_predator.key] == 0.0
    assert analysis.enemy_speed_by_key[prey.key] == 0.0


def test_event_gate_includes_own_one_step_approach(monkeypatch) -> None:
    monkeypatch.setenv("BOT_REPLAY_ENEMY_MOTION", "event_gated")
    monkeypatch.setenv("BOT_REPLAY_DYNAMIC_HORIZON", "1")
    strategy = ReplayDominanceStrategy()
    own = OwnBlob(blob_id=0, x=30.0, y=30.0, radius=1.0)
    predator = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=32.31,
        y=30.0,
        radius=1.11,
    )
    node = SearchNode(
        own_blobs=(own,),
        enemies=(predator,),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )
    strategy._rival_values = {1: 0.0}

    analysis = strategy._proxy_analysis(
        node=node,
        foods=(),
        viruses=(),
        arena_size=60.0,
    )

    # Enemy speed alone is smaller than the initial 1.20 ordinary gap, but
    # our toward action closes the rest before the next observation.
    assert math.isclose(
        math.dist(own.pos, predator.pos) - predator.radius,
        1.2,
    )
    assert player_speed(predator.radius) < 1.2
    assert math.isclose(
        analysis.enemy_speed_by_key[predator.key],
        player_speed(predator.radius),
    )


def test_split_attack_envelope_does_not_add_parent_ordinary_speed() -> None:
    strategy = ReplayDominanceStrategy()
    enemy_radius = 4.0
    split_reach = _split_attack_reach(enemy_radius)
    enemy_speed = player_speed(enemy_radius)
    epsilon = 1e-6

    outside = strategy._exclusive_threat_margin(
        distance=split_reach + epsilon,
        ordinary_radius=enemy_radius,
        split_attack_radius=split_reach,
        ordinary_reach=enemy_speed,
    )
    inside = strategy._exclusive_threat_margin(
        distance=split_reach - epsilon,
        ordinary_radius=enemy_radius,
        split_attack_radius=split_reach,
        ordinary_reach=enemy_speed,
    )

    assert math.isclose(outside, epsilon, abs_tol=1e-12)
    assert math.isclose(inside, -epsilon, abs_tol=1e-12)


def test_event_gate_accounts_for_forward_split_child_motion(monkeypatch) -> None:
    monkeypatch.setenv("BOT_REPLAY_ENEMY_MOTION", "event_gated")
    monkeypatch.setenv("BOT_REPLAY_DYNAMIC_HORIZON", "1")
    strategy = ReplayDominanceStrategy()
    own = OwnBlob(blob_id=0, x=30.0, y=30.0, radius=1.8)
    predator = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=37.0,
        y=30.0,
        radius=1.41,
    )
    node = SearchNode(
        own_blobs=(own,),
        enemies=(predator,),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )
    strategy._rival_values = {1: 0.0}

    analysis = strategy._proxy_analysis(
        node=node,
        foods=(),
        viruses=(),
        arena_size=60.0,
    )
    threat = analysis.split_threats_by_blob[0][0]

    assert threat.normal_ordinary_radius is None
    assert threat.split_ordinary_radius == predator.radius
    assert math.isclose(
        analysis.enemy_speed_by_key[predator.key],
        player_speed(predator.radius),
    )


def test_threat_reachable_model_moves_predators_but_freezes_prey(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BOT_REPLAY_ENEMY_MOTION", "threat_reachable")
    monkeypatch.setenv("BOT_REPLAY_DYNAMIC_HORIZON", "1")
    strategy = ReplayDominanceStrategy()
    own = OwnBlob(blob_id=0, x=30.0, y=30.0, radius=2.0)
    predator = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=50.0,
        y=30.0,
        radius=4.0,
    )
    prey = EnemyBlob(
        player_id=2,
        blob_id=0,
        x=30.0,
        y=36.0,
        radius=1.0,
    )
    node = SearchNode(
        own_blobs=(own,),
        enemies=(predator, prey),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )
    strategy._rival_values = {1: 0.0, 2: 0.0}

    analysis = strategy._proxy_analysis(
        node=node,
        foods=(),
        viruses=(),
        arena_size=60.0,
    )

    assert math.isclose(
        analysis.enemy_speed_by_key[predator.key],
        player_speed(predator.radius),
    )
    assert analysis.enemy_speed_by_key[prey.key] == 0.0


def test_static_food_sweep_finds_shared_heading_covering_multiple_foods() -> None:
    strategy = ReplayDominanceStrategy()
    own = OwnBlob(blob_id=0, x=30.0, y=30.0, radius=1.0)
    node = SearchNode(
        own_blobs=(own,),
        enemies=(),
        score=0.0,
        first_direction=(0.0, 1.0),
        first_split=False,
        first_reason="keep",
        last_direction=(0.0, 1.0),
    )
    distance = 6.0
    foods = tuple(
        FoodModel(
            food_id=food_id,
            pos=(
                own.x + distance * math.cos(angle),
                own.y + distance * math.sin(angle),
            ),
        )
        for food_id, angle in enumerate((-0.1, 0.1))
    )

    directions = strategy._static_food_sweep_directions(
        node=node,
        foods=foods,
        horizon=8,
    )

    assert len(directions) == 1
    direction = directions[0]
    assert direction[0] > 0.99
    assert abs(direction[1]) < 0.07
    travel = 8 * player_speed(own.radius)
    for food in foods:
        assert (
            strategy._point_segment_distance(
                food.pos[0],
                food.pos[1],
                own.x,
                own.y,
                own.x + direction[0] * travel,
                own.y + direction[1] * travel,
            )
            <= own.radius
        )


def test_static_food_sweep_is_added_only_for_new_static_coverage(monkeypatch) -> None:
    monkeypatch.setenv("BOT_REPLAY_STATIC_FOOD_SWEEP", "1")
    strategy = ReplayDominanceStrategy()
    own = OwnBlob(blob_id=0, x=30.0, y=30.0, radius=0.9)
    node = SearchNode(
        own_blobs=(own,),
        enemies=(),
        score=0.0,
        first_direction=(0.0, 1.0),
        first_split=False,
        first_reason="keep",
        last_direction=(0.0, 1.0),
    )
    distance = 7.5
    foods = tuple(
        FoodModel(
            food_id=food_id,
            pos=(
                own.x + distance * math.cos(angle),
                own.y + distance * math.sin(angle),
            ),
        )
        for food_id, angle in enumerate((0.2, 0.4))
    )

    actions = strategy._candidate_actions(
        node=node,
        foods=foods,
        food_targets=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
        allow_split=False,
    )

    assert any(action.reason == "food_sweep" for action in actions)
    assert strategy._static_food_sweep_added is True
    assert strategy._static_food_sweep_predicted_advantage == 1

    aligned_foods = tuple(
        FoodModel(food_id=food.food_id, pos=(food.pos[0], own.y)) for food in foods
    )
    strategy._static_food_sweep_added = False
    aligned_actions = strategy._candidate_actions(
        node=node,
        foods=aligned_foods,
        food_targets=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
        allow_split=False,
    )

    assert all(action.reason != "food_sweep" for action in aligned_actions)
    assert strategy._static_food_sweep_added is False
    assert strategy._static_food_sweep_predicted_advantage == 0


def test_predator_risk_uses_reachable_region_not_observed_straight_line() -> None:
    strategy = ReplayDominanceStrategy()
    own = OwnBlob(blob_id=0, x=30.0, y=30.0, radius=2.0)
    base_predator = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=42.0,
        y=30.0,
        radius=4.0,
    )
    scores = []
    for observed_direction in ((0.0, 1.0), (0.0, -1.0)):
        node = SearchNode(
            own_blobs=(own,),
            enemies=(replace(base_predator, direction=observed_direction),),
            score=0.0,
            first_direction=(1.0, 0.0),
            first_split=False,
            first_reason="keep",
            last_direction=(1.0, 0.0),
        )
        strategy._rival_values = {1: 0.0}
        analysis = strategy._proxy_analysis(
            node=node,
            foods=(),
            viruses=(),
            arena_size=60.0,
        )
        scores.append(
            strategy._approximate_action_value(
                node=node,
                action=Action((1.0, 0.0), reason="toward_threat"),
                foods=(),
                arena_size=60.0,
                proxy_analysis=analysis,
            )
        )

    assert math.isclose(scores[0], scores[1], abs_tol=1e-12)


def test_coherent_enemy_scenario_cannot_chase_separated_fragments_at_once() -> None:
    strategy = ReplayDominanceStrategy()
    predator = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=50.0,
        y=50.0,
        radius=4.0,
    )
    blobs = [
        ProxyBlobMotion(
            blob_id=blob_id,
            source_blob_id=blob_id,
            start_x=x,
            start_y=50.0,
            x=x,
            y=50.0,
            radius=2.0,
            speed=0.0,
        )
        for blob_id, x in enumerate((35.0, 65.0))
    ]
    threats = tuple(
        (
            ProxyThreat(
                source_blob_id=blob.blob_id,
                enemy=predator,
                source_radius=blob.radius,
                normal_ordinary_radius=predator.radius,
                normal_split_attack_radius=None,
                split_ordinary_radius=predator.radius,
                split_split_attack_radius=None,
                away_x=0.0,
                away_y=0.0,
                initial_margin=11.0,
                motion_index=0,
            ),
        )
        for blob in blobs
    )

    survivals = strategy._coherent_reachable_survivals(
        blobs=blobs,
        threats_by_blob=threats,
        enemy_speeds=(8.0,),
        arena_size=100.0,
        dynamic_fraction=1.0,
        dynamic_steps=1,
    )

    # The selected scenario pursues one side. The same predator therefore
    # cannot also occupy the opposite edge of its reachable disk that round.
    assert min(survivals) < 0.2
    assert max(survivals) > 0.6


def test_experimental_cohesion_prefers_motion_that_reforms_the_group(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BOT_REPLAY_COHESION", "1")
    strategy = ReplayDominanceStrategy()
    node = SearchNode(
        own_blobs=(
            OwnBlob(blob_id=0, x=30.0, y=30.0, radius=3.0),
            OwnBlob(blob_id=1, x=20.0, y=30.0, radius=1.0),
        ),
        enemies=(),
        score=0.0,
        first_direction=(0.0, 1.0),
        first_split=False,
        first_reason="keep",
        last_direction=(0.0, 1.0),
    )
    analysis = strategy._proxy_analysis(
        node=node,
        foods=(),
        viruses=(),
        arena_size=60.0,
    )

    reform = strategy._approximate_action_value(
        node=node,
        action=Action((1.0, 0.0), reason="reform"),
        foods=(),
        arena_size=60.0,
        proxy_analysis=analysis,
    )
    disperse = strategy._approximate_action_value(
        node=node,
        action=Action((-1.0, 0.0), reason="disperse"),
        foods=(),
        arena_size=60.0,
        proxy_analysis=analysis,
    )

    assert reform > disperse


def test_replay_dominance_keeps_exact_search_as_an_offline_oracle() -> None:
    strategy = ReplayDominanceStrategy()

    assert strategy._uses_compute_time_bank() is True
    assert strategy.depth == 1
    assert strategy.width == 1
    assert strategy.compute_budget_seconds == 0.0
    assert strategy.max_turn_seconds == 0.003
    assert strategy.exact_min_blobs == 1
    assert strategy._transition_budget(1) == 6
    assert strategy._transition_budget(2) == 3
    assert strategy._transition_budget(4) == 2
    assert strategy._transition_budget(16) == 2
    assert strategy._transition_budget(1, 12) == 2
    assert strategy._actions_per_node_limit(0) == 6
    assert strategy._actions_per_node_limit(1) == 1


def test_replay_dominance_exact_action_width_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("BOT_REPLAY_EXACT_ROOT_ACTION_LIMIT", "4")
    monkeypatch.setenv("BOT_REPLAY_EXACT_DEEPER_ACTION_LIMIT", "2")
    monkeypatch.setenv("BOT_REPLAY_EXACT_MAX_BLOBS", "3")
    monkeypatch.setenv("BOT_REPLAY_EXACT_MIN_BLOBS", "2")

    strategy = ReplayDominanceStrategy()

    assert strategy._actions_per_node_limit(0) == 4
    assert strategy._actions_per_node_limit(1) == 2
    assert strategy.exact_max_blobs == 3
    assert strategy.exact_min_blobs == 2


def test_replay_dominance_preserves_explicit_depth_and_width_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BOT_RECEDING_HORIZON_DEPTH", "4")
    monkeypatch.setenv("BOT_RECEDING_HORIZON_WIDTH", "3")

    strategy = ReplayDominanceStrategy()

    assert strategy.depth == 4
    assert strategy.width == 3


def test_replay_dominance_stops_before_generating_an_unusable_depth() -> None:
    assert (
        ReplayDominanceStrategy._depth_start_stop_reason(
            depth_index=1,
            transitions_evaluated=6,
            transition_budget=6,
            uses_time_bank=True,
            deadline=float("inf"),
        )
        == "transition_budget"
    )
    assert (
        ReplayDominanceStrategy._depth_start_stop_reason(
            depth_index=1,
            transitions_evaluated=5,
            transition_budget=6,
            uses_time_bank=True,
            deadline=0.0,
        )
        == "deadline"
    )


def test_exposed_radii_include_future_virus_fragments() -> None:
    strategy = ReplayDominanceStrategy()
    own = OwnBlob(blob_id=0, x=30.0, y=30.0, radius=math.sqrt(32.0))
    virus = VirusModel(virus_id=1, pos=(34.0, 30.0), radius=1.5)
    without_virus = [radius for _, radius in strategy._exposed_own_radii((own,))]
    radii = [radius for _, radius in strategy._exposed_own_radii((own,), (virus,))]

    assert own.radius in radii
    assert own.radius / math.sqrt(2.0) in radii
    assert math.sqrt((own.mass + 1.5**2) / 16.0) in radii
    assert any(radius < 1.5 for radius in radii)
    assert all(radius >= own.radius / math.sqrt(2.0) for radius in without_virus)


def test_enemy_memory_predicts_every_track_once_before_authoritative_overwrite(
    monkeypatch,
) -> None:
    strategy = ReplayDominanceStrategy()
    strategy.enemy_tracks[(1, 0)] = EnemyTrack(
        player_id=1,
        blob_id=0,
        x=10.0,
        y=10.0,
        radius=2.0,
        direction=(1.0, 0.0),
        last_seen_round=4,
    )
    visible = SimpleNamespace(
        player_id=1,
        blob_id=0,
        pos=(12.0, 10.0),
        radius=2.0,
        merge_cooldown=0,
    )
    context = SimpleNamespace(
        game=SimpleNamespace(
            state=SimpleNamespace(
                round=5,
                visible_blobs=(visible,),
                view_center=(30.0, 30.0),
                vision_size=60.0,
            )
        )
    )

    calls = 0
    original_speed = player_speed

    def counted_speed(radius):
        nonlocal calls
        calls += 1
        return original_speed(radius)

    monkeypatch.setattr(
        "strategies.receding_horizon.player_speed",
        counted_speed,
    )
    enemies = strategy._update_enemy_memory(
        context,
        (OwnBlob(blob_id=0, x=30.0, y=30.0, radius=1.0),),
        60.0,
    )

    assert calls == 1
    assert enemies[0].pos == visible.pos


def test_replay_dominance_planning_uses_only_the_current_anonymous_view() -> None:
    strategy = ReplayDominanceStrategy()
    strategy.enemy_tracks[(1, 42)] = EnemyTrack(
        player_id=1,
        blob_id=42,
        x=10.0,
        y=10.0,
        radius=4.0,
        direction=(1.0, 0.0),
        last_seen_round=4,
    )
    visible = SimpleNamespace(
        player_id=2,
        blob_id=99,
        pos=(20.0, 30.0),
        radius=2.0,
        merge_cooldown=3,
    )
    context = SimpleNamespace(
        game=SimpleNamespace(
            state=SimpleNamespace(visible_blobs=(visible,)),
        )
    )

    enemies = strategy._planning_enemies(
        context,
        (OwnBlob(blob_id=0, x=30.0, y=30.0, radius=1.0),),
        60.0,
        (),
    )

    assert enemies == (
        EnemyBlob(
            player_id=2,
            blob_id=0,
            x=20.0,
            y=30.0,
            radius=2.0,
            merge_cooldown=3,
        ),
    )
    assert set(strategy.enemy_tracks) == {(1, 42)}


def test_replay_dominance_does_not_reorder_by_semantic_family() -> None:
    strategy = ReplayDominanceStrategy()
    ordered = strategy._order_root_actions(
        (
            Action((1.0, 0.0), reason="escape"),
            Action((0.9, 0.1), reason="escape_tangent"),
            Action((0.0, 1.0), reason="rival_prey"),
            Action((-1.0, 0.0), reason="virus_harvest"),
            Action((0.0, -1.0), reason="keep"),
        )
    )

    assert [action.reason for action in ordered] == [
        "escape",
        "escape_tangent",
        "rival_prey",
        "virus_harvest",
        "keep",
    ]
    assert strategy._actions_per_node_limit(0) == 6
    assert strategy._actions_per_node_limit(1) == 1


def test_replay_dominance_second_root_avoids_official_virus_massacre() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=4, angular_samples=10)
    strategy._rival_values = {0: 1.0}
    own = OwnBlob(blob_id=125, x=53.87, y=11.25, radius=5.8)
    predator_after_pop = EnemyBlob(
        player_id=0,
        blob_id=93,
        x=39.0,
        y=17.7,
        radius=5.45,
        direction=(-1.0, 0.0),
    )
    virus = VirusModel(
        virus_id=31,
        pos=(50.2472306260258, 16.532899639036557),
        radius=1.5,
    )
    node = SearchNode(
        own_blobs=(own,),
        enemies=(predator_after_pop,),
        score=0.0,
        first_direction=(0.0, 1.0),
        first_split=False,
        first_reason="keep",
        last_direction=(0.0, 1.0),
    )
    actions = strategy._candidate_actions(
        node=node,
        foods=(),
        food_targets=(),
        viruses=(virus,),
        arena_size=60.0,
        first_step=True,
        angle_offset=609,
    )
    results = [
        strategy._step(
            node=node,
            action=action,
            foods=(),
            viruses=(virus,),
            arena_size=60.0,
            first_step=True,
            safety_weight=2.0,
            aggression=1.0,
        ).node
        for action in actions[: strategy.minimum_root_actions]
    ]

    assert (
        "virus"
        not in max(
            results,
            key=strategy._terminal_score,
        ).first_reason
    )


def test_replay_dominance_continues_safe_virus_chain_when_late_and_fragmented() -> None:
    strategy = ReplayDominanceStrategy()
    own_blobs = (
        OwnBlob(blob_id=0, x=10.0, y=10.0, radius=2.0),
        *(
            OwnBlob(
                blob_id=index,
                x=20.0 + index * 0.1,
                y=20.0,
                radius=0.7,
            )
            for index in range(1, 12)
        ),
    )
    virus = VirusModel(virus_id=7, pos=(12.0, 10.0), radius=1.5)

    node = SearchNode(
        own_blobs=own_blobs,
        enemies=(),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )

    actions = strategy._candidate_actions(
        node=node,
        foods=(),
        food_targets=(),
        viruses=(virus,),
        arena_size=60.0,
        first_step=True,
    )

    assert any(action.reason == "virus_harvest" for action in actions)


def test_replay_dominance_evaluates_scoreboard_rival_before_ordinary_actions() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    strategy._rival_values = {1: 1.0}
    own = OwnBlob(blob_id=0, x=10.0, y=10.0, radius=2.0)
    rival = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=13.0,
        y=10.0,
        radius=1.0,
    )
    node = SearchNode(
        own_blobs=(own,),
        enemies=(rival,),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )

    actions = strategy._candidate_actions(
        node=node,
        foods=(),
        food_targets=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
    )

    assert any(action.reason in {"prey", "split_prey"} for action in actions[:3])


def test_replay_dominance_keeps_safe_split_candidates_in_late_lead() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    strategy._rival_values = {1: 1.0}
    own = OwnBlob(blob_id=0, x=10.0, y=10.0, radius=3.0)
    rival = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=14.0,
        y=10.0,
        radius=1.0,
    )
    node = SearchNode(
        own_blobs=(own,),
        enemies=(rival,),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )

    actions = strategy._candidate_actions(
        node=node,
        foods=(),
        food_targets=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
    )

    assert any(action.reason == "split_prey" for action in actions)


def test_replay_semantic_split_does_not_cross_product_unrelated_actions(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BOT_REPLAY_SEMANTIC_SPLIT", "1")
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=8)
    node = SearchNode(
        own_blobs=(OwnBlob(blob_id=0, x=30.0, y=30.0, radius=3.0),),
        enemies=(),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )

    actions = strategy._candidate_actions(
        node=node,
        foods=(),
        food_targets=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
    )

    assert not any(action.split for action in actions)
    assert not any(action.reason == "split_keep" for action in actions)


def test_replay_semantic_split_keeps_reachable_prey_event(monkeypatch) -> None:
    monkeypatch.setenv("BOT_REPLAY_SEMANTIC_SPLIT", "1")
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=8)
    own = OwnBlob(blob_id=0, x=30.0, y=30.0, radius=3.0)
    prey = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=38.0,
        y=30.0,
        radius=1.0,
    )
    node = SearchNode(
        own_blobs=(own,),
        enemies=(prey,),
        score=0.0,
        first_direction=(0.0, 1.0),
        first_split=False,
        first_reason="keep",
        last_direction=(0.0, 1.0),
    )

    actions = strategy._candidate_actions(
        node=node,
        foods=(),
        food_targets=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
    )

    split_reasons = {action.reason for action in actions if action.split}
    assert split_reasons == {"split_prey"}


def test_replay_semantic_split_restores_mass_improving_escape(monkeypatch) -> None:
    monkeypatch.setenv("BOT_REPLAY_SEMANTIC_SPLIT", "1")
    monkeypatch.setenv("BOT_REPLAY_DYNAMIC_HORIZON", "1")
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=8)
    node = SearchNode(
        own_blobs=(OwnBlob(blob_id=0, x=30.0, y=30.0, radius=1.5),),
        enemies=(
            EnemyBlob(
                player_id=1,
                blob_id=0,
                x=28.0,
                y=30.0,
                radius=3.0,
            ),
        ),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )

    actions = strategy._candidate_actions(
        node=node,
        foods=(),
        food_targets=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
    )

    recovery = tuple(action for action in actions if action.reason == "split_escape")
    assert len(recovery) == 1
    assert recovery[0].split
    assert recovery[0].direction[0] > 0.9


def test_replay_semantic_split_omits_escape_when_no_split_retains_mass(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BOT_REPLAY_SEMANTIC_SPLIT", "1")
    monkeypatch.setenv("BOT_REPLAY_DYNAMIC_HORIZON", "1")
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=8)
    node = SearchNode(
        own_blobs=(OwnBlob(blob_id=0, x=30.0, y=30.0, radius=2.0),),
        enemies=(
            EnemyBlob(
                player_id=1,
                blob_id=0,
                x=24.0,
                y=30.0,
                radius=3.0,
            ),
        ),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )

    actions = strategy._candidate_actions(
        node=node,
        foods=(),
        food_targets=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
    )

    assert not any(action.reason == "split_escape" for action in actions)


def test_replay_family_refine_preserves_active_roles_with_bounded_width() -> None:
    strategy = ReplayDominanceStrategy()
    coarse_scored = tuple(
        (
            Action(
                (math.cos(index * math.pi / 6), math.sin(index * math.pi / 6)),
                split=reason.startswith("split_"),
                reason=reason,
            ),
            20.0 - index,
        )
        for index, reason in enumerate(
            (
                "angle",
                "steer",
                "nearest_food",
                "food_cluster",
                "escape",
                "escape_tangent",
                "prey",
                "split_prey",
                "virus_harvest",
                "center",
                "keep",
            )
        )
    )

    refined, remainder = strategy._family_refine_partition(
        coarse_scored,
        limit=7,
    )

    refined_families = {strategy._action_family(action) for action in refined}
    assert {"baseline", "escape", "prey_split"} <= refined_families
    assert len(refined) == 7
    assert sum(not action.split for action in refined) >= 2
    assert len(refined) + len(remainder) == len(coarse_scored)


def test_replay_family_refine_keeps_orthogonal_escape_when_pressures_cancel(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BOT_REPLAY_SEMANTIC_SPLIT", "1")
    monkeypatch.setenv("BOT_REPLAY_FAMILY_REFINE", "1")
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=8)
    node = SearchNode(
        own_blobs=(OwnBlob(blob_id=0, x=30.0, y=30.0, radius=2.0),),
        enemies=(
            EnemyBlob(player_id=1, blob_id=0, x=20.0, y=30.0, radius=3.0),
            EnemyBlob(player_id=2, blob_id=0, x=40.0, y=30.0, radius=3.0),
        ),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )

    actions = strategy._candidate_actions(
        node=node,
        foods=(),
        food_targets=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
    )

    assert any(abs(action.direction[0]) < 0.1 for action in actions[:7])
    assert abs(actions[0].direction[0]) < 0.1
    assert len(strategy._root_proxy_scores) == strategy._root_proxy_refined


def test_replay_semantic_split_checks_capture_feasibility_beyond_normal_prey_limit(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BOT_REPLAY_SEMANTIC_SPLIT", "1")
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=8)
    own = OwnBlob(blob_id=0, x=30.0, y=30.0, radius=4.0)
    enemies = (
        EnemyBlob(player_id=1, blob_id=0, x=35.0, y=30.0, radius=2.6),
        EnemyBlob(player_id=2, blob_id=0, x=30.0, y=35.0, radius=2.6),
        EnemyBlob(player_id=3, blob_id=0, x=25.0, y=30.0, radius=2.6),
        EnemyBlob(player_id=4, blob_id=0, x=30.0, y=38.0, radius=2.5),
    )
    node = SearchNode(
        own_blobs=(own,),
        enemies=enemies,
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )

    analysis = strategy._proxy_analysis(
        node=node,
        foods=(),
        viruses=(),
        arena_size=60.0,
    )
    assert len(analysis.prey) == 4
    assert all(
        not strategy._split_can_capture(
            node,
            target.enemy,
            strategy._intercept_direction(own, target.enemy),
        )
        for target in analysis.prey[:3]
    )
    assert strategy._split_can_capture(
        node,
        analysis.prey[3].enemy,
        strategy._intercept_direction(own, analysis.prey[3].enemy),
    )

    actions = strategy._candidate_actions(
        node=node,
        foods=(),
        food_targets=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
    )

    assert any(action.reason == "split_prey" for action in actions)


def test_replay_action_family_separates_split_events_from_directions() -> None:
    strategy = ReplayDominanceStrategy()

    assert (
        strategy._action_family(Action((1.0, 0.0), split=True, reason="split_keep"))
        == "split_event"
    )
    assert (
        strategy._action_family(Action((1.0, 0.0), split=True, reason="split_center"))
        == "split_event"
    )
    assert (
        strategy._action_family(Action((1.0, 0.0), split=True, reason="split_prey"))
        == "prey_split"
    )
    assert (
        strategy._action_family(Action((1.0, 0.0), split=True, reason="split_farm"))
        == "resource_split"
    )


def test_replay_dominance_scores_wall_clamp_by_actual_movement() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    own = OwnBlob(blob_id=0, x=2.0, y=10.0, radius=2.0)

    blocked = strategy._move_own_blobs((own,), (-1.0, 0.0), 60.0)
    unblocked = strategy._move_own_blobs((own,), (1.0, 0.0), 60.0)

    assert blocked.efficiency == 0.0
    assert unblocked.efficiency == 1.0


def test_replay_step_moves_each_own_blob_once(monkeypatch) -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    node = SearchNode(
        own_blobs=(OwnBlob(blob_id=0, x=20.0, y=20.0, radius=2.0),),
        enemies=(),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )
    original = strategy._move_own
    calls = 0

    def counted_move(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(strategy, "_move_own", counted_move)
    strategy._step(
        node=node,
        action=Action((1.0, 0.0), reason="keep"),
        foods=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
        safety_weight=1.3,
        aggression=1.0,
    )

    assert calls == len(node.own_blobs)


def test_replay_reuses_each_transition_hazard_for_child_utility(
    monkeypatch,
) -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=8)
    own = SimpleNamespace(
        blob_id=0,
        pos=(20.0, 20.0),
        radius=1.0,
        merge_cooldown=0,
    )
    state = SimpleNamespace(
        round=100,
        max_rounds=1400,
        rankings=[0, 1, 2, 3, 4, 5, 6, 7],
        me=SimpleNamespace(player_id=0, blobs={0: own}),
        map=SimpleNamespace(size=60.0),
        visible_blobs=(),
        visible_food=(FoodModel(food_id=1, pos=(22.0, 20.0)),),
        visible_viruses=(),
        view_center=(20.0, 20.0),
        vision_size=40.0,
    )
    context = SimpleNamespace(
        game=SimpleNamespace(state=state),
        query=SimpleNamespace(update={}),
    )
    original = strategy._hazard_summary
    calls = 0

    def counted_hazard(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(strategy, "_hazard_summary", counted_hazard)
    decision = strategy._choose(context, deadline=float("inf"), turn_budget=1.0)

    # The parent value needs one hazard summary. Each exact transition then
    # computes its child summary once and passes it into child utility.
    assert calls == 1 + decision.diagnostics["transitions_evaluated"]


def test_replay_candidate_analysis_builds_one_shared_proxy() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    strategy._profile_active = True
    strategy._rival_values = {1: 1.0}
    own = OwnBlob(blob_id=0, x=20.0, y=20.0, radius=3.0)
    prey = EnemyBlob(player_id=1, blob_id=0, x=24.0, y=20.0, radius=1.0)
    virus = VirusModel(virus_id=7, pos=(22.0, 20.0), radius=1.5)
    node = SearchNode(
        own_blobs=(own,),
        enemies=(prey,),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )

    foods = ()
    viruses = (virus,)
    strategy._candidate_actions(
        node=node,
        foods=foods,
        food_targets=(),
        viruses=viruses,
        arena_size=60.0,
        first_step=True,
    )

    counts = strategy._profile_counts
    assert counts["proxy_analysis_nodes"] == 1
    assert counts["proxy_virus_targets"] == 1
    assert counts["proxy_prey_targets"] == 1
    assert counts["virus_consumability_calls"] == 1


def test_proxy_projection_caches_each_segment_delta_once() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    node = SearchNode(
        own_blobs=(OwnBlob(blob_id=0, x=20.0, y=20.0, radius=3.0),),
        enemies=(),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )

    movement = strategy._proxy_project_action(
        node=node,
        action=Action((0.6, 0.8), split=True, reason="split_probe"),
        arena_size=60.0,
        horizon=8,
    )

    assert len(movement.blobs) == 2
    for blob in movement.blobs:
        assert blob.delta_x == blob.x - blob.start_x
        assert blob.delta_y == blob.y - blob.start_y
        assert blob.length_sq == blob.delta_x**2 + blob.delta_y**2


def test_proxy_short_path_matches_one_round_physics_for_all_motion_modes() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    cases = (
        (
            OwnBlob(blob_id=0, x=20.0, y=20.0, radius=3.0),
            Action((0.6, 0.8), split=False, reason="normal"),
        ),
        (
            OwnBlob(blob_id=0, x=30.0, y=30.0, radius=1.8),
            Action((1.0, 0.0), split=True, reason="split"),
        ),
        (
            OwnBlob(
                blob_id=0,
                x=20.0,
                y=20.0,
                radius=3.0,
                eject_vx=0.8,
                eject_vy=-0.3,
            ),
            Action((0.0, 1.0), split=False, reason="inherited_eject"),
        ),
        (
            OwnBlob(blob_id=0, x=57.8, y=30.0, radius=2.0),
            Action((1.0, 0.0), split=False, reason="wall"),
        ),
    )

    for own, action in cases:
        unit = action.direction
        node = SearchNode(
            own_blobs=(own,),
            enemies=(),
            score=0.0,
            first_direction=unit,
            first_split=action.split,
            first_reason=action.reason,
            last_direction=unit,
        )
        projected = strategy._proxy_project_action(
            node=node,
            action=action,
            arena_size=60.0,
            horizon=8,
        )
        initial = strategy._apply_split([own], unit, 60.0) if action.split else [own]
        expected = {
            blob.blob_id: strategy._move_own(blob, unit, 60.0) for blob in initial
        }

        assert len(projected.blobs) == len(expected)
        for blob in projected.blobs:
            one_round = expected[blob.blob_id]
            assert len(blob.short_path) == 2
            assert math.isclose(blob.short_path[-1][0], one_round.x)
            assert math.isclose(blob.short_path[-1][1], one_round.y)
            assert strategy._proxy_short_position(blob) == blob.short_path[-1]


def test_disabled_replay_profile_does_not_read_inner_timers(monkeypatch) -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    strategy._profile_every_n = 0
    strategy._profile_active = False
    node = SearchNode(
        own_blobs=(OwnBlob(blob_id=0, x=20.0, y=20.0, radius=2.0),),
        enemies=(),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )

    def unexpected_timer():
        raise AssertionError("disabled inner profiling must not read the clock")

    monkeypatch.setattr(
        "strategies.receding_horizon.perf_counter",
        unexpected_timer,
    )
    strategy._candidate_actions(
        node=node,
        foods=(),
        food_targets=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
    )
    strategy._step(
        node=node,
        action=Action((1.0, 0.0), reason="keep"),
        foods=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
        safety_weight=1.3,
        aggression=1.0,
    )


def test_replay_dominance_prices_prey_by_clamped_closing_speed() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    wall_blocked_own = OwnBlob(blob_id=0, x=2.0, y=20.0, radius=2.0)
    wall_fleeing = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=1.0,
        y=30.0,
        radius=1.0,
        direction=(0.0, 1.0),
    )
    corner_own = OwnBlob(blob_id=1, x=2.0, y=10.0, radius=2.0)
    corner_pinned = replace(
        wall_fleeing,
        blob_id=1,
        y=1.0,
        direction=(0.0, -1.0),
    )

    unreachable = strategy._prey_capture_probability(
        wall_blocked_own, wall_fleeing, 60.0
    )
    reachable = strategy._prey_capture_probability(corner_own, corner_pinned, 60.0)

    assert unreachable < 1e-10
    assert 0.3 < reachable < 0.5


def test_replay_dominance_state_value_rewards_absolute_mass_linearly() -> None:
    strategy = ReplayDominanceStrategy()

    def utility(mass: float) -> float:
        node = SearchNode(
            own_blobs=(OwnBlob(blob_id=0, x=30.0, y=30.0, radius=math.sqrt(mass)),),
            enemies=(),
            score=0.0,
            first_direction=(1.0, 0.0),
            first_split=False,
            first_reason="keep",
            last_direction=(1.0, 0.0),
        )
        return strategy._search_utility(
            node,
            foods=(),
            viruses=(),
            arena_size=60.0,
            safety_weight=1.0,
        )

    assert math.isclose(utility(15.0) - utility(5.0), 1000.0)
    assert math.isclose(utility(60.0) - utility(50.0), 1000.0)


def test_replay_dominance_rival_capture_value_is_continuous_at_contact() -> None:
    strategy = ReplayDominanceStrategy()
    strategy._rival_values = {1: 0.5}
    own = OwnBlob(blob_id=0, x=20.0, y=30.0, radius=2.0)
    rival = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=22.0,
        y=30.0,
        radius=1.0,
    )

    before = SearchNode(
        own_blobs=(own,),
        enemies=(rival,),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="rival_prey",
        last_direction=(1.0, 0.0),
    )
    after = replace(
        before,
        own_blobs=(replace(own, radius=math.sqrt(own.mass + rival.mass)),),
        enemies=(),
    )

    before_value = strategy._search_utility(
        before,
        foods=(),
        viruses=(),
        arena_size=60.0,
        safety_weight=1.0,
    )
    after_value = strategy._search_utility(
        after,
        foods=(),
        viruses=(),
        arena_size=60.0,
        safety_weight=1.0,
    )

    assert math.isclose(before_value, after_value, rel_tol=1e-9)


def test_replay_dominance_merges_before_virus_like_engine_failure_replay() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    own_blobs = [
        OwnBlob(blob_id=0, x=10.0, y=10.0, radius=5.06),
        *(
            OwnBlob(
                blob_id=index,
                x=8.0 + (index % 4),
                y=8.0 + (index // 4),
                radius=1.4,
            )
            for index in range(1, 16)
        ),
    ]
    virus = VirusModel(virus_id=50, pos=(10.0, 10.0), radius=1.5)
    consumed: set[int] = set()
    expected_mass = sum(blob.mass for blob in own_blobs) + virus.radius**2
    stabilised = strategy._stabilise_own_blobs(own_blobs, 60.0)

    after, _, _, own_consumed = strategy._resolve_own_viruses(
        own_blobs=stabilised,
        viruses=(virus,),
        consumed_virus_ids=consumed,
        arena_size=60.0,
    )

    assert consumed == {50}
    assert own_consumed == 1
    assert len(stabilised) == 1
    assert len(after) == 16
    assert all(
        math.isclose(blob.radius, math.sqrt(expected_mass / 16), rel_tol=1e-9)
        for blob in after
    )
    assert math.isclose(sum(blob.mass for blob in after), expected_mass)


def test_replay_dominance_prices_virus_fragment_survival_without_banning_it() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    virus = VirusModel(virus_id=7, pos=(5.0, 30.0), radius=1.5)
    own = OwnBlob(blob_id=0, x=4.0, y=30.0, radius=2.0)
    predator = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=14.0,
        y=30.0,
        radius=3.0,
    )
    trapped = SearchNode(
        own_blobs=(own,),
        enemies=(predator,),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )
    open_own = OwnBlob(blob_id=0, x=30.0, y=30.0, radius=2.0)
    open_virus = VirusModel(virus_id=7, pos=(31.0, 30.0), radius=1.5)
    open_predator = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=40.0,
        y=30.0,
        radius=3.0,
    )
    open_space = replace(
        trapped,
        own_blobs=(open_own,),
        enemies=(open_predator,),
    )

    trapped_retention = strategy._virus_retained_mass_fraction(
        trapped, own, virus, 60.0
    )
    open_retention = strategy._virus_retained_mass_fraction(
        open_space, open_own, open_virus, 60.0
    )
    actions = strategy._candidate_actions(
        node=trapped,
        foods=(),
        food_targets=(),
        viruses=(virus,),
        arena_size=60.0,
        first_step=True,
    )

    assert 0.0 < trapped_retention < open_retention < 1.0
    assert any(action.reason == "virus_harvest" for action in actions)


def test_replay_dominance_penalises_wall_only_when_predator_blocks_retreat() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    edge_blob = OwnBlob(blob_id=0, x=12.0, y=4.0, radius=2.0)
    center_blob = OwnBlob(blob_id=0, x=12.0, y=15.0, radius=2.0)
    predator = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=18.0,
        y=5.0,
        radius=3.0,
    )

    def utility(own: OwnBlob, enemies: tuple[EnemyBlob, ...]) -> float:
        node = SearchNode(
            own_blobs=(own,),
            enemies=enemies,
            score=0.0,
            first_direction=(1.0, 0.0),
            first_split=False,
            first_reason="keep",
            last_direction=(1.0, 0.0),
        )
        return strategy._search_utility(
            node,
            foods=(),
            viruses=(),
            arena_size=60.0,
            safety_weight=1.0,
        )

    safe_edge = utility(edge_blob, ())
    trapped_edge = utility(edge_blob, (predator,))
    open_center = utility(center_blob, (predator,))

    assert trapped_edge < open_center
    assert trapped_edge < safe_edge


def test_replay_proxy_prices_wall_continuation_before_exact_search() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    strategy.proxy_horizon = 8
    strategy.proxy_refine_limit = 64
    own = OwnBlob(blob_id=0, x=55.0, y=55.0, radius=1.0)
    predator = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=51.0,
        y=51.0,
        radius=2.0,
    )
    node = SearchNode(
        own_blobs=(own,),
        enemies=(predator,),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )

    actions = strategy._candidate_actions(
        node=node,
        foods=(),
        food_targets=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
    )
    reasons = [action.reason for action in actions]

    # The long proxy may prefer moving inward behind the predator over a
    # literal escape vector; both coordinates must stop pushing into the wall.
    assert actions[0].direction[0] <= 0.0
    assert actions[0].direction[1] <= 0.0
    assert "escape_wide_tangent" in reasons[:6]
    assert strategy._safety_weight(rank_position=7, progress=0.0) == 1.3


def test_replay_coarse_rank_projects_each_fragment_when_threatened() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    own_blobs = tuple(
        OwnBlob(blob_id=index, x=48.0, y=23.0 + index * 2.0, radius=1.0)
        for index in range(8)
    )
    predator = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=53.0,
        y=30.0,
        radius=2.5,
    )
    node = SearchNode(
        own_blobs=own_blobs,
        enemies=(predator,),
        score=0.0,
        first_direction=(0.0, 1.0),
        first_split=False,
        first_reason="keep",
        last_direction=(0.0, 1.0),
    )
    analysis = strategy._proxy_analysis(
        node=node,
        foods=(),
        viruses=(),
        arena_size=60.0,
    )
    geometry = strategy._node_geometry(node)

    away = strategy._coarse_action_value(
        node=node,
        action=Action((-1.0, 0.0), reason="away"),
        arena_size=60.0,
        proxy_analysis=analysis,
        node_geometry=geometry,
    )
    toward = strategy._coarse_action_value(
        node=node,
        action=Action((1.0, 0.0), reason="toward"),
        arena_size=60.0,
        proxy_analysis=analysis,
        node_geometry=geometry,
    )

    assert away > toward


def test_replay_dominance_proxy_promotes_high_value_prey_without_family_slot() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=2, angular_samples=8)
    strategy._rival_values = {1: 1.0}
    own = OwnBlob(blob_id=0, x=20.0, y=20.0, radius=4.0)
    prey = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=25.0,
        y=20.0,
        radius=2.5,
        direction=(0.0, 1.0),
    )
    node = SearchNode(
        own_blobs=(own,),
        enemies=(prey,),
        score=0.0,
        first_direction=(0.0, 1.0),
        first_split=False,
        first_reason="keep",
        last_direction=(0.0, 1.0),
    )

    actions = strategy._candidate_actions(
        node=node,
        foods=(),
        food_targets=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
    )

    assert "prey" in actions[0].reason
    strategy._audit_every_n = 1
    strategy._audit_root_candidate_ranking(
        node=node,
        actions=actions,
        foods=(),
        viruses=(),
        arena_size=60.0,
        safety_weight=1.3,
        aggression=1.0,
        transition_budget=2,
    )
    strategy._run_pending_audit()
    assert strategy._audit_samples == 1
    assert strategy._audit_last_exact_rank is not None


def test_replay_audit_uses_best_admissible_action_and_isolates_caches(
    monkeypatch,
) -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    strategy._audit_every_n = 1
    strategy._current_round = 0
    own = OwnBlob(blob_id=0, x=20.0, y=20.0, radius=2.0)
    node = SearchNode(
        own_blobs=(own,),
        enemies=(),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )
    actions = (
        Action((1.0, 0.0), split=True, reason="fatal_high"),
        Action((0.0, 1.0), reason="safe"),
    )
    original_utility_cache = strategy._utility_cache

    def fake_step(*, action, **_kwargs):
        score = 100.0 if action.reason == "fatal_high" else 10.0
        return StepResult(
            replace(node, score=score),
            fatal=action.reason == "fatal_high",
        )

    monkeypatch.setattr(strategy, "_step", fake_step)
    elapsed = strategy._audit_root_candidate_ranking(
        node=node,
        actions=actions,
        foods=(),
        viruses=(),
        arena_size=60.0,
        safety_weight=1.3,
        aggression=1.0,
        transition_budget=2,
    )
    strategy._run_pending_audit()

    assert elapsed == 0.0
    assert strategy._audit_last_raw_rank == 1
    assert strategy._audit_last_exact_rank == 2
    assert strategy._audit_last_fatal_count == 1
    assert strategy._utility_cache is original_utility_cache


def test_replay_audit_runs_after_and_does_not_change_search_decision() -> None:
    own = SimpleNamespace(
        blob_id=0,
        pos=(20.0, 20.0),
        radius=3.0,
        merge_cooldown=0,
    )
    prey = SimpleNamespace(
        player_id=1,
        blob_id=0,
        pos=(24.0, 20.0),
        radius=1.0,
        merge_cooldown=0,
    )
    state = SimpleNamespace(
        round=100,
        max_rounds=1400,
        rankings=[0, 1, 2, 3, 4, 5, 6, 7],
        me=SimpleNamespace(player_id=0, blobs={0: own}),
        map=SimpleNamespace(size=60.0),
        visible_blobs=(prey,),
        visible_food=(FoodModel(food_id=1, pos=(22.0, 20.0)),),
        visible_viruses=(),
        view_center=(20.0, 20.0),
        vision_size=40.0,
    )
    context = SimpleNamespace(
        game=SimpleNamespace(state=state),
        query=SimpleNamespace(update={}),
    )
    normal = ReplayDominanceStrategy(depth=1, width=1, angular_samples=8)
    audited = ReplayDominanceStrategy(depth=1, width=1, angular_samples=8)
    audited._audit_every_n = 1

    expected = normal._choose(context, deadline=float("inf"), turn_budget=1.0)
    actual = audited._choose(context, deadline=float("inf"), turn_budget=1.0)

    assert (
        actual.direction,
        actual.split,
        actual.reason,
        actual.score,
    ) == (
        expected.direction,
        expected.split,
        expected.reason,
        expected.score,
    )
    assert audited._audit_samples == 1


def test_approximate_fallback_preserves_escape_semantics() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    strategy.proxy_refine_limit = 64
    strategy.previous_direction = (0.0, 1.0)
    own = SimpleNamespace(
        blob_id=0,
        pos=(30.0, 30.0),
        radius=1.0,
        merge_cooldown=0,
    )
    predator = SimpleNamespace(
        player_id=1,
        blob_id=0,
        pos=(35.0, 30.0),
        radius=3.0,
        merge_cooldown=0,
    )
    state = SimpleNamespace(
        round=100,
        rankings=[0, 1, 2, 3, 4, 5, 6, 7],
        me=SimpleNamespace(player_id=0, blobs={0: own}),
        map=SimpleNamespace(size=60.0),
        visible_blobs=(predator,),
        visible_food=(),
        visible_viruses=(),
        view_center=(30.0, 30.0),
        vision_size=60.0,
    )
    context = SimpleNamespace(
        game=SimpleNamespace(state=state),
        query=SimpleNamespace(update={}),
    )

    decision = strategy._time_budget_fallback(context)

    assert "escape" in decision.reason
    assert decision.target_kind == "escape"
    assert decision.direction[0] < -0.7
    assert len(decision.diagnostics["proxy_top_actions"]) <= 5


def test_turn_preparation_computes_exposed_radii_once(monkeypatch) -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    own = SimpleNamespace(
        blob_id=0,
        pos=(30.0, 30.0),
        radius=2.0,
        merge_cooldown=0,
    )
    enemies = tuple(
        SimpleNamespace(
            player_id=index + 1,
            blob_id=0,
            pos=(5.0 + index * 2.0, 5.0),
            radius=1.0 + index * 0.05,
            merge_cooldown=0,
        )
        for index in range(16)
    )
    state = SimpleNamespace(
        round=100,
        rankings=list(range(17)),
        me=SimpleNamespace(player_id=0, blobs={0: own}),
        map=SimpleNamespace(size=60.0),
        visible_blobs=enemies,
        visible_food=(),
        visible_viruses=(),
        view_center=(30.0, 30.0),
        vision_size=60.0,
    )
    context = SimpleNamespace(
        game=SimpleNamespace(state=state),
        query=SimpleNamespace(update={}),
    )
    original = strategy._exposed_own_radii
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(strategy, "_exposed_own_radii", counted)

    turn = strategy._prepare_turn(context)

    assert turn is not None
    assert len(turn.node.enemies) == strategy.max_enemies
    assert calls == 1


def test_shadow_force_rank_changes_only_the_configured_round(monkeypatch) -> None:
    monkeypatch.setenv("BOT_REPLAY_SHADOW_FORCE_ROUND", "20")
    monkeypatch.setenv("BOT_REPLAY_SHADOW_FORCE_RANK", "2")
    strategy = ReplayDominanceStrategy()
    strategy._root_proxy_scores = (
        (Action((1.0, 0.0), reason="keep"), 10.0),
        (Action((0.0, 1.0), split=True, reason="split_prey"), 9.0),
    )
    original = StrategyDecision(direction=(1.0, 0.0), reason="keep", score=10.0)

    strategy._current_round = 19
    assert strategy._shadow_forced_decision(original) is original

    strategy._current_round = 20
    forced = strategy._shadow_forced_decision(original)

    assert forced.direction == (0.0, 1.0)
    assert forced.split is True
    assert forced.reason == "split_prey"
    assert forced.score == 9.0
    assert forced.diagnostics["shadow_forced_rank"] == 2


def test_enemy_prediction_chases_nearest_edible_fragment_not_nearest_blob() -> None:
    strategy = ReplayDominanceStrategy()
    large = OwnBlob(blob_id=0, x=50.0, y=50.0, radius=12.0)
    small = OwnBlob(blob_id=1, x=40.0, y=50.0, radius=4.0)
    enemy = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=55.0,
        y=50.0,
        radius=6.0,
    )

    exact = strategy._move_enemies((enemy,), [large, small], 100.0)[0]

    proxy_blobs = [
        ProxyBlobMotion(
            blob_id=blob.blob_id,
            source_blob_id=blob.blob_id,
            start_x=blob.x,
            start_y=blob.y,
            x=blob.x,
            y=blob.y,
            radius=blob.radius,
            speed=0.0,
        )
        for blob in (large, small)
    ]
    proxy = strategy._proxy_enemy_motions(
        (enemy,),
        proxy_blobs,
        horizon=1,
        arena_size=100.0,
        hunter_masks=(0b01,),
        predator_masks=(0b10,),
    )[0]

    assert exact.direction[0] < 0.0
    assert proxy.direction[0] < 0.0


def test_enemy_prediction_single_scan_matches_reference_ordering() -> None:
    strategy = ReplayDominanceStrategy()
    rng = random.Random(20260714)

    for _ in range(500):
        own_blobs = [
            OwnBlob(
                blob_id=index,
                x=rng.uniform(1.0, 59.0),
                y=rng.uniform(1.0, 59.0),
                radius=rng.uniform(0.5, 8.0),
            )
            for index in range(rng.randint(1, 16))
        ]
        enemy = EnemyBlob(
            player_id=1,
            blob_id=0,
            x=rng.uniform(1.0, 59.0),
            y=rng.uniform(1.0, 59.0),
            radius=rng.uniform(0.5, 8.0),
            direction=(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)),
        )
        reference_prey = min(
            (own for own in own_blobs if can_eat_player_blob(enemy.radius, own.radius)),
            key=lambda own: squared_distance(enemy.pos, own.pos),
            default=None,
        )
        reference_hunter = min(
            (own for own in own_blobs if can_eat_player_blob(own.radius, enemy.radius)),
            key=lambda own: squared_distance(enemy.pos, own.pos),
            default=None,
        )

        moved = strategy._move_enemies((enemy,), own_blobs, 60.0)[0]
        target = reference_prey or reference_hunter
        if target is None:
            continue
        expected_sign = 1.0 if reference_prey is not None else -1.0
        target_vector = (target.x - enemy.x, target.y - enemy.y)
        assert (
            moved.direction[0] * target_vector[0]
            + moved.direction[1] * target_vector[1]
        ) * expected_sign > 0.0


def test_utility_identity_front_cache_keeps_structural_reuse(monkeypatch) -> None:
    strategy = ReplayDominanceStrategy()
    node = SearchNode(
        own_blobs=(OwnBlob(blob_id=0, x=30.0, y=30.0, radius=2.0),),
        enemies=(),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )
    equivalent = replace(node, score=123.0, first_reason="equivalent")
    calls = 0

    def compute(*args, **kwargs):
        nonlocal calls
        calls += 1
        return 42.0

    monkeypatch.setattr(strategy, "_search_utility", compute)
    kwargs = {
        "foods": (),
        "viruses": (),
        "arena_size": 60.0,
        "safety_weight": 1.0,
    }

    assert strategy._cached_search_utility(node, **kwargs) == 42.0
    assert strategy._cached_search_utility(node, **kwargs) == 42.0
    assert strategy._cached_search_utility(equivalent, **kwargs) == 42.0
    assert calls == 1


def test_replay_dominance_does_not_hide_trapped_fragment_behind_safe_center() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    anchor = OwnBlob(blob_id=0, x=30.0, y=30.0, radius=3.0)
    trapped_fragment = OwnBlob(blob_id=1, x=58.5, y=1.5, radius=1.5)
    predator = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=51.0,
        y=9.0,
        radius=3.0,
    )

    safe_node = SearchNode(
        own_blobs=(anchor, trapped_fragment),
        enemies=(),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )
    exposed_node = replace(safe_node, enemies=(predator,))
    safe = strategy._search_utility(
        safe_node,
        foods=(),
        viruses=(),
        arena_size=60.0,
        safety_weight=1.0,
    )
    exposed = strategy._search_utility(
        exposed_node,
        foods=(),
        viruses=(),
        arena_size=60.0,
        safety_weight=1.0,
    )

    assert exposed < safe


def test_replay_dominance_ignores_virus_that_decay_makes_unreachable() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    virus = VirusModel(virus_id=7, pos=(22.0, 10.0), radius=1.5)
    threshold_mass = virus.radius * virus.radius * 1.1
    own = OwnBlob(
        blob_id=0,
        x=10.0,
        y=10.0,
        radius=math.sqrt(threshold_mass * 1.01),
    )
    node = SearchNode(
        own_blobs=(own,),
        enemies=(),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )

    actions = strategy._candidate_actions(
        node=node,
        foods=(),
        food_targets=(),
        viruses=(virus,),
        arena_size=60.0,
        first_step=True,
    )

    assert all(action.reason != "virus_harvest" for action in actions)


def test_replay_dominance_stabilisation_matches_engine_transition() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    rng = random.Random(20260712)

    for case_index in range(40):
        count = rng.choice((2, 4, 8, 16))
        center_x = rng.choice((3.0, 30.0, 57.0))
        center_y = rng.choice((3.0, 30.0, 57.0))
        own_blobs = []
        engine_blobs = {}
        for blob_id in range(count):
            radius = rng.uniform(0.55, 1.8)
            x = min(max(center_x + rng.uniform(-3.0, 3.0), radius), 60.0 - radius)
            y = min(max(center_y + rng.uniform(-3.0, 3.0), radius), 60.0 - radius)
            cooldown = rng.choice((0, 0, 3, 17))
            eject_vx = rng.uniform(-0.25, 0.25)
            eject_vy = rng.uniform(-0.25, 0.25)
            own_blobs.append(
                OwnBlob(
                    blob_id=blob_id,
                    x=x,
                    y=y,
                    radius=radius,
                    merge_cooldown=cooldown,
                    eject_vx=eject_vx,
                    eject_vy=eject_vy,
                )
            )
            engine_blobs[blob_id] = BlobState(
                blob_id=blob_id,
                x=x,
                y=y,
                radius=radius,
                merge_cooldown=cooldown,
                eject_vx=eject_vx,
                eject_vy=eject_vy,
            )

        player = PlayerState(player_id=0, team_id=0)
        player.blobs = engine_blobs
        state = SimpleNamespace(
            players={0: player},
            map=SimpleNamespace(size=60.0),
        )
        StateMutator(state)._stabilise_same_player_blobs()
        expected = [player.blobs[key] for key in sorted(player.blobs)]
        actual = strategy._stabilise_own_blobs(own_blobs, 60.0)

        assert [blob.blob_id for blob in actual] == [
            blob.blob_id for blob in expected
        ], case_index
        for modelled, authoritative in zip(actual, expected, strict=True):
            assert math.isclose(modelled.x, authoritative.x, abs_tol=1e-12)
            assert math.isclose(modelled.y, authoritative.y, abs_tol=1e-12)
            assert math.isclose(modelled.radius, authoritative.radius, abs_tol=1e-12)
            assert modelled.merge_cooldown == authoritative.merge_cooldown
            assert math.isclose(
                modelled.eject_vx,
                authoritative.eject_vx,
                abs_tol=1e-12,
            )
            assert math.isclose(
                modelled.eject_vy,
                authoritative.eject_vy,
                abs_tol=1e-12,
            )


def test_enemy_separation_mutable_work_rows_match_frozen_replace_oracle() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    rng = random.Random(20260715)

    def frozen_replace_oracle(
        enemies: tuple[EnemyBlob, ...],
        arena_size: float,
        iterations: int,
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
                    nx, ny = (
                        (1.0, 0.0)
                        if distance <= 1e-9
                        else (dx / distance, dy / distance)
                    )
                    overlap = minimum - distance
                    total_mass = first.mass + second.mass
                    first_move = overlap * second.mass / total_mass
                    second_move = overlap * first.mass / total_mass
                    by_key[first_key] = replace(
                        first,
                        x=min(
                            max(first.x - nx * first_move, first.radius),
                            arena_size - first.radius,
                        ),
                        y=min(
                            max(first.y - ny * first_move, first.radius),
                            arena_size - first.radius,
                        ),
                    )
                    by_key[second_key] = replace(
                        second,
                        x=min(
                            max(second.x + nx * second_move, second.radius),
                            arena_size - second.radius,
                        ),
                        y=min(
                            max(second.y + ny * second_move, second.radius),
                            arena_size - second.radius,
                        ),
                    )
                    changed = True
            if not changed:
                break
        return tuple(by_key[key] for key in sorted(by_key))

    for _ in range(100):
        count = rng.choice((2, 4, 8, 16))
        enemies = tuple(
            EnemyBlob(
                player_id=rng.randrange(1, 4),
                blob_id=blob_id,
                x=rng.uniform(0.6, 8.0),
                y=rng.uniform(0.6, 8.0),
                radius=rng.uniform(0.5, 1.8),
                direction=(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)),
                stale_rounds=rng.randrange(0, 3),
                merge_cooldown=rng.randrange(0, 20),
            )
            for blob_id in range(count)
        )
        iterations = rng.randrange(1, 5)

        expected = frozen_replace_oracle(enemies, 60.0, iterations)
        actual = strategy._separate_enemy_blobs(enemies, 60.0, iterations)

        assert actual == expected


def test_nonmerging_enemy_stabilisation_matches_two_four_pass_sequence() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    rng = random.Random(20260716)

    for _ in range(50):
        enemies = tuple(
            EnemyBlob(
                player_id=1 + index // 8,
                blob_id=index,
                x=rng.uniform(2.0, 58.0),
                y=rng.uniform(2.0, 58.0),
                radius=rng.uniform(0.5, 1.8),
                direction=(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)),
                merge_cooldown=rng.randrange(1, 20),
            )
            for index in range(16)
        )
        by_player: dict[int, list[EnemyBlob]] = {}
        for enemy in enemies:
            by_player.setdefault(enemy.player_id, []).append(enemy)
        expected = []
        for player_id in sorted(by_player):
            group = list(strategy._apply_attraction(by_player[player_id], 60.0))
            group = list(strategy._merge_enemy_blobs(tuple(group), 60.0))
            group = list(strategy._separate_enemy_blobs(tuple(group), 60.0))
            group = list(strategy._merge_enemy_blobs(tuple(group), 60.0))
            group = list(strategy._separate_enemy_blobs(tuple(group), 60.0))
            expected.extend(group)

        actual = strategy._stabilise_enemy_blobs(enemies, 60.0)

        assert actual == tuple(sorted(expected, key=lambda enemy: enemy.key))


def test_virus_fragment_layout_cache_is_bounded_and_recomputation_is_exact() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    original = strategy._virus_fragment_layout(0.5, 16)
    assert strategy._virus_fragment_layout(0.5, 16) is original

    for index in range(strategy._VIRUS_FRAGMENT_LAYOUT_CACHE_LIMIT + 20):
        strategy._virus_fragment_layout(0.6 + index / 1000.0, 16)

    assert len(strategy._virus_fragment_layout_cache) == (
        strategy._VIRUS_FRAGMENT_LAYOUT_CACHE_LIMIT
    )
    assert strategy._virus_fragment_layout(0.5, 16) == original


def test_replay_dominance_resolves_virus_before_food_like_engine() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    virus = VirusModel(virus_id=9, pos=(10.0, 10.0), radius=1.5)
    food = FoodModel(food_id=4, pos=(10.0, 10.0))
    threshold_mass = virus.radius * virus.radius * 1.1
    own = OwnBlob(
        blob_id=0,
        x=10.0,
        y=10.0,
        radius=math.sqrt(threshold_mass - 0.01),
    )
    node = SearchNode(
        own_blobs=(own,),
        enemies=(),
        score=0.0,
        first_direction=(0.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(0.0, 0.0),
    )

    result = strategy._step(
        node=node,
        action=Action((0.0, 0.0), reason="keep"),
        foods=(food,),
        viruses=(virus,),
        arena_size=60.0,
        first_step=True,
        safety_weight=1.0,
        aggression=1.0,
    )

    assert result.node.projected_food == 1
    assert result.node.consumed_virus_ids == frozenset()
    assert result.node.own_blobs[0].mass > threshold_mass
