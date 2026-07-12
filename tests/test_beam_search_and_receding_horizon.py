from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from beam_search_core import (  # noqa: E402
    Action,
    BlobState,
    BeamSearchPlanner,
    FoodState,
    StrategyConfig,
    Vec2,
    WorldState,
    can_eat,
    feature_vector,
    generate_actions,
    is_threat,
    profile_config,
    split_can_hit_prey,
)
from strategies.receding_horizon import (  # noqa: E402
    Action as RecedingHorizonAction,
    ThreatAwareRecedingHorizonStrategy,
    EnemyBlob,
    EnemyTrack,
    OwnBlob,
    SearchNode as RecedingHorizonSearchNode,
    _split_attack_reach,
)
from lib.models.food_model import FoodModel  # noqa: E402
from lib.models.blob_model import BlobModel, VisibleBlobModel  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from strategies.unified_deterministic import UnifiedDeterministicStrategy  # noqa: E402


def make_state(
    me: BlobState,
    enemies: tuple[BlobState, ...] = (),
    food: tuple[FoodState, ...] = (),
) -> WorldState:
    return WorldState(
        round_number=10,
        max_rounds=1400,
        arena_size=60.0,
        player_id=0,
        self_blobs=(me,),
        enemies=enemies,
        food=food,
        viruses=(),
        rankings=(0, 1, 2),
    )


def test_mass_eat_threshold() -> None:
    assert can_eat(1.1, 1.0)
    assert not can_eat(1.09, 1.0)
    assert is_threat(1.0, 1.1)


def test_escape_from_visible_threat() -> None:
    config = profile_config("survival")
    me = BlobState(0, 0, Vec2(10.0, 10.0), 1.0, is_self=True)
    enemy = BlobState(1, 0, Vec2(12.0, 10.0), 1.5)
    food = FoodState(Vec2(13.0, 10.0))
    state = make_state(me, enemies=(enemy,), food=(food,))
    action = BeamSearchPlanner(config).choose_action(state)
    assert action.dx < 0.0
    assert not action.split


def test_food_cluster_direction_when_safe() -> None:
    config = profile_config("farmer")
    me = BlobState(0, 0, Vec2(10.0, 10.0), 1.2, is_self=True)
    food = (FoodState(Vec2(15.0, 10.0)), FoodState(Vec2(15.4, 10.1)), FoodState(Vec2(15.2, 9.7)))
    state = make_state(me, food=food)
    action = BeamSearchPlanner(config).choose_action(state)
    assert action.dx > 0.0


def test_split_candidate_requires_reachable_prey() -> None:
    config = profile_config("hunter")
    me = BlobState(0, 0, Vec2(10.0, 10.0), 3.0, is_self=True)
    prey = BlobState(1, 0, Vec2(14.0, 10.0), 1.0)
    state = make_state(me, enemies=(prey,))
    split_action = Action(1.0, 0.0, True, "split")
    assert split_can_hit_prey(state, split_action, config)
    assert any(a.split for a in generate_actions(state, config))


def test_feature_vector_length_is_stable() -> None:
    config = StrategyConfig()
    me = BlobState(0, 0, Vec2(10.0, 10.0), 1.0, is_self=True)
    state = make_state(me)
    assert len(feature_vector(state, config)) == 30


def test_threat_aware_receding_horizon_split_matches_engine_geometry() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=1, width=1, angular_samples=8)
    parent = OwnBlob(blob_id=4, x=10.0, y=10.0, radius=2.0)

    split = sorted(
        strategy._apply_split([parent], (1.0, 0.0), arena_size=60.0),
        key=lambda blob: blob.blob_id,
    )

    assert len(split) == 2
    assert math.isclose(split[0].radius, math.sqrt(2.0))
    assert math.isclose(split[1].radius, math.sqrt(2.0))
    assert math.isclose(split[1].x, 10.0 + 2.0 * math.sqrt(2.0) + 1e-4)
    assert math.isclose(split[1].eject_vx, 1.6)
    assert split[0].merge_cooldown == split[1].merge_cooldown == 18


def test_enemy_memory_retains_threat_to_only_the_small_fragment() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(
        depth=1,
        width=1,
        angular_samples=4,
    )
    own_blobs = (
        OwnBlob(blob_id=0, x=10.0, y=10.0, radius=3.0),
        OwnBlob(blob_id=1, x=12.0, y=10.0, radius=1.0),
    )
    strategy.enemy_tracks[(1, 0)] = EnemyTrack(
        player_id=1,
        blob_id=0,
        x=40.0,
        y=40.0,
        radius=1.2,
        direction=(0.0, 0.0),
        last_seen_round=9,
    )
    state = SimpleNamespace(
        round=10,
        visible_blobs=(),
        view_center=(10.0, 10.0),
        vision_size=20.0,
    )

    enemies = strategy._update_enemy_memory(
        SimpleNamespace(game=SimpleNamespace(state=state)),
        own_blobs,
        60.0,
    )

    assert len(enemies) == 1
    assert enemies[0].player_id == 1
    assert can_eat(enemies[0].radius, own_blobs[1].radius)
    assert not can_eat(enemies[0].radius, own_blobs[0].radius)


def test_unified_enemy_memory_retains_threat_to_only_the_small_fragment() -> None:
    strategy = UnifiedDeterministicStrategy()
    own_blobs = (
        OwnBlob(blob_id=0, x=10.0, y=10.0, radius=3.0),
        OwnBlob(blob_id=1, x=12.0, y=10.0, radius=1.0),
    )
    strategy.enemy_tracks[(1, 0)] = EnemyTrack(
        player_id=1,
        blob_id=0,
        x=40.0,
        y=40.0,
        radius=1.2,
        direction=(0.0, 0.0),
        last_seen_round=9,
    )
    state = SimpleNamespace(
        round=10,
        visible_blobs=(),
        view_center=(10.0, 10.0),
        vision_size=20.0,
    )

    enemies = strategy._update_enemy_memory(
        SimpleNamespace(game=SimpleNamespace(state=state)),
        own_blobs,
        60.0,
    )

    assert len(enemies) == 1
    assert can_eat(enemies[0].radius, own_blobs[1].radius)


def test_unified_value_does_not_penalise_collecting_reachable_food() -> None:
    strategy = UnifiedDeterministicStrategy()
    blobs = tuple(
        OwnBlob(
            blob_id=index,
            x=10.0 + (index % 4) * 3.0,
            y=10.0 + (index // 4) * 3.0,
            radius=0.9,
            merge_cooldown=10,
        )
        for index in range(16)
    )
    food = FoodModel(food_id=7, pos=blobs[0].pos)
    before = RecedingHorizonSearchNode(
        own_blobs=blobs,
        enemies=(),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )
    grown = OwnBlob(
        blob_id=blobs[0].blob_id,
        x=blobs[0].x,
        y=blobs[0].y,
        radius=math.sqrt(blobs[0].mass + 0.15**2),
        merge_cooldown=10,
    )
    after = RecedingHorizonSearchNode(
        own_blobs=(grown, *blobs[1:]),
        enemies=(),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
        eaten_food_ids=frozenset({food.food_id}),
    )

    assert strategy._state_value(after, (food,), (), 60.0) >= strategy._state_value(
        before, (food,), (), 60.0
    )


def test_unified_value_does_not_penalise_capturing_reachable_prey() -> None:
    strategy = UnifiedDeterministicStrategy()
    strategy._rival_values = {1: 0.5}
    own = (
        OwnBlob(blob_id=0, x=28.0, y=30.0, radius=2.0),
        OwnBlob(blob_id=1, x=35.0, y=30.0, radius=2.0),
        OwnBlob(blob_id=2, x=30.0, y=25.0, radius=2.0),
        OwnBlob(blob_id=3, x=30.0, y=35.0, radius=2.0),
    )
    prey = EnemyBlob(player_id=1, blob_id=0, x=30.0, y=30.0, radius=1.0)
    before = RecedingHorizonSearchNode(
        own_blobs=own,
        enemies=(prey,),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="prey",
        last_direction=(1.0, 0.0),
    )
    eater = OwnBlob(
        blob_id=0,
        x=own[0].x,
        y=own[0].y,
        radius=math.sqrt(own[0].mass + prey.mass),
    )
    after = RecedingHorizonSearchNode(
        own_blobs=(eater, *own[1:]),
        enemies=(),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="prey",
        last_direction=(1.0, 0.0),
        captured_enemy_ids=frozenset({prey.key}),
    )

    assert strategy._state_value(after, (), (), 60.0) >= strategy._state_value(
        before, (), (), 60.0
    )



def test_unified_enemy_siblings_share_one_player_move_direction() -> None:
    strategy = UnifiedDeterministicStrategy()
    enemies = (
        EnemyBlob(player_id=1, blob_id=0, x=10.0, y=10.0, radius=1.0, direction=(1.0, 0.0)),
        EnemyBlob(player_id=1, blob_id=1, x=20.0, y=20.0, radius=1.0, direction=(1.0, 0.0)),
    )

    moved = strategy._move_enemies(enemies, [], 60.0)

    assert math.isclose(moved[0].x - enemies[0].x, moved[1].x - enemies[1].x)
    assert math.isclose(moved[0].y - enemies[0].y, moved[1].y - enemies[1].y)


def test_unified_path_cost_survives_the_second_search_depth() -> None:
    strategy = UnifiedDeterministicStrategy()
    node = RecedingHorizonSearchNode(
        own_blobs=(OwnBlob(blob_id=0, x=59.0, y=30.0, radius=1.0),),
        enemies=(),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )
    first = strategy._transition(
        node=node,
        action=RecedingHorizonAction((1.0, 0.0), reason="keep"),
        foods=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
    ).node
    second = strategy._transition(
        node=first,
        action=RecedingHorizonAction((-1.0, 0.0), reason="turn"),
        foods=(),
        viruses=(),
        arena_size=60.0,
        first_step=False,
    ).node

    assert first.control_cost > 0.0
    assert second.control_cost >= first.control_cost


def test_unified_root_screen_is_invariant_to_candidate_order() -> None:
    strategy = UnifiedDeterministicStrategy()
    node = RecedingHorizonSearchNode(
        own_blobs=(OwnBlob(blob_id=0, x=30.0, y=30.0, radius=1.0),),
        enemies=(),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )
    actions = (
        RecedingHorizonAction((0.0, 1.0), reason="north"),
        RecedingHorizonAction((-1.0, 0.0), reason="west"),
        RecedingHorizonAction((1.0, 0.0), reason="east"),
    )
    strategy._screening_context = lambda *args: (0.0, 0.0)  # type: ignore[method-assign]
    strategy._cheap_action_score = lambda *args: 0.0  # type: ignore[method-assign]

    forward = strategy._select_actions_no_reservation(
        node=node,
        actions=actions,
        foods=(),
        viruses=(),
        arena_size=60.0,
        limit=2,
        require_diversity=True,
    )
    reverse = strategy._select_actions_no_reservation(
        node=node,
        actions=tuple(reversed(actions)),
        foods=(),
        viruses=(),
        arena_size=60.0,
        limit=2,
        require_diversity=True,
    )

    assert forward == reverse


def test_threat_aware_receding_horizon_captures_safely_while_leading_late() -> None:
    own = BlobModel(blob_id=0, pos=(10.0, 30.0), radius=2.2)
    prey = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(13.6, 30.0),
        radius=1.0,
    )
    state = SimpleNamespace(
        me=SimpleNamespace(player_id=0, x=10.0, y=30.0, blobs={0: own}),
        visible_blobs=[prey],
        visible_food=[],
        visible_viruses=[],
        map=SimpleNamespace(size=60.0),
        round=1300,
        max_rounds=1400,
        rankings=[0, 1, 2, 3, 4, 5, 6, 7],
        view_center=(10.0, 30.0),
        vision_size=20.0,
    )
    strategy = ThreatAwareRecedingHorizonStrategy(depth=3, width=4, angular_samples=18)
    strategy.compute_budget_seconds = 100.0
    strategy.max_turn_seconds = 1.0

    decision = strategy.choose(
        StrategyContext(
            game=SimpleNamespace(state=state),
            query=SimpleNamespace(update={}),
        )
    )

    assert decision.diagnostics["projected_captures"] >= 1
    assert decision.reason in {"keep", "prey", "split_prey"}
    assert decision.diagnostics["projected_captures"] == 1


def test_threat_aware_receding_horizon_rejects_split_when_a_neutral_enemy_can_eat_fragments() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=3, width=4, angular_samples=18)
    own = OwnBlob(blob_id=0, x=10.0, y=30.0, radius=2.2)
    newly_dangerous_enemy = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=16.0,
        y=30.0,
        radius=2.4,
    )
    node = RecedingHorizonSearchNode(
        own_blobs=(own,),
        enemies=(newly_dangerous_enemy,),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )

    unsplit = strategy._step(
        node=node,
        action=RecedingHorizonAction((1.0, 0.0), split=False, reason="keep"),
        foods=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
        safety_weight=1.0,
        aggression=1.0,
    )
    split = strategy._step(
        node=node,
        action=RecedingHorizonAction((1.0, 0.0), split=True, reason="unsafe_split"),
        foods=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
        safety_weight=1.0,
        aggression=1.0,
    )

    assert split.node.total_mass < unsplit.node.total_mass
    assert strategy._terminal_score(split.node) < strategy._terminal_score(unsplit.node)
    assert split.fatal


def test_threat_aware_receding_horizon_resolves_predator_growth_cascade() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=1, width=1, angular_samples=8)
    small = OwnBlob(blob_id=0, x=10.0, y=30.0, radius=1.0)
    larger = OwnBlob(blob_id=1, x=11.4, y=30.0, radius=1.3)
    predator = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=10.0,
        y=30.0,
        radius=1.1,
    )

    survivors, enemies, score, captures = strategy._resolve_interactions(
        [small, larger],
        (predator,),
        set(),
    )

    assert not survivors
    assert math.isclose(enemies[0].radius, math.sqrt(1.1**2 + 1.0**2 + 1.3**2))
    assert score < -1_000.0
    assert captures == 0


def test_threat_aware_receding_horizon_separates_split_fragments_clamped_at_wall() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=1, width=1, angular_samples=8)
    parent = OwnBlob(blob_id=0, x=58.0, y=30.0, radius=2.0)
    split = strategy._apply_split([parent], (1.0, 0.0), arena_size=60.0)
    moved = [strategy._move_own(blob, (1.0, 0.0), arena_size=60.0) for blob in split]

    stabilised = strategy._stabilise_own_blobs(moved, arena_size=60.0)

    assert len(stabilised) == 2
    assert math.dist(stabilised[0].pos, stabilised[1].pos) > 0.0
    assert all(blob.x <= 60.0 - blob.radius for blob in stabilised)


def test_threat_aware_receding_horizon_prices_enemy_fragments_after_they_merge() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=1, width=1, angular_samples=8)
    fragments = (
        EnemyBlob(
            player_id=1,
            blob_id=0,
            x=15.0,
            y=30.0,
            radius=1.0,
            merge_cooldown=0,
        ),
        EnemyBlob(
            player_id=1,
            blob_id=1,
            x=16.9,
            y=30.0,
            radius=1.0,
            merge_cooldown=0,
        ),
    )
    own_fragment = OwnBlob(blob_id=0, x=12.0, y=30.0, radius=1.2)

    merged = strategy._merge_enemy_blobs(fragments, arena_size=60.0)
    penalty, _, _ = strategy._risk_score(
        [own_fragment],
        merged,
        safety_weight=1.0,
        arena_size=60.0,
    )

    assert len(merged) == 1
    assert math.isclose(merged[0].radius, math.sqrt(2.0))
    assert penalty > 0.0


def test_threat_aware_receding_horizon_prices_merge_then_split_before_merge() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(
        depth=1,
        width=1,
        angular_samples=8,
    )
    fragments = (
        EnemyBlob(
            player_id=1,
            blob_id=0,
            x=15.0,
            y=30.0,
            radius=1.0,
            merge_cooldown=7,
        ),
        EnemyBlob(
            player_id=1,
            blob_id=1,
            x=16.9,
            y=30.0,
            radius=1.0,
            merge_cooldown=7,
        ),
    )
    own = OwnBlob(blob_id=0, x=10.0, y=30.0, radius=1.2)

    envelopes = strategy._future_enemy_envelopes(fragments)
    penalty, margin, unavoidable = strategy._risk_score(
        [own],
        fragments,
        safety_weight=1.0,
        arena_size=60.0,
    )

    assert len(envelopes) == 1
    assert math.isclose(envelopes[0].radius, math.sqrt(2.0))
    assert penalty > 0.0
    assert margin < math.inf
    assert not unavoidable


def test_threat_aware_receding_horizon_prices_enemy_split_reach_not_only_current_radius() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=1, width=1, angular_samples=8)
    own = OwnBlob(blob_id=0, x=10.0, y=10.0, radius=1.0)
    enemy = EnemyBlob(player_id=1, blob_id=0, x=16.0, y=10.0, radius=2.0)

    penalty, margin, unavoidable = strategy._risk_score([own], (enemy,), safety_weight=1.0)

    assert 6.0 > enemy.radius  # Safe from ordinary overlap this round.
    assert _split_attack_reach(enemy.radius) > 6.0
    assert penalty > 400.0
    assert margin < 0.0
    assert unavoidable


def test_threat_aware_receding_horizon_keeps_farther_predator_ahead_of_harmless_enemy_limit() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=1, width=1, angular_samples=8)
    own = (OwnBlob(blob_id=0, x=30.0, y=30.0, radius=1.0),)
    harmless = [
        EnemyBlob(
            player_id=index + 1,
            blob_id=0,
            x=30.5 + index * 0.1,
            y=30.0,
            radius=0.5,
        )
        for index in range(12)
    ]
    predator = EnemyBlob(
        player_id=20,
        blob_id=0,
        x=39.0,
        y=30.0,
        radius=2.0,
    )

    selected = sorted(
        (*harmless, predator),
        key=lambda enemy: strategy._enemy_priority(enemy, own, (30.0, 30.0)),
    )[: strategy.max_enemies]

    assert predator in selected
    assert len(selected) == strategy.max_enemies


def test_threat_aware_receding_horizon_can_collect_food_touching_a_safe_wall() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=1, width=1, angular_samples=8)
    own = OwnBlob(blob_id=0, x=2.0, y=30.0, radius=1.0)
    node = RecedingHorizonSearchNode(
        own_blobs=(own,),
        enemies=(),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )
    edge_food = FoodModel(food_id=1, pos=(0.2, 30.0))

    toward_wall = strategy._step(
        node=node,
        action=RecedingHorizonAction((-1.0, 0.0), reason="edge_food"),
        foods=(edge_food,),
        viruses=(),
        arena_size=60.0,
        first_step=True,
        safety_weight=1.0,
        aggression=1.0,
    )
    away_from_wall = strategy._step(
        node=node,
        action=RecedingHorizonAction((1.0, 0.0), reason="away"),
        foods=(edge_food,),
        viruses=(),
        arena_size=60.0,
        first_step=True,
        safety_weight=1.0,
        aggression=1.0,
    )

    assert toward_wall.node.projected_food == 1
    assert away_from_wall.node.projected_food == 0
    assert strategy._terminal_score(toward_wall.node) > strategy._terminal_score(away_from_wall.node)


def test_threat_aware_receding_horizon_penalises_a_move_clamped_by_the_wall() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=3, width=4, angular_samples=18)
    own = OwnBlob(blob_id=0, x=59.0, y=30.0, radius=1.0)
    node = RecedingHorizonSearchNode(
        own_blobs=(own,),
        enemies=(),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )

    blocked = strategy._step(
        node=node,
        action=RecedingHorizonAction((1.0, 0.0), reason="keep"),
        foods=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
        safety_weight=1.0,
        aggression=1.0,
    )
    inward = strategy._step(
        node=node,
        action=RecedingHorizonAction((-1.0, 0.0), reason="wall_escape"),
        foods=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
        safety_weight=1.0,
        aggression=1.0,
    )

    assert blocked.node.own_blobs[0].pos == own.pos
    assert inward.node.own_blobs[0].x < own.x
    assert strategy._terminal_score(inward.node) > strategy._terminal_score(blocked.node)


def test_threat_aware_receding_horizon_does_not_repeat_an_outward_wall_direction() -> None:
    blob = BlobModel(blob_id=0, pos=(59.0, 30.0), radius=1.0)
    state = SimpleNamespace(
        me=SimpleNamespace(player_id=0, x=59.0, y=30.0, blobs={0: blob}),
        visible_blobs=[],
        visible_food=[],
        visible_viruses=[],
        map=SimpleNamespace(size=60.0),
        round=10,
        max_rounds=1400,
        rankings=[0, 1, 2, 3, 4, 5, 6, 7],
        view_center=(50.0, 30.0),
        vision_size=20.0,
    )
    context = StrategyContext(
        game=SimpleNamespace(state=state),
        query=SimpleNamespace(update={}),
    )
    strategy = ThreatAwareRecedingHorizonStrategy(depth=3, width=4, angular_samples=18)
    # This regression checks wall behavior, not the host scheduler's ability
    # to complete a three-millisecond search while the full suite is busy.
    strategy.max_turn_seconds = 0.1
    strategy.compute_budget_seconds = 140.0
    strategy.previous_direction = (1.0, 0.0)

    decisions = [strategy.choose(context) for _ in range(3)]

    assert all(decision.direction != (1.0, 0.0) for decision in decisions)
    assert all(not decision.diagnostics["search_timed_out"] for decision in decisions)


def test_threat_aware_receding_horizon_still_prices_a_predator_near_the_wall() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=1, width=1, angular_samples=8)
    own = OwnBlob(blob_id=0, x=1.0, y=30.0, radius=1.0)
    interior_predator = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=6.0,
        y=30.0,
        radius=1.5,
    )
    penalty, margin, unavoidable = strategy._risk_score(
        [own],
        (interior_predator,),
        safety_weight=1.0,
    )

    assert penalty > 0.0
    assert margin > 0.0
    assert not unavoidable


def test_threat_aware_receding_horizon_wall_risk_depends_on_whether_it_blocks_retreat() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=3, width=4, angular_samples=18)
    trapped = OwnBlob(blob_id=0, x=1.0, y=30.0, radius=1.0)
    trapped_predator = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=6.0,
        y=30.0,
        radius=1.5,
    )
    open_space = OwnBlob(blob_id=0, x=20.0, y=30.0, radius=1.0)
    open_space_predator = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=25.0,
        y=30.0,
        radius=1.5,
    )

    trapped_penalty, _, _ = strategy._risk_score(
        [trapped],
        (trapped_predator,),
        safety_weight=1.0,
        arena_size=60.0,
    )
    open_penalty, _, _ = strategy._risk_score(
        [open_space],
        (open_space_predator,),
        safety_weight=1.0,
        arena_size=60.0,
    )

    assert strategy._wall_trap_factor(trapped, trapped_predator, 60.0) == 1.0
    assert strategy._wall_trap_factor(open_space, open_space_predator, 60.0) == 0.0
    assert trapped_penalty > open_penalty


def test_threat_aware_receding_horizon_turn_budget_tracks_remaining_time_bank() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=3, width=4, angular_samples=18)
    strategy.compute_budget_seconds = 7.2
    strategy.max_turn_seconds = 0.005

    initial = strategy._turn_budget_seconds(round_number=0, max_rounds=1400)
    assert math.isclose(initial, 0.005)

    strategy.compute_spent_seconds = 7.0
    constrained = strategy._turn_budget_seconds(round_number=700, max_rounds=1400)
    assert math.isclose(constrained, 0.2 / 700)

    strategy.compute_spent_seconds = 7.2
    assert strategy._turn_budget_seconds(round_number=1000, max_rounds=1400) == 0.0


def test_threat_aware_receding_horizon_anytime_root_considers_escape_before_angle_grid() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=3, width=4, angular_samples=18)
    own = OwnBlob(blob_id=0, x=10.0, y=30.0, radius=1.0)
    predator = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=16.0,
        y=30.0,
        radius=2.0,
    )
    node = RecedingHorizonSearchNode(
        own_blobs=(own,),
        enemies=(predator,),
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
        angle_offset=7,
    )

    reasons = [action.reason for action in actions]
    assert reasons[0] == "escape"
    assert reasons.index("angle") > reasons.index("center")


def test_receding_horizon_endgame_adaptation_can_be_disabled_in_isolation() -> None:
    adaptive = ThreatAwareRecedingHorizonStrategy(endgame_adaptation=True)
    neutral = ThreatAwareRecedingHorizonStrategy(endgame_adaptation=False)

    assert adaptive._safety_weight(rank_position=1, progress=0.8) == 1.8
    assert adaptive._aggression(rank_position=1, progress=0.8) == 0.72
    assert adaptive._aggression(rank_position=6, progress=0.8) == 1.45
    assert neutral._safety_weight(rank_position=1, progress=0.8) == 1.0
    assert neutral._aggression(rank_position=1, progress=0.8) == 1.0
    assert neutral._aggression(rank_position=6, progress=0.8) == 1.0
