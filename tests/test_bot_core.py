from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from bot_core import (  # noqa: E402
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
from strategies.champion import (  # noqa: E402
    ChampionStrategy,
    EnemyBlob,
    OwnBlob,
    _split_attack_reach,
)


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


def test_radius_eat_threshold() -> None:
    assert can_eat(1.2, 1.0)
    assert not can_eat(1.19, 1.0)
    assert is_threat(1.0, 1.2)


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


def test_champion_split_matches_engine_geometry() -> None:
    strategy = ChampionStrategy(depth=1, width=1, angular_samples=8)
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


def test_champion_prices_enemy_split_reach_not_only_current_radius() -> None:
    strategy = ChampionStrategy(depth=1, width=1, angular_samples=8)
    own = OwnBlob(blob_id=0, x=10.0, y=10.0, radius=1.0)
    enemy = EnemyBlob(player_id=1, blob_id=0, x=16.0, y=10.0, radius=2.0)

    penalty, margin, unavoidable = strategy._risk_score([own], (enemy,), safety_weight=1.0)

    assert 6.0 > enemy.radius  # Safe from ordinary overlap this round.
    assert _split_attack_reach(enemy.radius) > 6.0
    assert penalty > 400.0
    assert margin < 0.0
    assert unavoidable
