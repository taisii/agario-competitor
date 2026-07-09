from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from bot_core import (  # noqa: E402
    Action,
    BlobState,
    FoodState,
    StrategyConfig,
    Vec2,
    VirusState,
    WorldState,
    _predict_enemies,
    _resolve_food,
    _resolve_player_eating,
    simulate_step,
    speed_for_radius,
    split_can_hit_prey,
    virus_risk,
)
from lib.models.blob_model import BlobModel, VisibleBlobModel  # noqa: E402
from lib.models.food_model import FoodModel  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from strategies.beam_hunter import BeamHunterStrategy, SimOwnBlob  # noqa: E402
from strategies.beam_survival import (  # noqa: E402
    BeamNode as SurvivalBeamNode,
    BeamSurvivalStrategy,
    SimBlob as SurvivalEnemyBlob,
    SimOwnBlob as SurvivalOwnBlob,
)
from strategies.features import can_eat_player_blob, extract_visible_features  # noqa: E402
from strategies.food_greedy import FoodGreedyStrategy  # noqa: E402
from strategies.potential_hunter import PotentialHunterStrategy  # noqa: E402


def _world(
    own: tuple[BlobState, ...],
    *,
    enemies: tuple[BlobState, ...] = (),
    food: tuple[FoodState, ...] = (),
    viruses: tuple[VirusState, ...] = (),
) -> WorldState:
    return WorldState(
        round_number=10,
        max_rounds=1400,
        arena_size=60.0,
        player_id=0,
        self_blobs=own,
        enemies=enemies,
        food=food,
        viruses=viruses,
        rankings=(0, 1),
    )


def _game(
    own: tuple[BlobModel, ...],
    *,
    enemies: tuple[VisibleBlobModel, ...] = (),
    food: tuple[FoodModel, ...] = (),
) -> SimpleNamespace:
    total_mass = sum(blob.radius * blob.radius for blob in own)
    center_x = sum(blob.pos[0] * blob.radius * blob.radius for blob in own) / total_mass
    center_y = sum(blob.pos[1] * blob.radius * blob.radius for blob in own) / total_mass
    me = SimpleNamespace(
        player_id=0,
        x=center_x,
        y=center_y,
        radius=math.sqrt(total_mass),
        alive=True,
        blobs={blob.blob_id: blob for blob in own},
    )
    state = SimpleNamespace(
        me=me,
        visible_blobs=list(enemies),
        visible_food=list(food),
        visible_viruses=[],
        map=SimpleNamespace(size=60.0),
        round=10,
        max_rounds=1400,
        rankings=[0, 1],
    )
    return SimpleNamespace(state=state)


def test_visible_features_measure_threat_from_vulnerable_fragment() -> None:
    large = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=3.0)
    small = BlobModel(blob_id=1, pos=(13.0, 10.0), radius=1.0)
    mixed_enemy = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(11.0, 10.0),
        radius=1.5,
    )

    features = extract_visible_features(_game((large, small), enemies=(mixed_enemy,)))

    assert len(features.predators) == 1
    assert not features.prey
    assert features.predators[0].nearest_own_blob.blob_id == small.blob_id


def test_all_strategies_share_engine_mass_ratio_threshold() -> None:
    assert can_eat_player_blob(1.1, 1.0)
    assert not can_eat_player_blob(1.09, 1.0)


def test_food_greedy_aims_from_fragment_that_can_reach_food_first() -> None:
    left = BlobModel(blob_id=0, pos=(0.9, 10.0), radius=0.9)
    right = BlobModel(blob_id=1, pos=(10.0, 10.0), radius=0.9)
    food = FoodModel(food_id=1, pos=(9.0, 10.0))
    game = _game((left, right), food=(food,))

    decision = FoodGreedyStrategy().choose(
        StrategyContext(game=game, query=SimpleNamespace())
    )

    assert decision.direction[0] < 0.0


def test_visible_features_choose_food_nearest_any_real_fragment() -> None:
    left = BlobModel(blob_id=0, pos=(1.0, 10.0), radius=0.9)
    right = BlobModel(blob_id=1, pos=(19.0, 10.0), radius=0.9)
    center_near = FoodModel(food_id=1, pos=(9.0, 10.0))
    fragment_near = FoodModel(food_id=2, pos=(19.5, 10.0))

    features = extract_visible_features(
        _game((left, right), food=(center_near, fragment_near))
    )

    assert features.nearest_food is not None
    assert features.nearest_food.food_id == fragment_near.food_id
    assert math.isclose(features.nearest_food_distance or -1.0, 0.5)


def test_beam_hunter_split_at_capacity_preserves_existing_blobs() -> None:
    strategy = BeamHunterStrategy(depth=1, width=1, angular_samples=4)
    blobs = [
        SimOwnBlob(blob_id=index, x=5.0 + index * 3.0, y=20.0, radius=2.0)
        for index in range(15)
    ]

    split = strategy._apply_split(blobs, (1.0, 0.0), arena_size=60.0)

    assert len(split) == 16
    assert set(range(15)).issubset(blob.blob_id for blob in split)
    assert sum(blob.blob_id >= 15 for blob in split) == 1


def test_beam_survival_does_not_use_aggregate_radius_as_eating_power() -> None:
    strategy = BeamSurvivalStrategy(depth=1, width=1, angular_samples=4)
    node = SurvivalBeamNode(
        own_blobs=(
            SurvivalOwnBlob(blob_id=0, x=10.0, y=10.0, radius=1.0),
            SurvivalOwnBlob(blob_id=1, x=12.0, y=10.0, radius=1.0),
        ),
        enemies=(
            SurvivalEnemyBlob(
                player_id=1,
                blob_id=0,
                x=14.0,
                y=10.0,
                radius=1.1,
            ),
        ),
        score=0.0,
        first_direction=(1.0, 0.0),
        last_direction=(1.0, 0.0),
    )

    # sqrt(1^2 + 1^2) could eat radius 1.1, but neither real fragment can.
    assert strategy._nearest_prey_pair(node) is None


def test_potential_hunter_does_not_split_beyond_engine_reach() -> None:
    strategy = PotentialHunterStrategy()
    primary = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=2.0)
    prey = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(17.5, 10.0),
        radius=0.8,
    )

    plan = strategy._best_prey(
        primary=primary,
        own_blob_count=1,
        enemies=(prey,),
        viruses=(),
        early=True,
    )

    assert plan is not None
    assert not plan.split


def test_enemy_prediction_preserves_partial_aggression_speed() -> None:
    own = BlobState(0, 0, Vec2(10.0, 10.0), 1.0, is_self=True)
    enemy = BlobState(1, 0, Vec2(15.0, 10.0), 1.5)
    config = StrategyConfig(predicted_enemy_aggression=0.5)

    predicted = _predict_enemies(_world((own,), enemies=(enemy,)), config)[0]

    expected_x = enemy.pos.x - speed_for_radius(enemy.radius) * 0.5
    assert math.isclose(predicted.pos.x, expected_x)


def test_simulation_respects_engine_minimum_mass() -> None:
    own = BlobState(0, 0, Vec2(10.0, 10.0), 0.9, is_self=True)

    result, _ = simulate_step(
        _world((own,)),
        Action(1.0, 0.0),
        StrategyConfig(),
    )

    assert math.isclose(result.self_blobs[0].radius, 0.9)


def test_simulation_uses_food_center_inside_blob_rule() -> None:
    own = BlobState(0, 0, Vec2(10.0, 10.0), 1.0, is_self=True)
    next_x = own.pos.x + speed_for_radius(own.radius)
    food = FoodState(Vec2(next_x + 1.08, own.pos.y))

    result, info = simulate_step(
        _world((own,), food=(food,)),
        Action(1.0, 0.0),
        StrategyConfig(),
    )

    assert len(result.food) == 1
    assert info["food_gain_mass"] == 0.0


def test_shared_simulation_gives_contested_food_to_largest_blob() -> None:
    own = BlobState(0, 0, Vec2(10.0, 10.0), 1.0, is_self=True)
    enemy = BlobState(1, 0, Vec2(10.0, 10.0), 2.0)

    own_after, enemies_after, food_after, own_gain = _resolve_food(
        [own],
        [enemy],
        (FoodState(Vec2(10.0, 10.0)),),
    )

    assert not food_after
    assert own_gain == 0.0
    assert math.isclose(own_after[0].radius, own.radius)
    assert enemies_after[0].radius > enemy.radius


def test_shared_simulation_applies_predator_growth_cascade() -> None:
    own = [
        BlobState(0, 0, Vec2(10.0, 10.0), 0.8, is_self=True),
        BlobState(0, 1, Vec2(10.0, 10.0), 1.1, is_self=True),
    ]
    enemy = [BlobState(1, 0, Vec2(10.0, 10.0), 1.1)]

    own_after, enemies_after, _gain, lost = _resolve_player_eating(own, enemy)

    assert not own_after
    assert lost == 2
    assert enemies_after[0].radius > 1.1


def test_harmless_virus_has_no_risk() -> None:
    harmless = _world(
        (BlobState(0, 0, Vec2(10.0, 10.0), 1.0, is_self=True),),
        viruses=(VirusState(Vec2(10.5, 10.0), 1.5),),
    )
    dangerous = _world(
        (BlobState(0, 0, Vec2(10.0, 10.0), 2.0, is_self=True),),
        viruses=(VirusState(Vec2(10.5, 10.0), 1.5),),
    )

    assert virus_risk(harmless) == 0.0
    assert virus_risk(dangerous) > 0.0


def test_shared_beam_split_applies_to_every_eligible_blob() -> None:
    own = (
        BlobState(0, 0, Vec2(10.0, 10.0), 2.0, is_self=True),
        BlobState(0, 1, Vec2(10.0, 20.0), 2.0, is_self=True),
    )

    result, _ = simulate_step(
        _world(own),
        Action(1.0, 0.0, split=True),
        StrategyConfig(),
    )

    assert len(result.self_blobs) == 4


def test_shared_beam_finds_split_from_non_largest_fragment() -> None:
    own = (
        BlobState(0, 0, Vec2(10.0, 10.0), 3.0, is_self=True),
        BlobState(0, 1, Vec2(20.0, 10.0), 2.0, is_self=True),
    )
    prey = BlobState(1, 0, Vec2(25.0, 10.0), 0.8)

    assert split_can_hit_prey(
        _world(own, enemies=(prey,)),
        Action(1.0, 0.0, split=True),
        StrategyConfig(),
    )
