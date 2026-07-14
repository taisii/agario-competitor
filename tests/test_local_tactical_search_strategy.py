from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from strategies.local_tactical_search import (  # noqa: E402
    LocalTarget,
    LocalTacticalSearchReferenceStrategy,
    LocalTacticalSearchStrategy,
)
from strategies.features import can_eat_player_blob  # noqa: E402
from strategies.receding_horizon import Action, EnemyBlob, OwnBlob, SearchNode  # noqa: E402
from strategies.registry import create_strategy, submission_strategy_spec  # noqa: E402
from lib.models.food_model import FoodModel  # noqa: E402
from lib.models.virus_model import VirusModel  # noqa: E402


def _node(*, enemy: EnemyBlob | None = None) -> SearchNode:
    return SearchNode(
        own_blobs=(OwnBlob(blob_id=0, x=30.0, y=30.0, radius=4.0),),
        enemies=() if enemy is None else (enemy,),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )


def test_local_tactical_search_is_a_separate_submission_strategy() -> None:
    strategy = create_strategy("local_tactical_search")

    assert isinstance(strategy, LocalTacticalSearchStrategy)
    assert strategy.name == "local_tactical_search"
    assert (
        submission_strategy_spec("local_tactical_search").submission.strategy_class
        == "LocalTacticalSearchStrategy"
    )


def test_reference_planner_keeps_the_widest_two_step_search() -> None:
    strategy = create_strategy("local_tactical_search_reference")

    assert isinstance(strategy, LocalTacticalSearchReferenceStrategy)
    assert strategy.depth == 1
    assert strategy.width == 32
    assert strategy._actions_per_node_limit(0) == 64
    assert strategy._LOCAL_STATE_LIMIT == 32
    assert strategy._TARGET_DIRECTION_LIMIT == 12
    assert strategy.local_continuation_prior == 4.0
    assert not strategy._uses_compute_time_bank()


def test_turn_cost_only_discourages_movement_with_a_backward_component() -> None:
    strategy = LocalTacticalSearchStrategy()

    assert strategy._turn_cost((1.0, 0.0), (1.0, 0.0)) == 0.0
    assert strategy._turn_cost((1.0, 0.0), (0.0, 1.0)) == 0.0
    assert 0.0 < strategy._turn_cost((1.0, 0.0), (-0.5, 0.5))
    assert strategy._turn_cost((1.0, 0.0), (-1.0, 0.0)) == 2.0


def test_production_planner_is_shallow_and_wide() -> None:
    strategy = LocalTacticalSearchStrategy()

    assert strategy.depth == 1
    assert strategy.width == 6
    assert strategy.angular_samples == 6
    assert strategy._LOCAL_ROOT_LIMIT == 6
    assert strategy._actions_per_node_limit(0) == 2
    assert strategy._DEEP_DIRECTION_LIMIT == 3
    assert not strategy._uses_compute_time_bank()


def test_local_continuation_rank_remains_in_exact_root_score() -> None:
    strategy = LocalTacticalSearchStrategy(depth=1)
    strategy.local_continuation_prior = 4.0
    top = Action((1.0, 0.0), reason="prey")
    bottom = Action((-1.0, 0.0), reason="angle")
    strategy._local_root_scores = ((top, 20.0), (bottom, 10.0))
    node = _node()

    top_result = strategy._step(
        node=node,
        action=top,
        foods=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
        safety_weight=1.3,
        aggression=1.0,
    )
    bottom_result = strategy._step(
        node=node,
        action=bottom,
        foods=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
        safety_weight=1.3,
        aggression=1.0,
    )

    assert top_result.node.score > bottom_result.node.score + 3.0


def test_local_enemy_set_keeps_nearest_and_split_reachable_predator() -> None:
    strategy = LocalTacticalSearchStrategy()
    distant_predator = EnemyBlob(
        player_id=7,
        blob_id=0,
        x=30.0,
        y=7.0,
        radius=7.0,
    )
    node = _node(enemy=distant_predator)

    assert distant_predator in strategy._local_enemies(node)


def test_local_state_advance_is_clamped_to_the_arena() -> None:
    assert LocalTacticalSearchStrategy._advance_local_position(
        (4.0, 30.0),
        (-1.0, 0.0),
        4.0,
        60.0,
    ) == (4.0, 30.0)


def test_food_field_uses_inverse_square_distance_without_turn_smoothing() -> None:
    strategy = LocalTacticalSearchStrategy()
    targets = (
        LocalTarget("food", (0.0, 1.0), 0.01, (0, 1)),
        LocalTarget("food", (0.0, -3.0), 0.01, (0, 2)),
    )

    direction = strategy._food_field_direction(
        (0.0, 0.0),
        targets,
    )

    assert direction == (0.0, 1.0)


def test_food_field_is_a_root_candidate_for_exact_validation() -> None:
    strategy = LocalTacticalSearchStrategy()
    field_direction = strategy._food_field_direction(
        (30.0, 30.0),
        (LocalTarget("food", (31.0, 32.0), 0.01, (0, 1)),),
    )
    actions = strategy._candidate_actions(
        node=_node(),
        foods=(FoodModel(food_id=1, pos=(31.0, 32.0)),),
        food_targets=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
    )

    expected_key = strategy._action_key(Action(field_direction))
    assert any(strategy._action_key(action) == expected_key for action in actions)


def test_local_targets_use_current_virus_expected_mass_api() -> None:
    strategy = LocalTacticalSearchStrategy()

    targets = strategy._local_targets(
        _node(),
        foods=(),
        viruses=(VirusModel(virus_id=7, pos=(34.0, 30.0), radius=1.5),),
        arena_size=60.0,
    )

    assert any(target.kind == "virus" and target.identity == (1, 7) for target in targets)


def test_almost_lethal_larger_neighbour_is_treated_as_future_predator() -> None:
    strategy = LocalTacticalSearchStrategy()
    own = OwnBlob(blob_id=0, x=30.0, y=30.0, radius=1.259)
    enemy = EnemyBlob(
        player_id=5,
        blob_id=0,
        x=31.1,
        y=30.0,
        radius=1.372,
    )
    node = SearchNode(
        own_blobs=(own,),
        enemies=(enemy,),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )

    assert not can_eat_player_blob(enemy.radius, own.radius)
    assert strategy._is_near_future_predator(own, enemy)
    assert strategy._near_future_predator_escape(node) == (-1.0, 0.0)


def test_exact_root_score_prefers_separation_from_future_predator() -> None:
    strategy = LocalTacticalSearchStrategy(depth=1)
    node = SearchNode(
        own_blobs=(OwnBlob(blob_id=0, x=30.0, y=30.0, radius=1.259),),
        enemies=(
            EnemyBlob(
                player_id=5,
                blob_id=0,
                x=31.1,
                y=30.0,
                radius=1.372,
            ),
        ),
        score=0.0,
        first_direction=(0.0, 1.0),
        first_split=False,
        first_reason="keep",
        last_direction=(0.0, 1.0),
    )

    toward = strategy._step(
        node=node,
        action=Action((1.0, 0.0), reason="angle"),
        foods=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
        safety_weight=1.3,
        aggression=1.0,
    )
    away = strategy._step(
        node=node,
        action=Action((-1.0, 0.0), reason="future_predator_escape"),
        foods=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
        safety_weight=1.3,
        aggression=1.0,
    )

    assert away.node.score > toward.node.score
