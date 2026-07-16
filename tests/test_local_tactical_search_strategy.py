from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from strategies.local_tactical_search import (  # noqa: E402
    LocalRolloutState,
    LocalTacticalSearchStrategy,
)
from strategies.features import can_eat_player_blob, player_speed  # noqa: E402
from strategies.receding_horizon import (  # noqa: E402
    Action,
    EnemyBlob,
    OwnBlob,
    SearchNode,
    _split_chain_attack_reach,
)
from strategies.registry import create_strategy, submission_strategy_spec  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from lib.config.player import MASS_DECAY_RATE  # noqa: E402
from lib.models.blob_model import BlobModel  # noqa: E402
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
    assert strategy._LOCAL_FOOD_LIMIT == 10
    assert strategy._LOCAL_ROOT_LIMIT == 12
    assert strategy._actions_per_node_limit(0) == 2
    assert strategy._TARGET_DIRECTION_LIMIT == 4
    assert strategy._DEEP_DIRECTION_LIMIT == 5
    assert strategy.proxy_refine_limit == 6
    assert strategy.proxy_min_refine == 6
    assert strategy.proxy_coarse_after_seconds == 4.0
    assert not strategy._uses_compute_time_bank()


def test_coarse_safety_mode_skips_the_local_rollout() -> None:
    strategy = LocalTacticalSearchStrategy()
    strategy._competition_coarse_mode = True

    def fail_if_called(**_kwargs):
        raise AssertionError("coarse safety mode must not run the local rollout")

    strategy._rank_roots_by_local_dp = fail_if_called  # type: ignore[method-assign]

    actions = strategy._candidate_actions(
        node=_node(),
        foods=(),
        food_targets=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
    )

    assert actions


def test_coarse_safety_mode_keeps_local_semantic_targets() -> None:
    strategy = LocalTacticalSearchStrategy()
    strategy._competition_coarse_mode = True
    prey = EnemyBlob(7, 0, 33.0, 30.0, 1.0)

    actions = strategy._candidate_actions(
        node=_node(enemy=prey),
        foods=(),
        food_targets=(),
        viruses=(VirusModel(virus_id=7, pos=(30.0, 34.0), radius=1.5),),
        arena_size=60.0,
        first_step=True,
    )

    reasons = {action.reason for action in actions}
    assert "local_prey_probe" in reasons
    assert "local_virus_probe" in reasons


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


def test_fatal_exact_step_skips_features_that_require_a_primary_blob() -> None:
    strategy = LocalTacticalSearchStrategy(depth=1)
    node = SearchNode(
        own_blobs=(OwnBlob(0, 30.0, 30.0, 1.0),),
        enemies=(EnemyBlob(7, 0, 30.0, 30.0, 4.0),),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )

    result = strategy._step(
        node=node,
        action=Action((1.0, 0.0), reason="toward_predator"),
        foods=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
        safety_weight=1.3,
        aggression=1.0,
    )

    assert result.fatal
    assert not result.node.own_blobs


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


def test_food_field_is_a_root_candidate_for_exact_validation() -> None:
    strategy = LocalTacticalSearchStrategy()
    field_direction, _ = strategy._root_food_field_and_targets(
        (30.0, 30.0),
        (FoodModel(food_id=1, pos=(31.0, 32.0)),),
        frozenset(),
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

    assert any(
        target.kind == "virus" and target.identity == (1, 7) for target in targets
    )


def test_prey_target_does_not_prepay_a_later_virus_option() -> None:
    strategy = LocalTacticalSearchStrategy()
    node = SearchNode(
        own_blobs=(OwnBlob(blob_id=0, x=30.0, y=30.0, radius=1.5),),
        enemies=(
            EnemyBlob(
                player_id=3,
                blob_id=0,
                x=32.0,
                y=30.0,
                radius=0.7,
            ),
        ),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )
    virus = VirusModel(virus_id=7, pos=(34.0, 30.0), radius=1.5)

    without_virus = next(
        target
        for target in strategy._local_targets(node, (), (), 60.0)
        if target.kind == "prey"
    )
    with_virus = next(
        target
        for target in strategy._local_targets(node, (), (virus,), 60.0)
        if target.kind == "prey"
    )

    assert with_virus.value == without_virus.value


def test_local_transition_credits_only_food_crossed_by_the_path() -> None:
    strategy = LocalTacticalSearchStrategy()
    state = LocalRolloutState(
        own_blobs=(OwnBlob(blob_id=0, x=30.0, y=30.0, radius=1.0),),
        eaten_food_ids=frozenset(),
    )
    foods = (
        FoodModel(food_id=1, pos=(30.8, 30.0)),
        FoodModel(food_id=2, pos=(30.8, 32.0)),
    )

    evaluation = strategy._local_evaluation_context(
        node=_node(),
        foods=foods,
        viruses=(),
        arena_size=60.0,
        local_enemies=(),
    )
    advanced, reward = strategy._advance_local_state(
        evaluation,
        state,
        (1.0, 0.0),
    )

    assert reward == 0.15 * 0.15
    assert advanced.eaten_food_ids == frozenset({1})


def test_food_contact_events_are_order_independent_and_growth_can_unlock_food() -> None:
    strategy = LocalTacticalSearchStrategy()
    state = LocalRolloutState(
        (OwnBlob(blob_id=0, x=30.0, y=30.0, radius=1.0),),
        frozenset(),
    )
    first = FoodModel(food_id=1, pos=(30.5, 30.0))
    unlocked = FoodModel(food_id=2, pos=(30.0, 31.005))

    def advance(foods):
        evaluation = strategy._local_evaluation_context(
            node=_node(),
            foods=foods,
            viruses=(),
            arena_size=60.0,
            local_enemies=(),
        )
        return strategy._advance_local_state(evaluation, state, (1.0, 0.0))

    forward, forward_gain = advance((first, unlocked))
    reverse, reverse_gain = advance((unlocked, first))

    assert forward == reverse
    assert forward.eaten_food_ids == frozenset({1, 2})
    assert forward_gain == reverse_gain == 2.0 * 0.15**2


def test_growth_wall_clamp_updates_the_remaining_piecewise_path() -> None:
    strategy = LocalTacticalSearchStrategy()
    state = LocalRolloutState(
        (OwnBlob(blob_id=0, x=1.0, y=30.0, radius=1.0),),
        frozenset(),
    )
    evaluation = strategy._local_evaluation_context(
        node=_node(),
        foods=(
            FoodModel(food_id=1, pos=(1.0, 30.0)),
            FoodModel(food_id=2, pos=(2.016, 30.0)),
        ),
        viruses=(),
        arena_size=60.0,
        local_enemies=(),
    )

    advanced, _ = strategy._advance_local_state(
        evaluation,
        state,
        (-1.0, 0.0),
    )

    assert advanced.eaten_food_ids == frozenset({1, 2})


def test_dense_food_event_queue_avoids_quadratic_rescans() -> None:
    strategy = LocalTacticalSearchStrategy()
    blobs = tuple(
        OwnBlob(
            blob_id=index,
            x=10.0 + index % 4 * 3.0,
            y=10.0 + index // 4 * 3.0,
            radius=0.9,
            merge_cooldown=10,
        )
        for index in range(16)
    )
    foods = tuple(FoodModel(food_id=index, pos=(10.0, 10.0)) for index in range(16))
    node = SearchNode(
        own_blobs=blobs,
        enemies=(),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )
    evaluation = strategy._local_evaluation_context(
        node=node,
        foods=foods,
        viruses=(),
        arena_size=60.0,
        local_enemies=(),
    )

    advanced, _ = strategy._advance_local_state(
        evaluation,
        LocalRolloutState(blobs, frozenset()),
        (1.0, 0.0),
    )

    assert len(advanced.eaten_food_ids) == 16
    # The spatial broad phase must retain every contact while sending only
    # swept-capsule candidates to the exact event queue.  The previous full
    # scan performed 16*16 initial checks plus one shrinking rescan per food.
    assert evaluation.contact_checks < 16 * 16 + sum(range(16))


def test_dense_food_and_sixteen_fragment_decision_stays_bounded() -> None:
    strategy = LocalTacticalSearchStrategy()
    blobs = tuple(
        BlobModel(
            blob_id=index,
            pos=(20.0 + index % 4 * 3.0, 20.0 + index // 4 * 3.0),
            radius=0.9,
            merge_cooldown=10,
        )
        for index in range(16)
    )
    total_mass = sum(blob.radius**2 for blob in blobs)
    center = (
        sum(blob.pos[0] * blob.radius**2 for blob in blobs) / total_mass,
        sum(blob.pos[1] * blob.radius**2 for blob in blobs) / total_mass,
    )
    state = SimpleNamespace(
        me=SimpleNamespace(
            player_id=0,
            x=center[0],
            y=center[1],
            radius=math.sqrt(total_mass),
            alive=True,
            blobs={blob.blob_id: blob for blob in blobs},
        ),
        visible_blobs=[],
        visible_food=[
            FoodModel(food_id=index, pos=(20.0, 20.0)) for index in range(16)
        ],
        visible_viruses=[],
        map=SimpleNamespace(size=60.0),
        round=100,
        max_rounds=1400,
        rankings=[0, 1, 2, 3, 4, 5, 6, 7],
        view_center=center,
        vision_size=20.0,
    )

    decision = strategy.choose(
        StrategyContext(
            game=SimpleNamespace(state=state),
            query=SimpleNamespace(update={}),
        )
    )

    assert decision.direction != (0.0, 0.0)
    assert 0 < decision.diagnostics["local_contact_checks"] < 100_000


def test_local_split_preserves_both_fragments_and_total_mass() -> None:
    strategy = LocalTacticalSearchStrategy()
    node = _node()
    start = LocalRolloutState(
        own_blobs=node.own_blobs,
        eaten_food_ids=frozenset(),
    )
    evaluation = strategy._local_evaluation_context(
        node=node,
        foods=(),
        viruses=(),
        arena_size=60.0,
        local_enemies=(),
    )

    advanced, reward = strategy._advance_local_state(
        evaluation,
        start,
        (1.0, 0.0),
        split=True,
    )

    assert reward == 0.0
    assert advanced.split_performed
    assert len(advanced.own_blobs) == 2
    assert math.isclose(
        advanced.mass,
        start.mass * (1.0 - MASS_DECAY_RATE),
    )


def test_local_transition_cache_uses_the_exact_state_for_one_turn() -> None:
    strategy = LocalTacticalSearchStrategy()
    node = _node()
    start = LocalRolloutState(node.own_blobs, frozenset())
    evaluation = strategy._local_evaluation_context(
        node=node,
        foods=(),
        viruses=(),
        arena_size=60.0,
        local_enemies=(),
    )

    first, _ = strategy._advance_local_state(evaluation, start, (1.0, 0.0))
    second, _ = strategy._advance_local_state(evaluation, start, (1.0, 0.0))

    assert first is second
    assert len(evaluation.transition_cache) == 1


def test_terminal_capture_can_use_a_non_primary_fragment() -> None:
    strategy = LocalTacticalSearchStrategy()
    enemy = EnemyBlob(
        player_id=7,
        blob_id=0,
        x=31.0,
        y=30.0,
        radius=1.5,
    )
    state = LocalRolloutState(
        own_blobs=(
            OwnBlob(blob_id=0, x=8.0, y=8.0, radius=4.0),
            OwnBlob(blob_id=1, x=30.0, y=30.0, radius=2.0),
        ),
        eaten_food_ids=frozenset(),
    )

    source, probability = strategy._best_local_capture(state, enemy, 60.0)

    assert source is not None
    assert source.blob_id == 1
    assert probability > 0.0


def test_rollout_stabilises_overlapping_own_fragments() -> None:
    strategy = LocalTacticalSearchStrategy()
    state = LocalRolloutState(
        (
            OwnBlob(0, 29.5, 30.0, 2.0, merge_cooldown=10),
            OwnBlob(1, 30.5, 30.0, 2.0, merge_cooldown=10),
        ),
        frozenset(),
    )
    evaluation = strategy._local_evaluation_context(
        node=_node(),
        foods=(),
        viruses=(),
        arena_size=60.0,
        local_enemies=(),
    )

    advanced, _ = strategy._advance_local_state(
        evaluation,
        state,
        (0.0, 1.0),
    )

    first, second = advanced.own_blobs
    assert math.dist(first.pos, second.pos) >= first.radius + second.radius


def test_one_threatened_split_fragment_is_a_mass_penalty_not_global_fatal() -> None:
    strategy = LocalTacticalSearchStrategy()
    predator = EnemyBlob(
        player_id=7,
        blob_id=0,
        x=31.0,
        y=30.0,
        radius=4.0,
    )
    node = _node(enemy=predator)
    state = LocalRolloutState(
        own_blobs=(
            OwnBlob(blob_id=0, x=30.0, y=30.0, radius=2.0),
            OwnBlob(blob_id=1, x=10.0, y=10.0, radius=2.0),
        ),
        eaten_food_ids=frozenset(),
        split_performed=True,
    )
    evaluation = strategy._local_evaluation_context(
        node=node,
        foods=(),
        viruses=(),
        arena_size=60.0,
        local_enemies=(predator,),
    )

    safety = strategy._local_safety_value(
        evaluation,
        state=state,
        depth=1,
    )
    assert safety is not None
    assert safety < 0.0


def test_food_actually_eaten_on_path_can_unlock_terminal_virus_option() -> None:
    strategy = LocalTacticalSearchStrategy()
    node = SearchNode(
        own_blobs=(
            OwnBlob(
                blob_id=0,
                x=30.0,
                y=30.0,
                radius=math.sqrt(2.69),
            ),
        ),
        enemies=(),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )
    food = FoodModel(food_id=1, pos=(30.8, 30.0))
    virus = VirusModel(virus_id=7, pos=(32.0, 30.0), radius=1.5)
    start = LocalRolloutState(
        own_blobs=node.own_blobs,
        eaten_food_ids=frozenset(),
    )
    evaluation = strategy._local_evaluation_context(
        node=node,
        foods=(food,),
        viruses=(virus,),
        arena_size=60.0,
        local_enemies=(),
    )

    before = strategy._terminal_option_value(
        evaluation,
        state=start,
    )
    advanced, _ = strategy._advance_local_state(
        evaluation,
        start,
        (1.0, 0.0),
    )
    after = strategy._terminal_option_value(
        evaluation,
        state=advanced,
    )

    assert before < 0.1
    assert after > 1.0


def test_terminal_options_take_best_virus_instead_of_summing_them() -> None:
    strategy = LocalTacticalSearchStrategy()
    node = _node()
    state = LocalRolloutState(
        own_blobs=node.own_blobs,
        eaten_food_ids=frozenset(),
    )
    first = VirusModel(virus_id=7, pos=(34.0, 30.0), radius=1.5)
    second = VirusModel(virus_id=8, pos=(34.0, 30.0), radius=1.5)

    one_evaluation = strategy._local_evaluation_context(
        node=node,
        foods=(),
        viruses=(first,),
        arena_size=60.0,
        local_enemies=(),
    )
    two_evaluation = strategy._local_evaluation_context(
        node=node,
        foods=(),
        viruses=(first, second),
        arena_size=60.0,
        local_enemies=(),
    )
    one = strategy._terminal_option_value(
        one_evaluation,
        state=state,
    )
    two = strategy._terminal_option_value(
        two_evaluation,
        state=state,
    )

    assert two == one


def test_terminal_capture_and_followup_are_atomic_options() -> None:
    strategy = LocalTacticalSearchStrategy()
    enemy = EnemyBlob(7, 0, 31.0, 30.0, 1.0)
    node = _node(enemy=enemy)
    state = LocalRolloutState(node.own_blobs, frozenset())
    evaluation = strategy._local_evaluation_context(
        node=node,
        foods=(),
        viruses=(VirusModel(virus_id=7, pos=(32.0, 30.0), radius=1.5),),
        arena_size=60.0,
        local_enemies=(),
    )
    strategy._prey_capture_probability = lambda *_args: 1.0  # type: ignore[method-assign]
    strategy._virus_expected_mass = lambda *_args: 10.0  # type: ignore[method-assign]

    value = strategy._terminal_option_value(evaluation, state=state)

    assert value == 10.0


def test_post_capture_virus_projection_removes_the_captured_enemy() -> None:
    strategy = LocalTacticalSearchStrategy()
    enemy = EnemyBlob(7, 0, 31.0, 30.0, 0.9)
    node = SearchNode(
        own_blobs=(OwnBlob(0, 30.0, 30.0, 1.5),),
        enemies=(enemy,),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )
    evaluation = strategy._local_evaluation_context(
        node=node,
        foods=(),
        viruses=(VirusModel(virus_id=7, pos=(32.0, 30.0), radius=1.5),),
        arena_size=60.0,
        local_enemies=(),
    )
    seen_projection: list[tuple[tuple[tuple[int, int], ...], tuple[int, ...]]] = []

    def expected_mass(projected, *_args):
        seen_projection.append(
            (
                tuple(other.key for other in projected.enemies),
                tuple(blob.blob_id for blob in projected.own_blobs),
            )
        )
        return 5.0

    strategy._prey_capture_probability = lambda *_args: 1.0  # type: ignore[method-assign]
    strategy._virus_expected_mass = expected_mass  # type: ignore[method-assign]
    strategy._stabilise_own_blobs = (  # type: ignore[method-assign]
        lambda _blobs, _arena: [OwnBlob(99, 31.0, 30.0, 2.0)]
    )

    strategy._terminal_option_value(
        evaluation,
        state=LocalRolloutState(node.own_blobs, frozenset()),
    )

    assert seen_projection == [((), (99,))]


def test_terminal_forage_uses_the_nearest_fragment() -> None:
    strategy = LocalTacticalSearchStrategy()
    state = LocalRolloutState(
        (
            OwnBlob(0, 8.0, 8.0, 4.0),
            OwnBlob(1, 30.0, 30.0, 1.0),
        ),
        frozenset(),
    )
    evaluation = strategy._local_evaluation_context(
        node=_node(),
        foods=(FoodModel(food_id=1, pos=(30.5, 30.0)),),
        viruses=(),
        arena_size=60.0,
        local_enemies=(),
    )

    value = strategy._terminal_option_value(evaluation, state=state)

    assert value == 0.15**2


def test_local_targets_only_keep_positive_two_step_opportunities() -> None:
    strategy = LocalTacticalSearchStrategy()
    node = SearchNode(
        own_blobs=(OwnBlob(0, 30.0, 30.0, 1.5),),
        enemies=(
            EnemyBlob(7, 0, 32.0, 30.0, 0.5),
            EnemyBlob(8, 0, 40.0, 30.0, 0.5),
        ),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )

    targets = strategy._local_targets(
        node,
        foods=(),
        viruses=(VirusModel(virus_id=7, pos=(40.0, 30.0), radius=1.0),),
        arena_size=60.0,
    )

    identities = {target.identity for target in targets}
    assert (9, 0) in identities
    assert (10, 0) not in identities
    assert (1, 7) not in identities


def test_setup_probe_cannot_pool_food_from_mutually_exclusive_routes() -> None:
    strategy = LocalTacticalSearchStrategy()
    target_mass = 0.84 / 1.2
    node = SearchNode(
        own_blobs=(OwnBlob(0, 30.0, 30.0, 0.9),),
        enemies=(EnemyBlob(7, 0, 31.2, 30.0, math.sqrt(target_mass)),),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )
    foods = (
        FoodModel(food_id=1, pos=(30.0, 31.0)),
        FoodModel(food_id=2, pos=(30.0, 29.0)),
    )

    targets = strategy._local_targets(node, foods, (), 60.0)

    assert not any(target.kind == "prey" for target in targets)


def test_setup_probe_cannot_eat_food_after_target_contact() -> None:
    strategy = LocalTacticalSearchStrategy()
    node = SearchNode(
        own_blobs=(OwnBlob(0, 30.0, 30.0, 1.0),),
        enemies=(),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )
    virus = VirusModel(
        virus_id=7,
        pos=(33.0, 30.0),
        radius=0.915607,
    )
    food = FoodModel(food_id=1, pos=(33.8, 30.0))

    targets = strategy._local_targets(node, (food,), (virus,), 60.0)

    assert not any(target.kind == "virus" for target in targets)


def test_active_viruses_are_filtered_once_to_local_option_range() -> None:
    strategy = LocalTacticalSearchStrategy()
    near = VirusModel(virus_id=1, pos=(35.0, 30.0), radius=1.5)
    far = VirusModel(virus_id=2, pos=(59.0, 30.0), radius=1.5)

    assert strategy._relevant_local_viruses(_node(), (near, far)) == (near,)


def test_predator_cannot_move_and_split_in_the_same_tick() -> None:
    strategy = LocalTacticalSearchStrategy()
    own = OwnBlob(0, 20.0, 30.0, 1.0)
    enemy = EnemyBlob(7, 0, 20.0, 30.0, 2.0)
    chain_reach = _split_chain_attack_reach(enemy.radius, own.radius)
    speed = player_speed(enemy.radius)
    enemy = EnemyBlob(7, 0, own.x + chain_reach + 0.25 * speed, own.y, 2.0)
    node = SearchNode(
        own_blobs=(own,),
        enemies=(enemy,),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )
    state = LocalRolloutState(node.own_blobs, frozenset())
    evaluation = strategy._local_evaluation_context(
        node=node,
        foods=(),
        viruses=(),
        arena_size=60.0,
        local_enemies=(enemy,),
    )

    assert math.dist(own.pos, enemy.pos) < chain_reach + speed
    assert strategy._local_safety_value(evaluation, state=state, depth=1) is not None


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


def test_advisor_physical_root_matches_full_step_physics_randomised() -> None:
    rng = random.Random(20260715)

    for _ in range(30):
        own_count = rng.choice((1, 2, 4, 8, 16))
        own = tuple(
            OwnBlob(
                blob_id=index,
                x=rng.uniform(5.0, 55.0),
                y=rng.uniform(5.0, 55.0),
                radius=rng.uniform(0.65, 2.8),
                merge_cooldown=rng.randrange(0, 20),
                eject_vx=rng.uniform(-0.3, 0.3),
                eject_vy=rng.uniform(-0.3, 0.3),
            )
            for index in range(own_count)
        )
        enemies = tuple(
            EnemyBlob(
                player_id=1 + index // 2,
                blob_id=index,
                x=rng.uniform(5.0, 55.0),
                y=rng.uniform(5.0, 55.0),
                radius=rng.uniform(0.65, 3.2),
                direction=(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)),
                merge_cooldown=rng.randrange(0, 20),
            )
            for index in range(rng.randrange(0, 7))
        )
        node = SearchNode(
            own_blobs=own,
            enemies=enemies,
            score=0.0,
            first_direction=(1.0, 0.0),
            first_split=False,
            first_reason="keep",
            last_direction=(1.0, 0.0),
        )
        action = Action(
            (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)),
            split=rng.choice((False, True)),
            reason="parity_probe",
        )
        foods = tuple(
            FoodModel(
                food_id=index,
                pos=(rng.uniform(0.0, 60.0), rng.uniform(0.0, 60.0)),
            )
            for index in range(4)
        )
        viruses = tuple(
            VirusModel(
                virus_id=index,
                pos=(rng.uniform(3.0, 57.0), rng.uniform(3.0, 57.0)),
                radius=1.5,
            )
            for index in range(2)
        )
        full = LocalTacticalSearchStrategy()
        physical = LocalTacticalSearchStrategy()

        full._step(
            node=node,
            action=action,
            foods=foods,
            viruses=viruses,
            arena_size=60.0,
            first_step=True,
            safety_weight=1.0,
            aggression=1.0,
        )
        physical._advisor_physical_root_step(
            node=node,
            action=action,
            foods=foods,
            viruses=viruses,
            arena_size=60.0,
            first_step=True,
            safety_weight=1.0,
            aggression=1.0,
        )
        action_key = full._action_key(action)
        expected = full.root_transition_results[action_key]
        actual = physical.root_transition_results[action_key]

        assert actual.fatal == expected.fatal
        assert actual.movement_efficiency == expected.movement_efficiency
        assert actual.hazard_summary == expected.hazard_summary
        assert actual.node.own_blobs == expected.node.own_blobs
        assert actual.node.enemies == expected.node.enemies
        assert actual.node.eaten_food_ids == expected.node.eaten_food_ids
        assert actual.node.consumed_virus_ids == expected.node.consumed_virus_ids
        assert actual.node.projected_food == expected.node.projected_food
        assert actual.node.projected_captures == expected.node.projected_captures
        assert actual.node.projected_viruses == expected.node.projected_viruses
        assert actual.node.min_safety_margin == expected.node.min_safety_margin
        assert (
            physical.root_transition_summaries[action_key]
            == (full.root_transition_summaries[action_key])
        )
