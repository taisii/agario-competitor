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
from engine.state.blob_state import BlobState  # noqa: E402
from engine.state.player_state import PlayerState  # noqa: E402
from engine.state.state_mutator import StateMutator  # noqa: E402
from strategies.receding_horizon import (  # noqa: E402
    Action,
    EnemyBlob,
    EnemyTrack,
    OwnBlob,
    ReplayDominanceStrategy,
    SearchNode,
    StepResult,
)
from strategies.registry import (  # noqa: E402
    available_strategy_names,
    create_strategy,
)


def test_legacy_receding_horizon_names_resolve_without_duplicate_list_entries() -> None:
    assert create_strategy("champion").name == "threat_aware_receding_horizon"
    assert "champion" not in available_strategy_names()


def test_replay_dominance_is_a_distinct_registered_strategy() -> None:
    strategy = create_strategy("replay_dominance")

    assert isinstance(strategy, ReplayDominanceStrategy)
    assert strategy.name == "replay_dominance"
    assert "replay_dominance" in available_strategy_names()


def test_replay_dominance_compares_two_roots_before_deadline() -> None:
    strategy = ReplayDominanceStrategy()

    assert strategy.minimum_root_actions == 2
    assert strategy._required_actions_for_depth(0) == 2
    assert strategy._required_actions_for_depth(1) == 1


def test_replay_dominance_uses_blob_scaled_deterministic_transition_budget() -> None:
    strategy = ReplayDominanceStrategy()

    assert strategy._uses_compute_time_bank() is True
    assert strategy.depth == 1
    assert strategy.width == 1
    assert strategy._transition_budget(1) == 6
    assert strategy._transition_budget(2) == 3
    assert strategy._transition_budget(4) == 2
    assert strategy._transition_budget(16) == 2
    assert strategy._transition_budget(1, 12) == 2


def test_replay_dominance_preserves_explicit_depth_and_width_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BOT_RECEDING_HORIZON_DEPTH", "4")
    monkeypatch.setenv("BOT_RECEDING_HORIZON_WIDTH", "3")

    strategy = ReplayDominanceStrategy()

    assert strategy.depth == 4
    assert strategy.width == 3


def test_replay_dominance_stops_before_generating_an_unusable_depth() -> None:
    assert ReplayDominanceStrategy._depth_start_stop_reason(
        depth_index=1,
        transitions_evaluated=6,
        transition_budget=6,
        uses_time_bank=True,
        deadline=float("inf"),
    ) == "transition_budget"
    assert ReplayDominanceStrategy._depth_start_stop_reason(
        depth_index=1,
        transitions_evaluated=5,
        transition_budget=6,
        uses_time_bank=True,
        deadline=0.0,
    ) == "deadline"


def test_enemy_memory_threat_model_includes_future_virus_fragments() -> None:
    strategy = ReplayDominanceStrategy()
    own = OwnBlob(blob_id=0, x=30.0, y=30.0, radius=math.sqrt(32.0))
    virus = VirusModel(virus_id=1, pos=(34.0, 30.0), radius=1.5)
    without_virus = [
        radius for _, radius in strategy._exposed_own_radii((own,))
    ]
    radii = [
        radius
        for _, radius in strategy._exposed_own_radii((own,), (virus,))
    ]

    assert own.radius in radii
    assert own.radius / math.sqrt(2.0) in radii
    assert math.sqrt((own.mass + 1.5**2) / 16.0) in radii
    assert any(radius < 1.5 for radius in radii)
    assert all(radius >= own.radius / math.sqrt(2.0) for radius in without_virus)

    small_stale = EnemyTrack(1, 1, 40.0, 30.0, 1.32, (-1.0, 0.0), 0)
    sweeping_stale = replace(small_stale, blob_id=2, radius=4.1)

    assert not strategy._stale_enemy_can_threaten_transition(
        small_stale, (own,), (virus,)
    )
    assert strategy._stale_enemy_can_threaten_transition(
        sweeping_stale, (own,), (virus,)
    )


def test_enemy_memory_does_not_predict_track_replaced_by_visible_blob(
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

    def fail_if_predicted(_radius):
        raise AssertionError("visible track prediction must be skipped")

    monkeypatch.setattr(
        "strategies.receding_horizon.player_speed",
        fail_if_predicted,
    )
    enemies = strategy._update_enemy_memory(
        context,
        (OwnBlob(blob_id=0, x=30.0, y=30.0, radius=1.0),),
        60.0,
    )

    assert enemies[0].pos == visible.pos


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

    assert "virus" not in max(
        results,
        key=strategy._terminal_score,
    ).first_reason


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
    reachable = strategy._prey_capture_probability(
        corner_own, corner_pinned, 60.0
    )

    assert unreachable < 1e-10
    assert 0.3 < reachable < 0.5


def test_replay_dominance_state_value_rewards_absolute_mass_linearly() -> None:
    strategy = ReplayDominanceStrategy()

    def utility(mass: float) -> float:
        node = SearchNode(
            own_blobs=(
                OwnBlob(blob_id=0, x=30.0, y=30.0, radius=math.sqrt(mass)),
            ),
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

    after, _, _ = strategy._resolve_own_viruses(
        own_blobs=stabilised,
        viruses=(virus,),
        consumed_virus_ids=consumed,
        arena_size=60.0,
    )

    assert consumed == {50}
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
