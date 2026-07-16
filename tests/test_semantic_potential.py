from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from lib.models.blob_model import BlobModel, VisibleBlobModel  # noqa: E402
from lib.models.food_model import FoodModel  # noqa: E402
from lib.models.virus_model import VirusModel  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
import strategies.semantic_potential as semantic_module  # noqa: E402
from strategies.semantic_potential import (  # noqa: E402
    SemanticLookaheadStrategy,
    SemanticPotentialStrategy,
    TargetMemory,
    _continuation_candidate,
    _fan_weight,
    _project_action_blobs,
    _project_one_step_outcome,
)


def _enemy(
    *,
    player_id: int,
    pos: tuple[float, float],
    radius: float,
    blob_id: int = 0,
) -> VisibleBlobModel:
    return VisibleBlobModel(
        player_id=player_id,
        team_id=player_id,
        blob_id=blob_id,
        pos=pos,
        radius=radius,
    )


def _context(
    own: tuple[BlobModel, ...],
    *,
    foods: tuple[FoodModel, ...] = (),
    viruses: tuple[VirusModel, ...] = (),
    enemies: tuple[VisibleBlobModel, ...] = (),
    arena_size: float = 60.0,
    round_number: int = 0,
    max_rounds: int = 1400,
) -> StrategyContext:
    state = SimpleNamespace(
        me=SimpleNamespace(player_id=0, blobs={blob.blob_id: blob for blob in own}),
        visible_food=list(foods),
        visible_viruses=list(viruses),
        visible_blobs=list(enemies),
        map=SimpleNamespace(size=arena_size),
        round=round_number,
        max_rounds=max_rounds,
    )
    return StrategyContext(
        game=SimpleNamespace(state=state),
        query=SimpleNamespace(),
    )


def test_candidate_set_contains_each_available_semantic_slot_once() -> None:
    strategy = SemanticPotentialStrategy()
    own = (BlobModel(blob_id=0, pos=(4.0, 30.0), radius=3.0),)
    foods = (
        FoodModel(food_id=1, pos=(6.0, 30.0)),
        FoodModel(food_id=2, pos=(4.0, 34.0)),
    )
    viruses = (
        VirusModel(virus_id=1, pos=(8.0, 28.0), radius=1.5),
        VirusModel(virus_id=2, pos=(8.0, 34.0), radius=1.5),
    )
    enemies = (
        _enemy(player_id=1, pos=(9.0, 30.0), radius=1.0),
        _enemy(player_id=2, pos=(10.0, 26.0), radius=5.0),
    )

    decision = strategy.choose(
        _context(own, foods=foods, viruses=viruses, enemies=enemies)
    )

    assert set(decision.diagnostics["candidate_scores"]) == {
        "continue",
        "nearest_food",
        "second_food",
        "nearest_virus",
        "second_virus",
        "capture_enemy",
        "split_capture",
        "wall_avoidance",
        "escape",
    }
    assert decision.diagnostics["candidate_count"] == 9


def test_field_can_prefer_second_food_leading_to_a_dense_region() -> None:
    strategy = SemanticPotentialStrategy()
    strategy._last_direction = (0.0, 1.0)
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=1.0),)
    foods = (
        FoodModel(food_id=1, pos=(27.0, 30.0)),
        FoodModel(food_id=2, pos=(33.0, 30.0)),
        FoodModel(food_id=3, pos=(34.0, 29.0)),
        FoodModel(food_id=4, pos=(34.0, 31.0)),
        FoodModel(food_id=5, pos=(35.0, 30.0)),
        FoodModel(food_id=6, pos=(35.0, 32.0)),
    )

    decision = strategy.choose(_context(own, foods=foods))

    assert decision.reason == "second_food"
    assert decision.target_id == "2"
    assert decision.direction[0] > 0.0
    assert decision.diagnostics["selected_components"]["food"] > 0.0


def test_lookahead_keeps_quiet_small_mass_food_collection_one_ply() -> None:
    strategy = SemanticLookaheadStrategy()
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=1.0),)
    foods = (FoodModel(food_id=1, pos=(33.0, 30.0)),)

    decision = strategy.choose(_context(own, foods=foods))

    assert decision.reason == "nearest_food"
    assert decision.diagnostics["lookahead"] == {
        "enabled": False,
        "searched_roots": 0,
        "root_limit": 4,
        "estimated_root_work": 10,
        "work_budget": 800,
        "skipped_reason": "irrelevant",
        "food_scale": 1.0,
        "search_configured": True,
        "food_scaling_configured": True,
        "selected": None,
    }


def test_lookahead_skips_expensive_fragmented_tactical_position() -> None:
    strategy = SemanticLookaheadStrategy()
    own = tuple(
        BlobModel(
            blob_id=index,
            pos=(20.0 + (index % 4), 20.0 + (index // 4)),
            radius=1.0,
        )
        for index in range(16)
    )
    prey = _enemy(player_id=1, pos=(25.0, 25.0), radius=0.5)
    foods = tuple(
        FoodModel(food_id=index, pos=(30.0 + index * 0.1, 30.0))
        for index in range(40)
    )

    decision = strategy.choose(_context(own, foods=foods, enemies=(prey,)))

    lookahead = decision.diagnostics["lookahead"]
    assert not lookahead["enabled"]
    assert lookahead["searched_roots"] == 0
    assert lookahead["root_limit"] == 0
    assert lookahead["estimated_root_work"] > lookahead["work_budget"]
    assert lookahead["skipped_reason"] == "work_budget"


def test_lookahead_continuation_falls_back_after_root_consumes_target() -> None:
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=2.0),)

    candidate = _continuation_candidate(
        own=own,
        foods=(),
        viruses=(),
        memory=TargetMemory(kind="food", pos=(31.0, 30.0)),
        fallback=(0.0, 1.0),
    )

    assert candidate.family == "continue"
    assert candidate.target_kind == "momentum"
    assert candidate.direction == (0.0, 1.0)


def test_lookahead_searches_every_bounded_root_in_tactical_position() -> None:
    strategy = SemanticLookaheadStrategy()
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=3.0),)
    prey = _enemy(player_id=1, pos=(35.0, 30.0), radius=1.0)

    decision = strategy.choose(_context(own, enemies=(prey,)))

    lookahead = decision.diagnostics["lookahead"]
    assert lookahead["enabled"]
    assert 1 <= lookahead["searched_roots"] <= min(
        4,
        decision.diagnostics["candidate_count"],
    )
    assert lookahead["selected"]["child_count"] >= 1
    assert lookahead["selected"]["scenario_count"] in {1, 2}


def test_lookahead_ignores_visible_enemy_outside_tactical_horizon() -> None:
    strategy = SemanticLookaheadStrategy()
    own = (BlobModel(blob_id=0, pos=(10.0, 10.0), radius=2.0),)
    distant_prey = _enemy(player_id=1, pos=(50.0, 50.0), radius=1.0)

    decision = strategy.choose(_context(own, enemies=(distant_prey,)))

    assert not decision.diagnostics["lookahead"]["enabled"]
    assert decision.diagnostics["lookahead"]["searched_roots"] == 0


def test_lookahead_refreshes_consumed_root_target_before_second_ply() -> None:
    strategy = SemanticLookaheadStrategy()
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=2.0),)
    food = FoodModel(food_id=1, pos=(32.0, 30.0))
    nearby_enemy = _enemy(player_id=1, pos=(35.0, 30.0), radius=1.0)

    decision = strategy.choose(
        _context(own, foods=(food,), enemies=(nearby_enemy,))
    )

    assert decision.diagnostics["lookahead"]["enabled"]
    assert decision.diagnostics["lookahead"]["searched_roots"] >= 1


def test_lookahead_reduces_food_routing_value_only_at_large_mass() -> None:
    foods = (
        FoodModel(food_id=1, pos=(34.0, 30.0)),
        FoodModel(food_id=2, pos=(35.0, 30.0)),
        FoodModel(food_id=3, pos=(36.0, 30.0)),
    )
    small = SemanticLookaheadStrategy().choose(
        _context(
            (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=2.0),),
            foods=foods,
        )
    )
    large = SemanticLookaheadStrategy().choose(
        _context(
            (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=7.0),),
            foods=foods,
        )
    )
    unscaled_large = SemanticPotentialStrategy().choose(
        _context(
            (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=7.0),),
            foods=foods,
        )
    )

    assert small.diagnostics["lookahead"]["food_scale"] == 1.0
    assert 0.65 <= large.diagnostics["lookahead"]["food_scale"] < 0.75
    assert (
        large.diagnostics["selected_components"]["intent"]
        < unscaled_large.diagnostics["selected_components"]["intent"]
    )


def test_lookahead_features_can_be_ablated_independently(monkeypatch) -> None:
    monkeypatch.setenv("SEMANTIC_LOOKAHEAD_SEARCH", "0")
    monkeypatch.setenv("SEMANTIC_LARGE_MASS_FOOD_SCALE", "0")
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=7.0),)
    food = FoodModel(food_id=1, pos=(34.0, 30.0))
    prey = _enemy(player_id=1, pos=(38.0, 30.0), radius=1.0)

    baseline = SemanticPotentialStrategy().choose(
        _context(own, foods=(food,), enemies=(prey,))
    )
    ablated = SemanticLookaheadStrategy().choose(
        _context(own, foods=(food,), enemies=(prey,))
    )

    assert ablated.reason == baseline.reason
    assert ablated.score == baseline.score
    assert not ablated.diagnostics["lookahead"]["search_configured"]
    assert not ablated.diagnostics["lookahead"]["food_scaling_configured"]


def test_immediate_predator_excludes_catastrophic_resource_direction() -> None:
    strategy = SemanticPotentialStrategy()
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=1.0),)
    predator = _enemy(player_id=1, pos=(39.8, 30.0), radius=3.0)
    food = FoodModel(food_id=1, pos=(40.8, 30.0))

    decision = strategy.choose(_context(own, foods=(food,), enemies=(predator,)))

    assert decision.reason == "escape"
    assert decision.direction[0] < 0.0
    assert decision.diagnostics["safe_candidate_count"] >= 1


def test_quiet_edge_keeps_unblocked_tangent_without_wall_avoidance() -> None:
    strategy = SemanticPotentialStrategy()
    strategy._last_direction = (0.0, 1.0)
    own = (BlobModel(blob_id=0, pos=(1.0, 30.0), radius=1.0),)

    decision = strategy.choose(_context(own))

    assert "wall_avoidance" not in decision.diagnostics["candidate_scores"]
    assert decision.reason == "continue"
    assert decision.direction == (0.0, 1.0)
    assert decision.diagnostics["selected_components"]["wall"] == 0.0


def test_hard_clipped_quiet_continuation_recovers_without_general_wall_bias() -> None:
    strategy = SemanticPotentialStrategy()
    strategy._last_direction = (-1.0, 0.0)
    own = (BlobModel(blob_id=0, pos=(1.0, 30.0), radius=1.0),)

    decision = strategy.choose(_context(own))

    assert "wall_avoidance" not in decision.diagnostics["candidate_scores"]
    assert "continue" not in decision.diagnostics["candidate_scores"]
    assert decision.reason == "boundary_recovery"
    assert decision.direction[0] > 0.0


def test_corner_food_unreachable_by_current_blob_is_not_a_route() -> None:
    strategy = SemanticPotentialStrategy()
    strategy._last_direction = (1.0, -1.0)
    own = (BlobModel(blob_id=0, pos=(55.0, 5.0), radius=5.0),)
    unreachable = FoodModel(food_id=1, pos=(60.0, 0.0))
    reachable = FoodModel(food_id=2, pos=(48.0, 8.0))

    decision = strategy.choose(_context(own, foods=(unreachable, reachable)))

    assert decision.target_id == "2"
    assert decision.direction[0] < 0.0
    assert decision.diagnostics["unreachable_stationary_resources"] == 1


def test_wall_deflection_exists_only_when_predator_blocks_escape() -> None:
    strategy = SemanticPotentialStrategy()
    own = (BlobModel(blob_id=0, pos=(1.0, 30.0), radius=1.0),)
    predator = _enemy(player_id=1, pos=(8.0, 30.0), radius=3.0)

    decision = strategy.choose(_context(own, enemies=(predator,)))

    assert decision.reason == "wall_avoidance"
    assert abs(decision.direction[1]) > 0.9
    assert decision.diagnostics["selected_components"]["wall"] < 0.0


def test_food_route_beats_empty_momentum_in_quiet_state() -> None:
    strategy = SemanticPotentialStrategy()
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=1.0),)
    food = FoodModel(food_id=1, pos=(25.0, 30.0))

    decision = strategy.choose(_context(own, foods=(food,)))

    assert decision.reason == "nearest_food"
    assert decision.direction[0] < 0.0


def test_committed_food_target_prevents_nearby_target_oscillation() -> None:
    strategy = SemanticPotentialStrategy()
    strategy._last_direction = (0.0, 1.0)
    first_own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=1.0),)
    committed = FoodModel(food_id=1, pos=(25.0, 30.0))
    other = FoodModel(food_id=2, pos=(36.0, 30.0))
    first = strategy.choose(_context(first_own, foods=(committed, other)))

    second_own = (BlobModel(blob_id=0, pos=(29.0, 30.0), radius=1.0),)
    same_committed_position = FoodModel(food_id=99, pos=(25.0, 30.0))
    newly_closer = FoodModel(food_id=3, pos=(31.5, 30.0))
    second = strategy.choose(
        _context(second_own, foods=(same_committed_position, newly_closer))
    )

    assert first.target_id == "1"
    assert first.direction[0] < 0.0
    assert second.reason == "continue"
    assert second.target_id == "99"
    assert second.direction[0] < 0.0


def test_fragment_is_valid_prey_even_when_whole_enemy_player_is_too_large() -> None:
    strategy = SemanticPotentialStrategy()
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=3.0),)
    fragments = (
        _enemy(player_id=1, blob_id=0, pos=(34.0, 30.0), radius=1.0),
        _enemy(player_id=1, blob_id=1, pos=(39.0, 30.0), radius=2.5),
    )
    safe_food = FoodModel(food_id=1, pos=(25.0, 30.0))

    decision = strategy.choose(_context(own, foods=(safe_food,), enemies=fragments))

    assert "capture_enemy" in decision.diagnostics["candidate_scores"]
    assert decision.reason == "capture_enemy"
    assert decision.target_id in {"1:0", "1:1"}
    assert not decision.split


def test_growth_phase_prioritises_food_before_virus_threshold() -> None:
    strategy = SemanticPotentialStrategy()
    strategy._last_direction = (0.0, 1.0)
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=1.0),)
    food = FoodModel(food_id=1, pos=(25.0, 30.0))
    virus = VirusModel(virus_id=1, pos=(35.0, 30.0), radius=1.5)

    decision = strategy.choose(_context(own, foods=(food,), viruses=(virus,)))

    assert decision.diagnostics["mass_phase"] == "growth"
    assert decision.target_kind == "food"
    assert decision.direction[0] < 0.0


def test_mixed_phase_prefers_consumable_virus_over_equally_distant_food() -> None:
    strategy = SemanticPotentialStrategy()
    strategy._last_direction = (0.0, 1.0)
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=2.0),)
    food = FoodModel(food_id=1, pos=(25.0, 30.0))
    virus = VirusModel(virus_id=1, pos=(35.0, 30.0), radius=1.5)

    decision = strategy.choose(_context(own, foods=(food,), viruses=(virus,)))

    assert decision.diagnostics["mass_phase"] == "mixed"
    assert decision.target_kind == "virus"
    assert decision.direction[0] > 0.0


def test_hunter_phase_ignores_food_when_safe_enemy_is_available() -> None:
    strategy = SemanticPotentialStrategy()
    strategy._last_direction = (0.0, 1.0)
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=4.0),)
    food = FoodModel(food_id=1, pos=(27.0, 30.0))
    prey = _enemy(player_id=1, pos=(36.0, 30.0), radius=1.5)

    decision = strategy.choose(_context(own, foods=(food,), enemies=(prey,)))

    assert decision.diagnostics["mass_phase"] == "hunter"
    assert decision.target_kind == "prey"
    assert decision.direction[0] > 0.0


def test_growth_phase_chases_edible_enemy_instead_of_distant_food() -> None:
    strategy = SemanticPotentialStrategy()
    strategy._last_direction = (0.0, 1.0)
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=1.0),)
    food = FoodModel(food_id=1, pos=(25.0, 30.0))
    prey = _enemy(player_id=1, pos=(34.0, 30.0), radius=0.5)

    decision = strategy.choose(_context(own, foods=(food,), enemies=(prey,)))

    assert decision.diagnostics["mass_phase"] == "growth"
    assert decision.target_kind == "prey"
    assert decision.direction[0] > 0.0


def test_moving_prey_is_led_using_its_previous_direction() -> None:
    strategy = SemanticPotentialStrategy()
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=1.0),)
    previous_prey = _enemy(player_id=1, pos=(32.0, 29.0), radius=0.5)
    current_prey = _enemy(player_id=1, pos=(32.0, 30.0), radius=0.5)

    strategy.choose(_context(own, enemies=(previous_prey,)))
    decision = strategy.choose(_context(own, enemies=(current_prey,)))

    assert decision.reason == "intercept_enemy"
    assert decision.direction[0] > 0.0
    assert decision.direction[1] > 0.0


def test_prey_moving_away_faster_than_hunter_is_not_a_capture_candidate() -> None:
    strategy = SemanticPotentialStrategy()
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=1.0),)
    previous_prey = _enemy(player_id=1, pos=(32.0, 30.0), radius=0.5)
    current_prey = _enemy(player_id=1, pos=(33.0, 30.0), radius=0.5)

    strategy.choose(_context(own, enemies=(previous_prey,)))
    decision = strategy.choose(_context(own, enemies=(current_prey,)))

    assert "intercept_enemy" not in decision.diagnostics["candidate_scores"]
    assert "capture_enemy" not in decision.diagnostics["candidate_scores"]
    assert decision.target_kind != "prey"


def test_crossing_prey_remains_a_capture_candidate_when_distance_can_close() -> None:
    strategy = SemanticPotentialStrategy()
    own = (BlobModel(blob_id=0, pos=(31.74, 3.57), radius=1.10),)
    previous_prey = _enemy(player_id=6, pos=(36.663, 11.601), radius=0.972)
    current_prey = _enemy(player_id=6, pos=(36.61, 12.62), radius=0.972)

    strategy.choose(_context(own, enemies=(previous_prey,)))
    decision = strategy.choose(_context(own, enemies=(current_prey,)))

    assert decision.reason == "intercept_enemy"
    assert decision.target_kind == "prey"
    assert decision.direction[0] > 0.0
    assert decision.direction[1] > 0.0


def test_closing_predator_outside_near_term_reach_does_not_force_early_escape() -> None:
    strategy = SemanticPotentialStrategy()
    own = (BlobModel(blob_id=0, pos=(20.0, 30.0), radius=1.0),)
    previous_predator = _enemy(player_id=1, pos=(41.0, 30.0), radius=3.0)
    current_predator = _enemy(player_id=1, pos=(40.0, 30.0), radius=3.0)

    strategy.choose(_context(own, enemies=(previous_predator,)))
    decision = strategy.choose(_context(own, enemies=(current_predator,)))

    assert "escape" not in decision.diagnostics["candidate_scores"]


def test_reachable_prey_adds_and_selects_single_split_capture() -> None:
    strategy = SemanticPotentialStrategy()
    strategy._last_direction = (0.0, 1.0)
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=3.0),)
    prey = _enemy(player_id=1, pos=(37.0, 30.0), radius=1.5)

    decision = strategy.choose(_context(own, enemies=(prey,)))

    assert decision.reason == "split_capture"
    assert decision.split
    assert decision.diagnostics["split_depth"] == 1


def test_observed_motion_enables_split_that_overlaps_next_enemy_position() -> None:
    strategy = SemanticPotentialStrategy()
    own = (BlobModel(blob_id=0, pos=(52.12, 8.07), radius=3.326),)
    previous_prey = _enemy(
        player_id=4,
        pos=(42.906, 5.954),
        radius=1.040,
    )
    current_prey = _enemy(
        player_id=4,
        pos=(43.22, 6.92),
        radius=1.040,
    )

    strategy.choose(_context(own, enemies=(previous_prey,)))
    decision = strategy.choose(_context(own, enemies=(current_prey,)))

    assert decision.reason == "split_capture"
    assert decision.split
    assert decision.diagnostics["secured_one_step_mass"]["enemy"] > 1.0


def test_observed_motion_does_not_enable_split_when_child_cannot_eat_target() -> None:
    strategy = SemanticPotentialStrategy()
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=1.449),)
    previous_prey = _enemy(player_id=4, pos=(34.0, 29.0), radius=1.064)
    current_prey = _enemy(player_id=4, pos=(34.5, 30.0), radius=1.064)

    strategy.choose(_context(own, enemies=(previous_prey,)))
    decision = strategy.choose(_context(own, enemies=(current_prey,)))

    assert "split_capture" not in decision.diagnostics["candidate_scores"]
    assert not decision.split


def test_split_harvests_reachable_subset_of_large_fragmented_player() -> None:
    strategy = SemanticPotentialStrategy()
    own = (BlobModel(blob_id=0, pos=(20.0, 30.0), radius=4.02),)
    fragments = tuple(
        _enemy(player_id=1, blob_id=index, pos=(x, 30.0), radius=2.12)
        for index, x in enumerate(
            (27.5, 29.5, 31.5, 33.5, 38.0, 40.0, 42.0, 44.0, 46.0, 48.0)
        )
    )

    decision = strategy.choose(_context(own, enemies=fragments))

    secured = decision.diagnostics["secured_one_step_mass"]["enemy"]
    whole_player_mass = sum(blob.radius * blob.radius for blob in fragments)
    assert decision.reason == "split_capture"
    assert decision.split
    assert 3.0 * 2.12**2 * 0.99 < secured < whole_player_mass


def test_virus_contact_is_resolved_before_split_capture() -> None:
    strategy = SemanticPotentialStrategy()
    own = (BlobModel(blob_id=0, pos=(20.0, 30.0), radius=3.0),)
    prey = _enemy(player_id=1, pos=(27.0, 30.0), radius=1.5)
    virus = VirusModel(virus_id=1, pos=(26.5, 30.0), radius=1.5)

    outcome = _project_one_step_outcome(
        own=own,
        direction=(1.0, 0.0),
        foods=(),
        viruses=(virus,),
        enemies=(prey,),
        arena_size=60.0,
    )
    decision = strategy.choose(_context(own, viruses=(virus,), enemies=(prey,)))

    assert outcome.virus_mass_gained == 2.25
    assert outcome.enemy_mass_gained == 0.0
    assert outcome.own_mass_lost > 0.0
    assert not decision.split


def test_non_split_virus_contact_is_scored_after_fragmentation() -> None:
    strategy = SemanticPotentialStrategy()
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=3.0),)
    virus = VirusModel(virus_id=1, pos=(33.0, 30.0), radius=1.5)
    nearby_player = _enemy(player_id=1, pos=(35.0, 30.0), radius=2.5)

    outcome = _project_one_step_outcome(
        own=own,
        direction=(1.0, 0.0),
        split=False,
        foods=(),
        viruses=(virus,),
        enemies=(nearby_player,),
        arena_size=60.0,
    )
    decision = strategy.choose(
        _context(own, viruses=(virus,), enemies=(nearby_player,))
    )

    assert outcome.virus_mass_gained == 2.25
    assert outcome.own_mass_lost > 0.0
    assert decision.target_kind != "virus"


def test_very_large_blob_adds_multi_split_capture_for_distant_prey() -> None:
    strategy = SemanticPotentialStrategy()
    strategy._last_direction = (0.0, 1.0)
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=6.0),)
    prey = _enemy(player_id=1, pos=(49.0, 30.0), radius=1.0)

    decision = strategy.choose(_context(own, enemies=(prey,)))

    assert decision.reason == "multi_split_capture"
    assert decision.split
    assert decision.diagnostics["split_depth"] == 2

    after_first_split = _project_action_blobs(
        own,
        decision.direction,
        split=True,
        arena_size=60.0,
    )
    continuation = strategy.choose(_context(after_first_split, enemies=(prey,)))

    assert continuation.reason == "split_capture"
    assert continuation.split
    assert continuation.diagnostics["split_depth"] == 1


def test_split_capture_is_rejected_when_fragments_enter_predator_reach() -> None:
    strategy = SemanticPotentialStrategy()
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=3.0),)
    prey = _enemy(player_id=1, pos=(37.0, 30.0), radius=1.5)
    predator = _enemy(player_id=2, pos=(35.0, 30.0), radius=5.0)

    decision = strategy.choose(_context(own, enemies=(prey, predator)))

    assert not decision.split
    assert decision.reason == "escape"
    assert decision.direction[0] < 0.0


def test_safety_reserve_turns_away_before_any_candidate_is_catastrophic() -> None:
    strategy = SemanticPotentialStrategy()
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=1.0),)
    predator = _enemy(player_id=1, pos=(41.0, 30.0), radius=3.0)
    tempting_food = FoodModel(food_id=1, pos=(40.0, 30.0))

    decision = strategy.choose(
        _context(own, foods=(tempting_food,), enemies=(predator,))
    )

    assert decision.diagnostics["safe_candidate_count"] == 3
    assert 0.0 < decision.diagnostics["current_safety_margin"] < 3.0
    assert decision.reason == "escape"
    assert decision.direction[0] < 0.0
    assert decision.diagnostics["selected_safety_margin"] >= 3.0


def test_directional_fan_has_angular_and_three_turn_radial_boundaries() -> None:
    source = BlobModel(blob_id=0, pos=(30.0, 30.0), radius=1.0)
    east = (1.0, 0.0)

    centered = _fan_weight(source, (32.5, 30.0), east)
    near_edge = _fan_weight(source, (32.3, 31.2), east)
    outside_angle = _fan_weight(source, (32.0, 32.0), east)
    outside_horizon = _fan_weight(source, (40.0, 30.0), east)

    assert centered > near_edge > 0.0
    assert outside_angle == 0.0
    assert outside_horizon == 0.0


def test_directional_field_caps_tiny_food_sources(monkeypatch) -> None:
    strategy = SemanticPotentialStrategy()
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=1.0),)
    foods = tuple(
        FoodModel(food_id=index, pos=(20.0 + index, 30.0))
        for index in range(20)
    )
    observed_counts: list[int] = []
    original = semantic_module._directional_potential

    def record_food_count(**kwargs):
        observed_counts.append(len(kwargs["foods"]))
        return original(**kwargs)

    monkeypatch.setattr(semantic_module, "_directional_potential", record_food_count)

    strategy.choose(_context(own, foods=foods))

    assert observed_counts
    assert max(observed_counts) == semantic_module.MAX_DIRECTIONAL_FOODS


def test_quiet_candidates_skip_unused_macro_projection(monkeypatch) -> None:
    strategy = SemanticPotentialStrategy()
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=1.0),)
    foods = (
        FoodModel(food_id=1, pos=(25.0, 30.0)),
        FoodModel(food_id=2, pos=(35.0, 30.0)),
    )
    projection_calls = 0
    original = semantic_module._project_blobs

    def count_projection(*args, **kwargs):
        nonlocal projection_calls
        projection_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(semantic_module, "_project_blobs", count_projection)

    decision = strategy.choose(_context(own, foods=foods))

    # Each non-split candidate needs only its one-step movement. The former
    # unconditional four-step wall projection would double this count.
    assert projection_calls == decision.diagnostics["candidate_count"]
