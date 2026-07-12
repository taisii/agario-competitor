from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from lib.config.player import EAT_SIZE_RATIO, MASS_DECAY_RATE  # noqa: E402
from lib.models.blob_model import BlobModel, VisibleBlobModel  # noqa: E402
from lib.models.food_model import FoodModel  # noqa: E402
from lib.models.virus_model import VirusModel  # noqa: E402
from strategies.base import StrategyContext, StrategyDecision  # noqa: E402
from strategies.features import (  # noqa: E402
    can_consume_virus,
    can_eat_player_blob,
    extract_visible_features,
    virus_center_clearance,
)
from strategies.greedy import FoodGreedyStrategy  # noqa: E402
from strategies.potential_field import PotentialFieldHunterStrategy  # noqa: E402
from strategies.virus_farming import VirusHunterStrategy  # noqa: E402


def _game(
    own: tuple[BlobModel, ...],
    *,
    enemies: tuple[VisibleBlobModel, ...] = (),
    food: tuple[FoodModel, ...] = (),
    viruses: tuple[VirusModel, ...] = (),
    rankings: tuple[int, ...] = (0, 1),
    round_number: int = 10,
    max_rounds: int = 1400,
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
        visible_viruses=list(viruses),
        map=SimpleNamespace(size=60.0),
        round=round_number,
        max_rounds=max_rounds,
        rankings=list(rankings),
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


def test_virus_consumption_uses_engine_strict_mass_threshold() -> None:
    threshold_radius = math.sqrt(1.5 * 1.5 * EAT_SIZE_RATIO)

    assert not can_consume_virus(threshold_radius, 1.5)
    assert can_consume_virus(threshold_radius + 1e-6, 1.5)


def test_virus_collision_requires_center_containment() -> None:
    """Partial circle overlap no longer pops a blob as of agario-kit 2026.1.13."""

    blob_pos = (10.0, 10.0)
    blob_radius = 2.0
    virus_pos = (13.0, 10.0)
    virus_radius = 1.5

    assert math.dist(blob_pos, virus_pos) < blob_radius + virus_radius
    assert virus_center_clearance(blob_pos, blob_radius, virus_pos) == 1.0


def test_virus_hunter_times_contact_at_virus_center_boundary() -> None:
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=3.0)
    virus = VirusModel(virus_id=7, pos=(14.5, 10.0), radius=1.5)

    decision = VirusHunterStrategy().choose(
        StrategyContext(
            game=_game((own,), viruses=(virus,)),
            query=SimpleNamespace(),
        )
    )

    assert decision.target_kind == "virus"
    assert decision.diagnostics["virus_contact_distance"] == 1.5
    assert decision.diagnostics["turns_to_contact"] == 2


def test_virus_hunter_prioritises_reachable_virus_over_nearby_food() -> None:
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=2.0)
    food = FoodModel(food_id=1, pos=(9.0, 10.0))
    virus = VirusModel(virus_id=7, pos=(18.0, 10.0), radius=1.5)

    decision = VirusHunterStrategy().choose(
        StrategyContext(
            game=_game((own,), food=(food,), viruses=(virus,)),
            query=SimpleNamespace(),
        )
    )

    assert decision.direction[0] > 0.0
    assert decision.target_kind == "virus"
    assert decision.target_id == "7"
    assert decision.reason == "reachable_virus"


def test_virus_hunter_aims_from_fragment_that_can_consume_virus() -> None:
    capable = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=2.0)
    nearer_but_small = BlobModel(blob_id=1, pos=(20.0, 10.0), radius=1.0)
    virus = VirusModel(virus_id=7, pos=(18.0, 10.0), radius=1.5)

    decision = VirusHunterStrategy().choose(
        StrategyContext(
            game=_game((capable, nearer_but_small), viruses=(virus,)),
            query=SimpleNamespace(),
        )
    )

    assert decision.direction[0] > 0.0
    assert decision.diagnostics["hunter_blob_id"] == capable.blob_id


def test_virus_hunter_grows_when_decay_prevents_reaching_virus() -> None:
    virus = VirusModel(virus_id=7, pos=(18.0, 10.0), radius=1.5)
    threshold_mass = virus.radius * virus.radius * EAT_SIZE_RATIO
    barely_capable = BlobModel(
        blob_id=0,
        pos=(10.0, 10.0),
        radius=math.sqrt(threshold_mass / (1.0 - MASS_DECAY_RATE)),
    )
    food = FoodModel(food_id=1, pos=(9.0, 10.0))

    decision = VirusHunterStrategy().choose(
        StrategyContext(
            game=_game((barely_capable,), food=(food,), viruses=(virus,)),
            query=SimpleNamespace(),
        )
    )

    assert decision.direction[0] < 0.0
    assert decision.target_kind == "food"
    assert decision.diagnostics["virus_hunter_mode"] == "grow_for_virus"
    assert decision.diagnostics["virus_unavailable_reason"] == "mass_decays_before_contact"


def test_virus_hunter_escapes_immediate_predator_before_pursuit() -> None:
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=2.0)
    predator = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(12.0, 10.0),
        radius=3.0,
    )
    virus = VirusModel(virus_id=7, pos=(18.0, 10.0), radius=1.5)

    decision = VirusHunterStrategy().choose(
        StrategyContext(
            game=_game((own,), enemies=(predator,), viruses=(virus,)),
            query=SimpleNamespace(),
        )
    )

    assert decision.direction[0] < 0.0
    assert decision.target_kind == "escape"
    assert decision.diagnostics["virus_hunter_mode"] == "emergency_escape"


def test_virus_hunter_rejects_virus_that_creates_edible_fragments() -> None:
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=2.0)
    virus = VirusModel(virus_id=7, pos=(18.0, 10.0), radius=1.5)
    post_split_predator = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(18.5, 10.0),
        radius=1.2,
    )

    decision = VirusHunterStrategy().choose(
        StrategyContext(
            game=_game((own,), enemies=(post_split_predator,), viruses=(virus,)),
            query=SimpleNamespace(),
        )
    )

    assert decision.target_kind != "virus"
    assert decision.diagnostics["virus_hunter_mode"] == "grow_for_virus"
    assert decision.diagnostics["virus_unavailable_reason"] == "post_split_predator_risk"
    assert decision.diagnostics["post_split_rejected_pairs"] == 1


def test_virus_hunter_ignores_enemy_too_small_to_eat_projected_fragments() -> None:
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=2.0)
    virus = VirusModel(virus_id=7, pos=(18.0, 10.0), radius=1.5)
    harmless_enemy = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(18.5, 10.0),
        radius=0.4,
    )

    decision = VirusHunterStrategy().choose(
        StrategyContext(
            game=_game((own,), enemies=(harmless_enemy,), viruses=(virus,)),
            query=SimpleNamespace(),
        )
    )

    assert decision.target_kind == "virus"
    assert decision.diagnostics["post_split_predator_count"] == 0


def test_virus_hunter_accepts_growth_only_virus_at_blob_cap() -> None:
    hunter = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=2.0)
    fragments = tuple(
        BlobModel(
            blob_id=index,
            pos=(10.0, 20.0 + index),
            radius=0.5,
            merge_cooldown=12,
        )
        for index in range(1, 16)
    )
    virus = VirusModel(virus_id=7, pos=(18.0, 10.0), radius=1.5)
    nearby_enemy = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(18.5, 10.0),
        radius=1.2,
    )

    decision = VirusHunterStrategy().choose(
        StrategyContext(
            game=_game(
                (hunter, *fragments),
                enemies=(nearby_enemy,),
                viruses=(virus,),
            ),
            query=SimpleNamespace(),
        )
    )

    assert decision.target_kind == "virus"
    assert decision.diagnostics["projected_pieces_created"] == 1
    assert decision.diagnostics["post_split_safety_margin"] is None


def test_virus_hunter_steers_out_of_unsafe_virus_contact() -> None:
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=2.0)
    virus = VirusModel(virus_id=7, pos=(12.8, 10.0), radius=1.5)
    post_split_predator = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(13.2, 10.0),
        radius=1.2,
    )

    decision = VirusHunterStrategy().choose(
        StrategyContext(
            game=_game((own,), enemies=(post_split_predator,), viruses=(virus,)),
            query=SimpleNamespace(),
        )
    )

    assert decision.direction[0] < 0.0
    assert decision.target_kind == "avoid_virus"
    assert decision.diagnostics["virus_hunter_mode"] == "avoid_unsafe_virus"
    assert decision.diagnostics["selected_collision_clearance"] > (
        decision.diagnostics["fallback_collision_clearance"]
    )


def test_virus_hunter_predicts_upcoming_fragment_merge_before_virus() -> None:
    fragments = tuple(
        BlobModel(
            blob_id=index,
            pos=(10.0 + (index % 4) * 2.36, 10.0 + (index // 4) * 2.36),
            radius=1.18,
            merge_cooldown=8,
        )
        for index in range(16)
    )
    strategy = VirusHunterStrategy()

    consumers = strategy._projected_consumers(fragments)

    assert len(consumers) == 1
    assert consumers[0].merge_cooldown == 8
    assert math.isclose(
        consumers[0].radius * consumers[0].radius,
        sum(blob.radius * blob.radius for blob in fragments),
    )


def test_virus_hunter_preserves_target_mass_instead_of_refragmenting() -> None:
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=7.0)
    virus = VirusModel(virus_id=7, pos=(18.0, 10.0), radius=1.5)

    decision = VirusHunterStrategy().choose(
        StrategyContext(
            game=_game((own,), viruses=(virus,)),
            query=SimpleNamespace(),
        )
    )

    assert decision.target_kind != "virus"
    assert decision.diagnostics["virus_unavailable_reason"] == "mass_target_preservation"
    assert decision.diagnostics["mass_target_rejected_pairs"] == 1


def test_virus_hunter_latches_virus_preservation_when_already_top_two() -> None:
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=math.sqrt(20.0))
    virus = VirusModel(virus_id=7, pos=(18.0, 10.0), radius=1.5)

    decision = VirusHunterStrategy().choose(
        StrategyContext(
            game=_game(
                (own,),
                viruses=(virus,),
                rankings=(0, 1, 2),
                round_number=800,
            ),
            query=SimpleNamespace(),
        )
    )

    assert decision.target_kind != "virus"
    assert decision.diagnostics["virus_unavailable_reason"] == (
        "mass_target_preservation"
    )
    assert decision.diagnostics["mass_preservation_reason"] == (
        "competitive_position"
    )


def test_virus_hunter_still_uses_virus_to_recover_from_third_place() -> None:
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=math.sqrt(20.0))
    virus = VirusModel(virus_id=7, pos=(18.0, 10.0), radius=1.5)

    decision = VirusHunterStrategy().choose(
        StrategyContext(
            game=_game(
                (own,),
                viruses=(virus,),
                rankings=(1, 2, 0),
                round_number=800,
            ),
            query=SimpleNamespace(),
        )
    )

    assert decision.target_kind == "virus"
    assert decision.diagnostics["mass_target_latched"] is False


def test_virus_hunter_uses_neutral_growth_value_function() -> None:
    assert VirusHunterStrategy()._growth.endgame_adaptation is False


def test_virus_hunter_suppresses_voluntary_split_after_preservation_latches() -> None:
    strategy = VirusHunterStrategy()
    strategy._mass_target_reached = True
    proposed = StrategyDecision(
        direction=(1.0, 0.0),
        split=True,
        target_kind="prey",
        reason="split_prey",
    )

    decision = strategy._suppress_preservation_split(proposed)

    assert decision.direction == proposed.direction
    assert decision.split is False
    assert decision.reason == proposed.reason
    assert decision.diagnostics["split_suppressed_reason"] == (
        "mass_preservation"
    )


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


def test_potential_hunter_does_not_split_beyond_engine_reach() -> None:
    strategy = PotentialFieldHunterStrategy()
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
