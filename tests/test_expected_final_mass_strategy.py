from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from strategies.expected_final_mass import ExpectedFinalMassStrategy  # noqa: E402
from lib.models.virus_model import VirusModel  # noqa: E402
from strategies.features import normalise  # noqa: E402
from strategies.receding_horizon import (  # noqa: E402
    Action,
    EnemyBlob,
    OwnBlob,
    SearchNode,
    _split_attack_reach,
    _split_chain_attack_reach,
)
from strategies.registry import create_strategy, submission_strategy_spec  # noqa: E402


def _node(*, enemy: EnemyBlob | None = None) -> SearchNode:
    return SearchNode(
        own_blobs=(OwnBlob(blob_id=0, x=30.0, y=30.0, radius=3.0),),
        enemies=() if enemy is None else (enemy,),
        score=0.0,
        first_direction=(1.0, 0.0),
        first_split=False,
        first_reason="keep",
        last_direction=(1.0, 0.0),
    )


def test_expected_final_mass_is_registered_and_submission_capable() -> None:
    strategy = create_strategy("expected_final_mass")

    assert isinstance(strategy, ExpectedFinalMassStrategy)
    assert strategy.name == "expected_final_mass"
    assert (
        submission_strategy_spec("expected_final_mass").submission.strategy_class
        == "ExpectedFinalMassStrategy"
    )


def test_expected_final_mass_safety_does_not_change_with_ordinal_rank() -> None:
    strategy = ExpectedFinalMassStrategy()

    assert strategy._safety_weight(1, 1.0) == 1.3
    assert strategy._safety_weight(8, 0.0) == 1.3


def test_recovery_value_is_smooth_and_increases_with_time_remaining() -> None:
    strategy = ExpectedFinalMassStrategy()
    strategy._max_rounds = 1400

    strategy._current_round = 1400
    final_round = strategy._recovery_terminal_mass()
    strategy._current_round = 700
    midpoint = strategy._recovery_terminal_mass()
    strategy._current_round = 0
    opening = strategy._recovery_terminal_mass()

    assert 0.8 < final_round < 1.0
    assert 15.0 < midpoint < 20.0
    assert 29.0 < opening < 30.0
    assert final_round < midpoint < opening


def test_expected_utility_values_recovery_instead_of_negative_death_mass() -> None:
    strategy = ExpectedFinalMassStrategy()
    strategy._max_rounds = 1400
    predator = EnemyBlob(
        player_id=1,
        blob_id=0,
        x=30.0,
        y=30.0,
        radius=5.0,
    )
    node = _node(enemy=predator)

    strategy._current_round = 0
    early = strategy._search_utility(
        node,
        foods=(),
        viruses=(),
        arena_size=60.0,
        safety_weight=1.3,
    )
    strategy._hazard_summary_cache.clear()
    strategy._current_round = 1400
    late = strategy._search_utility(
        node,
        foods=(),
        viruses=(),
        arena_size=60.0,
        safety_weight=1.3,
    )

    assert early > late > 0.0


def test_leader_imitation_is_compared_as_the_first_exact_root() -> None:
    strategy = ExpectedFinalMassStrategy(depth=1, width=2, angular_samples=4)
    strategy._leader_action = Action(
        direction=(0.0, -1.0),
        split=False,
        reason="leader_imitation_59",
    )

    actions = strategy._candidate_actions(
        node=_node(),
        foods=(),
        food_targets=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
    )

    assert actions[0] == strategy._leader_action
    assert len(actions) >= strategy.minimum_root_actions


def test_visible_chain_weight_never_exceeds_real_acquired_mass() -> None:
    strategy = ExpectedFinalMassStrategy()
    strategy._max_rounds = 1400

    for round_number in (0, 700, 1400):
        strategy._current_round = round_number
        weights = strategy._opportunity_weights()
        assert math.isclose(weights[0], 1.0)
        assert all(0.0 <= weight <= 1.0 for weight in weights)


def test_virus_retention_is_evaluated_at_contact_not_current_position() -> None:
    """Match 19229: the leader swept every fragment after a late virus pop."""

    strategy = ExpectedFinalMassStrategy(depth=1, width=2, angular_samples=4)
    own = OwnBlob(
        blob_id=0,
        x=20.0,
        y=30.0,
        radius=math.sqrt(35.4),
    )
    predator = EnemyBlob(
        player_id=5,
        blob_id=0,
        x=45.0,
        y=30.0,
        radius=8.0,
    )
    node = replace(_node(enemy=predator), own_blobs=(own,))
    far_from_predator = VirusModel(
        virus_id=1,
        pos=(26.0, 30.0),
        radius=1.5,
    )
    beside_predator = VirusModel(
        virus_id=2,
        pos=(34.0, 30.0),
        radius=1.5,
    )

    safer_retention = strategy._virus_retained_mass_fraction(
        node,
        own,
        far_from_predator,
        60.0,
    )
    exposed_retention = strategy._virus_retained_mass_fraction(
        node,
        own,
        beside_predator,
        60.0,
    )

    assert 0.0 <= exposed_retention < safer_retention <= 1.0


def test_split_chain_reach_accumulates_each_engine_split_transition() -> None:
    """Matches 19574/19599: two split commands crossed a one-split-safe gap."""

    predator_radius = 6.68
    post_virus_piece_radius = 1.66
    observed_gap = 21.78

    assert _split_attack_reach(predator_radius) < observed_gap
    assert (
        _split_chain_attack_reach(
            predator_radius,
            post_virus_piece_radius,
            max_splits=2,
        )
        > observed_gap
    )


def test_comparable_rival_can_chain_split_through_a_nearby_virus_pop() -> None:
    """Match 19599: 41.7 radius-sum was swept after popping beside 44.6."""

    strategy = ExpectedFinalMassStrategy(depth=1, width=2, angular_samples=4)
    own = OwnBlob(
        blob_id=0,
        x=8.42,
        y=53.54,
        radius=6.457,
    )
    rival = EnemyBlob(
        player_id=5,
        blob_id=0,
        x=13.92,
        y=32.46,
        radius=6.68,
    )
    node = replace(_node(enemy=rival), own_blobs=(own,))
    virus = VirusModel(
        virus_id=42,
        pos=(12.996942159714951, 58.49882968715043),
        radius=1.5,
    )

    retained = strategy._virus_retained_mass_fraction(
        node,
        own,
        virus,
        60.0,
    )

    assert 0.0 < retained < 0.5


def test_imminent_low_retention_virus_promotes_verified_non_contact_moves() -> None:
    """Matches 19574/19599 popped although the chosen label was not virus."""

    strategy = ExpectedFinalMassStrategy(depth=1, width=2, angular_samples=4)
    strategy._leader_action = Action(
        direction=(0.4, -0.9),
        reason="leader_imitation_24",
    )
    own = OwnBlob(
        blob_id=9,
        x=33.169,
        y=10.426,
        radius=3.544,
    )
    rival = EnemyBlob(
        player_id=5,
        blob_id=66,
        x=20.698,
        y=14.740,
        radius=3.597,
    )
    node = replace(_node(enemy=rival), own_blobs=(own,))
    viruses = (
        VirusModel(virus_id=21, pos=(35.844, 7.318), radius=1.5),
        VirusModel(virus_id=29, pos=(32.083, 6.095), radius=1.5),
    )

    actions = strategy._candidate_actions(
        node=node,
        foods=(),
        food_targets=(),
        viruses=viruses,
        arena_size=60.0,
        first_step=True,
    )

    assert [action.reason for action in actions[:2]] == [
        "dangerous_virus_escape",
        "dangerous_virus_escape_tangent",
    ]
    for action in actions[:2]:
        moved = strategy._move_own(own, action.direction, 60.0)
        assert all(math.dist(moved.pos, virus.pos) > moved.radius for virus in viruses)


def test_cornered_predator_escape_promotes_wall_tangents_before_imitation() -> None:
    """Match 19212: radial escape was clamped into the bottom-left corner."""

    strategy = ExpectedFinalMassStrategy(depth=1, width=2, angular_samples=4)
    strategy._leader_action = Action(
        direction=(-1.0, 1.0),
        reason="leader_imitation_9",
    )
    own = OwnBlob(blob_id=0, x=3.0, y=57.0, radius=3.0)
    predator = EnemyBlob(
        player_id=3,
        blob_id=0,
        x=14.0,
        y=46.0,
        radius=5.0,
    )
    node = replace(_node(enemy=predator), own_blobs=(own,))

    actions = strategy._candidate_actions(
        node=node,
        foods=(),
        food_targets=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
    )

    assert [action.reason for action in actions[:2]] == [
        "urgent_wall_tangent",
        "urgent_wall_tangent",
    ]
    direct_efficiency = strategy._movement_efficiency(
        node.own_blobs,
        (-1.0, 1.0),
        60.0,
    )
    assert all(
        strategy._movement_efficiency(
            node.own_blobs,
            action.direction,
            60.0,
        )
        > direct_efficiency
        for action in actions[:2]
    )


def test_visible_fragments_of_leader_are_not_mistaken_for_safe_prey() -> None:
    """Match 19229: two visible pieces hid a 16-piece leader about to merge."""

    strategy = ExpectedFinalMassStrategy(depth=1, width=2, angular_samples=4)
    strategy._own_player_id = 7
    strategy._scoreboard_leader_player_id = 5
    strategy._leader_action = Action(
        direction=(-0.3, 0.95),
        reason="leader_imitation_59",
    )
    own = OwnBlob(
        blob_id=0,
        x=53.0,
        y=18.0,
        radius=math.sqrt(33.5),
    )
    fragments = (
        EnemyBlob(
            player_id=5,
            blob_id=0,
            x=46.0,
            y=25.0,
            radius=2.0,
            merge_cooldown=8,
        ),
        EnemyBlob(
            player_id=5,
            blob_id=1,
            x=49.9,
            y=25.0,
            radius=2.0,
            merge_cooldown=8,
        ),
    )
    node = replace(_node(), own_blobs=(own,), enemies=fragments)

    actions = strategy._candidate_actions(
        node=node,
        foods=(),
        food_targets=(),
        viruses=(),
        arena_size=60.0,
        first_step=True,
    )

    assert [action.reason for action in actions[:2]] == [
        "leader_merge_escape",
        "leader_merge_escape_tangent",
    ]
    assert all(not action.split for action in actions)
    toward_fragments = normalise((47.95 - own.x, 25.0 - own.y))
    assert all(
        action.direction[0] * toward_fragments[0]
        + action.direction[1] * toward_fragments[1]
        < 0.0
        for action in actions[:2]
    )

    strategy._max_rounds = 1400
    strategy._current_round = 1072
    toward_node = replace(
        node,
        own_blobs=(replace(own, x=52.4, y=18.8),),
    )
    away_node = replace(
        node,
        own_blobs=(replace(own, x=53.6, y=17.2),),
    )
    toward_value = strategy._search_utility(
        toward_node,
        foods=(),
        viruses=(),
        arena_size=60.0,
        safety_weight=1.3,
    )
    away_value = strategy._search_utility(
        away_node,
        foods=(),
        viruses=(),
        arena_size=60.0,
        safety_weight=1.3,
    )

    assert away_value > toward_value
