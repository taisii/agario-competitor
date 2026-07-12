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
    OwnBlob,
    ReplayDominanceStrategy,
    SearchNode,
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


def test_replay_dominance_does_not_bypass_beam_with_direct_virus_mode() -> None:
    strategy = ReplayDominanceStrategy()
    own = OwnBlob(blob_id=0, x=10.0, y=10.0, radius=2.0)
    virus = VirusModel(virus_id=7, pos=(12.0, 10.0), radius=1.5)

    decision = strategy._direct_virus_decision(
        own_blobs=(own,),
        enemies=(),
        viruses=(virus,),
        arena_size=60.0,
        rank_position=1,
        progress=0.9,
    )

    assert decision is None


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
        allow_split=False,
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
        allow_split=True,
    )

    assert actions[0].reason in {"rival_prey", "split_rival_prey"}


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
        # The base policy passes False for a late top-two player.
        allow_split=False,
    )

    assert any(action.reason == "split_rival_prey" for action in actions)


def test_replay_dominance_scores_wall_clamp_by_actual_movement() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    own = OwnBlob(blob_id=0, x=2.0, y=10.0, radius=2.0)

    assert strategy._movement_efficiency((own,), (-1.0, 0.0), 60.0) == 0.0
    assert strategy._movement_efficiency((own,), (1.0, 0.0), 60.0) == 1.0


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

    after, _, _, _ = strategy._resolve_own_viruses(
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


def test_replay_dominance_virus_potential_rewards_safe_approach_without_mode() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    virus = VirusModel(virus_id=7, pos=(20.0, 10.0), radius=1.5)

    def node_at(x: float) -> SearchNode:
        return SearchNode(
            own_blobs=(OwnBlob(blob_id=0, x=x, y=10.0, radius=2.0),),
            enemies=(),
            score=0.0,
            first_direction=(1.0, 0.0),
            first_split=False,
            first_reason="keep",
            last_direction=(1.0, 0.0),
        )

    far = strategy._virus_potential(node_at(8.0), (virus,), 60.0)
    near = strategy._virus_potential(node_at(12.0), (virus,), 60.0)

    assert near > far > 0.0


def test_replay_dominance_prices_virus_fragment_survival_without_banning_it() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
    virus = VirusModel(virus_id=7, pos=(4.0, 30.0), radius=1.5)
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
    open_virus = VirusModel(virus_id=7, pos=(30.0, 30.0), radius=1.5)
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
    actions = strategy._virus_actions(
        node=trapped,
        viruses=(virus,),
        arena_size=60.0,
        limit=3,
    )

    assert 0.0 < trapped_retention < open_retention < 1.0
    assert actions == [Action((0.0, 0.0), reason="virus_harvest")]
    assert strategy._virus_potential(trapped, (virus,), 60.0) < (
        strategy._virus_potential(open_space, (open_virus,), 60.0)
    )


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

    safe_edge = strategy._position_value(
        [edge_blob], (), (), set(), 60.0, 1.0
    )
    trapped_edge = strategy._position_value(
        [edge_blob], (predator,), (), set(), 60.0, 1.0
    )
    open_center = strategy._position_value(
        [center_blob], (predator,), (), set(), 60.0, 1.0
    )

    assert trapped_edge < open_center
    assert safe_edge == 0.0


def test_replay_dominance_keeps_wide_escape_routes_in_anytime_prefix() -> None:
    strategy = ReplayDominanceStrategy(depth=1, width=1, angular_samples=4)
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
        allow_split=False,
    )
    reasons = [action.reason for action in actions]

    assert reasons[:5].count("escape_wide_tangent") == 2
    assert strategy._safety_weight(rank_position=7, progress=0.0) == 1.3


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

    safe = strategy._position_value(
        [anchor, trapped_fragment], (), (), set(), 60.0, 1.0
    )
    exposed = strategy._position_value(
        [anchor, trapped_fragment], (predator,), (), set(), 60.0, 1.0
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

    assert strategy._virus_potential(node, (virus,), 60.0) == 0.0
    assert strategy._virus_actions(
        node=node,
        viruses=(virus,),
        arena_size=60.0,
        limit=3,
    ) == []


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
