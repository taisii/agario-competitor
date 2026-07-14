from __future__ import annotations

import math
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from engine.state.blob_state import BlobState as EngineBlobState  # noqa: E402
from engine.state.game_state import GameState  # noqa: E402
from engine.state.player_state import PlayerState  # noqa: E402
from engine.state.state_mutator import StateMutator  # noqa: E402
from lib.config.player import (  # noqa: E402
    BASE_PLAYER_SPEED,
    EAT_SIZE_RATIO,
    MASS_DECAY_RATE,
    MIN_PLAYER_SPEED,
    PLAYER_SPEED_RADIUS_FACTOR,
    SAME_PLAYER_OVERLAP_EPSILON,
    STARTING_RADIUS,
)
from lib.interact.map import Map  # noqa: E402
from lib.models.blob_model import VisibleBlobModel  # noqa: E402
from lib.models.virus_model import VirusModel  # noqa: E402
from simulation.rules import (  # noqa: E402
    can_consume_virus,
    circle_intersects_square,
    decayed_mass_after_turns,
    movement_speed,
    virus_replacement_positions,
)
from strategies.base import StrategyContext  # noqa: E402
from strategies.features import can_consume_virus as feature_can_consume_virus  # noqa: E402
from strategies.potential_field import player_speed as potential_field_speed  # noqa: E402
from strategies.receding_horizon import (  # noqa: E402
    EnemyBlob,
    EnemyTrack,
    OwnBlob,
    ThreatAwareRecedingHorizonStrategy,
)
from strategies.virus_farming import VirusHunterStrategy  # noqa: E402


def _engine_state(*, own_radius: float, enemy_radius: float) -> GameState:
    state = object.__new__(GameState)
    state.round = 1
    state.players = {0: PlayerState(0, 0), 1: PlayerState(1, 1)}
    state.players[0].blobs = {0: EngineBlobState(0, 10.0, 10.0, own_radius)}
    state.players[1].blobs = {0: EngineBlobState(0, 30.0, 30.0, enemy_radius)}
    state.map = Map()
    state.event_history = []
    state.private_event_history = []
    return state


def test_circle_vision_uses_corner_distance_not_expanded_bounding_box() -> None:
    assert not circle_intersects_square(
        circle_x=56.0,
        circle_y=56.0,
        circle_radius=1.0,
        square_center_x=50.0,
        square_center_y=50.0,
        square_size=10.0,
    )
    assert circle_intersects_square(
        circle_x=55.6,
        circle_y=55.6,
        circle_radius=1.0,
        square_center_x=50.0,
        square_center_y=50.0,
        square_size=10.0,
    )


def test_shared_physics_primitives_match_installed_engine() -> None:
    authoritative = _engine_state(own_radius=2.4, enemy_radius=1.0)
    mutator = StateMutator(authoritative)
    radius = authoritative.players[0].blobs[0].radius
    virus_radius = 1.5

    expected_speed = mutator._movement_speed(radius)
    assert movement_speed(
        radius,
        base_speed=BASE_PLAYER_SPEED,
        radius_factor=PLAYER_SPEED_RADIUS_FACTOR,
        minimum_speed=MIN_PLAYER_SPEED,
    ) == expected_speed
    assert potential_field_speed(radius) == expected_speed
    assert VirusHunterStrategy()._speed(radius) == expected_speed

    expected_can_consume = mutator._can_consume_virus(
        authoritative.players[0].blobs[0],
        virus_radius,
    )
    assert can_consume_virus(
        radius,
        virus_radius,
        eat_size_ratio=EAT_SIZE_RATIO,
    ) is expected_can_consume
    assert feature_can_consume_virus(radius, virus_radius) is expected_can_consume

    positions = virus_replacement_positions(
        center_x=10.0,
        center_y=11.0,
        piece_radius=0.7,
        piece_count=7,
        overlap_epsilon=SAME_PLAYER_OVERLAP_EPSILON,
    )
    assert positions == tuple(mutator._replacement_positions(10.0, 11.0, 0.7, 7))


def test_projected_decay_matches_repeated_engine_rounds() -> None:
    authoritative = _engine_state(own_radius=3.0, enemy_radius=STARTING_RADIUS)
    mutator = StateMutator(authoritative)
    for _ in range(9):
        mutator._apply_mass_decay()

    assert math.isclose(
        decayed_mass_after_turns(
            3.0**2,
            9,
            decay_rate=MASS_DECAY_RATE,
            minimum_radius=STARTING_RADIUS,
        ),
        authoritative.players[0].blobs[0].mass,
        rel_tol=1e-12,
    )


def test_enemy_memory_keeps_blob_outside_vision_corner() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=1, width=1, angular_samples=4)
    strategy.enemy_tracks[(1, 0)] = EnemyTrack(
        player_id=1,
        blob_id=0,
        x=56.0,
        y=56.0,
        radius=1.0,
        direction=(0.0, 0.0),
        last_seen_round=1,
    )
    state = SimpleNamespace(
        round=2,
        visible_blobs=(),
        view_center=(50.0, 50.0),
        vision_size=10.0,
    )
    context = StrategyContext(game=SimpleNamespace(state=state), query=SimpleNamespace(update={}))

    strategy._update_enemy_memory(
        context,
        (OwnBlob(blob_id=0, x=40.0, y=40.0, radius=2.0),),
        arena_size=60.0,
    )

    assert (1, 0) in strategy.enemy_tracks


def test_enemy_memory_does_not_treat_public_visible_index_as_identity() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=1, width=1, angular_samples=4)
    strategy.enemy_tracks[(1, 42)] = EnemyTrack(
        player_id=1,
        blob_id=42,
        x=30.0,
        y=30.0,
        radius=2.0,
        direction=(1.0, 0.0),
        last_seen_round=10,
    )
    # agario-kit 2026.1.14 reassigns this public index from the currently
    # visible subset; it may change even though the physical blob did not.
    visible = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(30.8, 30.0),
        radius=2.0,
    )
    state = SimpleNamespace(
        round=11,
        visible_blobs=(visible,),
        view_center=(30.0, 30.0),
        vision_size=20.0,
    )

    enemies = strategy._update_enemy_memory(
        StrategyContext(
            game=SimpleNamespace(state=state),
            query=SimpleNamespace(update={}),
        ),
        (OwnBlob(blob_id=0, x=20.0, y=20.0, radius=1.0),),
        arena_size=60.0,
    )

    assert set(strategy.enemy_tracks) == {(1, 42)}
    assert enemies[0].blob_id == 42
    assert enemies[0].pos == visible.pos


def test_enemy_matching_maximises_cardinality_before_geometric_cost() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=1, width=1, angular_samples=4)
    tracks = {
        (1, 10): EnemyTrack(1, 10, 20.0, 30.0, 1.0, (0.0, 0.0), 10),
        (1, 11): EnemyTrack(1, 11, 28.0, 30.0, 1.0, (0.0, 0.0), 10),
    }
    observations = (
        VisibleBlobModel(
            player_id=1,
            team_id=1,
            blob_id=0,
            pos=(23.0, 30.0),
            radius=1.0,
        ),
        VisibleBlobModel(
            player_id=1,
            team_id=1,
            blob_id=1,
            pos=(16.9, 30.0),
            radius=1.0,
        ),
    )

    matches = strategy._match_enemy_observations(tracks, observations)

    assert matches == {0: (1, 11), 1: (1, 10)}


def test_enemy_memory_uses_one_post_prediction_assignment_without_track_growth() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=1, width=1, angular_samples=4)
    strategy.enemy_tracks = {
        (1, 10): EnemyTrack(1, 10, 14.0, 30.0, 1.0, (-1.0, 0.0), 10),
        (1, 11): EnemyTrack(1, 11, 17.0, 30.0, 1.0, (-1.0, 0.0), 10),
    }
    strategy.last_moves[1] = ((-1.0, 0.0), False)
    visible = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(15.1, 30.0),
        radius=0.9,
    )
    state = SimpleNamespace(
        round=11,
        visible_blobs=(visible,),
        view_center=(20.0, 30.0),
        vision_size=10.0,
    )

    strategy._update_enemy_memory(
        StrategyContext(
            game=SimpleNamespace(state=state),
            query=SimpleNamespace(update={}),
        ),
        (OwnBlob(blob_id=0, x=30.0, y=30.0, radius=0.5),),
        arena_size=60.0,
    )

    assert set(strategy.enemy_tracks) == {(1, 10), (1, 11)}
    matched = strategy.enemy_tracks[(1, 11)]
    assert (matched.x, matched.y) == visible.pos
    assert strategy.enemy_tracks[(1, 11)].last_seen_round == 11
    assert strategy.enemy_tracks[(1, 10)].last_seen_round == 10
    assert strategy._next_enemy_track_id == 0


def test_enemy_matching_sixteen_by_sixteen_is_bounded() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=1, width=1, angular_samples=4)
    tracks = {
        (1, index): EnemyTrack(
            1,
            index,
            30.0 + index * 0.01,
            30.0,
            1.0,
            (0.0, 0.0),
            10,
        )
        for index in range(16)
    }
    observations = tuple(
        VisibleBlobModel(
            player_id=1,
            team_id=1,
            blob_id=index,
            pos=(30.0 + (15 - index) * 0.01, 30.0),
            radius=1.0,
        )
        for index in range(16)
    )

    started = perf_counter()
    matches = strategy._match_enemy_observations(tracks, observations)
    elapsed = perf_counter() - started

    assert len(matches) == 16
    assert len(set(matches.values())) == 16
    assert elapsed < 0.05


def test_enemy_memory_retires_tracks_consumed_by_an_observed_merge() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=1, width=1, angular_samples=4)
    strategy.enemy_tracks = {
        (1, 10): EnemyTrack(1, 10, 15.5, 30.0, 1.0, (0.0, 0.0), 10, 0),
        (1, 11): EnemyTrack(1, 11, 13.5, 30.0, 1.0, (0.0, 0.0), 10, 0),
    }
    strategy.last_moves[1] = ((0.0, 0.0), False)
    merged = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(14.5, 30.0),
        # The ready fragments merge first, then the r=sqrt(2) blob consumes a
        # legal r=1 prey in the same engine transition.
        radius=math.sqrt(3.0),
        merge_cooldown=0,
    )
    state = SimpleNamespace(
        round=11,
        visible_blobs=(merged,),
        view_center=(20.0, 30.0),
        vision_size=10.0,
    )

    strategy._update_enemy_memory(
        StrategyContext(
            game=SimpleNamespace(state=state),
            query=SimpleNamespace(update={}),
        ),
        (OwnBlob(blob_id=0, x=30.0, y=30.0, radius=2.0),),
        arena_size=60.0,
    )

    assert len(strategy.enemy_tracks) == 1
    survivor = next(iter(strategy.enemy_tracks.values()))
    assert (survivor.x, survivor.y) == merged.pos
    assert survivor.radius == merged.radius


def test_enemy_memory_accepts_food_sized_growth_after_an_observed_merge() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=1, width=1, angular_samples=4)
    strategy.enemy_tracks = {
        (1, 10): EnemyTrack(1, 10, 15.5, 30.0, 1.0, (0.0, 0.0), 10, 0),
        (1, 11): EnemyTrack(1, 11, 13.5, 30.0, 1.0, (0.0, 0.0), 10, 0),
    }
    strategy.last_moves[1] = ((0.0, 0.0), False)
    merged = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(14.5, 30.0),
        radius=math.sqrt(2.0 + 0.15**2),
        merge_cooldown=0,
    )
    state = SimpleNamespace(
        round=11,
        visible_blobs=(merged,),
        view_center=(20.0, 30.0),
        vision_size=10.0,
    )

    strategy._update_enemy_memory(
        StrategyContext(
            game=SimpleNamespace(state=state),
            query=SimpleNamespace(update={}),
        ),
        (OwnBlob(blob_id=0, x=30.0, y=30.0, radius=2.0),),
        arena_size=60.0,
    )

    assert len(strategy.enemy_tracks) == 1


def test_enemy_memory_does_not_merge_tracks_into_an_unrelated_giant() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=1, width=1, angular_samples=4)
    strategy.enemy_tracks = {
        (1, 10): EnemyTrack(1, 10, 15.5, 30.0, 1.0, (0.0, 0.0), 10, 0),
        (1, 11): EnemyTrack(1, 11, 13.5, 30.0, 1.0, (0.0, 0.0), 10, 0),
    }
    strategy.last_moves[1] = ((0.0, 0.0), False)
    unrelated = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        # Assignment to the nearby survivor is possible, but the center does
        # not explain the complete ready/touching component.
        pos=(15.5, 30.0),
        radius=4.0,
        merge_cooldown=0,
    )
    state = SimpleNamespace(
        round=11,
        visible_blobs=(unrelated,),
        view_center=(20.0, 30.0),
        vision_size=10.0,
    )

    strategy._update_enemy_memory(
        StrategyContext(
            game=SimpleNamespace(state=state),
            query=SimpleNamespace(update={}),
        ),
        (OwnBlob(blob_id=0, x=30.0, y=30.0, radius=5.0),),
        arena_size=60.0,
    )

    assert len(strategy.enemy_tracks) == 2


def test_enemy_memory_accepts_merge_then_multiple_prey_chain_growth() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=1, width=1, angular_samples=4)
    strategy.enemy_tracks = {
        (1, 10): EnemyTrack(1, 10, 15.5, 30.0, 1.0, (0.0, 0.0), 10, 0),
        (1, 11): EnemyTrack(1, 11, 13.5, 30.0, 1.0, (0.0, 0.0), 10, 0),
    }
    strategy.last_moves[1] = ((0.0, 0.0), False)
    chained = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(14.5, 30.0),
        radius=2.0,
        merge_cooldown=0,
    )
    state = SimpleNamespace(
        round=11,
        visible_blobs=(chained,),
        view_center=(20.0, 30.0),
        vision_size=10.0,
    )

    strategy._update_enemy_memory(
        StrategyContext(
            game=SimpleNamespace(state=state),
            query=SimpleNamespace(update={}),
        ),
        (OwnBlob(blob_id=0, x=30.0, y=30.0, radius=3.0),),
        arena_size=60.0,
    )

    assert len(strategy.enemy_tracks) == 1
    assert next(iter(strategy.enemy_tracks.values())).radius == 2.0


def test_enemy_memory_split_keeps_one_parent_id_and_creates_one_child_id() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=1, width=1, angular_samples=4)
    strategy.enemy_tracks = {
        (1, 10): EnemyTrack(
            1,
            10,
            16.0,
            30.0,
            math.sqrt(2.0),
            (0.0, 0.0),
            10,
            0,
        )
    }
    strategy.last_moves[1] = ((0.0, 0.0), True)
    children = tuple(
        VisibleBlobModel(
            player_id=1,
            team_id=1,
            blob_id=index,
            pos=(15.0 + 2.0 * index, 30.0),
            radius=1.0,
            merge_cooldown=18,
        )
        for index in range(2)
    )
    state = SimpleNamespace(
        round=11,
        visible_blobs=children,
        view_center=(16.0, 30.0),
        vision_size=20.0,
    )

    strategy._update_enemy_memory(
        StrategyContext(
            game=SimpleNamespace(state=state),
            query=SimpleNamespace(update={}),
        ),
        (OwnBlob(blob_id=0, x=30.0, y=30.0, radius=3.0),),
        arena_size=60.0,
    )

    assert len(strategy.enemy_tracks) == 2
    assert (1, 10) in strategy.enemy_tracks
    assert len(set(strategy.enemy_tracks) - {(1, 10)}) == 1


def test_merge_ready_tracks_are_not_retired_when_mass_remains_split() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=1, width=1, angular_samples=4)
    strategy.enemy_tracks = {
        (1, 10): EnemyTrack(1, 10, 15.5, 30.0, 1.0, (0.0, 0.0), 10, 0),
        (1, 11): EnemyTrack(1, 11, 13.5, 30.0, 1.0, (0.0, 0.0), 10, 0),
    }
    strategy.last_moves[1] = ((0.0, 0.0), True)
    fragments = tuple(
        VisibleBlobModel(
            player_id=1,
            team_id=1,
            blob_id=index,
            pos=(13.5 + 2.0 * index, 30.0),
            radius=1.0,
            merge_cooldown=18,
        )
        for index in range(2)
    )
    state = SimpleNamespace(
        round=11,
        visible_blobs=fragments,
        view_center=(15.0, 30.0),
        vision_size=20.0,
    )

    strategy._update_enemy_memory(
        StrategyContext(
            game=SimpleNamespace(state=state),
            query=SimpleNamespace(update={}),
        ),
        (OwnBlob(blob_id=0, x=30.0, y=30.0, radius=3.0),),
        arena_size=60.0,
    )

    assert set(strategy.enemy_tracks) == {(1, 10), (1, 11)}


def test_independent_assigned_fragment_does_not_block_merge_retirement() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=1, width=1, angular_samples=4)
    strategy.enemy_tracks = {
        (1, 10): EnemyTrack(1, 10, 13.43, 30.0, 0.9, (0.0, 0.0), 10, 0),
        (1, 11): EnemyTrack(1, 11, 15.5, 30.0, 1.1, (0.0, 0.0), 10, 0),
        (1, 12): EnemyTrack(1, 12, 18.5, 30.0, 1.0, (0.0, 0.0), 10, 10),
    }
    strategy.last_moves[1] = ((0.0, 0.0), False)
    observations = (
        VisibleBlobModel(
            player_id=1,
            team_id=1,
            blob_id=0,
            pos=(14.67, 30.0),
            radius=math.sqrt(2.02),
            merge_cooldown=0,
        ),
        VisibleBlobModel(
            player_id=1,
            team_id=1,
            blob_id=1,
            pos=(18.5, 30.0),
            radius=1.0,
            merge_cooldown=9,
        ),
    )
    state = SimpleNamespace(
        round=11,
        visible_blobs=observations,
        view_center=(20.0, 30.0),
        vision_size=10.0,
    )

    strategy._update_enemy_memory(
        StrategyContext(
            game=SimpleNamespace(state=state),
            query=SimpleNamespace(update={}),
        ),
        (OwnBlob(blob_id=0, x=30.0, y=30.0, radius=3.0),),
        arena_size=60.0,
    )

    assert set(strategy.enemy_tracks) == {(1, 11), (1, 12)}
    assert strategy.enemy_tracks[(1, 11)].radius == observations[0].radius
    assert strategy.enemy_tracks[(1, 12)].radius == observations[1].radius


def test_assigned_observations_inside_one_ready_component_block_merge_retirement() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=1, width=1, angular_samples=4)
    strategy.enemy_tracks = {
        (1, 10): EnemyTrack(1, 10, 13.43, 30.0, 0.9, (0.0, 0.0), 10, 0),
        (1, 11): EnemyTrack(1, 11, 15.5, 30.0, 1.1, (0.0, 0.0), 10, 0),
        (1, 12): EnemyTrack(1, 12, 17.7, 30.0, 1.0, (0.0, 0.0), 10, 0),
    }
    strategy.last_moves[1] = ((0.0, 0.0), True)
    observations = (
        VisibleBlobModel(
            player_id=1,
            team_id=1,
            blob_id=0,
            pos=(14.67, 30.0),
            radius=math.sqrt(2.02),
            merge_cooldown=18,
        ),
        VisibleBlobModel(
            player_id=1,
            team_id=1,
            blob_id=1,
            pos=(17.7, 30.0),
            radius=1.0,
            merge_cooldown=18,
        ),
    )

    matches = strategy._match_enemy_observations(
        strategy.enemy_tracks,
        observations,
    )
    retired = strategy._retired_merged_tracks(
        strategy.enemy_tracks,
        observations,
        matches,
    )

    assert matches == {0: (1, 11), 1: (1, 12)}
    assert retired == set()


def test_receding_virus_collision_does_not_split_self_when_enemy_is_larger() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=1, width=1, angular_samples=4)
    corner = (2.1, 2.1)
    own = OwnBlob(blob_id=0, x=30.0, y=30.0, radius=2.0)
    enemy = EnemyBlob(player_id=1, blob_id=0, x=corner[0], y=corner[1], radius=2.1)
    virus = VirusModel(virus_id=7, pos=corner, radius=1.5)
    consumed: set[int] = set()

    own_after, enemies_after, penalty, own_consumed = strategy._resolve_own_viruses(
        own_blobs=[own],
        enemies=(enemy,),
        viruses=(virus,),
        consumed_virus_ids=consumed,
        arena_size=60.0,
    )

    enemies_after = strategy._stabilise_enemy_blobs(enemies_after, 60.0)

    authoritative = _engine_state(own_radius=own.radius, enemy_radius=enemy.radius)
    authoritative.players[1].blobs[0].x = corner[0]
    authoritative.players[1].blobs[0].y = corner[1]
    authoritative.map.viruses = [virus]
    mutator = StateMutator(authoritative)
    mutator._resolve_viruses()
    mutator._stabilise_same_player_blobs()
    expected = tuple(
        authoritative.players[1].blobs[blob_id]
        for blob_id in sorted(authoritative.players[1].blobs)
    )

    assert own_after == [own]
    assert own_consumed == 0
    assert len(enemies_after) == len(expected) == 16
    assert math.isclose(
        sum(blob.mass for blob in enemies_after),
        enemy.mass + virus.radius**2,
    )
    for actual, wanted in zip(enemies_after, expected):
        assert actual.blob_id == wanted.blob_id
        assert math.isclose(actual.x, wanted.x, abs_tol=1e-12)
        assert math.isclose(actual.y, wanted.y, abs_tol=1e-12)
        assert math.isclose(actual.radius, wanted.radius, abs_tol=1e-12)
    assert consumed == {7}
    assert penalty == 0.0
