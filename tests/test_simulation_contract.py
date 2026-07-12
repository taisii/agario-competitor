from __future__ import annotations

import math
import sys
from pathlib import Path
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
from strategies.potential_field import _speed as potential_field_speed  # noqa: E402
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


def test_receding_virus_collision_does_not_split_self_when_enemy_is_larger() -> None:
    strategy = ThreatAwareRecedingHorizonStrategy(depth=1, width=1, angular_samples=4)
    corner = (2.1, 2.1)
    own = OwnBlob(blob_id=0, x=30.0, y=30.0, radius=2.0)
    enemy = EnemyBlob(player_id=1, blob_id=0, x=corner[0], y=corner[1], radius=2.1)
    virus = VirusModel(virus_id=7, pos=corner, radius=1.5)
    consumed: set[int] = set()

    own_after, enemies_after, score, penalty = strategy._resolve_own_viruses(
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
    assert score == 0.0
    assert penalty == 0.0
