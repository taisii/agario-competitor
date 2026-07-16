from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from lib.models.blob_model import BlobModel, VisibleBlobModel  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from strategies.semantic_offensive_beam import (  # noqa: E402
    PursuitMemory,
    SemanticOffensiveBeamStrategy,
    _attack_beam_value,
    _trap_geometry,
)
from strategies.semantic_potential import DirectionCandidate  # noqa: E402


def _enemy(
    *,
    player_id: int,
    blob_id: int,
    pos: tuple[float, float],
    radius: float,
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
    round_number: int,
    enemies: tuple[VisibleBlobModel, ...] = (),
) -> StrategyContext:
    state = SimpleNamespace(
        me=SimpleNamespace(player_id=0, blobs={blob.blob_id: blob for blob in own}),
        visible_food=[],
        visible_viruses=[],
        visible_blobs=list(enemies),
        rankings=[0, 1, 2],
        round=round_number,
        max_rounds=1400,
        map=SimpleNamespace(size=60.0),
    )
    return StrategyContext(
        game=SimpleNamespace(state=state),
        query=SimpleNamespace(),
    )


def test_corner_cutoff_is_a_bounded_attack_candidate(monkeypatch) -> None:
    monkeypatch.setenv("SEMANTIC_OFFENSIVE_CORNER", "1")
    strategy = SemanticOffensiveBeamStrategy()
    own = (BlobModel(blob_id=0, pos=(32.0, 45.0), radius=4.0),)
    target = _enemy(
        player_id=1,
        blob_id=0,
        pos=(55.0, 55.0),
        radius=1.5,
    )

    decision = strategy.choose(
        _context(own, round_number=500, enemies=(target,))
    )

    scores = decision.diagnostics["candidate_scores"]
    assert "corner_cutoff_enemy" in scores
    assert decision.diagnostics["candidate_count"] <= 8


def test_far_prey_does_not_add_a_direct_pursuit_root() -> None:
    strategy = SemanticOffensiveBeamStrategy()
    own = (BlobModel(blob_id=0, pos=(10.0, 10.0), radius=1.5),)
    target = _enemy(
        player_id=1,
        blob_id=0,
        pos=(40.0, 10.0),
        radius=1.0,
    )

    decision = strategy.choose(
        _context(own, round_number=100, enemies=(target,))
    )

    assert "pursuit_enemy" not in decision.diagnostics["candidate_scores"]


def test_pursuit_memory_survives_brief_vision_loss_then_expires(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SEMANTIC_OFFENSIVE_MEMORY", "1")
    strategy = SemanticOffensiveBeamStrategy()
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=4.0),)
    target = _enemy(
        player_id=1,
        blob_id=0,
        pos=(38.0, 30.0),
        radius=1.5,
    )

    strategy.choose(_context(own, round_number=100, enemies=(target,)))
    hidden = strategy.choose(_context(own, round_number=104))
    expired = strategy.choose(_context(own, round_number=107))

    assert "pursuit_memory" in hidden.diagnostics["candidate_scores"]
    assert "pursuit_memory" not in expired.diagnostics["candidate_scores"]


def test_visible_predator_clears_pursuit_memory(monkeypatch) -> None:
    monkeypatch.setenv("SEMANTIC_OFFENSIVE_MEMORY", "1")
    strategy = SemanticOffensiveBeamStrategy()
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=4.0),)
    prey = _enemy(
        player_id=1,
        blob_id=0,
        pos=(38.0, 30.0),
        radius=1.5,
    )
    predator = _enemy(
        player_id=2,
        blob_id=0,
        pos=(25.0, 30.0),
        radius=5.0,
    )

    strategy.choose(_context(own, round_number=100, enemies=(prey,)))
    threatened = strategy.choose(
        _context(own, round_number=101, enemies=(predator,))
    )
    hidden = strategy.choose(_context(own, round_number=102))

    assert threatened.diagnostics["offensive_beam"]["pursuit_player_id"] is None
    assert "pursuit_memory" not in hidden.diagnostics["candidate_scores"]


def test_corner_geometry_rewards_inward_pin_more_than_outward_chase() -> None:
    inward_pin = _trap_geometry(
        hunter_pos=(45.0, 52.0),
        target_pos=(55.0, 55.0),
        target_radius=1.5,
        arena_size=60.0,
    )
    outward_chase = _trap_geometry(
        hunter_pos=(58.0, 58.0),
        target_pos=(55.0, 55.0),
        target_radius=1.5,
        arena_size=60.0,
    )

    assert inward_pin > 0.0
    assert inward_pin > outward_chase


def test_stronger_target_owner_reduces_attack_beam_value() -> None:
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=4.0),)
    target = _enemy(
        player_id=1,
        blob_id=0,
        pos=(38.0, 30.0),
        radius=1.5,
    )
    strong_fragment = _enemy(
        player_id=1,
        blob_id=1,
        pos=(45.0, 30.0),
        radius=5.0,
    )
    candidate = DirectionCandidate(
        family="capture_enemy",
        direction=(1.0, 0.0),
        target_kind="prey",
        target_id="1:0",
        target_pos=target.pos,
    )
    memory = PursuitMemory(
        player_id=1,
        pos=target.pos,
        velocity=(0.0, 0.0),
        target_mass=target.radius * target.radius,
        last_seen_round=100,
    )

    safe_value = _attack_beam_value(
        candidate=candidate,
        own=own,
        enemies=(target,),
        arena_size=60.0,
        previous_directions={},
        memory=memory,
        current_round=100,
    )
    strong_owner_value = _attack_beam_value(
        candidate=candidate,
        own=own,
        enemies=(target, strong_fragment),
        arena_size=60.0,
        previous_directions={},
        memory=memory,
        current_round=100,
    )

    assert strong_owner_value < safe_value


def test_existing_capture_routes_receive_the_same_pursuit_bonus() -> None:
    own = (BlobModel(blob_id=0, pos=(30.0, 30.0), radius=4.0),)
    target = _enemy(
        player_id=1,
        blob_id=0,
        pos=(38.0, 30.0),
        radius=1.5,
    )
    memory = PursuitMemory(
        player_id=1,
        pos=target.pos,
        velocity=(0.0, 0.0),
        target_mass=target.radius * target.radius,
        last_seen_round=100,
    )

    values = [
        _attack_beam_value(
            candidate=DirectionCandidate(
                family=family,
                direction=(1.0, 0.0),
                target_kind="prey",
                target_id="1:0",
                target_pos=target.pos,
                split=split,
            ),
            own=own,
            enemies=(target,),
            arena_size=60.0,
            previous_directions={},
            memory=memory,
            current_round=100,
        )
        for family, split in (
            ("capture_enemy", False),
            ("split_capture", True),
            ("multi_split_capture", True),
        )
    ]

    assert values[0] == values[1] == values[2]
