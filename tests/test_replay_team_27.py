from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from lib.models.blob_model import BlobModel, VisibleBlobModel  # noqa: E402
from lib.models.food_model import FoodModel  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from strategies.replay_team_27 import (  # noqa: E402
    PROFILE,
    ReplayTeam27Strategy,
)


def _game(
    own: tuple[BlobModel, ...],
    *,
    food: tuple[FoodModel, ...] = (),
    enemies: tuple[VisibleBlobModel, ...] = (),
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
    return SimpleNamespace(
        state=SimpleNamespace(
            me=me,
            visible_blobs=list(enemies),
            visible_food=list(food),
            visible_viruses=[],
            map=SimpleNamespace(size=60.0),
            round=10,
            max_rounds=1400,
            rankings=list(range(8)),
        )
    )


def _choose(strategy: ReplayTeam27Strategy, game: SimpleNamespace):
    return strategy.choose(
        StrategyContext(game=game, query=SimpleNamespace())
    )


def test_team27_uses_validated_all_match_profile() -> None:
    assert PROFILE.validation_passed
    assert PROFILE.source_matches == (11716, 11719, 11752)


def test_team27_moves_to_food_nearest_any_real_fragment() -> None:
    far = BlobModel(blob_id=0, pos=(5.0, 5.0), radius=1.0)
    near = BlobModel(blob_id=1, pos=(20.0, 20.0), radius=1.0)
    food = FoodModel(food_id=7, pos=(22.5, 19.0))

    decision = _choose(
        ReplayTeam27Strategy(),
        _game((far, near), food=(food,)),
    )

    assert decision.direction == (2.5, -1.0)
    assert decision.target_kind == "food"
    assert decision.diagnostics["origin_blob_id"] == near.blob_id
    assert not decision.split


def test_team27_keeps_food_priority_with_predator_and_prey_visible() -> None:
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=2.0)
    predator = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(12.0, 10.0),
        radius=3.0,
    )
    prey = VisibleBlobModel(
        player_id=2,
        team_id=2,
        blob_id=0,
        pos=(9.0, 10.0),
        radius=1.0,
    )
    food = FoodModel(food_id=7, pos=(10.0, 13.0))

    decision = _choose(
        ReplayTeam27Strategy(),
        _game((own,), food=(food,), enemies=(predator, prey)),
    )

    assert decision.direction == (0.0, 3.0)
    assert decision.target_kind == "food"
    assert not decision.split


def test_team27_retains_last_heading_without_visible_food() -> None:
    strategy = ReplayTeam27Strategy()
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=1.0)
    food = FoodModel(food_id=7, pos=(10.0, 13.0))
    _choose(strategy, _game((own,), food=(food,)))

    decision = _choose(strategy, _game((own,)))

    assert decision.direction == (0.0, 1.0)
    assert decision.reason == "team27_inertia_fallback"
    assert not decision.split


def test_team27_never_splits_for_reachable_prey() -> None:
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=3.0)
    prey = VisibleBlobModel(
        player_id=2,
        team_id=2,
        blob_id=0,
        pos=(13.0, 10.0),
        radius=1.0,
    )
    food = FoodModel(food_id=7, pos=(11.0, 10.0))

    decision = _choose(
        ReplayTeam27Strategy(),
        _game((own,), food=(food,), enemies=(prey,)),
    )

    assert not decision.split
