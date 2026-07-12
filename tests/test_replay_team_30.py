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
from strategies.replay_team_30 import (  # noqa: E402
    PROFILE,
    ReplayTeam30Strategy,
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
    return SimpleNamespace(
        state=SimpleNamespace(
            me=SimpleNamespace(
                player_id=0,
                x=center_x,
                y=center_y,
                radius=math.sqrt(total_mass),
                alive=True,
                blobs={blob.blob_id: blob for blob in own},
            ),
            visible_blobs=list(enemies),
            visible_food=list(food),
            visible_viruses=[],
            map=SimpleNamespace(size=60.0),
            round=10,
            max_rounds=1400,
            rankings=list(range(8)),
        )
    )


def _choose(strategy: ReplayTeam30Strategy, game: SimpleNamespace):
    return strategy.choose(StrategyContext(game=game, query=SimpleNamespace()))


def test_team30_profile_covers_both_validated_matches() -> None:
    assert PROFILE.validation_passed
    assert PROFILE.source_matches == (11724, 11756, 13931, 13935, 13937)


def test_team30_reproduces_raw_nearest_food_vector() -> None:
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=1.0)
    food = FoodModel(food_id=7, pos=(12.5, 9.0))

    decision = _choose(ReplayTeam30Strategy(), _game((own,), food=(food,)))

    assert decision.direction == (2.5, -1.0)
    assert decision.target_id == "7"
    assert not decision.split


def test_team30_chooses_food_nearest_any_actual_fragment() -> None:
    far = BlobModel(blob_id=0, pos=(5.0, 5.0), radius=1.0)
    near = BlobModel(blob_id=1, pos=(20.0, 20.0), radius=1.0)
    food = FoodModel(food_id=7, pos=(21.0, 22.0))

    decision = _choose(
        ReplayTeam30Strategy(),
        _game((far, near), food=(food,)),
    )

    assert decision.direction == (1.0, 2.0)
    assert decision.diagnostics["origin_blob_id"] == near.blob_id


def test_team30_ignores_predator_and_reachable_prey_when_food_exists() -> None:
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
        ReplayTeam30Strategy(),
        _game((own,), food=(food,), enemies=(predator, prey)),
    )

    assert decision.direction == (0.0, 3.0)
    assert decision.target_kind == "food"
    assert not decision.split


def test_team30_retains_normalised_food_heading_without_food() -> None:
    strategy = ReplayTeam30Strategy()
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=1.0)
    food = FoodModel(food_id=7, pos=(13.0, 14.0))
    _choose(strategy, _game((own,), food=(food,)))

    decision = _choose(strategy, _game((own,)))

    assert decision.direction == (0.6, 0.8)
    assert decision.reason == "team30_inertia_fallback"
    assert not decision.split
