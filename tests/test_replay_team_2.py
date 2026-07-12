from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from lib.models.blob_model import BlobModel, VisibleBlobModel  # noqa: E402
from lib.models.food_model import FoodModel  # noqa: E402
from lib.models.virus_model import VirusModel  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from strategies.replay_team_2 import ReplayTeam2Strategy  # noqa: E402


def _game(
    own: tuple[BlobModel, ...],
    *,
    enemies: tuple[VisibleBlobModel, ...] = (),
    food: tuple[FoodModel, ...] = (),
    viruses: tuple[VirusModel, ...] = (),
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
            visible_viruses=list(viruses),
            map=SimpleNamespace(size=60.0),
            round=10,
            max_rounds=1400,
            rankings=[0, 1],
        )
    )


def _choose(strategy: ReplayTeam2Strategy, game: SimpleNamespace):
    return strategy.choose(StrategyContext(game=game, query=SimpleNamespace()))


def test_team2_uses_food_nearest_to_any_real_fragment() -> None:
    large = BlobModel(blob_id=0, pos=(5.0, 5.0), radius=4.0)
    small = BlobModel(blob_id=1, pos=(20.0, 20.0), radius=1.0)
    food = FoodModel(food_id=7, pos=(21.5, 19.5))

    decision = _choose(ReplayTeam2Strategy(), _game((large, small), food=(food,)))

    assert decision.direction == (1.5, -0.5)
    assert decision.target_kind == "food"
    assert decision.diagnostics["origin_blob_id"] == small.blob_id
    assert not decision.split


def test_team2_keeps_nearest_food_authoritative_with_other_entities_visible() -> None:
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=3.0)
    predator = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(12.0, 10.0),
        radius=4.0,
    )
    prey = VisibleBlobModel(
        player_id=2,
        team_id=2,
        blob_id=0,
        pos=(9.0, 10.0),
        radius=1.0,
    )
    virus = VirusModel(virus_id=3, pos=(10.0, 9.0), radius=1.5)
    food = FoodModel(food_id=7, pos=(10.0, 12.0))

    decision = _choose(
        ReplayTeam2Strategy(),
        _game(
            (own,),
            enemies=(predator, prey),
            food=(food,),
            viruses=(virus,),
        ),
    )

    assert decision.direction == (0.0, 2.0)
    assert decision.target_kind == "food"
    assert not decision.split


def test_team2_never_splits_even_when_large_with_edible_prey() -> None:
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=5.0)
    prey = VisibleBlobModel(
        player_id=2,
        team_id=2,
        blob_id=0,
        pos=(12.0, 10.0),
        radius=1.0,
    )
    food = FoodModel(food_id=7, pos=(10.0, 11.0))

    decision = _choose(
        ReplayTeam2Strategy(),
        _game((own,), enemies=(prey,), food=(food,)),
    )

    assert not decision.split
    assert decision.direction == (0.0, 1.0)


def test_team2_retains_previous_heading_without_visible_food() -> None:
    strategy = ReplayTeam2Strategy()
    own = BlobModel(blob_id=0, pos=(10.0, 10.0), radius=1.0)
    food = FoodModel(food_id=7, pos=(10.0, 12.0))

    first = _choose(strategy, _game((own,), food=(food,)))
    fallback = _choose(strategy, _game((own,)))

    assert first.direction == (0.0, 2.0)
    assert fallback.direction == (0.0, 1.0)
    assert fallback.reason == "team2_inertia_fallback"
    assert not fallback.split
