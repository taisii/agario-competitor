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
from strategies.replay_team_28 import ReplayTeam28Strategy  # noqa: E402


def _game(
    *,
    own_radius: float = 1.0,
    food: tuple[FoodModel, ...] = (),
    enemies: tuple[VisibleBlobModel, ...] = (),
    viruses: tuple[VirusModel, ...] = (),
) -> SimpleNamespace:
    own = BlobModel(blob_id=0, pos=(30.0, 30.0), radius=own_radius)
    state = SimpleNamespace(
        me=SimpleNamespace(
            player_id=3,
            team_id=28,
            pos=(30.0, 30.0),
            radius=own_radius,
            alive=True,
            blobs={0: own},
        ),
        visible_blobs=list(enemies),
        visible_food=list(food),
        visible_viruses=list(viruses),
        map=SimpleNamespace(size=60.0),
        round=100,
        max_rounds=1400,
        rankings=list(range(8)),
    )
    return SimpleNamespace(state=state)


def _choose(strategy: ReplayTeam28Strategy, game: SimpleNamespace):
    return strategy.choose(StrategyContext(game=game, query=SimpleNamespace()))


def test_team_28_uses_validated_official_profile() -> None:
    strategy = ReplayTeam28Strategy()

    assert strategy.profile.team_id == 28
    assert strategy.profile.source_matches == (11716, 11753, 13936)
    assert strategy.profile.validation_passed is True


def test_team_28_chases_nearest_food() -> None:
    decision = _choose(
        ReplayTeam28Strategy(),
        _game(
            food=(
                FoodModel(food_id=1, pos=(34.0, 30.0)),
                FoodModel(food_id=2, pos=(20.0, 30.0)),
            )
        ),
    )

    assert decision.direction[0] > 0.99
    assert abs(decision.direction[1]) < 0.02


def test_team_28_ignores_predator_when_selecting_food() -> None:
    predator = VisibleBlobModel(
        player_id=1,
        team_id=9,
        blob_id=0,
        pos=(33.0, 30.0),
        radius=2.0,
    )
    decision = _choose(
        ReplayTeam28Strategy(),
        _game(
            food=(FoodModel(food_id=1, pos=(34.0, 30.0)),),
            enemies=(predator,),
        ),
    )

    assert decision.direction[0] > 0.99


def test_team_28_ignores_edible_virus_when_selecting_food() -> None:
    virus = VirusModel(virus_id=1, pos=(26.0, 30.0), radius=1.5)
    decision = _choose(
        ReplayTeam28Strategy(),
        _game(
            own_radius=2.0,
            food=(FoodModel(food_id=1, pos=(34.0, 30.0)),),
            viruses=(virus,),
        ),
    )

    assert decision.direction[0] > 0.99


def test_team_28_never_splits_with_safe_edible_prey() -> None:
    prey = VisibleBlobModel(
        player_id=1,
        team_id=9,
        blob_id=0,
        pos=(33.0, 30.0),
        radius=1.0,
    )
    decision = _choose(
        ReplayTeam28Strategy(),
        _game(
            own_radius=2.0,
            food=(FoodModel(food_id=1, pos=(34.0, 30.0)),),
            enemies=(prey,),
        ),
    )

    assert decision.split is False
