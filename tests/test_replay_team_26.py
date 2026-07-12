from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from lib.models.blob_model import BlobModel, VisibleBlobModel  # noqa: E402
from lib.models.food_model import FoodModel  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from strategies.replay_team_26 import ReplayTeam26Strategy  # noqa: E402


def _game(
    *,
    own_radius: float = 1.0,
    food: tuple[FoodModel, ...] = (),
    enemies: tuple[VisibleBlobModel, ...] = (),
) -> SimpleNamespace:
    own = BlobModel(blob_id=0, pos=(30.0, 30.0), radius=own_radius)
    state = SimpleNamespace(
        me=SimpleNamespace(
            player_id=4,
            team_id=26,
            pos=(30.0, 30.0),
            radius=own_radius,
            alive=True,
            blobs={0: own},
        ),
        visible_blobs=list(enemies),
        visible_food=list(food),
        visible_viruses=[],
        map=SimpleNamespace(size=60.0),
        round=100,
        max_rounds=1400,
        rankings=list(range(8)),
    )
    return SimpleNamespace(state=state)


def _choose(strategy: ReplayTeam26Strategy, game: SimpleNamespace):
    return strategy.choose(StrategyContext(game=game, query=SimpleNamespace()))


def test_team_26_uses_validated_official_profile() -> None:
    strategy = ReplayTeam26Strategy()

    assert strategy.profile.team_id == 26
    assert strategy.profile.source_matches == (11679, 11698)
    assert strategy.profile.validation_passed is True


def test_team_26_chases_nearest_food() -> None:
    decision = _choose(
        ReplayTeam26Strategy(),
        _game(
            food=(
                FoodModel(food_id=1, pos=(34.0, 30.0)),
                FoodModel(food_id=2, pos=(20.0, 30.0)),
            )
        ),
    )

    assert decision.direction[0] > 0.999
    assert abs(decision.direction[1]) < 0.001


def test_team_26_keeps_food_target_when_predator_is_visible() -> None:
    predator = VisibleBlobModel(
        player_id=1,
        team_id=9,
        blob_id=0,
        pos=(33.0, 30.0),
        radius=2.0,
    )
    decision = _choose(
        ReplayTeam26Strategy(),
        _game(
            food=(FoodModel(food_id=1, pos=(34.0, 30.0)),),
            enemies=(predator,),
        ),
    )

    assert decision.direction[0] > 0.999


def test_team_26_switches_immediately_when_nearest_food_changes() -> None:
    strategy = ReplayTeam26Strategy()
    first = _choose(
        strategy,
        _game(food=(FoodModel(food_id=1, pos=(34.0, 30.0)),)),
    )
    second = _choose(
        strategy,
        _game(food=(FoodModel(food_id=2, pos=(26.0, 30.0)),)),
    )

    assert first.direction[0] > 0.999
    assert second.direction[0] < -0.999


def test_team_26_never_splits() -> None:
    prey = VisibleBlobModel(
        player_id=1,
        team_id=9,
        blob_id=0,
        pos=(33.0, 30.0),
        radius=1.0,
    )
    decision = _choose(
        ReplayTeam26Strategy(),
        _game(
            own_radius=2.0,
            food=(FoodModel(food_id=1, pos=(34.0, 30.0)),),
            enemies=(prey,),
        ),
    )

    assert decision.split is False
