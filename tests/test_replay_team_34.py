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
from strategies.replay_team_34 import ReplayTeam34Strategy  # noqa: E402


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
            team_id=34,
            pos=(30.0, 30.0),
            radius=own_radius,
            alive=True,
            blobs={0: own},
        ),
        visible_blobs=list(enemies),
        visible_food=list(food),
        visible_viruses=[],
        map=SimpleNamespace(size=60.0),
        round=700,
        max_rounds=1400,
        rankings=list(range(8)),
    )
    return SimpleNamespace(state=state)


def _choose(strategy: ReplayTeam34Strategy, game: SimpleNamespace):
    return strategy.choose(StrategyContext(game=game, query=SimpleNamespace()))


def test_team_34_marks_single_trace_validation_failure() -> None:
    strategy = ReplayTeam34Strategy()

    assert strategy.profile.team_id == 34
    assert strategy.profile.source_matches == (11681, 13938)
    assert strategy.profile.validation_passed is False


def test_team_34_clear_regime_chases_nearest_food() -> None:
    decision = _choose(
        ReplayTeam34Strategy(),
        _game(
            food=(
                FoodModel(food_id=1, pos=(34.0, 30.0)),
                FoodModel(food_id=2, pos=(20.0, 30.0)),
            )
        ),
    )

    assert decision.direction[0] > 0.99


def test_team_34_initial_predator_regime_escapes() -> None:
    predator = VisibleBlobModel(
        player_id=1,
        team_id=9,
        blob_id=0,
        pos=(33.0, 30.0),
        radius=2.0,
    )
    decision = _choose(
        ReplayTeam34Strategy(),
        _game(
            food=(FoodModel(food_id=1, pos=(35.0, 30.0)),),
            enemies=(predator,),
        ),
    )

    assert decision.direction[0] < 0.0


def test_team_34_always_emits_unit_direction() -> None:
    strategy = ReplayTeam34Strategy()
    decision = _choose(
        strategy,
        _game(food=(FoodModel(food_id=1, pos=(34.0, 34.0)),)),
    )

    assert math.isclose(math.hypot(*decision.direction), 1.0)


def test_team_34_cannot_split_below_engine_mass_threshold() -> None:
    prey = VisibleBlobModel(
        player_id=1,
        team_id=9,
        blob_id=0,
        pos=(33.0, 30.0),
        radius=0.5,
    )
    decision = _choose(
        ReplayTeam34Strategy(),
        _game(own_radius=1.0, enemies=(prey,)),
    )

    assert decision.split is False
