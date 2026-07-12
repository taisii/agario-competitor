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
from strategies.replay_team_55 import ReplayTeam55Strategy  # noqa: E402


def _game(
    *,
    own_radius: float = 1.0,
    food: tuple[FoodModel, ...] = (),
    enemies: tuple[VisibleBlobModel, ...] = (),
) -> SimpleNamespace:
    own = BlobModel(blob_id=0, pos=(30.0, 30.0), radius=own_radius)
    state = SimpleNamespace(
        me=SimpleNamespace(
            player_id=0,
            team_id=55,
            x=30.0,
            y=30.0,
            radius=own_radius,
            alive=True,
            blobs={0: own},
        ),
        visible_blobs=list(enemies),
        visible_food=list(food),
        visible_viruses=[],
        map=SimpleNamespace(size=60.0),
        round=1000,
        max_rounds=1400,
        rankings=list(range(8)),
    )
    return SimpleNamespace(state=state)


def _choose(strategy: ReplayTeam55Strategy, game: SimpleNamespace):
    return strategy.choose(StrategyContext(game=game, query=SimpleNamespace()))


def _heading_bin(direction: tuple[float, float]) -> float:
    angle = math.atan2(direction[1], direction[0]) % math.tau
    return angle * 24.0 / math.tau


def test_team55_profile_covers_all_observed_matches() -> None:
    strategy = ReplayTeam55Strategy()

    assert strategy.profile.source_matches == (11679, 11725)


def test_team55_preserves_failed_direction_validation_flag() -> None:
    assert not ReplayTeam55Strategy().profile.validation_passed


def test_team55_direction_is_unit_length_and_on_observed_grid() -> None:
    decision = _choose(
        ReplayTeam55Strategy(),
        _game(food=(FoodModel(food_id=1, pos=(34.0, 32.0)),)),
    )

    assert math.isclose(math.hypot(*decision.direction), 1.0)
    assert math.isclose(_heading_bin(decision.direction), round(_heading_bin(decision.direction)))


def test_team55_every_regime_stays_on_twenty_four_heading_grid() -> None:
    predator = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(32.0, 30.0),
        radius=2.0,
    )
    strategy = ReplayTeam55Strategy()
    decisions = (
        _choose(strategy, _game(food=(FoodModel(food_id=1, pos=(35.0, 30.0)),))),
        _choose(strategy, _game(enemies=(predator,))),
    )

    assert all(
        math.isclose(_heading_bin(decision.direction), round(_heading_bin(decision.direction)))
        for decision in decisions
    )


def test_team55_never_splits_even_with_edible_prey() -> None:
    prey = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(32.0, 30.0),
        radius=1.0,
    )
    decision = _choose(
        ReplayTeam55Strategy(),
        _game(own_radius=4.0, enemies=(prey,)),
    )

    assert not decision.split
