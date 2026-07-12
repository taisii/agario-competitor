from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from lib.models.blob_model import BlobModel, VisibleBlobModel  # noqa: E402
from lib.models.food_model import FoodModel  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from strategies.replay_team_53 import ReplayTeam53Strategy  # noqa: E402


def _game(
    *,
    own_radius: float = 1.0,
    round_number: int = 100,
    food: tuple[FoodModel, ...] = (),
    enemies: tuple[VisibleBlobModel, ...] = (),
) -> SimpleNamespace:
    own = BlobModel(blob_id=0, pos=(30.0, 30.0), radius=own_radius)
    state = SimpleNamespace(
        me=SimpleNamespace(
            player_id=6,
            team_id=53,
            pos=(30.0, 30.0),
            radius=own_radius,
            alive=True,
            blobs={0: own},
        ),
        visible_blobs=list(enemies),
        visible_food=list(food),
        visible_viruses=[],
        map=SimpleNamespace(size=60.0),
        round=round_number,
        max_rounds=1400,
        rankings=list(range(8)),
    )
    return SimpleNamespace(state=state)


def _choose(strategy: ReplayTeam53Strategy, game: SimpleNamespace):
    return strategy.choose(StrategyContext(game=game, query=SimpleNamespace()))


def test_team_53_chases_nearest_food_in_clear_view() -> None:
    decision = _choose(
        ReplayTeam53Strategy(),
        _game(
            food=(
                FoodModel(food_id=1, pos=(35.0, 30.0)),
                FoodModel(food_id=2, pos=(20.0, 30.0)),
            )
        ),
    )

    assert decision.direction[0] > 0.99
    assert abs(decision.direction[1]) < 0.02
    assert decision.reason == "team_53_food_chase"


def test_team_53_uses_predator_field_before_food() -> None:
    predator = VisibleBlobModel(
        player_id=1,
        team_id=9,
        blob_id=0,
        pos=(33.0, 30.0),
        radius=2.0,
    )
    decision = _choose(
        ReplayTeam53Strategy(),
        _game(
            food=(FoodModel(food_id=1, pos=(35.0, 30.0)),),
            enemies=(predator,),
        ),
    )

    assert decision.direction[0] < -0.99
    assert decision.reason == "team_53_predator_escape"
    assert decision.split is False


def test_team_53_chases_safe_prey() -> None:
    prey = VisibleBlobModel(
        player_id=1,
        team_id=9,
        blob_id=0,
        pos=(34.0, 30.0),
        radius=1.0,
    )
    decision = _choose(
        ReplayTeam53Strategy(),
        _game(own_radius=2.0, enemies=(prey,)),
    )

    assert decision.direction[0] > 0.9
    assert decision.reason == "team_53_prey_chase"


def test_team_53_split_roll_is_deterministic_and_sparse() -> None:
    prey = VisibleBlobModel(
        player_id=1,
        team_id=9,
        blob_id=0,
        pos=(35.0, 30.0),
        radius=1.0,
    )
    first = ReplayTeam53Strategy()
    second = ReplayTeam53Strategy()
    first_results = [
        _choose(
            first,
            _game(own_radius=2.0, round_number=round_number, enemies=(prey,)),
        ).split
        for round_number in range(1000)
    ]
    second_results = [
        _choose(
            second,
            _game(own_radius=2.0, round_number=round_number, enemies=(prey,)),
        ).split
        for round_number in range(1000)
    ]

    assert first_results == second_results
    assert 20 <= sum(first_results) <= 70


def test_team_53_never_splits_while_predator_is_visible() -> None:
    prey = VisibleBlobModel(
        player_id=1,
        team_id=9,
        blob_id=0,
        pos=(34.0, 30.0),
        radius=1.0,
    )
    predator = VisibleBlobModel(
        player_id=2,
        team_id=10,
        blob_id=0,
        pos=(27.0, 30.0),
        radius=3.0,
    )
    strategy = ReplayTeam53Strategy()

    decisions = [
        _choose(
            strategy,
            _game(
                own_radius=2.0,
                round_number=round_number,
                enemies=(prey, predator),
            ),
        )
        for round_number in range(200)
    ]

    assert not any(decision.split for decision in decisions)
