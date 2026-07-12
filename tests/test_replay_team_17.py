from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from lib.models.blob_model import BlobModel, VisibleBlobModel  # noqa: E402
from lib.models.food_model import FoodModel  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from strategies.replay_team_17 import ReplayTeam17Strategy  # noqa: E402


def _game(
    *,
    own_radius: float = 1.0,
    food: tuple[FoodModel, ...] = (),
    enemies: tuple[VisibleBlobModel, ...] = (),
) -> SimpleNamespace:
    own = BlobModel(blob_id=0, pos=(30.0, 30.0), radius=own_radius)
    state = SimpleNamespace(
        me=SimpleNamespace(
            player_id=5,
            team_id=17,
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


def _choose(strategy: ReplayTeam17Strategy, game: SimpleNamespace):
    return strategy.choose(StrategyContext(game=game, query=SimpleNamespace()))


def test_team_17_uses_validated_official_profile() -> None:
    strategy = ReplayTeam17Strategy()

    assert strategy.profile.team_id == 17
    assert strategy.profile.source_matches == (11667, 11697, 11756, 13934, 13935, 13937, 13940)
    assert strategy.profile.validation_passed is True


def test_team_17_chases_nearest_food_when_safe() -> None:
    decision = _choose(
        ReplayTeam17Strategy(),
        _game(
            food=(
                FoodModel(food_id=1, pos=(34.0, 30.0)),
                FoodModel(food_id=2, pos=(20.0, 30.0)),
            )
        ),
    )

    assert decision.direction[0] > 0.85
    assert abs(decision.direction[1]) < 0.15


def test_team_17_prioritises_predator_escape_over_food() -> None:
    predator = VisibleBlobModel(
        player_id=1,
        team_id=9,
        blob_id=0,
        pos=(33.0, 30.0),
        radius=2.0,
    )
    decision = _choose(
        ReplayTeam17Strategy(),
        _game(
            food=(FoodModel(food_id=1, pos=(34.0, 30.0)),),
            enemies=(predator,),
        ),
    )

    assert decision.direction[0] < -0.85


def test_team_17_keeps_food_priority_when_safe_prey_is_visible() -> None:
    prey = VisibleBlobModel(
        player_id=1,
        team_id=9,
        blob_id=0,
        pos=(34.0, 30.0),
        radius=1.0,
    )
    decision = _choose(
        ReplayTeam17Strategy(),
        _game(
            own_radius=2.0,
            food=(FoodModel(food_id=1, pos=(27.0, 30.0)),),
            enemies=(prey,),
        ),
    )

    assert decision.direction[0] < -0.85


def test_team_17_never_splits_even_when_split_kill_is_available() -> None:
    prey = VisibleBlobModel(
        player_id=1,
        team_id=9,
        blob_id=0,
        pos=(33.0, 30.0),
        radius=1.0,
    )
    strategy = ReplayTeam17Strategy()

    decisions = [
        _choose(
            strategy,
            _game(own_radius=2.0, enemies=(prey,)),
        )
        for _ in range(20)
    ]

    assert not any(decision.split for decision in decisions)
