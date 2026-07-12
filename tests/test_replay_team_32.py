from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from lib.models.blob_model import BlobModel, VisibleBlobModel  # noqa: E402
from lib.models.food_model import FoodModel  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from strategies.replay_team_32 import ReplayTeam32Strategy  # noqa: E402


def _game(
    *,
    own_radius: float = 1.0,
    food: tuple[FoodModel, ...] = (),
    enemies: tuple[VisibleBlobModel, ...] = (),
) -> SimpleNamespace:
    own = BlobModel(blob_id=0, pos=(30.0, 30.0), radius=own_radius)
    state = SimpleNamespace(
        me=SimpleNamespace(
            player_id=1,
            team_id=32,
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


def _choose(strategy: ReplayTeam32Strategy, game: SimpleNamespace):
    return strategy.choose(StrategyContext(game=game, query=SimpleNamespace()))


def test_team_32_uses_all_sources_and_preserves_failed_gate() -> None:
    strategy = ReplayTeam32Strategy()

    assert strategy.profile.team_id == 32
    assert strategy.profile.source_matches == (11667, 11745, 11752)
    assert strategy.profile.validation_passed is False


def test_team_32_safe_initial_direction_is_food_attracted() -> None:
    decision = _choose(
        ReplayTeam32Strategy(),
        _game(food=(FoodModel(food_id=1, pos=(35.0, 30.0)),)),
    )

    assert decision.direction[0] > 0.8


def test_team_32_initial_predator_regime_escapes() -> None:
    predator = VisibleBlobModel(
        player_id=2,
        team_id=9,
        blob_id=0,
        pos=(33.0, 30.0),
        radius=2.0,
    )
    decision = _choose(
        ReplayTeam32Strategy(),
        _game(
            food=(FoodModel(food_id=1, pos=(35.0, 30.0)),),
            enemies=(predator,),
        ),
    )

    assert decision.direction[0] < 0.0


def test_team_32_safe_prey_can_override_opposed_food() -> None:
    prey = VisibleBlobModel(
        player_id=2,
        team_id=9,
        blob_id=0,
        pos=(34.0, 30.0),
        radius=1.0,
    )
    decision = _choose(
        ReplayTeam32Strategy(),
        _game(
            own_radius=2.0,
            food=(FoodModel(food_id=1, pos=(25.0, 30.0)),),
            enemies=(prey,),
        ),
    )

    assert decision.direction[0] > 0.0


def test_team_32_cannot_split_below_engine_mass_threshold() -> None:
    prey = VisibleBlobModel(
        player_id=2,
        team_id=9,
        blob_id=0,
        pos=(33.0, 30.0),
        radius=0.5,
    )
    decision = _choose(
        ReplayTeam32Strategy(),
        _game(own_radius=1.0, enemies=(prey,)),
    )

    assert decision.split is False
