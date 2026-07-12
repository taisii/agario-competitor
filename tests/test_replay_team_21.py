from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from lib.models.blob_model import BlobModel, VisibleBlobModel  # noqa: E402
from lib.models.food_model import FoodModel  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from strategies.replay_team_21 import ReplayTeam21Strategy  # noqa: E402


def _game(
    *,
    own_radius: float = 1.0,
    food: tuple[FoodModel, ...] = (),
    enemies: tuple[VisibleBlobModel, ...] = (),
) -> SimpleNamespace:
    own = BlobModel(blob_id=0, pos=(30.0, 30.0), radius=own_radius)
    state = SimpleNamespace(
        me=SimpleNamespace(
            player_id=2,
            team_id=21,
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


def _choose(strategy: ReplayTeam21Strategy, game: SimpleNamespace):
    return strategy.choose(StrategyContext(game=game, query=SimpleNamespace()))


def test_team_21_uses_all_sources_and_preserves_split_failure() -> None:
    strategy = ReplayTeam21Strategy()

    assert strategy.profile.team_id == 21
    assert strategy.profile.source_matches == (11646, 11654, 11698, 11719)
    assert strategy.profile.validation_passed is False


def test_team_21_chases_nearest_food_when_safe() -> None:
    decision = _choose(
        ReplayTeam21Strategy(),
        _game(
            food=(
                FoodModel(food_id=1, pos=(34.0, 30.0)),
                FoodModel(food_id=2, pos=(20.0, 30.0)),
            )
        ),
    )

    assert decision.direction[0] > 0.99


def test_team_21_prioritises_predator_escape() -> None:
    predator = VisibleBlobModel(
        player_id=1,
        team_id=9,
        blob_id=0,
        pos=(33.0, 30.0),
        radius=2.0,
    )
    decision = _choose(
        ReplayTeam21Strategy(),
        _game(
            food=(FoodModel(food_id=1, pos=(35.0, 30.0)),),
            enemies=(predator,),
        ),
    )

    assert decision.direction[0] < -0.9


def test_team_21_cannot_split_below_engine_mass_threshold() -> None:
    prey = VisibleBlobModel(
        player_id=1,
        team_id=9,
        blob_id=0,
        pos=(33.0, 30.0),
        radius=0.5,
    )
    decision = _choose(
        ReplayTeam21Strategy(),
        _game(own_radius=1.0, enemies=(prey,)),
    )

    assert decision.split is False


def test_team_21_profile_can_emit_mass_eligible_split() -> None:
    prey = VisibleBlobModel(
        player_id=1,
        team_id=9,
        blob_id=0,
        pos=(33.0, 30.0),
        radius=0.5,
    )
    decision = _choose(
        ReplayTeam21Strategy(),
        _game(
            own_radius=3.0,
            enemies=(prey,),
        ),
    )

    assert decision.split is True
