from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from lib.models.blob_model import BlobModel, VisibleBlobModel  # noqa: E402
from lib.models.food_model import FoodModel  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from strategies.replay_team_12 import ReplayTeam12Strategy  # noqa: E402


def _game(
    *,
    own_radius: float = 1.0,
    food: tuple[FoodModel, ...] = (),
    enemies: tuple[VisibleBlobModel, ...] = (),
) -> SimpleNamespace:
    own = BlobModel(blob_id=0, pos=(30.0, 30.0), radius=own_radius)
    state = SimpleNamespace(
        me=SimpleNamespace(
            player_id=3,
            team_id=12,
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


def _choose(strategy: ReplayTeam12Strategy, game: SimpleNamespace):
    return strategy.choose(StrategyContext(game=game, query=SimpleNamespace()))


def test_team_12_uses_all_matches_and_preserves_failed_gate() -> None:
    strategy = ReplayTeam12Strategy()

    assert strategy.profile.team_id == 12
    assert strategy.profile.source_matches == (11694, 11752, 11757)
    assert strategy.profile.validation_passed is False


def test_team_12_clear_regime_chases_nearest_food() -> None:
    decision = _choose(
        ReplayTeam12Strategy(),
        _game(
            food=(
                FoodModel(food_id=1, pos=(34.0, 30.0)),
                FoodModel(food_id=2, pos=(20.0, 30.0)),
            )
        ),
    )

    assert decision.direction[0] > 0.99


def test_team_12_initial_predator_regime_escapes() -> None:
    predator = VisibleBlobModel(
        player_id=1,
        team_id=9,
        blob_id=0,
        pos=(33.0, 30.0),
        radius=2.0,
    )
    decision = _choose(
        ReplayTeam12Strategy(),
        _game(
            food=(FoodModel(food_id=1, pos=(35.0, 30.0)),),
            enemies=(predator,),
        ),
    )

    assert decision.direction[0] < 0.0


def test_team_12_cannot_split_below_engine_mass_threshold() -> None:
    prey = VisibleBlobModel(
        player_id=1,
        team_id=9,
        blob_id=0,
        pos=(33.0, 30.0),
        radius=0.5,
    )
    decision = _choose(
        ReplayTeam12Strategy(),
        _game(own_radius=1.0, enemies=(prey,)),
    )

    assert decision.split is False


def test_team_12_profile_can_emit_high_mass_prey_split() -> None:
    prey = VisibleBlobModel(
        player_id=1,
        team_id=9,
        blob_id=0,
        pos=(33.0, 30.0),
        radius=1.0,
    )
    decision = _choose(
        ReplayTeam12Strategy(),
        _game(own_radius=5.0, enemies=(prey,)),
    )

    assert decision.split is True
