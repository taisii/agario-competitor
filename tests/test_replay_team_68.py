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
from strategies.replay_team_68 import (  # noqa: E402
    SOURCE_MATCHES,
    ReplayTeam68Strategy,
)


def _enemy(*, radius: float, pos: tuple[float, float]) -> VisibleBlobModel:
    return VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=pos,
        radius=radius,
    )


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
            team_id=68,
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


def _choose(strategy: ReplayTeam68Strategy, game: SimpleNamespace):
    return strategy.choose(StrategyContext(game=game, query=SimpleNamespace()))


def test_team68_uses_both_source_matches_and_preserves_split_failure() -> None:
    strategy = ReplayTeam68Strategy()

    assert SOURCE_MATCHES == (11654, 11697)
    assert strategy.profile.source_matches == SOURCE_MATCHES
    assert not strategy.profile.validation_passed


def test_team68_chases_nearest_food_without_actionable_entities() -> None:
    decision = _choose(
        ReplayTeam68Strategy(),
        _game(food=(FoodModel(food_id=1, pos=(35.0, 30.0)),)),
    )

    assert math.isclose(math.hypot(*decision.direction), 1.0)
    assert decision.direction[0] > 0.99
    assert decision.target_kind == "food"


def test_team68_ignores_predator_below_observed_escape_ratio() -> None:
    decision = _choose(
        ReplayTeam68Strategy(),
        _game(
            food=(FoodModel(food_id=1, pos=(35.0, 30.0)),),
            enemies=(_enemy(radius=1.2, pos=(32.0, 30.0)),),
        ),
    )

    assert decision.direction[0] > 0.99
    assert decision.target_kind == "food"


def test_team68_escapes_predator_above_observed_escape_ratio() -> None:
    decision = _choose(
        ReplayTeam68Strategy(),
        _game(
            food=(FoodModel(food_id=1, pos=(35.0, 30.0)),),
            enemies=(_enemy(radius=1.5, pos=(32.0, 30.0)),),
        ),
    )

    assert decision.direction[0] < -0.99
    assert decision.target_kind == "escape"


def test_team68_prey_priority_overrides_strong_predator() -> None:
    prey = _enemy(radius=0.5, pos=(30.0, 32.0))
    predator = VisibleBlobModel(
        player_id=2,
        team_id=2,
        blob_id=0,
        pos=(32.0, 30.0),
        radius=2.0,
    )
    decision = _choose(
        ReplayTeam68Strategy(),
        _game(enemies=(prey, predator)),
    )

    assert decision.direction[1] > 0.99
    assert decision.target_kind == "prey"


def test_team68_cannot_split_below_engine_mass_threshold() -> None:
    decision = _choose(
        ReplayTeam68Strategy(),
        _game(
            own_radius=1.0,
            enemies=(_enemy(radius=0.5, pos=(31.0, 30.0)),),
        ),
    )

    assert not decision.split
