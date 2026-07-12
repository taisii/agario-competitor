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
from strategies.replay_team_24 import ReplayTeam24Strategy  # noqa: E402


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
            team_id=24,
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


def _choose(strategy: ReplayTeam24Strategy, game: SimpleNamespace):
    return strategy.choose(StrategyContext(game=game, query=SimpleNamespace()))


def test_team24_profile_covers_all_observed_matches() -> None:
    strategy = ReplayTeam24Strategy()

    assert strategy.profile.source_matches == (11719, 11739, 11753)


def test_team24_preserves_failed_replay_validation_flag() -> None:
    assert not ReplayTeam24Strategy().profile.validation_passed


def test_team24_direction_is_unit_length() -> None:
    decision = _choose(
        ReplayTeam24Strategy(),
        _game(food=(FoodModel(food_id=1, pos=(35.0, 30.0)),)),
    )

    assert math.isclose(math.hypot(*decision.direction), 1.0)


def test_team24_tracks_reversed_food_direction() -> None:
    strategy = ReplayTeam24Strategy()
    east = _choose(
        strategy,
        _game(food=(FoodModel(food_id=1, pos=(35.0, 30.0)),)),
    )
    west = _choose(
        strategy,
        _game(food=(FoodModel(food_id=2, pos=(25.0, 30.0)),)),
    )

    assert east.direction[0] > 0.0
    assert west.direction[0] < 0.0


def test_team24_cannot_split_below_engine_mass_threshold() -> None:
    prey = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(32.0, 30.0),
        radius=0.5,
    )
    decision = _choose(
        ReplayTeam24Strategy(),
        _game(own_radius=1.0, enemies=(prey,)),
    )

    assert not decision.split


def test_team24_profile_can_emit_high_mass_prey_split() -> None:
    prey = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(32.0, 30.0),
        radius=1.0,
    )
    decision = _choose(
        ReplayTeam24Strategy(),
        _game(own_radius=3.0, enemies=(prey,)),
    )

    assert decision.split
