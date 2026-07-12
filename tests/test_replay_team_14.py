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
from strategies.replay_team_14 import ReplayTeam14Strategy  # noqa: E402


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
            team_id=14,
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
        round=900,
        max_rounds=1400,
        rankings=list(range(8)),
    )
    return SimpleNamespace(state=state)


def _choose(strategy: ReplayTeam14Strategy, game: SimpleNamespace):
    return strategy.choose(StrategyContext(game=game, query=SimpleNamespace()))


def test_team14_profile_covers_all_observed_matches() -> None:
    strategy = ReplayTeam14Strategy()

    assert strategy.profile.source_matches == (
        11646,
        11673,
        11694,
        11697,
        13933,
        13937,
        13939,
    )


def test_team14_preserves_failed_replay_validation_flag() -> None:
    assert not ReplayTeam14Strategy().profile.validation_passed


def test_team14_prey_regime_moves_toward_edible_blob() -> None:
    prey = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(34.0, 30.0),
        radius=1.0,
    )
    decision = _choose(
        ReplayTeam14Strategy(),
        _game(own_radius=3.0, enemies=(prey,)),
    )

    assert decision.direction[0] > 0.9
    assert math.isclose(math.hypot(*decision.direction), 1.0)


def test_team14_predator_regime_moves_away_from_threat() -> None:
    predator = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(34.0, 30.0),
        radius=3.0,
    )
    decision = _choose(
        ReplayTeam14Strategy(),
        _game(own_radius=1.0, enemies=(predator,)),
    )

    assert decision.direction[0] < 0.0
    assert math.isclose(math.hypot(*decision.direction), 1.0)


def test_team14_cannot_split_below_engine_mass_threshold() -> None:
    prey = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(32.0, 30.0),
        radius=0.5,
    )
    decision = _choose(
        ReplayTeam14Strategy(),
        _game(own_radius=1.0, enemies=(prey,)),
    )

    assert not decision.split


def test_team14_profile_can_emit_high_mass_prey_split() -> None:
    prey = VisibleBlobModel(
        player_id=1,
        team_id=1,
        blob_id=0,
        pos=(32.0, 30.0),
        radius=1.0,
    )
    decision = _choose(
        ReplayTeam14Strategy(),
        _game(own_radius=3.0, enemies=(prey,)),
    )

    assert decision.split
