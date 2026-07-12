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
from strategies.replay_team_77 import ReplayTeam77Strategy  # noqa: E402


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
            team_id=77,
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


def _choose(strategy: ReplayTeam77Strategy, game: SimpleNamespace):
    return strategy.choose(StrategyContext(game=game, query=SimpleNamespace()))


def test_team_77_uses_every_source_and_preserves_failed_gate() -> None:
    strategy = ReplayTeam77Strategy()

    assert strategy.profile.team_id == 77
    assert strategy.profile.source_matches == (11673, 11710, 11724)
    assert strategy.profile.validation_passed is False


def test_team_77_safe_direction_tracks_nearest_food() -> None:
    decision = _choose(
        ReplayTeam77Strategy(),
        _game(food=(FoodModel(food_id=1, pos=(30.0, 35.0)),)),
    )

    assert math.isclose(math.hypot(*decision.direction), 1.0)
    assert decision.direction[1] > 0.99


def test_team_77_safe_direction_quickly_retargets_reversed_food() -> None:
    strategy = ReplayTeam77Strategy()
    first = _choose(
        strategy,
        _game(food=(FoodModel(food_id=1, pos=(35.0, 30.0)),)),
    )
    second = _choose(
        strategy,
        _game(food=(FoodModel(food_id=2, pos=(25.0, 30.0)),)),
    )

    assert first.direction[0] > 0.99
    assert second.direction[0] < -0.99


def test_team_77_profile_records_no_observed_splits() -> None:
    strategy = ReplayTeam77Strategy()

    assert math.isinf(strategy.profile.split_threshold)
    assert strategy.profile.split_f1 == 1.0


def test_team_77_never_splits_even_with_mass_eligible_prey() -> None:
    prey = VisibleBlobModel(
        player_id=1,
        team_id=9,
        blob_id=0,
        pos=(31.0, 30.0),
        radius=1.0,
    )
    decision = _choose(
        ReplayTeam77Strategy(),
        _game(own_radius=10.0, enemies=(prey,)),
    )

    assert decision.split is False
