from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from lib.models.blob_model import BlobModel, VisibleBlobModel  # noqa: E402
from lib.models.food_model import FoodModel  # noqa: E402
from lib.models.virus_model import VirusModel  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from strategies.replay_team_58 import ReplayTeam58Strategy  # noqa: E402


def _game(
    *,
    own_radius: float = 1.0,
    food: tuple[FoodModel, ...] = (),
    enemies: tuple[VisibleBlobModel, ...] = (),
    viruses: tuple[VirusModel, ...] = (),
) -> SimpleNamespace:
    own = BlobModel(blob_id=0, pos=(30.0, 30.0), radius=own_radius)
    state = SimpleNamespace(
        me=SimpleNamespace(
            player_id=2,
            team_id=58,
            pos=(30.0, 30.0),
            radius=own_radius,
            alive=True,
            blobs={0: own},
        ),
        visible_blobs=list(enemies),
        visible_food=list(food),
        visible_viruses=list(viruses),
        map=SimpleNamespace(size=60.0),
        round=700,
        max_rounds=1400,
        rankings=list(range(8)),
    )
    return SimpleNamespace(state=state)


def _choose(strategy: ReplayTeam58Strategy, game: SimpleNamespace):
    return strategy.choose(StrategyContext(game=game, query=SimpleNamespace()))


def test_team_58_uses_all_seven_sources_and_preserves_failed_gate() -> None:
    strategy = ReplayTeam58Strategy()

    assert strategy.profile.team_id == 58
    assert strategy.profile.source_matches == (
        11667, 11694, 13939, 14015, 14048, 14054, 14062,
    )
    assert strategy.profile.validation_passed is False


def test_team_58_initial_safe_direction_is_food_attracted() -> None:
    decision = _choose(
        ReplayTeam58Strategy(),
        _game(food=(FoodModel(food_id=1, pos=(35.0, 30.0)),)),
    )

    assert decision.direction[0] > 0.8


def test_team_58_autonomous_profile_retargets_reversed_food() -> None:
    strategy = ReplayTeam58Strategy()
    first = _choose(
        strategy,
        _game(food=(FoodModel(food_id=1, pos=(35.0, 30.0)),)),
    )
    second = _choose(
        strategy,
        _game(food=(FoodModel(food_id=2, pos=(25.0, 30.0)),)),
    )

    assert first.direction[0] > 0.8
    assert second.direction[0] < -0.9


def test_team_58_cannot_split_below_engine_mass_threshold() -> None:
    virus = VirusModel(virus_id=1, pos=(32.0, 30.0), radius=1.5)
    decision = _choose(
        ReplayTeam58Strategy(),
        _game(own_radius=1.0, viruses=(virus,)),
    )

    assert decision.split is False


def test_team_58_profile_can_emit_observed_high_mass_prey_split() -> None:
    prey = VisibleBlobModel(
        player_id=1,
        team_id=9,
        blob_id=0,
        pos=(31.0, 30.0),
        radius=0.3,
    )
    virus = VirusModel(virus_id=1, pos=(32.0, 30.0), radius=1.5)
    decision = _choose(
        ReplayTeam58Strategy(),
        _game(own_radius=3.0, enemies=(prey,), viruses=(virus,)),
    )

    assert decision.split is True
