from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from lib.models.blob_model import BlobModel, VisibleBlobModel  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from strategies.replay_team_39 import ReplayTeam39Strategy  # noqa: E402


def _game(
    *,
    own_pos: tuple[float, float] = (30.0, 30.0),
    own_radius: float = 1.0,
    player_id: int = 2,
    round_number: int = 10,
    enemies: tuple[VisibleBlobModel, ...] = (),
) -> SimpleNamespace:
    own = BlobModel(blob_id=0, pos=own_pos, radius=own_radius)
    me = SimpleNamespace(
        player_id=player_id,
        pos=own_pos,
        radius=own_radius,
        alive=True,
        blobs={0: own},
    )
    state = SimpleNamespace(
        me=me,
        visible_blobs=list(enemies),
        visible_food=[],
        visible_viruses=[],
        map=SimpleNamespace(size=60.0),
        round=round_number,
        max_rounds=1400,
        rankings=list(range(8)),
    )
    return SimpleNamespace(state=state)


def _choose(strategy: ReplayTeam39Strategy, game: SimpleNamespace):
    return strategy.choose(StrategyContext(game=game, query=SimpleNamespace()))


def test_team_39_initial_heading_is_deterministic_and_unit_length() -> None:
    first = _choose(ReplayTeam39Strategy(), _game()).direction
    second = _choose(ReplayTeam39Strategy(), _game()).direction

    assert first == second
    assert math.hypot(*first) == 1.0


def test_team_39_retains_heading_without_threat_or_refresh() -> None:
    strategy = ReplayTeam39Strategy(heading_refresh_rate=0.0)
    first_game = _game(round_number=10)
    second_game = _game(round_number=11)

    first = _choose(strategy, first_game)
    second = _choose(strategy, second_game)

    assert second.direction == first.direction
    assert second.reason == "inertia"


def test_team_39_nearest_predator_overrides_opposed_stale_heading() -> None:
    strategy = ReplayTeam39Strategy(heading_refresh_rate=0.0)
    strategy._heading = (1.0, 0.0)
    strategy._last_round = 9
    predator = VisibleBlobModel(
        player_id=1,
        team_id=99,
        blob_id=0,
        pos=(33.0, 30.0),
        radius=2.0,
    )

    decision = _choose(strategy, _game(enemies=(predator,)))

    assert decision.direction[0] < 0.0
    assert decision.reason == "predator_avoidance"
    assert decision.target_kind == "escape"


def test_team_39_reflects_an_outward_heading_near_wall() -> None:
    strategy = ReplayTeam39Strategy(heading_refresh_rate=0.0)
    strategy._heading = (-1.0, 0.0)
    strategy._last_round = 9

    decision = _choose(
        strategy,
        _game(own_pos=(1.5, 30.0), own_radius=1.0),
    )

    assert decision.direction == (1.0, 0.0)
    assert decision.reason == "wall_reflection"


def test_team_39_never_splits_even_with_edible_prey() -> None:
    strategy = ReplayTeam39Strategy(heading_refresh_rate=0.0)
    prey = VisibleBlobModel(
        player_id=1,
        team_id=99,
        blob_id=0,
        pos=(32.0, 30.0),
        radius=1.0,
    )

    decision = _choose(
        strategy,
        _game(own_radius=2.0, enemies=(prey,)),
    )

    assert decision.split is False
