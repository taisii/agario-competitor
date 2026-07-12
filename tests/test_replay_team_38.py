from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from lib.models.blob_model import BlobModel  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from strategies.replay_team_38 import ReplayTeam38Strategy  # noqa: E402


def _game(*, player_id: int = 6, radius: float = 1.0) -> SimpleNamespace:
    blob = BlobModel(blob_id=0, pos=(30.0, 30.0), radius=radius)
    return SimpleNamespace(
        state=SimpleNamespace(
            me=SimpleNamespace(
                player_id=player_id,
                team_id=38,
                radius=radius,
                alive=True,
                blobs={0: blob},
            )
        )
    )


def _choose(strategy: ReplayTeam38Strategy, game: SimpleNamespace):
    return strategy.choose(StrategyContext(game=game, query=SimpleNamespace()))


def test_team_38_reproduces_observed_initial_headings() -> None:
    expected_bins = {3: 6, 6: 2, 7: 4}
    for player_id, expected_bin in expected_bins.items():
        decision = _choose(ReplayTeam38Strategy(), _game(player_id=player_id))
        angle = math.atan2(decision.direction[1], decision.direction[0]) % math.tau
        assert round(angle / (math.tau / 16)) % 16 == expected_bin


def test_team_38_every_direction_is_on_sixteen_bin_grid() -> None:
    strategy = ReplayTeam38Strategy()
    for _ in range(500):
        direction = _choose(strategy, _game()).direction
        angle = math.atan2(direction[1], direction[0]) % math.tau
        assert abs(angle / (math.tau / 16) - round(angle / (math.tau / 16))) < 1e-12


def test_team_38_sequence_is_deterministic_with_observed_hold_rate() -> None:
    first = ReplayTeam38Strategy()
    second = ReplayTeam38Strategy()
    a = [_choose(first, _game()).direction for _ in range(1000)]
    b = [_choose(second, _game()).direction for _ in range(1000)]
    holds = sum(left == right for left, right in zip(a, a[1:]))

    assert a == b
    assert 500 <= holds <= 610


def test_team_38_never_splits_below_mass_threshold() -> None:
    strategy = ReplayTeam38Strategy()
    assert not any(_choose(strategy, _game(radius=1.0)).split for _ in range(1000))


def test_team_38_split_is_rare_and_deterministic_when_eligible() -> None:
    first = ReplayTeam38Strategy()
    second = ReplayTeam38Strategy()
    a = [_choose(first, _game(radius=2.0)).split for _ in range(2000)]
    b = [_choose(second, _game(radius=2.0)).split for _ in range(2000)]

    assert a == b
    assert 5 <= sum(a) <= 35
