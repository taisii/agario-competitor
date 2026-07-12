from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from lib.models.blob_model import BlobModel  # noqa: E402
from strategies.base import StrategyContext  # noqa: E402
from strategies.replay_team_6 import ReplayTeam6Strategy  # noqa: E402


def _game(*, player_id: int = 2, radius: float = 1.0) -> SimpleNamespace:
    blob = BlobModel(blob_id=0, pos=(30.0, 30.0), radius=radius)
    state = SimpleNamespace(
        me=SimpleNamespace(
            player_id=player_id,
            team_id=6,
            radius=radius,
            alive=True,
            blobs={0: blob},
        )
    )
    return SimpleNamespace(state=state)


def _choose(strategy: ReplayTeam6Strategy, game: SimpleNamespace):
    return strategy.choose(StrategyContext(game=game, query=SimpleNamespace()))


def test_team_6_reproduces_observed_slot_initial_headings() -> None:
    player_two = _choose(ReplayTeam6Strategy(), _game(player_id=2)).direction
    player_seven = _choose(ReplayTeam6Strategy(), _game(player_id=7)).direction

    assert player_two == (1.0, 0.0)
    assert player_seven[0] == -1.0
    assert abs(player_seven[1]) < 1e-12


def test_team_6_every_direction_is_on_sixteen_bin_grid() -> None:
    strategy = ReplayTeam6Strategy()
    decisions = [_choose(strategy, _game()) for _ in range(500)]

    for decision in decisions:
        angle = math.atan2(decision.direction[1], decision.direction[0]) % math.tau
        bin_position = angle / (math.tau / 16)
        assert abs(bin_position - round(bin_position)) < 1e-12


def test_team_6_direction_sequence_is_deterministic_and_persistent() -> None:
    first = ReplayTeam6Strategy()
    second = ReplayTeam6Strategy()
    first_directions = [_choose(first, _game()).direction for _ in range(1000)]
    second_directions = [_choose(second, _game()).direction for _ in range(1000)]
    holds = sum(
        left == right
        for left, right in zip(first_directions, first_directions[1:])
    )

    assert first_directions == second_directions
    assert 550 <= holds <= 670


def test_team_6_never_splits_below_engine_mass_threshold() -> None:
    strategy = ReplayTeam6Strategy()
    decisions = [_choose(strategy, _game(radius=1.0)) for _ in range(1000)]

    assert not any(decision.split for decision in decisions)


def test_team_6_split_is_sparse_and_deterministic_when_mass_eligible() -> None:
    first = ReplayTeam6Strategy()
    second = ReplayTeam6Strategy()
    first_splits = [_choose(first, _game(radius=2.0)).split for _ in range(1000)]
    second_splits = [_choose(second, _game(radius=2.0)).split for _ in range(1000)]

    assert first_splits == second_splits
    assert 15 <= sum(first_splits) <= 50
