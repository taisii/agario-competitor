from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from strategies.registry import RandomOpponentStrategy  # noqa: E402
from entries.random_replay_opponent import selected_replay_team_id  # noqa: E402


def test_random_opponent_selection_is_paired_and_reproducible() -> None:
    candidates = ("food_greedy", "survival_greedy", "potential_field_hunter")
    left = RandomOpponentStrategy(candidates, base_seed=20260712, trial=3)
    right = RandomOpponentStrategy(candidates, base_seed=20260712, trial=3)

    assert [left._select_name(slot) for slot in range(1, 8)] == [
        right._select_name(slot) for slot in range(1, 8)
    ]
    assert len({left._select_name(slot) for slot in range(1, 8)}) > 1


def test_random_replay_selection_is_paired_per_trial_and_slot(monkeypatch) -> None:
    monkeypatch.setenv("BOT_RANDOM_SEED", "20260712")
    monkeypatch.setenv("BOT_BENCHMARK_TRIAL", "3")

    first = [selected_replay_team_id(player_id=slot) for slot in range(1, 8)]
    second = [selected_replay_team_id(player_id=slot) for slot in range(1, 8)]

    assert first == second
    assert len(set(first)) > 1
