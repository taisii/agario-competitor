from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from strategies.registry import (  # noqa: E402
    DEFAULT_RANDOM_OPPONENT_STRATEGIES,
    STRATEGY_SPECS,
    RandomOpponentStrategy,
    available_strategy_names,
    create_strategy,
    submission_strategy_names,
    submission_strategy_spec,
)


EXPECTED_STRATEGIES = (
    "event_driven_static_search",
    "replay_dominance",
    "semantic_lookahead",
    "semantic_potential",
    "static_retained_growth",
    "threat_aware_receding_horizon",
)


def test_catalog_contains_six_sparring_strategies() -> None:
    assert available_strategy_names() == EXPECTED_STRATEGIES
    assert tuple(sorted(STRATEGY_SPECS)) == EXPECTED_STRATEGIES
    assert submission_strategy_names() == EXPECTED_STRATEGIES
    assert set(DEFAULT_RANDOM_OPPONENT_STRATEGIES) == set(EXPECTED_STRATEGIES)


def test_random_opponents_are_paired_and_cover_the_whole_pool() -> None:
    left = RandomOpponentStrategy(
        DEFAULT_RANDOM_OPPONENT_STRATEGIES,
        base_seed=20260716,
        trial=3,
    )
    right = RandomOpponentStrategy(
        DEFAULT_RANDOM_OPPONENT_STRATEGIES,
        base_seed=20260716,
        trial=3,
    )

    left_names = [left._select_name(slot) for slot in range(1, 8)]
    right_names = [right._select_name(slot) for slot in range(1, 8)]

    assert left_names == right_names
    assert set(left_names) == set(DEFAULT_RANDOM_OPPONENT_STRATEGIES)
    assert sorted(left_names.count(name) for name in set(left_names)) == [
        1,
        1,
        1,
        1,
        1,
        2,
    ]


def test_trial_changes_the_randomized_slot_assignment() -> None:
    first = RandomOpponentStrategy(
        DEFAULT_RANDOM_OPPONENT_STRATEGIES,
        base_seed=20260716,
        trial=0,
    )
    second = RandomOpponentStrategy(
        DEFAULT_RANDOM_OPPONENT_STRATEGIES,
        base_seed=20260716,
        trial=1,
    )

    assert [first._select_name(slot) for slot in range(1, 8)] != [
        second._select_name(slot) for slot in range(1, 8)
    ]


def test_environment_override_must_stay_within_the_sparring_catalog(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "BOT_RANDOM_STRATEGIES",
        "semantic_potential,replay_dominance",
    )
    from strategies.registry import create_random_opponent_strategy

    strategy = create_random_opponent_strategy()
    assert strategy._candidates == ("semantic_potential", "replay_dominance")

    monkeypatch.setenv("BOT_RANDOM_STRATEGIES", "food_greedy")
    with pytest.raises(ValueError, match="Invalid BOT_RANDOM_STRATEGIES"):
        create_random_opponent_strategy()


def test_registry_import_does_not_eagerly_import_implementations() -> None:
    script = """
import sys
from strategies.registry import available_strategy_names

assert available_strategy_names()
implementation_modules = {
    "strategies.event_driven",
    "strategies.receding_horizon",
    "strategies.retained_growth",
    "strategies.semantic_potential",
}
assert implementation_modules.isdisjoint(sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "bots")},
        check=True,
    )


def test_every_catalog_entry_constructs_and_has_complete_submission_metadata() -> None:
    for name in available_strategy_names():
        assert create_strategy(name).name == name
        spec = submission_strategy_spec(name)
        bundle = spec.submission
        assert bundle is not None
        assert spec.factory_path.rpartition(":")[2] == bundle.strategy_class
        for source_module in bundle.source_modules:
            path = Path(source_module)
            assert not path.is_absolute()
            assert (ROOT / path).is_file()


def test_removed_strategy_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        create_strategy("food_greedy")
