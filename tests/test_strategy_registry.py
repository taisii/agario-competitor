from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from strategies.registry import (  # noqa: E402
    REPLAY_TEAM_IDS,
    STRATEGY_SPECS,
    RandomOpponentStrategy,
    RandomReplayOpponentStrategy,
    available_strategy_names,
    create_strategy,
    select_replay_team_id,
    submission_strategy_spec,
    submission_strategy_names,
)
import strategies.registry as registry  # noqa: E402


def test_random_opponent_selection_is_paired_and_reproducible() -> None:
    candidates = ("food_greedy", "survival_greedy", "potential_field_hunter")
    left = RandomOpponentStrategy(candidates, base_seed=20260712, trial=3)
    right = RandomOpponentStrategy(candidates, base_seed=20260712, trial=3)

    assert [left._select_name(slot) for slot in range(1, 8)] == [
        right._select_name(slot) for slot in range(1, 8)
    ]
    assert len({left._select_name(slot) for slot in range(1, 8)}) > 1


def test_registry_import_does_not_eagerly_import_strategy_implementations() -> None:
    script = """
import sys
from strategies.registry import available_strategy_names

assert available_strategy_names()
implementation_modules = {
    "strategies.greedy",
    "strategies.potential_field",
    "strategies.receding_horizon",
    "strategies.virus_farming",
}
assert implementation_modules.isdisjoint(sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "bots")},
        check=True,
    )


def test_strategy_catalog_matches_replay_modules_and_public_names() -> None:
    modules = {
        int(path.stem.removeprefix("replay_team_"))
        for path in (ROOT / "bots" / "strategies").glob("replay_team_*.py")
    }

    assert modules == set(REPLAY_TEAM_IDS)
    assert tuple(sorted(STRATEGY_SPECS)) == available_strategy_names()
    assert submission_strategy_names() == (
        "replay_dominance",
        "threat_aware_receding_horizon",
        "virus_hunter",
    )


def test_submission_metadata_is_complete_and_repository_relative() -> None:
    for name in submission_strategy_names():
        spec = submission_strategy_spec(name)
        bundle = spec.submission
        assert bundle is not None
        assert spec.factory_path.rpartition(":")[2] == bundle.strategy_class
        assert bundle.source_modules
        for source_module in bundle.source_modules:
            path = Path(source_module)
            assert not path.is_absolute()
            assert (ROOT / path).is_file()


def test_random_replay_strategy_selects_lazily_and_reports_once(monkeypatch) -> None:
    selected_names = []
    constructed_names = []
    decisions = []
    expected_team_id = select_replay_team_id(
        player_id=4,
        base_seed=20260712,
        trial=3,
    )

    class StubStrategy:
        name = f"replay_team_{expected_team_id}"

        def choose(self, context):
            decisions.append(context)
            return "decision"

    monkeypatch.setattr(
        registry,
        "create_strategy",
        lambda name: constructed_names.append(name) or StubStrategy(),
    )
    strategy = RandomReplayOpponentStrategy(
        base_seed=20260712,
        trial=3,
        on_selected=selected_names.append,
    )
    context = SimpleNamespace(
        game=SimpleNamespace(state=SimpleNamespace(me=SimpleNamespace(player_id=4)))
    )
    assert strategy.choose(context) == "decision"
    assert strategy.choose(context) == "decision"
    assert constructed_names == [f"replay_team_{expected_team_id}"]
    assert selected_names == [f"replay_team_{expected_team_id}"]
    assert decisions == [context, context]


def test_every_catalog_entry_constructs_the_declared_strategy() -> None:
    for name in available_strategy_names():
        assert create_strategy(name).name == name


@pytest.mark.parametrize(
    "name",
    (
        "beam_hunter",
        "beam_rl_balanced",
        "beam_rl_farmer",
        "beam_rl_hunter",
        "beam_rl_opportunist",
        "beam_rl_survival",
        "beam_rl_tuned",
        "beam_rl_value",
        "beam_survival",
        "unified_deterministic",
        "virus_farming_receding_horizon",
        "candidate_submission",
    ),
)
def test_removed_strategies_fail_explicitly(name: str) -> None:
    assert name not in available_strategy_names()
    with pytest.raises(ValueError, match="Unknown strategy"):
        create_strategy(name)


def test_random_replay_selection_is_paired_per_trial_and_slot() -> None:
    first = [
        select_replay_team_id(
            player_id=slot,
            base_seed=20260712,
            trial=3,
        )
        for slot in range(1, 8)
    ]
    second = [
        select_replay_team_id(
            player_id=slot,
            base_seed=20260712,
            trial=3,
        )
        for slot in range(1, 8)
    ]

    assert first == second
    assert len(set(first)) > 1
