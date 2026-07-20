from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from strategies.replay_opponents import (  # noqa: E402
    CloudReplayOpponent,
    CUSTOM_REPLAY_TEAM_IDS,
    OFFICIAL_CLOUD_LINEUPS,
    OFFICIAL_CLOUD_HIGH_DAMAGE_TEAM_IDS,
    OFFICIAL_CLOUD_PRESSURE_TEAM_IDS,
    OBSERVED_REPLAY_TEAM_IDS,
    REPLAY_OPPONENT_SPECS,
    REPLAY_TEAM_IDS,
    RANDOM_REPLAY_TEAM_IDS,
    RandomReplayOpponent,
    create_replay_opponent,
    create_cloud_pressure_opponent,
    select_cloud_replay_team_id,
    select_replay_team_id,
)
from strategies.registry import (  # noqa: E402
    STRATEGY_SPECS,
    RandomOpponentStrategy,
    available_strategy_names,
    create_strategy,
    submission_strategy_spec,
    submission_strategy_names,
)
import strategies.replay_opponents as replay_opponents  # noqa: E402


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


def test_replay_opponent_catalog_import_does_not_eagerly_import_opponents() -> None:
    script = """
import sys
from strategies.replay_opponents import REPLAY_OPPONENT_SPECS

assert REPLAY_OPPONENT_SPECS
implementation_modules = {
    "strategies.replay_imitation",
    "strategies.replay_opponent_policies",
    "strategies.replay_team_1",
}
assert implementation_modules.isdisjoint(sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "bots")},
        check=True,
    )


def test_strategy_and_replay_opponent_catalogs_have_distinct_public_names() -> None:
    modules = {
        int(path.stem.removeprefix("replay_team_"))
        for path in (ROOT / "bots" / "strategies").glob("replay_team_*.py")
    }

    assert modules == CUSTOM_REPLAY_TEAM_IDS
    assert tuple(sorted(STRATEGY_SPECS)) == available_strategy_names()
    assert available_strategy_names() == (
        "event_driven_static_search",
        "expected_final_mass",
        "food_greedy",
        "local_tactical_search",
        "outcome_teacher_hybrid",
        "potential_field_hunter",
        "potential_field_virus_farmer",
        "potential_tactical_hybrid",
        "replay_distilled",
        "replay_dominance",
        "semantic_lookahead",
        "semantic_potential",
        "static_option_growth",
        "static_retained_growth",
        "survival_greedy",
        "threat_aware_receding_horizon",
        "virus_hunter",
    )
    assert REPLAY_TEAM_IDS == (21,)
    assert tuple(sorted(REPLAY_OPPONENT_SPECS)) == REPLAY_TEAM_IDS
    entry_team_ids = {
        int(path.stem.removeprefix("replay_team_"))
        for path in (ROOT / "bots" / "entries").glob("replay_team_*.py")
    }
    assert entry_team_ids == set(REPLAY_TEAM_IDS)
    assert submission_strategy_names() == (
        "event_driven_static_search",
        "expected_final_mass",
        "local_tactical_search",
        "outcome_teacher_hybrid",
        "potential_tactical_hybrid",
        "replay_distilled",
        "replay_dominance",
        "semantic_lookahead",
        "semantic_potential",
        "static_option_growth",
        "static_retained_growth",
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
        replay_opponents,
        "create_replay_candidate",
        lambda team_id: constructed_names.append(team_id) or StubStrategy(),
    )
    strategy = RandomReplayOpponent(
        base_seed=20260712,
        trial=3,
        on_selected=selected_names.append,
    )
    context = SimpleNamespace(
        game=SimpleNamespace(state=SimpleNamespace(me=SimpleNamespace(player_id=4)))
    )
    assert strategy.choose(context) == "decision"
    assert strategy.choose(context) == "decision"
    assert constructed_names == [expected_team_id]
    assert selected_names == [f"replay_team_{expected_team_id}"]
    assert decisions == [context, context]


def test_every_catalog_entry_constructs_the_declared_strategy() -> None:
    for name in available_strategy_names():
        assert create_strategy(name).name == name


def test_every_replay_opponent_constructs_the_declared_team() -> None:
    for team_id, spec in REPLAY_OPPONENT_SPECS.items():
        assert create_replay_opponent(team_id).name == spec.name


def test_unknown_strategy_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        create_strategy("removed_strategy")


def test_replay_opponent_is_not_presented_as_a_candidate_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        create_strategy("replay_team_2")


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
    assert set(first) <= set(RANDOM_REPLAY_TEAM_IDS)
    assert len(set(first)) > 1


def test_random_replay_pool_covers_every_observed_enemy() -> None:
    selected = {
        select_replay_team_id(
            player_id=slot,
            base_seed=20260712,
            trial=trial,
        )
        for trial in range(256)
        for slot in range(1, 8)
    }

    assert RANDOM_REPLAY_TEAM_IDS == OBSERVED_REPLAY_TEAM_IDS
    assert selected == set(OBSERVED_REPLAY_TEAM_IDS)


def test_cloud_replay_lineups_are_complete_observed_cohorts() -> None:
    assert len(OFFICIAL_CLOUD_LINEUPS) == 30
    for lineup in OFFICIAL_CLOUD_LINEUPS:
        assert len(lineup) == 7
        assert len(set(lineup)) == 7
        assert 73 not in lineup
        assert set(lineup) <= set(OBSERVED_REPLAY_TEAM_IDS)


def test_cloud_replay_selection_uses_one_lineup_without_replacement() -> None:
    selected = tuple(
        select_cloud_replay_team_id(
            player_id=slot,
            target_slot=0,
            base_seed=20260719,
            trial=4,
        )
        for slot in range(1, 8)
    )
    repeated = tuple(
        select_cloud_replay_team_id(
            player_id=slot,
            target_slot=0,
            base_seed=20260719,
            trial=4,
        )
        for slot in range(1, 8)
    )

    assert selected == repeated
    assert len(set(selected)) == 7
    assert frozenset(selected) in {
        frozenset(lineup) for lineup in OFFICIAL_CLOUD_LINEUPS
    }


def test_cloud_replay_strategy_selects_lazily_and_reports_once(monkeypatch) -> None:
    selected_names = []
    constructed_names = []
    expected_team_id = select_cloud_replay_team_id(
        player_id=5,
        target_slot=0,
        base_seed=20260719,
        trial=2,
    )

    class StubStrategy:
        name = f"replay_team_{expected_team_id}"

        def choose(self, context):
            return "decision"

    monkeypatch.setattr(
        replay_opponents,
        "create_replay_candidate",
        lambda team_id: constructed_names.append(team_id) or StubStrategy(),
    )
    strategy = CloudReplayOpponent(
        target_slot=0,
        base_seed=20260719,
        trial=2,
        on_selected=selected_names.append,
    )
    context = SimpleNamespace(
        game=SimpleNamespace(state=SimpleNamespace(me=SimpleNamespace(player_id=5)))
    )

    assert strategy.choose(context) == "decision"
    assert strategy.choose(context) == "decision"
    assert constructed_names == [expected_team_id]
    assert selected_names == [f"replay_team_{expected_team_id}"]


def test_cloud_pressure_set_is_observed_and_failed_imitation() -> None:
    from strategies.replay_profiles import PROFILES

    assert OFFICIAL_CLOUD_HIGH_DAMAGE_TEAM_IDS <= set(OBSERVED_REPLAY_TEAM_IDS)
    assert all(
        not PROFILES[team_id].validation_passed
        for team_id in OFFICIAL_CLOUD_HIGH_DAMAGE_TEAM_IDS
    )
    assert OFFICIAL_CLOUD_HIGH_DAMAGE_TEAM_IDS <= OFFICIAL_CLOUD_PRESSURE_TEAM_IDS
    assert OFFICIAL_CLOUD_PRESSURE_TEAM_IDS <= set(OBSERVED_REPLAY_TEAM_IDS)


def test_cloud_pressure_substitutes_only_high_damage_team(monkeypatch) -> None:
    import strategies.receding_horizon as receding_horizon

    selected_team = next(iter(OFFICIAL_CLOUD_PRESSURE_TEAM_IDS))
    monkeypatch.setattr(
        replay_opponents,
        "select_cloud_replay_team_id",
        lambda **_kwargs: selected_team,
    )
    monkeypatch.setenv("BOT_RANDOM_SEED", "20260719")
    monkeypatch.setenv("BOT_BENCHMARK_TRIAL", "0")

    class StubDominance:
        name = "replay_dominance"

        def choose(self, context):
            return "decision"

    monkeypatch.setattr(
        receding_horizon,
        "ReplayDominanceStrategy",
        StubDominance,
    )
    strategy = create_cloud_pressure_opponent(target_slot=0)
    context = SimpleNamespace(
        game=SimpleNamespace(state=SimpleNamespace(me=SimpleNamespace(player_id=1)))
    )

    assert strategy.choose(context) == "decision"

    assert strategy.name == "replay_dominance"
