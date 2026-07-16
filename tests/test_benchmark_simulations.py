from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_simulations import (  # noqa: E402
    apply_benchmark_timeout_environment,
    paired_mass_comparisons,
    parse_factorial_cells,
    resolve_jobs,
    summarize_metrics,
    two_factor_mass_analysis,
)


def test_throughput_jobs_are_cpu_aware_but_memory_bounded() -> None:
    assert resolve_jobs(None, throughput=False, cpu_count=10) == 1
    assert resolve_jobs(None, throughput=True, cpu_count=10) == 4
    assert resolve_jobs(None, throughput=True, cpu_count=2) == 1
    assert resolve_jobs(8, throughput=True, cpu_count=10) == 8


def test_throughput_relaxes_every_player_without_strict_slot_override() -> None:
    env = {"AGARIO_LOCAL_RELAXED_PLAYER_IDS": "1,2,3"}

    apply_benchmark_timeout_environment(
        env,
        tracked_slots=(0,),
        throughput=True,
    )

    assert "AGARIO_LOCAL_RELAXED_PLAYER_IDS" not in env
    assert env["AGARIO_LOCAL_TURN_TIMEOUT_SECONDS"] == "60"
    assert env["AGARIO_LOCAL_CUMULATIVE_TIMEOUT_SECONDS"] == "600"


def test_fast_validation_relaxes_only_untracked_players() -> None:
    env = {}

    apply_benchmark_timeout_environment(
        env,
        tracked_slots=(0, 3),
        throughput=False,
    )

    assert env["AGARIO_LOCAL_RELAXED_PLAYER_IDS"] == "1,2,4,5,6,7"
    assert env["AGARIO_STRICT_TURN_TIMEOUT_SECONDS"] == "1"


def _row(variant: str, trial: int, mass: float, result: str = "SUCCESS"):
    return {
        "variant": variant,
        "trial": trial,
        "return_code": 0,
        "result_type": result,
        "tracked_mass_sum": mass,
        "tracked_peak_mass_sum": mass,
    }


def test_paired_mass_comparison_uses_trial_differences() -> None:
    rows = [
        _row("current", 0, 10.0),
        _row("candidate", 0, 13.0),
        _row("current", 1, 20.0),
        _row("candidate", 1, 25.0),
    ]

    comparison = paired_mass_comparisons(rows, bootstrap_samples=1000)[0]

    assert comparison["baseline_mass_mean"] == 15.0
    assert comparison["variant_mass_mean"] == 19.0
    assert comparison["paired_differences"] == [3.0, 5.0]
    assert comparison["one_sided_95_lower_bound"] == 3.0
    assert comparison["passed"] is True


def test_paired_mass_comparison_uses_final_not_peak_mass() -> None:
    baseline = _row("current", 0, 20.0)
    candidate = _row("candidate", 0, 25.0)
    baseline["tracked_peak_mass_sum"] = 100.0
    candidate["tracked_peak_mass_sum"] = 10.0

    comparison = paired_mass_comparisons(
        [baseline, candidate],
        bootstrap_samples=100,
    )[0]

    assert comparison["baseline_mass_mean"] == 20.0
    assert comparison["variant_mass_mean"] == 25.0
    assert comparison["paired_mean_difference"] == 5.0
    assert comparison["passed"] is True


def test_paired_mass_comparison_counts_bans_as_zero_and_fails_gate() -> None:
    rows = [
        _row("current", 0, 10.0),
        _row("candidate", 0, 100.0, result="PLAYER_BANNED"),
    ]

    comparison = paired_mass_comparisons(rows, bootstrap_samples=100)[0]

    assert comparison["variant_mass_mean"] == 0.0
    assert comparison["paired_mean_difference"] == -10.0
    assert comparison["valid_pairs"] == 0
    assert comparison["passed"] is False


def test_two_factor_analysis_keeps_conditional_and_interaction_effects() -> None:
    rows = [
        _row("base", 0, 10.0),
        _row("a", 0, 14.0),
        _row("b", 0, 8.0),
        _row("ab", 0, 20.0),
        _row("base", 1, 30.0),
        _row("a", 1, 35.0),
        _row("b", 1, 31.0),
        _row("ab", 1, 38.0),
    ]

    analysis = two_factor_mass_analysis(
        rows,
        cells={"base": "base", "a": "a", "b": "b", "ab": "ab"},
    )

    assert analysis["paired_trials"] == 2
    assert analysis["valid_paired_trials"] == 2
    assert analysis["cell_mass_means"] == {
        "base": 20.0,
        "a": 24.5,
        "b": 19.5,
        "ab": 29.0,
    }
    assert analysis["mean_effects"] == {
        "a_when_b_off": 4.5,
        "a_when_b_on": 9.5,
        "b_when_a_off": -0.5,
        "b_when_a_on": 4.5,
        "interaction": 5.0,
        "a_marginal": 7.0,
        "b_marginal": 2.0,
    }
    assert [row["interaction"] for row in analysis["trial_effects"]] == [8.0, 2.0]


def test_two_factor_analysis_counts_failed_cell_as_zero_and_reports_it() -> None:
    rows = [
        _row("base", 0, 10.0),
        _row("a", 0, 14.0),
        _row("b", 0, 8.0),
        _row("ab", 0, 99.0, result="PLAYER_BANNED"),
    ]

    analysis = two_factor_mass_analysis(
        rows,
        cells={"base": "base", "a": "a", "b": "b", "ab": "ab"},
    )

    assert analysis["valid_paired_trials"] == 0
    assert analysis["cell_mass_means"]["ab"] == 0.0
    assert analysis["trial_effects"][0]["interaction"] == -12.0


def test_parse_factorial_cells_requires_four_distinct_variants() -> None:
    assert parse_factorial_cells("base, a, b, ab") == {
        "base": "base",
        "a": "a",
        "b": "b",
        "ab": "ab",
    }

    try:
        parse_factorial_cells("base,a,b")
    except ValueError as error:
        assert "four variant names" in str(error)
    else:
        raise AssertionError("three factorial cells must be rejected")


def test_summarize_metrics_reads_engine_end_to_end_response_time(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "response_timings.json").write_text('{"0": 0.75, "1": 1.25, "2": 9.0}\n')

    summary = summarize_metrics(tmp_path, (0, 1))

    assert summary["response_cumulative_sum_seconds"] == 2.0
    assert summary["response_cumulative_mean_seconds"] == 1.0
    assert summary["response_cumulative_max_seconds"] == 1.25


def test_summarize_metrics_reports_peak_mass_for_each_tracked_slot(tmp_path) -> None:
    for slot, masses in ((0, (1.0, 5.0, 3.0)), (1, (2.0, 7.0))):
        submission = tmp_path / f"submission{slot}"
        submission.mkdir()
        (submission / "bot_metrics.jsonl").write_text(
            "".join(f'{{"my_mass":{mass}}}\n' for mass in masses)
        )

    summary = summarize_metrics(tmp_path, (0, 1))

    assert summary["tracked_peak_mass_sum"] == 12.0
    assert summary["tracked_peak_mass_mean"] == 6.0
    assert summary["tracked_peak_mass_max"] == 7.0
