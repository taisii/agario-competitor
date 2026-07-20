from scripts.benchmark_simulations import score_result, summarize


def test_successful_result_uses_slot_zero_final_mass_and_rank() -> None:
    result = score_result(
        3,
        {
            "result_type": "SUCCESS",
            "ranking": [2, 0, 1],
            "final_masses": {"0": 12.5, "1": 4.0, "2": 18.0},
        },
    )

    assert result == {
        "trial": 3,
        "result_type": "SUCCESS",
        "final_mass": 12.5,
        "rank": 2,
    }


def test_failed_result_counts_as_zero_mass() -> None:
    result = score_result(0, {"result_type": "PLAYER_BANNED"})

    assert result["final_mass"] == 0.0
    assert result["rank"] is None


def test_summary_ranks_by_mean_final_mass() -> None:
    summary = summarize(
        [
            {"result_type": "SUCCESS", "final_mass": 10.0, "rank": 1},
            {"result_type": "SUCCESS", "final_mass": 20.0, "rank": 3},
            {"result_type": "PLAYER_BANNED", "final_mass": 0.0, "rank": None},
        ]
    )

    assert summary == {
        "trials": 3,
        "successful_trials": 2,
        "mean_final_mass": 10.0,
        "mean_rank": 2.0,
        "top_one_rate": 1 / 3,
    }
