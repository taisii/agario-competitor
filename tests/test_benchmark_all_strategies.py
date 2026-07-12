from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_all_strategies import summarize_opponent_rows  # noqa: E402


def test_summary_requires_every_match_to_finish_with_a_rank() -> None:
    rows = [
        {
            "return_code": 0,
            "result_type": "SUCCESS",
            "tracked_ranks": [1],
        },
        {
            "return_code": 0,
            "result_type": "PLAYER_BANNED",
            "tracked_ranks": [],
        },
    ]

    summary = summarize_opponent_rows("timeout_opponent", rows)

    assert summary["wins"] == 1
    assert summary["matches"] == 2
    assert summary["successful_matches"] == 1
    assert summary["passed"] is False


def test_summary_passes_only_a_majority_over_all_requested_matches() -> None:
    rows = [
        {"return_code": 0, "result_type": "SUCCESS", "tracked_ranks": [1]},
        {"return_code": 0, "result_type": "SUCCESS", "tracked_ranks": [1]},
        {"return_code": 0, "result_type": "SUCCESS", "tracked_ranks": [2]},
    ]

    summary = summarize_opponent_rows("healthy_opponent", rows)

    assert summary["wins"] == 2
    assert summary["passed"] is True
