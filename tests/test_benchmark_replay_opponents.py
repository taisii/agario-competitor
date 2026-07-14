from __future__ import annotations

import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import benchmark_replay_opponents  # noqa: E402
from scripts.benchmark_replay_opponents import (  # noqa: E402
    run_cell,
    saved_opponent_names,
    summarize_opponent_rows,
)


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


def test_default_matrix_uses_only_the_curated_replay_panel() -> None:
    assert saved_opponent_names("replay_dominance") == (
        "replay_team_21",
    )


def test_summary_passes_only_a_majority_over_all_requested_matches() -> None:
    rows = [
        {"return_code": 0, "result_type": "SUCCESS", "tracked_ranks": [1]},
        {"return_code": 0, "result_type": "SUCCESS", "tracked_ranks": [1]},
        {"return_code": 0, "result_type": "SUCCESS", "tracked_ranks": [2]},
    ]

    summary = summarize_opponent_rows("healthy_opponent", rows)

    assert summary["wins"] == 2
    assert summary["passed"] is True


def test_matrix_cell_reuses_match_runner_and_global_job_limit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    semaphore = asyncio.Semaphore(3)
    captured: dict[str, object] = {}
    expected = [{"return_code": 0, "result_type": "SUCCESS"}]

    async def fake_run_all(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(benchmark_replay_opponents, "run_all", fake_run_all)
    monkeypatch.setattr(benchmark_replay_opponents, "write_outputs", lambda *_: None)

    actual = asyncio.run(
        run_cell(
            candidate="replay_dominance",
            opponent="replay_team_1",
            candidate_slot=7,
            trials=4,
            output_root=tmp_path,
            semaphore=semaphore,
            fast=True,
        )
    )

    assert actual == expected
    assert captured["semaphore"] is semaphore
    assert captured["fast"] is True
    assert captured["trials"] == 4
    assert captured["tracked_slots"] == (7,)
