from __future__ import annotations

import json

import pytest

from scripts.rank_replay_opponents import (
    TeamStrength,
    load_runtime_selected_team_ids,
    selected_team_ids,
)


def test_selection_requires_strength_and_tactical_activity() -> None:
    strengths = (
        TeamStrength(1, 4, 3.0, 0.75, 8.0, 2.0),
        TeamStrength(2, 8, 2.0, 1.0, 0.0, 0.0),
        TeamStrength(3, 2, 1.0, 1.0, 8.0, 2.0),
        TeamStrength(999, 8, 1.0, 1.0, 8.0, 2.0),
    )

    assert selected_team_ids(strengths) == (1,)


def test_runtime_report_rejects_a_selection_that_its_metrics_do_not_support(
    tmp_path,
) -> None:
    report = {
        "candidate_team_ids": [1, 3, 4, 9, 10, 21, 24, 31, 35, 49, 59],
        "runtime_gate": {
            "minimum_appearances": 8,
            "minimum_completion_rate": 1.0,
            "maximum_mean_rank": 4.5,
            "minimum_top_three_rate": 0.25,
        },
        "strengths": [
            {
                "team_id": 1,
                "appearances": 8,
                "completion_rate": 1.0,
                "mean_rank": 5.0,
                "top_three_rate": 0.0,
            }
        ],
        "selected_team_ids": [1],
    }
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="does not match"):
        load_runtime_selected_team_ids(path)
