from __future__ import annotations

import json
from pathlib import Path

from strategies.replay_opponents import (
    REPLAY_STRENGTH_CANDIDATE_TEAM_IDS,
    REPLAY_TEAM_IDS,
)


ROOT = Path(__file__).resolve().parents[1]


def test_measured_clone_strength_report_matches_the_active_catalog() -> None:
    report = json.loads(
        (ROOT / "docs" / "replay-opponent-runtime-strength.json").read_text()
    )

    assert tuple(report["candidate_team_ids"]) == REPLAY_STRENGTH_CANDIDATE_TEAM_IDS
    assert tuple(report["selected_team_ids"]) == REPLAY_TEAM_IDS
    assert all(
        item["result_type"] == "SUCCESS"
        for item in report["liveness"].values()
    )
    gate = report["runtime_gate"]
    strengths = {item["team_id"]: item for item in report["strengths"]}
    for team_id in REPLAY_TEAM_IDS:
        strength = strengths[team_id]
        assert strength["appearances"] >= gate["minimum_appearances"]
        assert strength["completion_rate"] >= gate["minimum_completion_rate"]
        assert strength["mean_rank"] <= gate["maximum_mean_rank"]
        assert strength["top_three_rate"] >= gate["minimum_top_three_rate"]
