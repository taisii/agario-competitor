from __future__ import annotations

from scripts.benchmark_replay_clone_strength import (
    RuntimeStrengthGate,
    league_layouts,
    summarise_runtime_strength,
)


def test_balanced_league_gives_each_live_team_eight_appearances() -> None:
    team_ids = (1, 3, 4, 9, 10, 21, 24, 31, 59)
    layouts = league_layouts(team_ids)

    assert len(layouts) == len(team_ids)
    assert all(len(layout) == 8 for layout in layouts)
    assert all(sum(team_id in layout for layout in layouts) == 8 for team_id in team_ids)
    assert all(
        sorted(slot for layout in layouts for slot, value in enumerate(layout) if value == team_id)
        == list(range(8))
        for team_id in team_ids
    )


def test_balanced_league_scales_to_all_eleven_strength_candidates() -> None:
    team_ids = (1, 3, 4, 9, 10, 21, 24, 31, 35, 49, 59)
    layouts = league_layouts(team_ids)

    assert len(layouts) == len(team_ids)
    assert all(sum(team_id in layout for layout in layouts) == 8 for team_id in team_ids)


def test_runtime_strength_requires_completion_and_above_median_results() -> None:
    team_ids = (1, 3)
    results = [
        ((1, 3), {"result_type": "SUCCESS", "ranking": [0, 1]}),
        ((1, 3), {"result_type": "SUCCESS", "ranking": [1, 0]}),
        ((1, 3), {"result_type": "PLAYER_BANNED", "ranking": []}),
    ]

    strengths = summarise_runtime_strength(team_ids, results)

    assert strengths[0].completion_rate == 2 / 3
    assert strengths[0].mean_rank is None
    assert not strengths[0].qualifies(RuntimeStrengthGate(minimum_appearances=3))
