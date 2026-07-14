from __future__ import annotations

import json
from pathlib import Path

from scripts.calibrate_expected_responses import (
    SCENARIO_LABELS,
    TwoStepSample,
    _action_regime_summary,
    _block_error_distribution,
    _conformal_quantile,
    _lomo_gain_precision_summary,
    _two_step_summary,
    build_report,
    classify_frame,
    extract_frames,
    simulate_match_two_step,
)


def _player(
    player_id: int,
    team_id: int,
    *,
    x: float,
    radius: float,
    blob_id: int = 0,
) -> dict[str, object]:
    return {
        "player_id": player_id,
        "team_id": team_id,
        "alive": True,
        "blobs": [
            {
                "blob_id": blob_id,
                "pos": [x, 30.0],
                "radius": radius,
                "merge_cooldown": 0,
            }
        ],
    }


def _write_replay(path: Path) -> None:
    own = _player(0, 73, x=30.0, radius=1.0)
    predator = _player(1, 21, x=34.0, radius=3.0, blob_id=99)
    events = [
        {
            "event_type": "event_game_started",
            "arena_size": 60.0,
            "vision_size": 20.0,
            "max_rounds": 2,
            "engine_version": "2026.1.13",
            "players": [own, predator],
        },
        {
            "event_type": "move_player",
            "player_id": 0,
            "direction": {"x": 1.0, "y": 0.0},
            "split": False,
        },
        {
            "event_type": "move_player",
            "player_id": 1,
            "direction": {"x": -1.0, "y": 0.0},
            "split": True,
        },
        {
            "event_type": "event_player_moved",
            "player_id": 0,
            "alive": True,
            "blobs": own["blobs"],
        },
        {
            "event_type": "event_player_moved",
            "player_id": 1,
            "alive": True,
            # A different public ID in the next snapshot must not affect the
            # previous command classification.
            "blobs": [
                {
                    "blob_id": 7,
                    "pos": [33.0, 30.0],
                    "radius": 3.0,
                    "merge_cooldown": 20,
                }
            ],
        },
    ]
    path.write_text(json.dumps(events), encoding="utf-8")


def test_extract_and_classify_prefers_split_match_before_angle(tmp_path: Path) -> None:
    path = tmp_path / "match-100-replay.json"
    _write_replay(path)

    frames = extract_frames(path)
    samples = classify_frame(frames[0])

    assert len(frames) == 1
    assert len(samples) == 1
    assert samples[0].scenario_id == SCENARIO_LABELS.index("adaptive")
    assert samples[0].split_match
    assert samples[0].angle_error_degrees == 0.0


def test_report_contains_match_and_leave_one_match_out_weights(tmp_path: Path) -> None:
    first = tmp_path / "match-100-replay.json"
    second = tmp_path / "match-101-replay.json"
    _write_replay(first)
    _write_replay(second)

    report = build_report(
        [first, second],
        replay_dirs=[tmp_path.resolve()],
    )

    assert report["overall"]["sample_count"] == 2
    assert report["overall"]["scenario_weights"]["adaptive"] == 1.0
    assert report["overall"]["split_accuracy"] == 1.0
    assert report["matches"]["100"]["sample_count"] == 1
    assert report["leave_one_match_out"]["100"]["training"]["sample_count"] == 1
    assert report["leave_one_match_out"]["100"]["holdout"]["sample_count"] == 1
    assert report["config"]["engine_versions"] == ["2026.1.13"]


def test_two_step_replay_uses_actual_actions_without_continuation_optimisation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "match-200-replay.json"
    own = _player(0, 73, x=30.0, radius=0.9)
    events = [
        {
            "event_type": "event_game_started",
            "arena_size": 60.0,
            "vision_size": 20.0,
            "max_rounds": 3,
            "engine_version": "2026.1.13",
            "players": [own],
        }
    ]
    for round_number in range(3):
        events.extend(
            (
                {
                    "event_type": "move_player",
                    "player_id": 0,
                    "direction": {"x": 1.0, "y": 0.0},
                    "split": False,
                },
                {
                    "event_type": "event_player_moved",
                    "player_id": 0,
                    "alive": True,
                    "blobs": [
                        {
                            "blob_id": round_number + 10,
                            "pos": [31.0 + round_number, 30.0],
                            "radius": 0.9,
                            "merge_cooldown": 0,
                        }
                    ],
                },
            )
        )
    path.write_text(json.dumps(events), encoding="utf-8")

    samples = simulate_match_two_step(
        extract_frames(path),
        scenario_weights=(1 / 6,) * 6,
    )

    assert len(samples) == 1
    assert abs(samples[0].predicted_final_mass - 0.81) < 1e-12
    assert samples[0].actual_final_mass == 0.81
    assert samples[0].absolute_error < 1e-12
    assert not samples[0].actual_gain
    assert samples[0].predicted_gain_probability == 0.0
    assert abs(samples[0].initial_mass - 0.81) < 1e-12
    assert samples[0].own_fragment_count == 1
    assert samples[0].visible_enemy_count == 0
    assert samples[0].relative_error < 1e-12
    assert not samples[0].any_action_split


def test_gain_precision_curve_selects_lowest_threshold_reaching_80_percent() -> None:
    samples = [
        TwoStepSample(1, 0, 1.0, 1.0, 0.0, 1.0, True),
        TwoStepSample(1, 1, 1.0, 1.0, 0.0, 0.5, True),
        TwoStepSample(1, 2, 1.0, 1.0, 0.0, 0.5, False),
        TwoStepSample(1, 3, 1.0, 1.0, 0.0, 0.0, False),
    ]

    summary = _two_step_summary(samples)

    assert summary["minimum_probability_for_80pct_precision"] == 1.0
    assert summary["frame_level_minimum_probability_for_80pct_precision"] is None
    assert summary["gain_positive_precision_curve"][1]["precision"] == 2 / 3
    assert summary["gain_positive_precision_curve"][2]["precision"] == 1.0
    assert (
        summary["gain_positive_precision_curve"][2][
            "precision_wilson_95_lower"
        ]
        < 0.8
    )


def test_frame_gain_threshold_requires_20_samples_and_wilson_lower_bound() -> None:
    samples = [
        TwoStepSample(1, index, 1.0, 1.0, 0.0, 0.5, True)
        for index in range(20)
    ] + [
        TwoStepSample(1, 20, 1.0, 1.0, 0.0, 0.0, False),
    ]

    summary = _two_step_summary(samples)

    assert summary["minimum_probability_for_80pct_precision"] == 0.0
    assert summary["frame_level_minimum_probability_for_80pct_precision"] == 0.5
    qualifying = summary["gain_positive_precision_curve"][1]
    assert qualifying["predicted_positive_count"] == 20
    assert qualifying["precision"] == 1.0
    assert qualifying["precision_wilson_95_lower"] >= 0.8


def test_match_block_summary_excludes_small_blocks_and_uses_one_score_each() -> None:
    by_match = {
        1: [
            TwoStepSample(1, index, 1.0, 1.0, float(index), 0.0, False)
            for index in range(20)
        ],
        2: [TwoStepSample(2, 0, 1.0, 1.0, 99.0, 0.0, False)],
    }

    summary = _block_error_distribution(by_match)

    assert summary["observed_match_block_count"] == 2
    assert summary["eligible_match_block_count"] == 1
    assert summary["excluded_match_block_count"] == 1
    assert (
        summary["absolute_within_match_q97_5_distribution"]["max"] == 19.0
    )
    assert not summary["blocks"]["2"]["eligible"]


def test_lomo_gain_precision_and_split_regimes_are_match_aware() -> None:
    by_match = {
        match_id: [
            TwoStepSample(
                match_id,
                index,
                1.0,
                1.0,
                0.0,
                0.5,
                True,
                first_action_split=match_id == 1,
            )
            for index in range(20)
        ]
        + [
            TwoStepSample(match_id, 20 + index, 1.0, 1.0, 0.0, 0.0, False)
            for index in range(5)
        ]
        for match_id in range(1, 4)
    }

    lomo = _lomo_gain_precision_summary(by_match)
    regimes = _action_regime_summary(by_match)

    assert lomo["minimum_probability_for_all_folds_80pct_lower_bound"] == 0.5
    assert regimes["any_action_split_count"] == 20
    assert regimes["split_any"]["frame_level"]["sample_count"] == 20
    assert regimes["no_split"]["frame_level"]["sample_count"] == 55


def test_conformal_quantile_uses_finite_sample_rank_and_mass_bins() -> None:
    assert _conformal_quantile([float(value) for value in range(1, 101)], 0.975) == 99.0
    sample = TwoStepSample(
        1,
        0,
        3.5,
        4.0,
        0.5,
        0.0,
        False,
        initial_mass=2.0,
        own_fragment_count=2,
        visible_enemy_count=3,
        relative_error=0.25,
    )

    summary = _two_step_summary([sample])

    assert summary["relative_final_mass_error"]["finite_sample_q97_5"] == 0.25
    assert summary["mass_bins"]["mass_lt_4"]["sample_count"] == 1
    assert summary["mass_bins"]["mass_4_to_16"]["sample_count"] == 0
    assert summary["counterfactual_global_bonferroni_bound"]["absolute_2x_q97_5"] == 1.0


def test_two_step_model_does_not_leak_round_plus_one_resource_visibility(
    tmp_path: Path,
) -> None:
    path = tmp_path / "match-201-replay.json"
    own = _player(0, 73, x=30.0, radius=2.0)
    events = [
        {
            "event_type": "event_game_started",
            "arena_size": 60.0,
            "vision_size": 20.0,
            "max_rounds": 3,
            "engine_version": "2026.1.13",
            "players": [own],
        },
        {
            "event_type": "move_player",
            "player_id": 0,
            "direction": {"x": 1.0, "y": 0.0},
            "split": False,
        },
        {
            "event_type": "event_virus_spawned",
            "viruses": [
                {"virus_id": 9, "pos": [32.0, 30.0], "radius": 1.5}
            ],
        },
        {
            "event_type": "event_player_moved",
            "player_id": 0,
            "alive": True,
            "blobs": own["blobs"],
        },
    ]
    for _ in range(2):
        events.extend(
            (
                {
                    "event_type": "move_player",
                    "player_id": 0,
                    "direction": {"x": 1.0, "y": 0.0},
                    "split": False,
                },
                {
                    "event_type": "event_player_moved",
                    "player_id": 0,
                    "alive": True,
                    "blobs": own["blobs"],
                },
            )
        )
    path.write_text(json.dumps(events), encoding="utf-8")

    frames = extract_frames(path)
    assert frames[0].viruses == ()
    assert len(frames[1].viruses) == 1

    samples = simulate_match_two_step(
        frames,
        scenario_weights=(1 / 6,) * 6,
    )

    assert samples[0].predicted_gain_probability == 0.0
