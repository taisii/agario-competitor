from __future__ import annotations

from strategies.replay_imitation import ImitationBlob, ImitationObservation
from strategies.replay_profiles import PROFILES
from strategies.replay_team_35 import ReplayTeam35Strategy
from scripts.evaluate_replay_team_35 import (
    HOLDOUT_MIN_MATCH_ID,
    partition_samples,
    profile_metadata_matches_validation,
    split_predictions,
    strict_validation_passed,
)


def observation(
    *,
    round_number: int = 100,
    own_radius: float = 3.0,
    prey_x: float = 20.0,
    prey_radius: float = 1.5,
    predator: bool = False,
    own_blob_count: int = 1,
    merge_cooldown: int = 0,
    player_id: int = 0,
) -> ImitationObservation:
    visible = [ImitationBlob(prey_x, 10.0, prey_radius, player_id=1)]
    if predator:
        visible.append(ImitationBlob(12.0, 10.0, 4.0, player_id=2))
    return ImitationObservation(
        round_number=round_number,
        max_rounds=1400,
        arena_size=60.0,
        own_blobs=tuple(
            ImitationBlob(
                10.0 + index,
                10.0,
                own_radius,
                player_id=player_id,
                merge_cooldown=merge_cooldown,
            )
            for index in range(own_blob_count)
        ),
        visible_blobs=tuple(visible),
        visible_food=(),
        visible_viruses=(),
    )


def test_team35_split_requires_replay_fitted_tactical_window() -> None:
    decision = ReplayTeam35Strategy._split_decision(
        observation(),
        (1.0, 0.0),
    )

    assert decision == (True, "aligned_prey_split")
    assert ReplayTeam35Strategy._split_decision(
        observation(prey_x=27.0), (1.0, 0.0)
    ) == (False, "prey_beyond_split_horizon")
    assert ReplayTeam35Strategy._split_decision(
        observation(prey_radius=2.1), (1.0, 0.0)
    ) == (False, "prey_not_small_enough")
    assert ReplayTeam35Strategy._split_decision(
        observation(), (0.0, 1.0)
    ) == (False, "prey_not_aligned")


def test_team35_split_rearms_after_eighteen_rounds() -> None:
    state = observation(round_number=100)

    assert ReplayTeam35Strategy._split_decision(
        state, (1.0, 0.0), last_split_round=83
    ) == (False, "split_rearming")
    assert ReplayTeam35Strategy._split_decision(
        state, (1.0, 0.0), last_split_round=82
    ) == (True, "aligned_prey_split")


def test_team35_strategy_resets_temporal_state_when_rounds_restart() -> None:
    strategy = ReplayTeam35Strategy()
    strategy._begin_observation(observation(round_number=1_399, player_id=4))
    strategy._last_split_round = 1_390
    strategy._previous_direction = (1.0, 0.0)

    trace_reset = strategy._begin_observation(observation(round_number=0, player_id=4))

    assert trace_reset is True
    assert strategy._last_observed_round == 0
    assert strategy._last_player_id == 4
    assert strategy._last_split_round == -10_000
    assert strategy._previous_direction == (0.0, 0.0)


def test_team35_strategy_resets_temporal_state_when_player_slot_changes() -> None:
    strategy = ReplayTeam35Strategy()
    strategy._begin_observation(observation(round_number=100, player_id=1))
    strategy._last_split_round = 95
    strategy._previous_direction = (1.0, 0.0)

    trace_reset = strategy._begin_observation(observation(round_number=101, player_id=7))

    assert trace_reset is True
    assert strategy._last_observed_round == 101
    assert strategy._last_player_id == 7
    assert strategy._last_split_round == -10_000
    assert strategy._previous_direction == (0.0, 0.0)


def test_team35_strategy_keeps_temporal_state_on_same_round_retry() -> None:
    strategy = ReplayTeam35Strategy()
    state = observation(round_number=100, player_id=5)
    strategy._begin_observation(state)
    strategy._last_split_round = 95
    strategy._previous_direction = (1.0, 0.0)

    trace_reset = strategy._begin_observation(state)

    assert trace_reset is False
    assert strategy._last_split_round == 95
    assert strategy._previous_direction == (1.0, 0.0)


def test_team35_never_attacks_while_predator_is_visible() -> None:
    assert ReplayTeam35Strategy._split_decision(
        observation(predator=True), (1.0, 0.0)
    ) == (False, "predator_visible")


def test_team35_split_requires_single_merge_ready_blob_above_radius_floor() -> None:
    direction = (1.0, 0.0)

    assert ReplayTeam35Strategy._split_decision(
        observation(own_blob_count=2), direction
    ) == (False, "requires_single_blob")
    assert ReplayTeam35Strategy._split_decision(
        observation(own_radius=2.499), direction
    ) == (False, "below_observed_radius_floor")
    assert ReplayTeam35Strategy._split_decision(
        observation(merge_cooldown=1), direction
    ) == (False, "merge_cooldown_active")
    assert ReplayTeam35Strategy._split_decision(
        observation(own_radius=2.5), direction
    ) == (True, "aligned_prey_split")


def test_team35_profile_records_current_official_cohort_and_holdout_metrics() -> None:
    profile = PROFILES[35]

    assert profile.source_matches == (
        13931,
        13935,
        13941,
        14054,
        40736,
        40754,
        40760,
        40763,
        40770,
        40773,
        40774,
    )
    assert profile.direction_median_error == 8.757172092092748
    assert profile.direction_within_30_rate == 0.7949293833107209
    assert profile.split_f1 == 0.6233915500620665
    assert profile.validation_passed is False


def test_team35_evaluation_partitions_whole_matches_without_leakage() -> None:
    before = type("Sample", (), {"match_id": HOLDOUT_MIN_MATCH_ID - 1})()
    after = type("Sample", (), {"match_id": HOLDOUT_MIN_MATCH_ID})()

    training, holdout = partition_samples([after, before])  # type: ignore[list-item]

    assert training == [before]
    assert holdout == [after]


def test_team35_split_rearm_is_independent_of_record_input_order() -> None:
    early = type(
        "Sample",
        (),
        {
            "match_id": 1,
            "player_id": 0,
            "round_number": 100,
            "split_features": (
                1.0, 0.0, 0.0, 0.0625, 0.3, 1.0, 1.0, 0.5,
                2.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0,
            ),
        },
    )()
    late = type(
        "Sample",
        (),
        {
            "match_id": 1,
            "player_id": 0,
            "round_number": 110,
            "split_features": early.split_features,
        },
    )()
    rule = (0.25, 0.8, 1.5, 0.9, 18)

    assert split_predictions([(late, 1.0), (early, 1.0)], rule) == [False, True]  # type: ignore[list-item]


def test_team35_strict_validation_requires_direction_precision_and_recall() -> None:
    direction = {"direction_pass": True}
    split = {"precision": 0.77, "recall": 0.81}

    assert strict_validation_passed(direction, split) is True
    assert strict_validation_passed(direction, split | {"precision": 0.69}) is False
    assert strict_validation_passed(direction, split | {"recall": 0.69}) is False
    assert strict_validation_passed({"direction_pass": False}, split) is False


def test_team35_profile_metadata_is_the_strict_holdout_result() -> None:
    profile = PROFILES[35]
    direction = {
        "direction_median_error_degrees": profile.direction_median_error,
        "direction_within_30_rate": profile.direction_within_30_rate,
    }
    split = {"f1": profile.split_f1}

    assert profile_metadata_matches_validation(
        profile,
        direction,
        split,
        passed=profile.validation_passed,  # type: ignore[arg-type]
    )
    assert not profile_metadata_matches_validation(
        profile,
        direction | {"direction_median_error_degrees": 99.0},
        split,  # type: ignore[arg-type]
        passed=profile.validation_passed,
    )
