from __future__ import annotations

from strategies.replay_imitation import (
    ImitationBlob,
    ImitationObservation,
    ImitationPoint,
    predict_direction,
)
from strategies.replay_profiles import PROFILES
from strategies.replay_team_1 import ReplayTeam1Strategy


def _observation(
    *,
    round_number: int = 100,
    own_blobs: tuple[ImitationBlob, ...] | None = None,
    visible_blobs: tuple[ImitationBlob, ...] = (),
    visible_food: tuple[ImitationPoint, ...] = (),
    visible_viruses: tuple[ImitationPoint, ...] = (),
) -> ImitationObservation:
    return ImitationObservation(
        round_number=round_number,
        max_rounds=1400,
        arena_size=60.0,
        own_blobs=(
            own_blobs
            if own_blobs is not None
            else (ImitationBlob(10.0, 10.0, 2.0, player_id=0),)
        ),
        visible_blobs=visible_blobs,
        visible_food=visible_food,
        visible_viruses=visible_viruses,
    )


def test_team1_tracks_food_exactly_without_predator() -> None:
    observation = _observation(
        visible_food=(ImitationPoint(10.0, 14.0), ImitationPoint(14.0, 10.0)),
    )

    assert predict_direction(PROFILES[1], observation) == (0.0, 1.0)
    assert predict_direction(PROFILES[1], observation, (-1.0, 0.0)) == (0.0, 1.0)


def test_team1_tracks_nearest_prey_exactly_without_predator() -> None:
    observation = _observation(
        visible_blobs=(ImitationBlob(14.0, 10.0, 1.0, player_id=1),),
        visible_food=(ImitationPoint(10.0, 11.0),),
    )

    assert predict_direction(PROFILES[1], observation) == (1.0, 0.0)


def test_team1_split_requires_merge_ready_resource_state() -> None:
    prey = ImitationBlob(14.0, 10.0, 1.0, player_id=1)
    split, reason = ReplayTeam1Strategy._split_decision(
        _observation(visible_blobs=(prey,))
    )
    blocked, blocked_reason = ReplayTeam1Strategy._split_decision(
        _observation(
            own_blobs=(
                ImitationBlob(
                    10.0,
                    10.0,
                    2.0,
                    player_id=0,
                    merge_cooldown=3,
                ),
            ),
            visible_blobs=(prey,),
        )
    )

    assert (split, reason) == (True, "merge_ready_resource_split")
    assert (blocked, blocked_reason) == (False, "no_merge_ready_blob")


def test_team1_treats_consumable_virus_as_split_resource() -> None:
    split, reason = ReplayTeam1Strategy._split_decision(
        _observation(visible_viruses=(ImitationPoint(13.0, 10.0, 1.5),))
    )

    assert (split, reason) == (True, "merge_ready_resource_split")


def test_team1_split_state_machine_rearms_after_fifteen_rounds() -> None:
    prey = ImitationBlob(14.0, 10.0, 1.0, player_id=1)
    blocked = ReplayTeam1Strategy._split_decision(
        _observation(round_number=114, visible_blobs=(prey,)),
        last_split_round=100,
    )
    rearmed = ReplayTeam1Strategy._split_decision(
        _observation(round_number=115, visible_blobs=(prey,)),
        last_split_round=100,
    )

    assert blocked == (False, "split_rearming")
    assert rearmed == (True, "merge_ready_resource_split")


def test_team1_split_rejects_resource_free_and_overfragmented_states() -> None:
    no_resource = ReplayTeam1Strategy._split_decision(_observation())
    overfragmented = ReplayTeam1Strategy._split_decision(
        _observation(
            own_blobs=tuple(
                ImitationBlob(10.0 + index, 10.0, 2.0, player_id=0)
                for index in range(9)
            ),
            visible_blobs=(ImitationBlob(14.0, 10.0, 1.0, player_id=1),),
        )
    )

    assert no_resource == (False, "no_visible_resource")
    assert overfragmented == (False, "fragment_cap")


def test_team1_empty_observation_cannot_split() -> None:
    assert ReplayTeam1Strategy._split_decision(
        _observation(own_blobs=(), visible_viruses=(ImitationPoint(13.0, 10.0, 1.5),))
    ) == (False, "no_split_sized_blob")


def test_team1_split_features_are_separate_across_fragments() -> None:
    prey = ImitationBlob(14.0, 10.0, 1.0, player_id=1)
    fragments = (
        ImitationBlob(10.0, 10.0, 2.5, player_id=0, merge_cooldown=5),
        ImitationBlob(11.0, 10.0, 1.0, player_id=0, merge_cooldown=0),
    )

    assert ReplayTeam1Strategy._split_decision(
        _observation(own_blobs=fragments, visible_blobs=(prey,))
    ) == (True, "merge_ready_resource_split")


def test_team1_allows_exactly_eight_fragments() -> None:
    fragments = tuple(
        ImitationBlob(
            10.0 + index,
            10.0,
            2.0 if index == 0 else 0.5,
            player_id=0,
        )
        for index in range(8)
    )
    prey = ImitationBlob(14.0, 10.0, 0.25, player_id=1)

    assert ReplayTeam1Strategy._split_decision(
        _observation(own_blobs=fragments, visible_blobs=(prey,))
    ) == (True, "merge_ready_resource_split")


def test_team1_resets_state_when_round_rewinds_for_a_new_match() -> None:
    strategy = ReplayTeam1Strategy()
    strategy._last_split_round = 1_200
    strategy._previous_direction = (0.0, 1.0)

    assert strategy._begin_observation(_observation(round_number=1_300)) is False
    assert strategy._begin_observation(_observation(round_number=0)) is True
    assert strategy._last_split_round == -10_000
    assert strategy._previous_direction == (0.0, 0.0)


def test_team1_same_round_retry_preserves_state_but_player_change_resets() -> None:
    strategy = ReplayTeam1Strategy()
    first = _observation(round_number=50)
    another_player = _observation(
        round_number=50,
        own_blobs=(ImitationBlob(10.0, 10.0, 2.0, player_id=7),),
    )

    assert strategy._begin_observation(first) is False
    strategy._last_split_round = 49
    assert strategy._begin_observation(first) is False
    assert strategy._last_split_round == 49
    assert strategy._begin_observation(another_player) is True
    assert strategy._last_split_round == -10_000


def test_team1_metadata_reports_chronological_holdout_not_in_sample_metrics() -> None:
    profile = PROFILES[1]

    assert len(profile.source_matches) == 20
    assert profile.direction_median_error == 8.537736462515939e-07
    assert profile.direction_within_30_rate == 0.7160957297043642
    assert profile.split_f1 == 0.48101265822784806
    assert profile.validation_passed is False
