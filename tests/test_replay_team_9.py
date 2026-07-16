from __future__ import annotations

from strategies.replay_imitation import ImitationBlob, ImitationObservation
from strategies.replay_opponents import create_replay_candidate
from strategies.replay_profiles import PROFILES
from strategies.replay_team_9 import ReplayTeam9Strategy


def _observation(
    *,
    round_number: int = 100,
    radius: float = 2.0,
    merge_cooldown: int = 0,
    blob_count: int = 1,
    prey: ImitationBlob | None = None,
    predator: ImitationBlob | None = None,
    player_id: int = 0,
) -> ImitationObservation:
    return ImitationObservation(
        round_number=round_number,
        max_rounds=1400,
        arena_size=60.0,
        own_blobs=tuple(
            ImitationBlob(
                10.0 + index,
                10.0,
                radius,
                player_id=player_id,
                blob_id=index,
                merge_cooldown=merge_cooldown,
            )
            for index in range(blob_count)
        ),
        visible_blobs=tuple(item for item in (prey, predator) if item is not None),
        visible_food=(),
        visible_viruses=(),
    )


def _prey(*, x: float = 20.0, radius: float = 0.6) -> ImitationBlob:
    return ImitationBlob(x, 10.0, radius, player_id=1)


def test_catalog_constructs_dedicated_team9_strategy() -> None:
    strategy = create_replay_candidate(9)

    assert isinstance(strategy, ReplayTeam9Strategy)
    assert strategy.name == "replay_team_9"


def test_split_state_requires_one_large_merge_ready_blob() -> None:
    assert ReplayTeam9Strategy._split_decision(
        _observation(blob_count=2, prey=_prey())
    ) == (False, "requires_single_blob")
    assert ReplayTeam9Strategy._split_decision(
        _observation(radius=1.99, prey=_prey())
    ) == (False, "below_observed_radius_floor")
    assert ReplayTeam9Strategy._split_decision(
        _observation(merge_cooldown=1, prey=_prey())
    ) == (False, "merge_cooldown_active")


def test_split_state_requires_close_sufficiently_small_prey() -> None:
    split = ReplayTeam9Strategy._split_decision(_observation(prey=_prey()))
    too_far = ReplayTeam9Strategy._split_decision(
        _observation(prey=_prey(x=25.1))
    )
    too_large = ReplayTeam9Strategy._split_decision(
        _observation(prey=_prey(radius=0.7))
    )
    no_prey = ReplayTeam9Strategy._split_decision(_observation())

    assert split == (True, "single_blob_prey_split")
    assert too_far == (False, "prey_too_far")
    assert too_large == (False, "prey_too_large")
    assert no_prey == (False, "no_visible_prey")


def test_split_state_rejects_visible_predator() -> None:
    decision = ReplayTeam9Strategy._split_decision(
        _observation(
            prey=_prey(),
            predator=ImitationBlob(8.0, 10.0, 3.0, player_id=2),
        )
    )

    assert decision == (False, "predator_visible")


def test_split_state_rearms_after_fifteen_rounds() -> None:
    blocked = ReplayTeam9Strategy._split_decision(
        _observation(round_number=114, prey=_prey()),
        last_split_round=100,
    )
    rearmed = ReplayTeam9Strategy._split_decision(
        _observation(round_number=115, prey=_prey()),
        last_split_round=100,
    )

    assert blocked == (False, "split_rearming")
    assert rearmed == (True, "single_blob_prey_split")


def test_profile_reports_measured_geometry_f1_not_hidden_rng_match() -> None:
    assert PROFILES[9].split_f1 == 0.4365482233502538


def test_team9_resets_temporal_state_when_rounds_restart() -> None:
    strategy = ReplayTeam9Strategy()
    strategy._begin_observation(_observation(round_number=1_399, player_id=4))
    strategy._last_split_round = 1_390
    strategy._previous_direction = (1.0, 0.0)

    trace_reset = strategy._begin_observation(
        _observation(round_number=0, player_id=4)
    )

    assert trace_reset is True
    assert strategy._last_observed_round == 0
    assert strategy._last_player_id == 4
    assert strategy._last_split_round == -10_000
    assert strategy._previous_direction == (0.0, 0.0)


def test_team9_resets_temporal_state_when_player_slot_changes() -> None:
    strategy = ReplayTeam9Strategy()
    strategy._begin_observation(_observation(round_number=100, player_id=2))
    strategy._last_split_round = 95
    strategy._previous_direction = (1.0, 0.0)

    trace_reset = strategy._begin_observation(
        _observation(round_number=101, player_id=6)
    )

    assert trace_reset is True
    assert strategy._last_player_id == 6
    assert strategy._last_split_round == -10_000
    assert strategy._previous_direction == (0.0, 0.0)


def test_team9_same_round_retry_keeps_temporal_state() -> None:
    strategy = ReplayTeam9Strategy()
    state = _observation(round_number=100, player_id=3)
    strategy._begin_observation(state)
    strategy._last_split_round = 95
    strategy._previous_direction = (1.0, 0.0)

    trace_reset = strategy._begin_observation(state)

    assert trace_reset is False
    assert strategy._last_split_round == 95
    assert strategy._previous_direction == (1.0, 0.0)
