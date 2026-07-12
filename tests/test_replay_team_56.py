from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from strategies.replay_imitation import (  # noqa: E402
    ImitationBlob,
    ImitationObservation,
)
from strategies.replay_team_56 import (  # noqa: E402
    DIRECTION_BINS,
    HIGH_MASS_CONTINUATION_RATE,
    HIGH_MASS_ONSET_RATE,
    HIGH_SPLIT_TOTAL_MASS,
    LOW_MASS_CONTINUATION_RATE,
    LOW_MASS_ONSET_RATE,
    MAX_AIMED_PREY_DISTANCE,
    MID_MASS_CONTINUATION_RATE,
    MID_MASS_ONSET_RATE,
    MID_SPLIT_TOTAL_MASS,
    MIN_SPLIT_TOTAL_MASS,
    ReplayTeam56Strategy,
)


def _blob(
    x: float,
    y: float,
    radius: float,
    *,
    player_id: int,
    blob_id: int = 0,
) -> ImitationBlob:
    return ImitationBlob(
        x,
        y,
        radius,
        player_id=player_id,
        blob_id=blob_id,
    )


def _observation(
    *,
    own: tuple[ImitationBlob, ...] | None = None,
    visible: tuple[ImitationBlob, ...] = (),
) -> ImitationObservation:
    return ImitationObservation(
        round_number=600,
        max_rounds=1400,
        arena_size=60.0,
        own_blobs=own
        if own is not None
        else (_blob(10.0, 10.0, 4.0, player_id=4),),
        visible_blobs=visible,
        visible_food=(),
        visible_viruses=(),
    )


def test_team56_profile_uses_all_observed_source_matches() -> None:
    strategy = ReplayTeam56Strategy()

    assert strategy.profile.source_matches == (11646, 11753, 11757)


def test_team56_normal_direction_uses_observed_twenty_four_way_grid() -> None:
    direction = ReplayTeam56Strategy._quantize_direction((0.8, 0.6))
    step = math.tau / DIRECTION_BINS
    angle = math.atan2(direction[1], direction[0])

    assert math.isclose(angle / step, round(angle / step), abs_tol=1e-12)
    assert math.isclose(math.hypot(*direction), 1.0)


def test_team56_never_splits_below_observed_mass_floor() -> None:
    assert ReplayTeam56Strategy._split_probability(
        MIN_SPLIT_TOTAL_MASS - 1e-6,
        False,
    ) == 0.0
    assert ReplayTeam56Strategy._split_probability(
        MIN_SPLIT_TOTAL_MASS - 1e-6,
        True,
    ) == 0.0


def test_team56_uses_measured_onset_rates_for_each_mass_band() -> None:
    assert ReplayTeam56Strategy._split_probability(
        MIN_SPLIT_TOTAL_MASS,
        False,
    ) == LOW_MASS_ONSET_RATE
    assert ReplayTeam56Strategy._split_probability(
        MID_SPLIT_TOTAL_MASS,
        False,
    ) == MID_MASS_ONSET_RATE
    assert ReplayTeam56Strategy._split_probability(
        HIGH_SPLIT_TOTAL_MASS,
        False,
    ) == HIGH_MASS_ONSET_RATE


def test_team56_uses_higher_continuation_rates_inside_split_bursts() -> None:
    assert ReplayTeam56Strategy._split_probability(
        MIN_SPLIT_TOTAL_MASS,
        True,
    ) == LOW_MASS_CONTINUATION_RATE
    assert ReplayTeam56Strategy._split_probability(
        MID_SPLIT_TOTAL_MASS,
        True,
    ) == MID_MASS_CONTINUATION_RATE
    assert ReplayTeam56Strategy._split_probability(
        HIGH_SPLIT_TOTAL_MASS,
        True,
    ) == HIGH_MASS_CONTINUATION_RATE


def test_team56_recognises_child_edible_prey_at_observed_aim_range() -> None:
    prey = _blob(
        10.0 + MAX_AIMED_PREY_DISTANCE,
        10.0,
        1.5,
        player_id=7,
    )

    candidate = ReplayTeam56Strategy._aimed_prey_candidate(
        _observation(visible=(prey,))
    )

    assert candidate is not None
    assert candidate[1] is prey
    assert math.isclose(candidate[2], MAX_AIMED_PREY_DISTANCE)


def test_team56_does_not_aim_at_prey_that_a_child_cannot_eat() -> None:
    too_large = _blob(14.0, 10.0, 3.0, player_id=7)

    candidate = ReplayTeam56Strategy._aimed_prey_candidate(
        _observation(visible=(too_large,))
    )

    assert candidate is None


def test_team56_state_roll_is_reproducible_and_bounded() -> None:
    observation = _observation()
    arguments = {
        "salt": 12345,
        "observation": observation,
        "total_mass": 16.0,
    }

    first = ReplayTeam56Strategy._state_roll(**arguments)
    second = ReplayTeam56Strategy._state_roll(**arguments)

    assert first == second
    assert 0.0 <= first < 1.0
