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
from strategies.replay_team_31 import (  # noqa: E402
    DIRECTION_BINS,
    FOUR_BLOB_SPLIT_RATE,
    MAX_SPLIT_PREY_DISTANCE,
    SINGLE_BLOB_SPLIT_RATE,
    TWO_BLOB_SPLIT_RATE,
    ReplayTeam31Strategy,
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
        round_number=400,
        max_rounds=1400,
        arena_size=60.0,
        own_blobs=own
        if own is not None
        else (_blob(10.0, 10.0, 3.0, player_id=2),),
        visible_blobs=visible,
        visible_food=(),
        visible_viruses=(),
    )


def test_team31_profile_uses_all_observed_source_matches() -> None:
    strategy = ReplayTeam31Strategy()

    assert strategy.profile.source_matches == (11667, 11710, 11724, 13934)


def test_team31_direction_is_snapped_to_observed_sixteen_way_grid() -> None:
    direction = ReplayTeam31Strategy._quantize_direction((0.8, 0.6))
    angle = math.atan2(direction[1], direction[0])
    step = math.tau / DIRECTION_BINS

    assert math.isclose(angle / step, round(angle / step), abs_tol=1e-12)
    assert math.isclose(math.hypot(*direction), 1.0)


def test_team31_split_rates_preserve_fragment_attack_stages() -> None:
    assert ReplayTeam31Strategy._split_rate(1) == SINGLE_BLOB_SPLIT_RATE
    assert ReplayTeam31Strategy._split_rate(2) == TWO_BLOB_SPLIT_RATE
    assert ReplayTeam31Strategy._split_rate(3) == TWO_BLOB_SPLIT_RATE
    assert ReplayTeam31Strategy._split_rate(4) == FOUR_BLOB_SPLIT_RATE


def test_team31_accepts_child_edible_prey_within_observed_range() -> None:
    prey = _blob(
        10.0 + MAX_SPLIT_PREY_DISTANCE,
        10.0,
        1.5,
        player_id=7,
    )

    candidate = ReplayTeam31Strategy._split_candidate(
        _observation(visible=(prey,))
    )

    assert candidate is not None
    assert candidate[1] is prey
    assert math.isclose(candidate[2], MAX_SPLIT_PREY_DISTANCE)


def test_team31_rejects_prey_that_a_split_child_cannot_eat() -> None:
    too_large = _blob(14.0, 10.0, 2.2, player_id=7)

    candidate = ReplayTeam31Strategy._split_candidate(
        _observation(visible=(too_large,))
    )

    assert candidate is None


def test_team31_predator_suppresses_split_even_with_valid_prey() -> None:
    prey = _blob(14.0, 10.0, 1.5, player_id=7)
    predator = _blob(16.0, 10.0, 4.0, player_id=9)

    candidate = ReplayTeam31Strategy._split_candidate(
        _observation(visible=(prey, predator))
    )

    assert candidate is None


def test_team31_split_roll_is_reproducible_and_bounded() -> None:
    arguments = {
        "round_number": 400,
        "player_id": 2,
        "blob_count": 1,
        "own_radius": 3.0,
        "prey_radius": 1.5,
        "prey_distance": 4.0,
    }

    first = ReplayTeam31Strategy._split_roll(**arguments)
    second = ReplayTeam31Strategy._split_roll(**arguments)

    assert first == second
    assert 0.0 <= first < 1.0
