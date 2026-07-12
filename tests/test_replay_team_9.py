from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bots"))

from strategies.replay_imitation import (  # noqa: E402
    ImitationBlob,
    ImitationObservation,
    ImitationPoint,
    predict_direction,
    predict_split,
)
from strategies.replay_team_9 import (  # noqa: E402
    PROFILE,
)


def _observation(
    *,
    own: tuple[ImitationBlob, ...] | None = None,
    enemies: tuple[ImitationBlob, ...] = (),
) -> ImitationObservation:
    return ImitationObservation(
        round_number=700,
        max_rounds=1400,
        arena_size=60.0,
        own_blobs=own or (ImitationBlob(20.0, 20.0, 3.0, player_id=0, blob_id=0),),
        visible_blobs=enemies,
        visible_food=(ImitationPoint(24.0, 20.0, entity_id=1),),
        visible_viruses=(),
    )


def test_team9_profile_covers_all_five_matches() -> None:
    assert PROFILE.source_matches == (11646, 11673, 11716, 11719, 11756)


def test_team9_direction_is_unit_length() -> None:
    direction = predict_direction(PROFILE, _observation())

    assert math.isclose(math.hypot(*direction), 1.0)


def test_team9_profile_has_replay_fitted_split_state_machine() -> None:
    assert PROFILE.split_rule == (0.65, 2.5, 0.14, 0.0625, 0.0, 0.0)
    assert PROFILE.split_cooldown_rounds == 90


def test_team9_close_prey_triggers_after_rearm() -> None:
    prey = ImitationBlob(26.0, 20.0, 1.0, player_id=2, blob_id=4)
    observation = _observation(enemies=(prey,))
    direction = predict_direction(PROFILE, observation)
    split, _ = predict_split(
        PROFILE,
        observation,
        (1.0, 0.0),
        direction,
        last_split_round=500,
    )
    assert split is True


def test_team9_predator_and_rearm_suppress_split() -> None:
    prey = ImitationBlob(26.0, 20.0, 1.0, player_id=2, blob_id=4)
    predator = ImitationBlob(18.0, 20.0, 4.0, player_id=3, blob_id=0)

    observation = _observation(enemies=(prey, predator))
    direction = predict_direction(PROFILE, observation)
    predator_blocked, _ = predict_split(
        PROFILE, observation, (1.0, 0.0), direction, last_split_round=500
    )
    safe_observation = _observation(enemies=(prey,))
    safe_direction = predict_direction(PROFILE, safe_observation)
    cooldown_blocked, _ = predict_split(
        PROFILE,
        safe_observation,
        (1.0, 0.0),
        safe_direction,
        last_split_round=650,
    )
    assert predator_blocked is False
    assert cooldown_blocked is False
