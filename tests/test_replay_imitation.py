from __future__ import annotations

import math
from pathlib import Path
import sys


BOTS_DIR = Path(__file__).resolve().parents[1] / "bots"
sys.path.insert(0, str(BOTS_DIR))

from strategies.replay_imitation import (
    FEATURE_NAMES,
    SPLIT_FEATURE_NAMES,
    ImitationBlob,
    ImitationObservation,
    ImitationPoint,
    ReplayProfile,
    direction_feature_vectors,
    predict_direction,
    predict_split,
)
from strategies.replay_profiles import PROFILES
from strategies.replay_opponents import OBSERVED_REPLAY_TEAM_IDS


def _observation() -> ImitationObservation:
    return ImitationObservation(
        round_number=100,
        max_rounds=1400,
        arena_size=60.0,
        own_blobs=(ImitationBlob(10.0, 10.0, 1.5, player_id=0),),
        visible_blobs=(
            ImitationBlob(8.0, 10.0, 2.0, player_id=1),
            ImitationBlob(13.0, 10.0, 0.5, player_id=2),
        ),
        visible_food=(ImitationPoint(10.0, 12.0),),
        visible_viruses=(ImitationPoint(15.0, 10.0, 1.5),),
    )


def _nearest_food_profile() -> ReplayProfile:
    weights = [0.0] * len(FEATURE_NAMES)
    weights[FEATURE_NAMES.index("nearest_food")] = 1.0
    return ReplayProfile(
        team_id=999,
        direction_weights=tuple(weights),
        split_weights=tuple(0.0 for _ in SPLIT_FEATURE_NAMES),
        split_threshold=math.inf,
    )


def test_direction_features_reconstruct_relations() -> None:
    vectors = dict(zip(FEATURE_NAMES, direction_feature_vectors(_observation())))
    assert vectors["nearest_food"] == (0.0, 1.0)
    assert vectors["nearest_predator_escape"] == (1.0, 0.0)
    assert vectors["nearest_prey"] == (1.0, 0.0)
    assert vectors["virus_escape"] == (-1.0, 0.0)


def test_profile_predicts_direction_and_respects_no_split() -> None:
    profile = _nearest_food_profile()
    direction = predict_direction(profile, _observation())
    split, score = predict_split(profile, _observation(), (1.0, 0.0), direction)
    assert abs(direction[0]) < 1e-12
    assert abs(direction[1] - 1.0) < 1e-12
    assert split is False
    assert score == 0.0


def test_stateful_split_rule_respects_rearm_interval() -> None:
    base = _nearest_food_profile()
    profile = ReplayProfile(
        team_id=49,
        direction_weights=base.direction_weights,
        split_weights=base.split_weights,
        split_threshold=math.inf,
        split_rule=(0.65, 1.5, 0.15, 0.125, 0.0, 0.0),
        split_cooldown_rounds=18,
    )
    observation = ImitationObservation(
        round_number=100,
        max_rounds=1400,
        arena_size=60.0,
        own_blobs=(ImitationBlob(10.0, 10.0, 2.0, player_id=0),),
        visible_blobs=(ImitationBlob(16.0, 10.0, 1.0, player_id=1),),
        visible_food=(),
        visible_viruses=(),
    )
    split, _ = predict_split(profile, observation, (1.0, 0.0), (1.0, 0.0))
    blocked, _ = predict_split(
        profile,
        observation,
        (1.0, 0.0),
        (1.0, 0.0),
        last_split_round=90,
    )
    assert split is True
    assert blocked is False


def test_direction_override_replaces_one_observation_regime() -> None:
    base = _nearest_food_profile()
    overrides = [[0.0] * len(FEATURE_NAMES) for _ in range(8)]
    overrides[3][FEATURE_NAMES.index("nearest_predator_escape")] = 1.0
    profile = ReplayProfile(
        team_id=35,
        direction_weights=base.direction_weights,
        split_weights=base.split_weights,
        split_threshold=math.inf,
        direction_override_weights=tuple(tuple(weights) for weights in overrides),
    )
    direction = predict_direction(profile, _observation())
    assert direction == (1.0, 0.0)


def test_fragmented_direction_weights_apply_only_after_split() -> None:
    base = _nearest_food_profile()
    fragmented = [[0.0] * len(FEATURE_NAMES) for _ in range(8)]
    fragmented[3][FEATURE_NAMES.index("nearest_predator_escape")] = 1.0
    profile = ReplayProfile(
        team_id=9,
        direction_weights=base.direction_weights,
        split_weights=base.split_weights,
        split_threshold=math.inf,
        fragmented_direction_weights=tuple(tuple(weights) for weights in fragmented),
    )
    single = _observation()
    split = ImitationObservation(
        round_number=single.round_number,
        max_rounds=single.max_rounds,
        arena_size=single.arena_size,
        own_blobs=single.own_blobs + (ImitationBlob(11.0, 10.0, 1.0, player_id=0),),
        visible_blobs=single.visible_blobs,
        visible_food=single.visible_food,
        visible_viruses=single.visible_viruses,
    )

    assert predict_direction(profile, single) == (0.0, 1.0)
    assert predict_direction(profile, split) == (1.0, 0.0)


def test_probabilistic_angle_grid_with_full_rate_is_deterministic() -> None:
    base = _nearest_food_profile()
    profile = ReplayProfile(
        team_id=58,
        direction_weights=base.direction_weights,
        split_weights=base.split_weights,
        split_threshold=math.inf,
        probabilistic_angle_bins=4,
        angle_grid_rates=(1.0,) * 8,
    )
    observation = ImitationObservation(
        round_number=123,
        max_rounds=1400,
        arena_size=60.0,
        own_blobs=(ImitationBlob(10.0, 10.0, 1.5, player_id=2),),
        visible_blobs=(),
        visible_food=(ImitationPoint(11.0, 12.0),),
        visible_viruses=(),
    )

    first = predict_direction(profile, observation)
    second = predict_direction(profile, observation)
    assert first == second
    assert abs(first[0]) < 1e-12
    assert abs(first[1] - 1.0) < 1e-12


def test_generated_profiles_cover_all_replay_opponents() -> None:
    assert set(PROFILES) == set(OBSERVED_REPLAY_TEAM_IDS)
    for profile in PROFILES.values():
        assert profile.source_matches
