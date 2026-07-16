from __future__ import annotations

"""Chronological holdout evaluation for the replay-derived team-35 clone."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bots"), str(ROOT / "scripts")]

from replay_imitation import (  # noqa: E402
    ReplaySample,
    _profile,
    _sample_regime,
    _unit,
    evaluate_profile,
    extract_samples,
)
from strategies.replay_imitation import FEATURE_NAMES, ReplayProfile  # noqa: E402
from strategies.replay_profiles import PROFILES  # noqa: E402
from strategies.replay_team_35 import (  # noqa: E402
    MAX_PREY_DISTANCE,
    MIN_PREY_ALIGNMENT,
    MIN_PREY_RADIUS_RATIO,
    MIN_SPLIT_RADIUS,
    SPLIT_REARM_ROUNDS,
)


TEAM_ID = 35
HOLDOUT_MIN_MATCH_ID = 29_800
MIN_VALIDATION_SPLIT_PRECISION = 0.70
MIN_VALIDATION_SPLIT_RECALL = 0.70
MATCH_IDS = (
    11_673, 11_716, 11_725, 11_739, 11_753,
    27_012, 27_064, 27_105, 27_138, 27_156, 27_173,
    27_202, 27_305, 27_311, 27_312, 27_335, 27_360,
    29_848, 29_855, 29_857, 29_858, 29_869, 29_875,
    29_878, 29_881, 29_882, 29_887, 29_900, 29_904,
)


def default_replay_paths() -> list[Path]:
    roots = (
        ROOT / ".agario/replays/official/latest-20",
        ROOT / ".agario/replays/official/current-submission-49",
        Path.home() / "Downloads",
    )
    paths: list[Path] = []
    for match_id in MATCH_IDS:
        path = next(
            (
                root / f"match-{match_id}-replay.json"
                for root in roots
                if (root / f"match-{match_id}-replay.json").exists()
            ),
            None,
        )
        if path is None:
            raise FileNotFoundError(f"missing team-35 replay {match_id}")
        paths.append(path)
    return paths


def load_samples(paths: Sequence[Path], jobs: int) -> list[ReplaySample]:
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        batches = executor.map(extract_samples, paths)
        return [sample for batch in batches for sample in batch if sample.team_id == TEAM_ID]


def partition_samples(
    samples: Sequence[ReplaySample],
) -> tuple[list[ReplaySample], list[ReplaySample]]:
    """Create a chronological match-level holdout with an explicit leak check."""

    training = [sample for sample in samples if sample.match_id < HOLDOUT_MIN_MATCH_ID]
    holdout = [sample for sample in samples if sample.match_id >= HOLDOUT_MIN_MATCH_ID]
    training_matches = {sample.match_id for sample in training}
    holdout_matches = {sample.match_id for sample in holdout}
    if not training or not holdout:
        raise ValueError("team-35 evaluation requires non-empty training and holdout cohorts")
    if training_matches & holdout_matches:
        raise ValueError("team-35 training and holdout matches overlap")
    return training, holdout


def predicted_directions(
    profile: ReplayProfile, samples: Sequence[ReplaySample]
) -> dict[tuple[int, int, int], tuple[float, float]]:
    previous_by_trace: dict[tuple[int, int], tuple[float, float]] = defaultdict(
        lambda: (0.0, 0.0)
    )
    result: dict[tuple[int, int, int], tuple[float, float]] = {}
    previous_index = FEATURE_NAMES.index("previous")
    previous_left_index = FEATURE_NAMES.index("previous_left")
    for sample in sorted(samples, key=lambda item: (item.match_id, item.player_id, item.round_number)):
        trace = (sample.match_id, sample.player_id)
        previous = previous_by_trace[trace]
        features = list(sample.direction_features)
        features[previous_index] = previous
        features[previous_left_index] = (-previous[1], previous[0])
        regime = _sample_regime(sample)
        override = profile.direction_override_weights[regime] if profile.direction_override_weights else ()
        weights = (
            override
            if override and any(override)
            else profile.regime_direction_weights[regime]
            if profile.regime_direction_weights
            else profile.direction_weights
        )
        direction = _unit(
            sum(weight * vector[0] for weight, vector in zip(weights, features)),
            sum(weight * vector[1] for weight, vector in zip(weights, features)),
        )
        result[(sample.match_id, sample.player_id, sample.round_number)] = direction
        previous_by_trace[trace] = direction
    return result


def tactical_records(
    profile: ReplayProfile, samples: Sequence[ReplaySample]
) -> list[tuple[ReplaySample, float]]:
    directions = predicted_directions(profile, samples)
    prey_index = FEATURE_NAMES.index("nearest_prey")
    return [
        (
            sample,
            sum(
                a * b
                for a, b in zip(
                    directions[(sample.match_id, sample.player_id, sample.round_number)],
                    sample.direction_features[prey_index],
                )
            ),
        )
        for sample in samples
    ]


def split_predictions(
    records: Sequence[tuple[ReplaySample, float]],
    rule: tuple[float, float, float, float, int],
) -> list[bool]:
    radius_min, distance_max, ratio_min, alignment_min, cooldown = rule
    last_split: dict[tuple[int, int], int] = defaultdict(lambda: -10_000)
    predictions = [False] * len(records)
    ordered = sorted(
        enumerate(records),
        key=lambda item: (
            item[1][0].match_id,
            item[1][0].player_id,
            item[1][0].round_number,
        ),
    )
    for original_index, (sample, alignment) in ordered:
        values = sample.split_features
        trace = (sample.match_id, sample.player_id)
        prediction = (
            values[3] <= 0.0625
            and values[4] >= radius_min
            and values[5] >= 1.0
            and values[6] > 0.5
            and values[7] <= distance_max
            and values[8] >= ratio_min
            and values[9] <= 0.0
            and alignment >= alignment_min
            and sample.round_number - last_split[trace] >= cooldown
        )
        predictions[original_index] = prediction
        if prediction:
            last_split[trace] = sample.round_number
    return predictions


def split_metrics(
    records: Sequence[tuple[ReplaySample, float]],
    rule: tuple[float, float, float, float, int],
) -> dict[str, float | int]:
    predictions = split_predictions(records, rule)
    labels = [sample.target_split for sample, _ in records]
    true_positive = sum(label and prediction for label, prediction in zip(labels, predictions))
    predicted_positive = sum(predictions)
    positive = sum(labels)
    precision = true_positive / max(predicted_positive, 1)
    recall = true_positive / max(positive, 1)
    return {
        "positive": positive,
        "predicted_positive": predicted_positive,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
    }


def fit_split_rule(
    profile: ReplayProfile, samples: Sequence[ReplaySample]
) -> tuple[float, float, float, float, int]:
    records = sorted(
        tactical_records(profile, samples),
        key=lambda item: (item[0].match_id, item[0].player_id, item[0].round_number),
    )
    eligible = [
        (sample, alignment)
        for sample, alignment in records
        if sample.split_features[3] <= 0.0625
        and sample.split_features[5] >= 1.0
        and sample.split_features[6] > 0.5
        and sample.split_features[9] <= 0.0
    ]
    positive = sum(sample.target_split for sample, _ in records)
    best_rule = (0.2, 2.0, 0.0, -1.0, 0)
    best_score = (-1.0, -1.0, -1.0)
    for radius_min in (0.2, 0.225, 0.25, 0.275, 0.3):
        for distance_max in (0.5, 0.6, 0.7, 0.8, 1.0, 2.0):
            for ratio_min in (0.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0):
                for alignment_min in (-1.0, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99):
                    for cooldown in (0, 5, 10, 15, 17, 18, 20, 30):
                        rule = (radius_min, distance_max, ratio_min, alignment_min, cooldown)
                        predictions = split_predictions(eligible, rule)
                        true_positive = sum(
                            sample.target_split and prediction
                            for (sample, _), prediction in zip(eligible, predictions)
                        )
                        predicted_positive = sum(predictions)
                        precision = true_positive / max(predicted_positive, 1)
                        recall = true_positive / max(positive, 1)
                        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
                        score = (f1, recall, precision)
                        if score > best_score:
                            best_score = score
                            best_rule = rule
    return best_rule


def turn_state(previous: tuple[float, float], current: tuple[float, float]) -> int:
    dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(previous, current))))
    angle = math.degrees(math.acos(dot)) if previous != (0.0, 0.0) else 180.0
    return 0 if angle <= 15.0 else 1 if angle <= 60.0 else 2


def transition_accuracy(profile: ReplayProfile, samples: Sequence[ReplaySample]) -> float:
    directions = predicted_directions(profile, samples)
    target_previous: dict[tuple[int, int], tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))
    predicted_previous: dict[tuple[int, int], tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))
    matches = total = 0
    for sample in sorted(samples, key=lambda item: (item.match_id, item.player_id, item.round_number)):
        trace = (sample.match_id, sample.player_id)
        predicted = directions[(sample.match_id, sample.player_id, sample.round_number)]
        if not sample.target_split:
            total += 1
            matches += turn_state(target_previous[trace], sample.target_direction) == turn_state(
                predicted_previous[trace], predicted
            )
        target_previous[trace] = sample.target_direction
        predicted_previous[trace] = predicted
    return matches / max(total, 1)


def cohort_metrics(profile: ReplayProfile, samples: Sequence[ReplaySample]) -> dict[str, object]:
    return evaluate_profile(profile, samples) | {"movement_transition_accuracy": transition_accuracy(profile, samples)}


def strict_validation_passed(
    direction_metrics: dict[str, object],
    split_results: dict[str, float | int],
) -> bool:
    return (
        bool(direction_metrics["direction_pass"])
        and float(split_results["precision"]) >= MIN_VALIDATION_SPLIT_PRECISION
        and float(split_results["recall"]) >= MIN_VALIDATION_SPLIT_RECALL
    )


def profile_metadata_matches_validation(
    profile: ReplayProfile,
    direction_metrics: dict[str, object],
    split_results: dict[str, float | int],
    passed: bool,
) -> bool:
    return (
        profile.direction_median_error is not None
        and math.isclose(
            profile.direction_median_error,
            float(direction_metrics["direction_median_error_degrees"]),
        )
        and profile.direction_within_30_rate is not None
        and math.isclose(
            profile.direction_within_30_rate,
            float(direction_metrics["direction_within_30_rate"]),
        )
        and profile.split_f1 is not None
        and math.isclose(profile.split_f1, float(split_results["f1"]))
        and profile.validation_passed is passed
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    samples = load_samples(default_replay_paths(), max(1, args.jobs))
    training, holdout = partition_samples(samples)
    fitted_training = _profile(TEAM_ID, training)
    fitted_all = _profile(TEAM_ID, samples)
    split_rule = fit_split_rule(fitted_training, training)
    implemented_rule = (
        MIN_SPLIT_RADIUS / 10.0,
        MAX_PREY_DISTANCE / 20.0,
        MIN_PREY_RADIUS_RATIO,
        MIN_PREY_ALIGNMENT,
        SPLIT_REARM_ROUNDS,
    )
    strict_training_direction = cohort_metrics(fitted_training, training)
    strict_holdout_direction = cohort_metrics(fitted_training, holdout)
    strict_training_split = split_metrics(
        tactical_records(fitted_training, training), implemented_rule
    )
    strict_holdout_split = split_metrics(
        tactical_records(fitted_training, holdout), implemented_rule
    )
    validation_passed = strict_validation_passed(
        strict_holdout_direction, strict_holdout_split
    )
    metadata_matches = profile_metadata_matches_validation(
        PROFILES[TEAM_ID],
        strict_holdout_direction,
        strict_holdout_split,
        validation_passed,
    )
    if not metadata_matches:
        raise RuntimeError("team-35 profile metadata does not match strict validation")
    report = {
        "training_matches": sorted({sample.match_id for sample in training}),
        "holdout_matches": sorted({sample.match_id for sample in holdout}),
        "sample_counts": {"training": len(training), "holdout": len(holdout)},
        "strict_validation": {
            "profile_fitted_on": "training_matches_only",
            "rule_fitted_on": "training_matches_only",
            "training_direction": strict_training_direction,
            "holdout_direction": strict_holdout_direction,
            "training_split": strict_training_split,
            "holdout_split": strict_holdout_split,
            "passed": validation_passed,
        },
        "final_retrained_descriptive_only": {
            "contains_holdout_training_data": True,
            "training": cohort_metrics(PROFILES[TEAM_ID], training),
            "holdout": cohort_metrics(PROFILES[TEAM_ID], holdout),
        },
        "tactical_split_training_only": {
            "rule": split_rule,
            "training": split_metrics(tactical_records(fitted_training, training), split_rule),
            "holdout": split_metrics(tactical_records(fitted_training, holdout), split_rule),
        },
        "profile_metadata": {
            "direction_median_error": PROFILES[TEAM_ID].direction_median_error,
            "direction_within_30_rate": PROFILES[TEAM_ID].direction_within_30_rate,
            "split_f1": PROFILES[TEAM_ID].split_f1,
            "validation_passed": PROFILES[TEAM_ID].validation_passed,
            "matches_strict_validation": metadata_matches,
        },
        "fitted_all_profile": repr(fitted_all),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
