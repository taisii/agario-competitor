from __future__ import annotations

"""Validate team 9's geometry rule against its rejected random surrogate."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import itertools
import math
import os
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bots"))

from scripts.replay_imitation import (  # noqa: E402
    ReplaySample,
    _f1,
    _tolerant_split_f1,
    extract_samples,
)
from strategies.replay_imitation import stable_unit_interval  # noqa: E402
from strategies.replay_profiles import PROFILES  # noqa: E402


TEAM_ID = 9
REJECTED_MIN_SPLIT_MASS = 2.0
REJECTED_SPLIT_RATE = 1.0 / 64.0
DEFAULT_REPLAY_DIRECTORIES = (
    ROOT / ".agario/replays/official/current-submission-49",
    Path.home() / "Downloads",
)
DEFAULT_HOLDOUT_MIN_MATCH_ID = 29_000


def resolve_replay_paths(
    directories: Sequence[Path],
    source_matches: Sequence[int] = PROFILES[TEAM_ID].source_matches,
) -> tuple[Path, ...]:
    """Resolve only the recorded source cohort, in directory preference order."""

    by_match: dict[int, Path] = {}
    for match_id in source_matches:
        path = next(
            (
                directory / f"match-{match_id}-replay.json"
                for directory in directories
                if (directory / f"match-{match_id}-replay.json").is_file()
            ),
            None,
        )
        if path is not None:
            by_match[match_id] = path
    missing = sorted(set(source_matches) - by_match.keys())
    if missing:
        roots = ", ".join(str(directory) for directory in directories)
        raise FileNotFoundError(
            f"missing team-9 source replays {missing}; searched: {roots}"
        )
    return tuple(by_match[match_id] for match_id in source_matches)


def _team_samples(path: Path) -> list[ReplaySample]:
    return [sample for sample in extract_samples(path) if sample.team_id == TEAM_ID]


def _eligible(sample: ReplaySample) -> bool:
    return (
        math.isclose(sample.split_features[3] * 16.0, 1.0)
        and (sample.split_features[4] * 10.0) ** 2 >= REJECTED_MIN_SPLIT_MASS
        and sample.split_features[5] > 0.0
    )


def _rejected_surrogate_split(sample: ReplaySample) -> bool:
    return _eligible(sample) and stable_unit_interval(
        TEAM_ID,
        sample.player_id,
        sample.round_number,
    ) < REJECTED_SPLIT_RATE


def _rule_predictions(
    samples: list[ReplaySample],
    parameters: tuple[float, float, float, float, int],
) -> list[bool]:
    distance_max, ratio_min, radius_min, predator_max, cooldown = parameters
    last_split: dict[tuple[int, int], int] = defaultdict(lambda: -10_000)
    predictions = []
    for sample in samples:
        key = (sample.match_id, sample.player_id)
        split = (
            math.isclose(sample.split_features[3] * 16.0, 1.0)
            and sample.split_features[4] * 10.0 >= radius_min
            and sample.split_features[5] > 0.0
            and sample.split_features[6] > 0.5
            and sample.split_features[7] <= distance_max
            and sample.split_features[8] >= ratio_min
            and sample.split_features[9] <= predator_max
            and sample.round_number - last_split[key] >= cooldown
        )
        predictions.append(split)
        if split:
            last_split[key] = sample.round_number
    return predictions


def partition_samples(
    samples: Sequence[ReplaySample],
    holdout_min_match_id: int,
) -> tuple[list[ReplaySample], list[ReplaySample]]:
    training = [
        sample for sample in samples if sample.match_id < holdout_min_match_id
    ]
    held_out = [
        sample for sample in samples if sample.match_id >= holdout_min_match_id
    ]
    training_matches = {sample.match_id for sample in training}
    held_out_matches = {sample.match_id for sample in held_out}
    overlap = sorted(training_matches & held_out_matches)
    if overlap:
        raise ValueError(f"training/holdout match overlap: {overlap}")
    if not training or not held_out:
        raise ValueError(
            "team-9 evaluation requires non-empty, disjoint training and holdout cohorts"
        )
    return training, held_out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay-dir",
        action="append",
        type=Path,
        dest="replay_dirs",
        help="Replay root, in preference order; repeat as needed",
    )
    parser.add_argument(
        "--holdout-min-match-id",
        type=int,
        default=DEFAULT_HOLDOUT_MIN_MATCH_ID,
    )
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    directories = tuple(args.replay_dirs or DEFAULT_REPLAY_DIRECTORIES)
    paths = resolve_replay_paths(directories)
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        samples = [
            sample
            for match_samples in executor.map(_team_samples, paths)
            for sample in match_samples
        ]
    samples.sort(key=lambda item: (item.match_id, item.round_number))
    training, held_out = partition_samples(samples, args.holdout_min_match_id)
    cohorts = {
        "training": training,
        "held_out": held_out,
        "all": samples,
    }

    training = cohorts["training"]
    training_labels = [sample.target_split for sample in training]
    fitted_rules = []
    for parameters in itertools.product(
        (0.4, 0.5, 0.6, 0.65, 0.75, 1.0, 2.0),
        (0.0, 1.5, 2.0, 2.5, 3.0, 4.0),
        (0.0, 1.0, math.sqrt(2.0), 2.0),
        (0.0, 1.0),
        (0, 5, 10, 15, 30, 60, 90),
    ):
        predictions = _rule_predictions(training, parameters)
        precision, recall, exact_f1 = _f1(training_labels, predictions)
        _, _, tolerant_f1 = _tolerant_split_f1(training, predictions)
        fitted_rules.append((exact_f1, tolerant_f1, precision, recall, parameters))
    print("best training geometry rules")
    for exact_f1, tolerant_f1, precision, recall, parameters in sorted(
        fitted_rules, reverse=True
    )[:10]:
        print(
            f"  parameters={parameters} precision={precision:.6f} "
            f"recall={recall:.6f} exact_f1={exact_f1:.6f} "
            f"tolerant_f1={tolerant_f1:.6f}"
        )

    selected_parameters = max(fitted_rules)[-1]
    for name in ("training", "held_out", "all"):
        cohort = cohorts[name]
        predictions = _rule_predictions(cohort, selected_parameters)
        precision, recall, exact_f1 = _f1(
            [sample.target_split for sample in cohort], predictions
        )
        _, _, tolerant_f1 = _tolerant_split_f1(cohort, predictions)
        print(
            f"selected geometry {name}: precision={precision:.6f} "
            f"recall={recall:.6f} exact_f1={exact_f1:.6f} "
            f"tolerant_f1={tolerant_f1:.6f} predictions={sum(predictions)}"
        )
    for name, cohort in cohorts.items():
        eligible = [sample for sample in cohort if _eligible(sample)]
        labels = [sample.target_split for sample in cohort]
        predictions = [_rejected_surrogate_split(sample) for sample in cohort]
        precision, recall, exact_f1 = _f1(labels, predictions)
        positives = sum(labels)
        state_violations = sum(
            sample.target_split and not _eligible(sample) for sample in cohort
        )
        prey_positives = sum(
            sample.target_split and sample.split_features[6] > 0.5
            for sample in cohort
        )
        print(
            f"{name}: matches={len({sample.match_id for sample in cohort})} "
            f"turns={len(cohort)} eligible={len(eligible)} "
            f"actual={positives} actual_rate={positives / len(eligible):.6f} "
            f"rejected_surrogate={sum(predictions)} "
            f"rejected_surrogate_rate={sum(predictions) / len(eligible):.6f} "
            f"exact_precision={precision:.6f} exact_recall={recall:.6f} "
            f"exact_f1={exact_f1:.6f} state_violations={state_violations} "
            f"prey_positives={prey_positives}"
        )
        for prey_visible in (False, True):
            subset = [
                sample
                for sample in eligible
                if (sample.split_features[6] > 0.5) is prey_visible
            ]
            subset_positives = sum(sample.target_split for sample in subset)
            print(
                f"  prey_visible={prey_visible}: eligible={len(subset)} "
                f"actual={subset_positives} "
                f"rate={subset_positives / len(subset) if subset else 0.0:.6f}"
            )
        positives_in_cohort = [sample for sample in cohort if sample.target_split]
        for feature_index, feature_name in ((7, "prey_distance"), (8, "prey_ratio")):
            values = sorted(sample.split_features[feature_index] for sample in positives_in_cohort)
            print(
                f"  positive_{feature_name}: min={values[0]:.6f} "
                f"median={values[len(values) // 2]:.6f} max={values[-1]:.6f}"
            )
        for sample in cohort:
            if sample.target_split and not _eligible(sample):
                print(
                    f"  state_violation match={sample.match_id} round={sample.round_number} "
                    f"blob_count={sample.split_features[3] * 16.0:.0f} "
                    f"radius={sample.split_features[4] * 10.0:.6f} "
                    f"merge_ready={sample.split_features[5]:.6f}"
                )


if __name__ == "__main__":
    main()
