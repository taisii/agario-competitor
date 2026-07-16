from __future__ import annotations

"""Chronological holdout analysis for the official team-1 clone."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bots")]

from scripts.replay_imitation import (  # noqa: E402
    ReplaySample,
    _f1,
    _tolerant_split_f1,
    evaluate_profile,
    extract_samples,
)
from strategies.replay_profiles import PROFILES  # noqa: E402
from strategies.replay_team_1 import (  # noqa: E402
    MAX_SPLIT_BLOB_COUNT,
    MIN_SPLIT_RADIUS,
    SPLIT_REARM_ROUNDS,
)


TEAM_ID = 1
DEFAULT_HOLDOUT_MIN_MATCH_ID = 29_800
TARGET_MODES = ("any", "prey", "virus", "resource", "safe")


def replay_paths(directories: Sequence[Path]) -> tuple[Path, ...]:
    """Resolve exactly the profile cohort, preferring the first supplied root."""

    paths: list[Path] = []
    missing: list[int] = []
    for match_id in PROFILES[TEAM_ID].source_matches:
        path = next(
            (
                directory / f"match-{match_id}-replay.json"
                for directory in directories
                if (directory / f"match-{match_id}-replay.json").exists()
            ),
            None,
        )
        if path is None:
            missing.append(match_id)
        else:
            paths.append(path)
    if missing:
        roots = ", ".join(map(str, directories))
        raise FileNotFoundError(f"missing team-1 matches {missing} under {roots}")
    return tuple(paths)


def extract_team_1(path: Path) -> list[ReplaySample]:
    return [sample for sample in extract_samples(path) if sample.team_id == TEAM_ID]


def load_samples(paths: Sequence[Path], jobs: int) -> list[ReplaySample]:
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        samples = [
            sample
            for batch in executor.map(extract_team_1, paths)
            for sample in batch
        ]
    return sorted(
        samples,
        key=lambda sample: (sample.match_id, sample.player_id, sample.round_number),
    )


def stateful_predictions(
    samples: Sequence[ReplaySample],
    *,
    minimum_radius: float,
    maximum_blob_count: int,
    cooldown: int,
    target_mode: str,
) -> list[bool]:
    last_split_by_trace: dict[tuple[int, int], int] = defaultdict(lambda: -10_000)
    predictions: list[bool] = []
    for sample in sorted(
        samples,
        key=lambda item: (item.match_id, item.player_id, item.round_number),
    ):
        key = (sample.match_id, sample.player_id)
        prey = sample.split_features[6] > 0.5
        predator = sample.split_features[9] > 0.5
        virus = sample.split_features[12] > 0.5
        target_matches = {
            "any": True,
            "prey": prey,
            "virus": virus,
            "resource": prey or virus,
            "safe": not predator,
        }[target_mode]
        split = (
            sample.split_features[4] * 10.0 >= minimum_radius
            and sample.split_features[3] * 16.0 <= maximum_blob_count
            and sample.split_features[5] > 0.0
            and target_matches
            and sample.round_number - last_split_by_trace[key] >= cooldown
        )
        predictions.append(split)
        if split:
            last_split_by_trace[key] = sample.round_number
    return predictions


def split_metrics(
    samples: Sequence[ReplaySample], predictions: Sequence[bool]
) -> dict[str, float | int]:
    labels = [sample.target_split for sample in samples]
    precision, recall, exact_f1 = _f1(labels, predictions)
    tolerant_precision, tolerant_recall, tolerant_f1 = _tolerant_split_f1(
        samples, predictions
    )
    return {
        "positive_count": sum(labels),
        "predicted_positive_count": sum(predictions),
        "precision": precision,
        "recall": recall,
        "f1": exact_f1,
        "tolerant_precision": tolerant_precision,
        "tolerant_recall": tolerant_recall,
        "tolerant_f1": tolerant_f1,
    }


def fit_split_rule(
    training: Sequence[ReplaySample],
) -> tuple[float, int, int, str]:
    best_rule = (MIN_SPLIT_RADIUS, MAX_SPLIT_BLOB_COUNT, SPLIT_REARM_ROUNDS, "resource")
    best_score = (-1.0, -1.0, -1.0)
    for radius in (math.sqrt(2.0), 2.0, 2.5, 3.0):
        for blob_count in (1, 2, 4, 8, 15):
            for cooldown in (5, 10, 15, 17, 18, 19, 20, 25, 30):
                for target_mode in TARGET_MODES:
                    rule = (radius, blob_count, cooldown, target_mode)
                    metrics = split_metrics(
                        training,
                        stateful_predictions(
                            training,
                            minimum_radius=radius,
                            maximum_blob_count=blob_count,
                            cooldown=cooldown,
                            target_mode=target_mode,
                        ),
                    )
                    score = (
                        float(metrics["tolerant_f1"]),
                        float(metrics["f1"]),
                        float(metrics["precision"]),
                    )
                    if score > best_score:
                        best_rule = rule
                        best_score = score
    return best_rule


def direction_metrics(samples: Sequence[ReplaySample]) -> dict[str, dict[str, object]]:
    baseline = replace(PROFILES[TEAM_ID], direction_override_weights=())
    keys = (
        "direction_median_error_degrees",
        "direction_p75_error_degrees",
        "direction_within_30_rate",
        "direction_over_90_rate",
        "direction_pass",
    )
    return {
        name: {key: evaluated[key] for key in keys}
        for name, evaluated in (
            ("fitted_baseline", evaluate_profile(baseline, samples)),
            ("resource_override", evaluate_profile(PROFILES[TEAM_ID], samples)),
        )
    }


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directories = args.replay_dirs or [
        ROOT / ".agario/replays/official/current-submission-49",
        Path.home() / "Downloads",
    ]
    samples = load_samples(replay_paths(directories), max(args.jobs, 1))
    training = [
        sample for sample in samples if sample.match_id < args.holdout_min_match_id
    ]
    holdout = [
        sample for sample in samples if sample.match_id >= args.holdout_min_match_id
    ]
    if not training or not holdout:
        raise SystemExit("chronological training and holdout cohorts are both required")

    fitted_rule = fit_split_rule(training)
    implemented_rule = (
        MIN_SPLIT_RADIUS,
        MAX_SPLIT_BLOB_COUNT,
        SPLIT_REARM_ROUNDS,
        "resource",
    )

    def evaluate_rule(
        cohort: Sequence[ReplaySample], rule: tuple[float, int, int, str]
    ) -> dict[str, float | int]:
        radius, blob_count, cooldown, target_mode = rule
        return split_metrics(
            cohort,
            stateful_predictions(
                cohort,
                minimum_radius=radius,
                maximum_blob_count=blob_count,
                cooldown=cooldown,
                target_mode=target_mode,
            ),
        )

    report = {
        "team_id": TEAM_ID,
        "replay_roots": [str(directory) for directory in directories],
        "training_matches": sorted({sample.match_id for sample in training}),
        "holdout_matches": sorted({sample.match_id for sample in holdout}),
        "sample_counts": {"training": len(training), "holdout": len(holdout)},
        "direction": {
            "training": direction_metrics(training),
            "holdout": direction_metrics(holdout),
        },
        "split": {
            "fitted_training_only_rule": fitted_rule,
            "implemented_rule": implemented_rule,
            "training": evaluate_rule(training, implemented_rule),
            "holdout": evaluate_rule(holdout, implemented_rule),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
