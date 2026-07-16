from __future__ import annotations

"""Fit a cheap semantic feature model to ReplayDominanceStrategy decisions."""

import argparse
from collections import defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
BOTS = ROOT / "bots"
sys.path.insert(0, str(BOTS))
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_expected_responses import extract_frames  # noqa: E402
from compare_strategy_decisions import _context  # noqa: E402
from replay_imitation import (  # noqa: E402
    FEATURE_NAMES,
    ReplaySample,
    _solve,
)
from strategies.receding_horizon import ReplayDominanceStrategy  # noqa: E402
from strategies.replay_imitation import (  # noqa: E402
    direction_feature_vectors,
    observation_from_context,
    split_feature_values,
)
from strategies.semantic_potential import SemanticPotentialStrategy  # noqa: E402


RESIDUAL_FEATURE_NAMES = ("semantic_direction",) + FEATURE_NAMES


def _player_id_for_team(started: dict[str, object], team_id: int) -> int:
    for raw_player in started["players"]:
        player = dict(raw_player)
        if int(player["team_id"]) == team_id:
            return int(player["player_id"])
    raise ValueError(f"team {team_id} is absent from replay")


def extract_teacher_samples(
    replay_path: Path,
    *,
    team_id: int,
    every_n: int,
) -> list[ReplaySample]:
    events = json.loads(replay_path.read_text(encoding="utf-8"))
    started = events[0]
    player_id = _player_id_for_team(started, team_id)
    max_rounds = int(started.get("max_rounds", 1400))
    match_id = int(replay_path.name.split("-")[1])
    teacher = ReplayDominanceStrategy()
    semantic = SemanticPotentialStrategy()
    previous_direction = (0.0, 0.0)
    samples: list[ReplaySample] = []

    for frame in extract_frames(replay_path):
        context = _context(
            frame,
            player_id=player_id,
            max_rounds=max_rounds,
        )
        if context is None:
            continue
        decision = teacher.choose(context)
        semantic_decision = semantic.choose(context)
        observation = observation_from_context(context)
        if frame.round_number % every_n == 0:
            samples.append(
                ReplaySample(
                    match_id=match_id,
                    team_id=team_id,
                    player_id=player_id,
                    round_number=frame.round_number,
                    direction_features=(
                        semantic_decision.direction,
                    )
                    + direction_feature_vectors(
                        observation,
                        previous_direction,
                    ),
                    split_features=split_feature_values(
                        observation,
                        previous_direction,
                        decision.direction,
                    ),
                    target_direction=decision.direction,
                    target_split=decision.split,
                )
            )
        previous_direction = decision.direction
    return samples


def _unit(vector: tuple[float, float]) -> tuple[float, float]:
    magnitude = math.hypot(*vector)
    if magnitude <= 1e-9 or not math.isfinite(magnitude):
        return (0.0, 0.0)
    return (vector[0] / magnitude, vector[1] / magnitude)


def _regime(sample: ReplaySample) -> int:
    return (
        (1 if sample.split_features[9] > 0.5 else 0)
        | (2 if sample.split_features[6] > 0.5 else 0)
        | (4 if sample.split_features[12] > 0.5 else 0)
    )


def _fit_direction(
    samples: list[ReplaySample],
    *,
    ridge: float = 0.25,
) -> tuple[float, ...]:
    size = len(RESIDUAL_FEATURE_NAMES)
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    target = [0.0 for _ in range(size)]
    for sample in samples:
        target_x, target_y = sample.target_direction
        for i, first in enumerate(sample.direction_features):
            target[i] += first[0] * target_x + first[1] * target_y
            for j, second in enumerate(sample.direction_features):
                matrix[i][j] += (
                    first[0] * second[0] + first[1] * second[1]
                )
    for index in range(size):
        matrix[index][index] += ridge
    return _solve(matrix, target)


def _fit_regimes(
    samples: list[ReplaySample],
    fallback: tuple[float, ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        _fit_direction(subset)
        if len(subset := [sample for sample in samples if _regime(sample) == index])
        >= 40
        else fallback
        for index in range(8)
    )


def _predict(
    sample: ReplaySample,
    weights: tuple[float, ...],
    regime_weights: tuple[tuple[float, ...], ...],
    previous_direction: tuple[float, float],
) -> tuple[float, float]:
    features = list(sample.direction_features)
    previous_index = 1 + FEATURE_NAMES.index("previous")
    previous_left_index = 1 + FEATURE_NAMES.index("previous_left")
    features[previous_index] = previous_direction
    features[previous_left_index] = (
        -previous_direction[1],
        previous_direction[0],
    )
    selected_weights = regime_weights[_regime(sample)] or weights
    return _unit(
        (
            sum(weight * vector[0] for weight, vector in zip(selected_weights, features)),
            sum(weight * vector[1] for weight, vector in zip(selected_weights, features)),
        )
    )


def _fit_residual_model(
    samples: list[ReplaySample],
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    weights = _fit_direction(samples)
    regime_weights = _fit_regimes(samples, weights)
    previous_by_trace: dict[tuple[int, int], tuple[float, float]] = defaultdict(
        lambda: (0.0, 0.0)
    )
    autonomous: list[ReplaySample] = []
    previous_index = 1 + FEATURE_NAMES.index("previous")
    previous_left_index = 1 + FEATURE_NAMES.index("previous_left")
    for sample in sorted(
        samples,
        key=lambda item: (item.match_id, item.player_id, item.round_number),
    ):
        key = (sample.match_id, sample.player_id)
        previous = previous_by_trace[key]
        features = list(sample.direction_features)
        features[previous_index] = previous
        features[previous_left_index] = (-previous[1], previous[0])
        autonomous_sample = replace(sample, direction_features=tuple(features))
        autonomous.append(autonomous_sample)
        previous_by_trace[key] = _predict(
            autonomous_sample,
            weights,
            regime_weights,
            previous,
        )
    weights = _fit_direction(autonomous)
    return weights, _fit_regimes(autonomous, weights)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return (
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def _evaluate_directions(
    samples: list[ReplaySample],
    weights: tuple[float, ...] | None = None,
    regime_weights: tuple[tuple[float, ...], ...] = (),
) -> dict[str, float | int]:
    angles: list[float] = []
    previous_by_trace: dict[tuple[int, int], tuple[float, float]] = defaultdict(
        lambda: (0.0, 0.0)
    )
    for sample in sorted(
        samples,
        key=lambda item: (item.match_id, item.player_id, item.round_number),
    ):
        key = (sample.match_id, sample.player_id)
        prediction = (
            sample.direction_features[0]
            if weights is None
            else _predict(
                sample,
                weights,
                regime_weights,
                previous_by_trace[key],
            )
        )
        previous_by_trace[key] = prediction
        dot = max(
            -1.0,
            min(
                1.0,
                prediction[0] * sample.target_direction[0]
                + prediction[1] * sample.target_direction[1],
            ),
        )
        angles.append(math.degrees(math.acos(dot)))
    return {
        "samples": len(angles),
        "direction_median_error_degrees": statistics.median(angles),
        "direction_p75_error_degrees": _percentile(angles, 0.75),
        "direction_within_30_rate": (
            sum(angle <= 30.0 for angle in angles) / max(len(angles), 1)
        ),
        "direction_over_90_rate": (
            sum(angle > 90.0 for angle in angles) / max(len(angles), 1)
        ),
    }


def distill(
    replay_paths: list[Path],
    *,
    team_id: int,
    train_matches: int,
    every_n: int,
) -> tuple[dict[str, object], dict[str, object]]:
    samples_by_match: dict[int, list[ReplaySample]] = {}
    for index, replay_path in enumerate(replay_paths, start=1):
        samples = extract_teacher_samples(
            replay_path,
            team_id=team_id,
            every_n=every_n,
        )
        if not samples:
            continue
        samples_by_match[samples[0].match_id] = samples
        print(
            f"[{index}/{len(replay_paths)}] {replay_path.name}: "
            f"{len(samples)} samples",
            flush=True,
        )

    match_ids = sorted(samples_by_match)
    if len(match_ids) < 2:
        raise ValueError("at least two replays containing the target team are required")
    split_index = min(max(1, train_matches), len(match_ids) - 1)
    training_ids = match_ids[:split_index]
    validation_ids = match_ids[split_index:]
    training = [
        sample
        for match_id in training_ids
        for sample in samples_by_match[match_id]
    ]
    validation = [
        sample
        for match_id in validation_ids
        for sample in samples_by_match[match_id]
    ]

    validation_weights, validation_regime_weights = _fit_residual_model(training)
    all_samples = [
        sample
        for match_id in match_ids
        for sample in samples_by_match[match_id]
    ]
    final_weights, final_regime_weights = _fit_residual_model(all_samples)
    report = {
        "teacher": "replay_dominance",
        "team_id": team_id,
        "every_n": every_n,
        "training_matches": training_ids,
        "validation_matches": validation_ids,
        "training_sample_count": len(training),
        "validation_sample_count": len(validation),
        "semantic_training_metrics": _evaluate_directions(training),
        "semantic_validation_metrics": _evaluate_directions(validation),
        "distilled_training_metrics": _evaluate_directions(
            training,
            validation_weights,
            validation_regime_weights,
        ),
        "distilled_validation_metrics": _evaluate_directions(
            validation,
            validation_weights,
            validation_regime_weights,
        ),
    }
    profile = {
        "teacher": "replay_dominance",
        "team_id": team_id,
        "feature_names": RESIDUAL_FEATURE_NAMES,
        "direction_weights": final_weights,
        "regime_direction_weights": final_regime_weights,
        "source_matches": match_ids,
    }
    return profile, report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_dir", type=Path)
    parser.add_argument("--team-id", type=int, default=73)
    parser.add_argument("--train-matches", type=int, default=20)
    parser.add_argument("--every-n", type=int, default=1)
    parser.add_argument(
        "--profile-out",
        type=Path,
        default=ROOT / ".agario/distillation/replay-dominance-profile.json",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=ROOT / ".agario/distillation/replay-dominance-report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    replay_paths = sorted(args.replay_dir.glob("match-*-replay.json"))
    if not replay_paths:
        raise SystemExit(f"no replay JSON files found in {args.replay_dir}")
    if args.every_n <= 0:
        raise SystemExit("--every-n must be positive")

    profile, report = distill(
        replay_paths,
        team_id=args.team_id,
        train_matches=args.train_matches,
        every_n=args.every_n,
    )
    args.profile_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.profile_out.write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics = report["distilled_validation_metrics"]
    print(
        "holdout: "
        f"median={metrics['direction_median_error_degrees']:.2f}deg "
        f"within30={metrics['direction_within_30_rate']:.2%} "
        f"over90={metrics['direction_over_90_rate']:.2%}"
    )
    print(f"wrote profile to {args.profile_out}")
    print(f"wrote report to {args.report_out}")


if __name__ == "__main__":
    main()
