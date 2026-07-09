from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from math import log1p
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Row:
    key: tuple[int, int, str]
    round_number: int
    max_rounds: int
    features: list[float]
    feature_names: list[str]
    total_mass: float
    rank_score: float


def rank_score(rankings: list[int], player_id: int) -> float:
    if not rankings:
        return 0.0
    try:
        idx = rankings.index(player_id)
    except ValueError:
        return 0.0
    if len(rankings) <= 1:
        return 1.0
    return 1.0 - idx / (len(rankings) - 1)


def read_rows(paths: Iterable[Path]) -> list[Row]:
    rows: list[Row] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                decision = payload.get("decision") or {}
                features = payload.get("features") or decision.get("features")
                feature_names = payload.get("feature_names") or decision.get("feature_names")
                if features is None:
                    continue
                player_id = int(payload.get("player_id", -1))
                key = (int(payload.get("pid", 0)), player_id, str(payload.get("config_name", payload.get("policy", ""))))
                rankings = [int(x) for x in payload.get("rankings", [])]
                rows.append(
                    Row(
                        key=key,
                        round_number=int(payload.get("round", 0)),
                        max_rounds=max(1, int(payload.get("max_rounds", 1400))),
                        features=[float(x) for x in features],
                        feature_names=[str(x) for x in feature_names],
                        total_mass=float(payload.get("total_mass", payload.get("mass", 0.0))),
                        rank_score=rank_score(rankings, player_id),
                    )
                )
    return rows


def build_discounted_returns(rows: list[Row], gamma: float) -> tuple[list[list[float]], list[float], list[str]]:
    by_episode: dict[tuple[int, int, str], list[Row]] = defaultdict(list)
    for row in rows:
        by_episode[row.key].append(row)

    x_items: list[list[float]] = []
    y_items: list[float] = []
    feature_names: list[str] | None = None

    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda row: row.round_number)
        if len(episode_rows) < 2:
            continue
        rewards: list[float] = []
        for current, nxt in zip(episode_rows, episode_rows[1:], strict=False):
            mass_delta = log1p(max(0.0, nxt.total_mass)) - log1p(max(0.0, current.total_mass))
            rank_delta = nxt.rank_score - current.rank_score
            rewards.append(2.5 * mass_delta + 1.2 * rank_delta - 0.001)
        last = episode_rows[-1]
        terminal = 2.0 * last.rank_score + 0.15 * log1p(max(0.0, last.total_mass))
        if last.round_number < 0.92 * last.max_rounds:
            terminal -= 1.5
        rewards.append(terminal)

        returns = [0.0] * len(episode_rows)
        running = 0.0
        for i in reversed(range(len(episode_rows))):
            running = rewards[i] + gamma * running
            returns[i] = running

        for row, ret in zip(episode_rows, returns, strict=True):
            feature_names = row.feature_names
            x_items.append(row.features)
            y_items.append(ret)

    if not x_items or feature_names is None:
        raise SystemExit("No usable training rows. Run rl/collect_rollouts.py first.")
    return x_items, y_items, feature_names


def fit_ridge(x: list[list[float]], y: list[float], ridge: float) -> tuple[list[float], float, dict[str, float]]:
    rows = len(x)
    cols = len(x[0])
    x_mean = [sum(row[col] for row in x) / rows for col in range(cols)]
    x_std = []
    for col in range(cols):
        variance = sum((row[col] - x_mean[col]) ** 2 for row in x) / rows
        x_std.append(max(variance ** 0.5, 1.0e-8))

    y_mean = sum(y) / len(y)
    y_std = (sum((item - y_mean) ** 2 for item in y) / len(y)) ** 0.5
    xz = [
        [(row[col] - x_mean[col]) / x_std[col] for col in range(cols)]
        for row in x
    ]
    yz = [item - y_mean for item in y]

    xtx = [
        [
            sum(row[i] * row[j] for row in xz) + (ridge if i == j else 0.0)
            for j in range(cols)
        ]
        for i in range(cols)
    ]
    xty = [sum(row[i] * target for row, target in zip(xz, yz, strict=True)) for i in range(cols)]
    weights_z = solve_linear_system(xtx, xty)
    weights = [weights_z[col] / x_std[col] for col in range(cols)]
    bias = y_mean - sum(x_mean[col] * weights[col] for col in range(cols))
    stats = {"rows": float(len(y)), "target_mean": y_mean, "target_std": y_std}
    return weights, bias, stats


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    aug = [row[:] + [value] for row, value in zip(matrix, vector, strict=True)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1.0e-12:
            raise SystemExit("Ridge system is singular; increase --ridge.")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_value = aug[col][col]
        for item in range(col, n + 1):
            aug[col][item] /= pivot_value
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            for item in range(col, n + 1):
                aug[row][item] -= factor * aug[col][item]
    return [aug[row][n] for row in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a linear value model from beam-bot rollouts.")
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("models/value_linear.json"))
    parser.add_argument("--gamma", type=float, default=0.985)
    parser.add_argument("--ridge", type=float, default=10.0)
    args = parser.parse_args()

    rows = read_rows(args.logs)
    x, y, names = build_discounted_returns(rows, args.gamma)
    weights, bias, stats = fit_ridge(x, y, args.ridge)

    payload = {
        "model_type": "linear_value_v1",
        "feature_names": names,
        "weights": [float(w) for w in weights],
        "bias": float(bias),
        "stats": stats,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(args.out), **stats}, indent=2))


if __name__ == "__main__":
    main()
