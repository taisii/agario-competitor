from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from dataclasses import dataclass
from math import exp, log, log1p
from pathlib import Path

PARAMS = [
    "weight_food",
    "weight_food_cluster",
    "weight_prey",
    "weight_threat",
    "weight_close_threat",
    "weight_virus",
    "weight_wall",
    "weight_fragmentation",
    "weight_split_risk",
    "prey_chase_max_distance",
    "split_min_radius_ratio",
    "split_reach_bonus",
    "split_gain_threshold",
    "late_game_aggression",
]

BASE = {
    "weight_food": 3.6,
    "weight_food_cluster": 2.1,
    "weight_prey": 5.8,
    "weight_threat": 14.0,
    "weight_close_threat": 21.0,
    "weight_virus": 6.2,
    "weight_wall": 3.0,
    "weight_fragmentation": 1.3,
    "weight_split_risk": 11.0,
    "prey_chase_max_distance": 8.0,
    "split_min_radius_ratio": 1.70,
    "split_reach_bonus": 1.7,
    "split_gain_threshold": 0.8,
    "late_game_aggression": 0.35,
}

DEFAULT_OPPONENT_POOL = [
    "1:bots/entries/beam_rl_tuned.py",
    "2:bots/entries/beam_survival.py",
    "2:bots/entries/potential_hunter.py",
    "3:bots/entries/random_opponent.py",
]


@dataclass(frozen=True)
class CandidateResult:
    score: float
    config: dict[str, float]
    log_path: Path


def sample_config(rng: random.Random, mean: dict[str, float], sigma: dict[str, float]) -> dict[str, float]:
    config: dict[str, float] = {"name": "tuned"}
    for key in PARAMS:
        m = max(1e-6, mean[key])
        s = max(1e-6, sigma[key])
        value = exp(rng.gauss(log(m), s))
        if key == "late_game_aggression":
            value = min(1.5, max(0.0, value))
        if key == "split_min_radius_ratio":
            value = min(2.2, max(1.55, value))
        config[key] = float(value)
    return config


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


def score_tuned_bot(log_path: Path) -> float:
    last_by_pid: dict[tuple[int, int], dict] = {}
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("config_name") != "tuned":
                continue
            key = (int(row.get("pid", 0)), int(row.get("player_id", -1)))
            last_by_pid[key] = row
    if not last_by_pid:
        return -1e9
    scores = []
    for row in last_by_pid.values():
        rs = rank_score([int(x) for x in row.get("rankings", [])], int(row.get("player_id", -1)))
        mass = float(row.get("total_mass", 0.0))
        round_number = int(row.get("round", 0))
        max_rounds = max(1, int(row.get("max_rounds", 1400)))
        early_death_penalty = 2.0 if round_number < 0.90 * max_rounds else 0.0
        scores.append(8.0 * rs + 0.4 * log1p(max(0.0, mass)) - early_death_penalty)
    return sum(scores) / len(scores)


def run_candidate(
    project_root: Path,
    workspace_root: Path,
    generation: int,
    index: int,
    config: dict[str, float],
    specs: list[str],
) -> CandidateResult:
    run_dir = workspace_root / f"g{generation:03d}_c{index:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    log_path = run_dir / "rollout.jsonl"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    env = os.environ.copy()
    env["BOT_CONFIG_JSON"] = str(config_path.resolve())
    env["BOT_LOG_PATH"] = str(log_path.resolve())
    command = ["uv", "run", "simulation", *specs, "--headless", "--workspace", str(run_dir / "match")]
    print("RUN", " ".join(command))
    subprocess.run(command, cwd=project_root, env=env, check=True)
    return CandidateResult(score=score_tuned_bot(log_path), config=config, log_path=log_path)


def update_distribution(elites: list[CandidateResult]) -> tuple[dict[str, float], dict[str, float]]:
    mean: dict[str, float] = {}
    sigma: dict[str, float] = {}
    for key in PARAMS:
        values = [max(1e-6, result.config[key]) for result in elites]
        logs = [log(v) for v in values]
        mu = sum(logs) / len(logs)
        var = sum((x - mu) ** 2 for x in logs) / max(1, len(logs) - 1)
        mean[key] = exp(mu)
        sigma[key] = max(0.06, var ** 0.5)
    return mean, sigma


def main() -> None:
    parser = argparse.ArgumentParser(description="CEM black-box RL tuner for beam-search StrategyConfig.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--workspace-root", type=Path, default=Path(".agario/cem"))
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--elite", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=Path, default=Path("models/cem_best_config.json"))
    parser.add_argument("--spec", action="append", default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    project_root = args.project_root.resolve()
    workspace_root = (project_root / args.workspace_root).resolve()
    specs = args.spec if args.spec else DEFAULT_OPPONENT_POOL
    mean = dict(BASE)
    sigma = {key: 0.35 for key in PARAMS}
    best: CandidateResult | None = None

    for generation in range(args.generations):
        results = []
        for index in range(args.population):
            config = sample_config(rng, mean, sigma)
            result = run_candidate(project_root, workspace_root, generation, index, config, specs)
            print(json.dumps({"generation": generation, "index": index, "score": result.score}, sort_keys=True))
            results.append(result)
            if best is None or result.score > best.score:
                best = result
        results.sort(key=lambda r: r.score, reverse=True)
        elites = results[: max(1, args.elite)]
        mean, sigma = update_distribution(elites)
        print("GENERATION_SUMMARY", json.dumps({"generation": generation, "best_score": results[0].score, "mean": mean}, sort_keys=True))

    if best is None:
        raise SystemExit("no candidate evaluated")
    out = (project_root / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best.config, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"best_score": best.score, "out": str(out), "source_log": str(best.log_path)}, indent=2))


if __name__ == "__main__":
    main()
