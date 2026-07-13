from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


CACHE_SPECS = {
    "utility": ("utility_hit", "utility_miss", "utility"),
    "virus_retention": (
        "virus_retention_hit",
        "virus_retention_miss",
        "virus_retention",
    ),
    "risk_envelope": (
        "risk_envelope_hit",
        "risk_envelope_miss",
        "risk_envelope",
    ),
    "node": ("node_hit", "node_miss", "cache_node"),
    "prey": ("prey_hit", "prey_miss", "cache_prey"),
    "split_prey": (
        "split_prey_hit",
        "split_prey_miss",
        "cache_split_prey",
    ),
    "virus": ("virus_hit", "virus_miss", "cache_virus"),
    "gradient": ("gradient_hit", "gradient_miss", "cache_gradient"),
    "hazard": ("hazard_hit", "hazard_miss", "cache_hazard"),
    "proxy": ("proxy_hit", "proxy_miss", "cache_proxy"),
    "virus_layout": (
        "virus_layout_hit",
        "virus_layout_miss",
        "cache_virus_layout",
    ),
}
SUPPORTED_SCHEMA_VERSION = 1
PROFILE_SECTIONS = (
    "phase_ms",
    "operation_inclusive_ms",
    "calls",
    "counts",
    "value_sums",
)
REQUIRED_PHASE_KEYS = frozenset(
    {
        "setup_and_search_control",
        "candidate_shared_analysis",
        "candidate_coarse_rank",
        "candidate_geometric_refine",
        "candidate_overhead",
        "search_parent_utility",
        "search_split_movement_efficiency",
        "search_physics",
        "search_child_utility",
        "search_shaping",
        "search_step_overhead",
        "audit_diagnostic",
        "fallback",
    }
)


@dataclass(frozen=True)
class ProfileSample:
    source: Path
    line_number: int
    row_round: object
    profile: dict[str, Any]

    @property
    def location(self) -> str:
        return f"{self.source}:{self.line_number}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate structured ReplayDominance decision profiles."
    )
    parser.add_argument("metrics", type=Path, nargs="+")
    return parser.parse_args()


def read_metrics(
    paths: list[Path],
) -> tuple[list[dict[str, Any]], list[ProfileSample], list[str]]:
    rows: list[dict[str, Any]] = []
    samples: list[ProfileSample] = []
    violations: list[str] = []
    for path in paths:
        path_rows = 0
        path_profiles = 0
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON: {error.msg}"
                    ) from error
                if not isinstance(row, dict):
                    violations.append(
                        f"{path}:{line_number}: metric row must be an object"
                    )
                    continue
                path_rows += 1
                rows.append(row)
                diagnostics = row.get("decision_diagnostics", {})
                if not isinstance(diagnostics, dict):
                    violations.append(
                        f"{path}:{line_number}: decision_diagnostics must be an object"
                    )
                    continue
                if "replay_profile" not in diagnostics:
                    continue
                profile = diagnostics["replay_profile"]
                if not isinstance(profile, dict):
                    violations.append(
                        f"{path}:{line_number}: replay_profile must be an object"
                    )
                    continue
                path_profiles += 1
                samples.append(
                    ProfileSample(path, line_number, row.get("round"), profile)
                )
        if path_rows == 0:
            violations.append(f"{path}: no metric rows")
        if path_profiles == 0:
            violations.append(f"{path}: no replay_profile samples")
    return rows, samples, violations


def _numeric_section(
    sample: ProfileSample,
    name: str,
    *,
    integer: bool,
    nonnegative: bool,
) -> tuple[dict[str, int | float], list[str]]:
    section = sample.profile.get(name)
    if not isinstance(section, dict):
        return {}, [f"{sample.location}: {name} must be an object"]
    values: dict[str, int | float] = {}
    violations: list[str] = []
    for key, value in section.items():
        valid_type = (
            isinstance(value, int) and not isinstance(value, bool)
            if integer
            else isinstance(value, (int, float)) and not isinstance(value, bool)
        )
        if not isinstance(key, str) or not valid_type:
            violations.append(
                f"{sample.location}: {name}.{key} must be "
                f"{'an integer' if integer else 'numeric'}"
            )
            continue
        if not math.isfinite(value):
            violations.append(f"{sample.location}: {name}.{key} must be finite")
            continue
        if nonnegative and value < 0:
            violations.append(
                f"{sample.location}: {name}.{key} must be nonnegative"
            )
            continue
        values[key] = value
    return values, violations


def validate_profile(sample: ProfileSample) -> list[str]:
    profile = sample.profile
    violations: list[str] = []
    if profile.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        violations.append(
            f"{sample.location}: unsupported schema_version="
            f"{profile.get('schema_version')!r}; expected {SUPPORTED_SCHEMA_VERSION}"
        )
    profile_round = profile.get("round")
    if profile_round != sample.row_round:
        violations.append(
            f"{sample.location}: profile round={profile_round!r} "
            f"!= metric round={sample.row_round!r}"
        )
    sample_every_n = profile.get("sample_every_n")
    if (
        not isinstance(sample_every_n, int)
        or isinstance(sample_every_n, bool)
        or sample_every_n <= 0
    ):
        violations.append(
            f"{sample.location}: sample_every_n must be a positive integer"
        )
    if (
        not isinstance(profile_round, int)
        or isinstance(profile_round, bool)
        or profile_round < 0
    ):
        violations.append(f"{sample.location}: round must be a nonnegative integer")
    elif isinstance(sample_every_n, int) and sample_every_n > 0:
        if profile_round % sample_every_n:
            violations.append(
                f"{sample.location}: round={profile_round} is not sampled by "
                f"sample_every_n={sample_every_n}"
            )

    sections: dict[str, dict[str, int | float]] = {}
    for name in PROFILE_SECTIONS:
        section, section_violations = _numeric_section(
            sample,
            name,
            integer=name in {"calls", "counts"},
            nonnegative=name != "value_sums",
        )
        sections[name] = section
        violations.extend(section_violations)

    phase_ms = sections["phase_ms"]
    inclusive_ms = sections["operation_inclusive_ms"]
    calls = sections["calls"]
    counts = sections["counts"]
    missing_phases = sorted(REQUIRED_PHASE_KEYS - phase_ms.keys())
    if missing_phases:
        violations.append(
            f"{sample.location}: phase_ms missing required keys="
            f"{','.join(missing_phases)}"
        )
    parent_calls = {name: calls.get(name, 0) for name in ("choose", "fallback")}
    active_parents = [name for name, count in parent_calls.items() if count]
    if sum(parent_calls.values()) != 1 or len(active_parents) != 1:
        violations.append(
            f"{sample.location}: exactly one choose/fallback call is required; "
            f"got {parent_calls}"
        )
    else:
        parent = active_parents[0]
        if parent not in inclusive_ms:
            violations.append(
                f"{sample.location}: operation_inclusive_ms missing parent {parent}"
            )
    expected_parent = sum(inclusive_ms.get(name, 0.0) for name in ("choose", "fallback"))
    phase_total = sum(phase_ms.values())
    if abs(phase_total - expected_parent) > 0.01:
        violations.append(
            f"{sample.location}: phase total {phase_total:.6f} "
            f"!= parent {expected_parent:.6f}"
        )

    for name, (hit_key, miss_key, call_key) in CACHE_SPECS.items():
        if call_key is None:
            continue
        lookups = counts.get(hit_key, 0) + counts.get(miss_key, 0)
        if calls.get(call_key, 0) != lookups:
            violations.append(
                f"{sample.location}: {name} calls={calls.get(call_key, 0)} "
                f"!= hits+misses={lookups}"
            )

    for prefix in ("base_candidate", "replay_candidate", "fallback_candidate"):
        raw = counts.get(f"{prefix}_raw", 0)
        accounted = sum(
            counts.get(f"{prefix}_{suffix}", 0)
            for suffix in ("unique", "zero_drops", "duplicate_drops")
        )
        if raw != accounted:
            violations.append(
                f"{sample.location}: {prefix} raw={raw} "
                f"!= unique+zero+duplicates={accounted}"
            )

    for key, nonzero in counts.items():
        if not key.endswith("_nonzero"):
            continue
        samples = counts.get(f"{key[:-8]}_samples", 0)
        if nonzero > samples:
            violations.append(
                f"{sample.location}: {key}={nonzero} exceeds samples={samples}"
            )
    return violations


def merge_numeric(
    target: Counter[str],
    source: dict[str, int | float],
) -> None:
    target.update(source)


def ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_runtime(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize hard compute headroom and observable search behaviour."""

    elapsed = [
        float(row["decision_elapsed_ms"])
        for row in rows
        if isinstance(row.get("decision_elapsed_ms"), (int, float))
    ]
    diagnostics = [
        value
        for row in rows
        if isinstance((value := row.get("decision_diagnostics")), dict)
    ]
    hard_remaining = [
        float(value["competition_compute_remaining_ms"])
        for value in diagnostics
        if isinstance(
            value.get("competition_compute_remaining_ms"),
            (int, float),
        )
    ]
    hard_budgets = [
        float(value["competition_compute_budget_ms"])
        for value in diagnostics
        if isinstance(value.get("competition_compute_budget_ms"), (int, float))
    ]
    candidate_counts = [
        int(value.get("fallback_candidates") or value.get("root_actions_generated") or 0)
        for value in diagnostics
    ]
    refined_counts = [
        int(value.get("proxy_candidates_refined") or 0)
        for value in diagnostics
    ]
    transitions = [
        int(value.get("transitions_evaluated") or 0)
        for value in diagnostics
    ]
    reasons = Counter(str(row.get("decision_reason") or "unknown") for row in rows)
    split_turns = sum(bool(row.get("decision_split")) for row in rows)
    direction_dots: list[float] = []
    for previous, current in zip(rows, rows[1:]):
        previous_direction = (
            previous.get("decision_direction_x"),
            previous.get("decision_direction_y"),
        )
        current_direction = (
            current.get("decision_direction_x"),
            current.get("decision_direction_y"),
        )
        if None in (*previous_direction, *current_direction):
            continue
        direction_dots.append(
            _clamp(
                float(previous_direction[0]) * float(current_direction[0])
                + float(previous_direction[1]) * float(current_direction[1]),
                -1.0,
                1.0,
            )
        )

    observed_round_gaps = 0
    respawn_like_events = 0
    previous_row: dict[str, Any] | None = None
    for row in rows:
        if previous_row is None:
            previous_row = row
            continue
        previous_round = previous_row.get("round")
        current_round = row.get("round")
        gap = 0
        if isinstance(previous_round, int) and isinstance(current_round, int):
            gap = max(0, current_round - previous_round - 1)
            observed_round_gaps += gap
        previous_mass = previous_row.get("my_mass")
        current_mass = row.get("my_mass")
        collapsed = (
            isinstance(previous_mass, (int, float))
            and isinstance(current_mass, (int, float))
            and previous_mass > 1.1
            and current_mass <= 1.1
            and current_mass < previous_mass * 0.55
        )
        if collapsed or (gap and isinstance(current_mass, (int, float)) and current_mass <= 1.1):
            respawn_like_events += 1
        previous_row = row

    return {
        "decision_count": len(elapsed),
        "decision_total_ms": sum(elapsed),
        "decision_avg_ms": ratio(sum(elapsed), len(elapsed)),
        "decision_p50_ms": percentile(elapsed, 0.50),
        "decision_p95_ms": percentile(elapsed, 0.95),
        "decision_p99_ms": percentile(elapsed, 0.99),
        "decision_max_ms": max(elapsed, default=0.0),
        "competition_budget_ms": max(hard_budgets, default=0.0),
        "competition_remaining_min_ms": min(hard_remaining, default=0.0),
        "competition_remaining_last_ms": hard_remaining[-1] if hard_remaining else 0.0,
        "competition_exhausted_samples": sum(value <= 0.0 for value in hard_remaining),
        "proxy_only_turns": sum(bool(value.get("proxy_only")) for value in diagnostics),
        "candidate_total": sum(candidate_counts),
        "candidate_avg": ratio(sum(candidate_counts), len(candidate_counts)),
        "refined_total": sum(refined_counts),
        "refined_avg": ratio(sum(refined_counts), len(refined_counts)),
        "exact_transitions": sum(transitions),
        "split_turns": split_turns,
        "prey_turns": sum(
            count for reason, count in reasons.items() if "prey" in reason
        ),
        "virus_turns": sum(
            count for reason, count in reasons.items() if "virus" in reason
        ),
        "escape_turns": sum(
            count for reason, count in reasons.items() if "escape" in reason
        ),
        "resource_turns": sum(
            ("food" in reason or "farm" in reason) * count
            for reason, count in reasons.items()
        ),
        "direction_reversals": sum(dot < -0.25 for dot in direction_dots),
        "direction_pairs": len(direction_dots),
        "observed_round_gaps": observed_round_gaps,
        "respawn_like_events": respawn_like_events,
        "reasons": reasons,
    }


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def main() -> None:
    rows, samples, violations = read_metrics(parse_args().metrics)
    for sample in samples:
        violations.extend(validate_profile(sample))
    if violations:
        raise SystemExit("profile invariant violations: " + "; ".join(violations))

    diagnostics = [row.get("decision_diagnostics", {}) for row in rows]
    profiles = [sample.profile for sample in samples]

    phase_ms: Counter[str] = Counter()
    inclusive_ms: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    value_sums: Counter[str] = Counter()
    for profile in profiles:
        merge_numeric(phase_ms, profile.get("phase_ms", {}))
        merge_numeric(
            inclusive_ms,
            profile.get("operation_inclusive_ms", {}),
        )
        merge_numeric(counts, profile.get("counts", {}))
        merge_numeric(value_sums, profile.get("value_sums", {}))

    stop_reasons = Counter(
        item.get("search_stop_reason", "fallback")
        for item in diagnostics
    )
    depths = Counter(item.get("depth", 0) for item in diagnostics)
    generated = sum(item.get("root_actions_generated", 0) for item in diagnostics)
    evaluated = sum(item.get("root_actions_evaluated", 0) for item in diagnostics)
    proxy_evaluated = sum(
        item.get("root_actions_proxy_evaluated", 0) for item in diagnostics
    )
    runtime = summarize_runtime(rows)

    print(f"turns={len(rows)} profiled_turns={len(profiles)}")
    print(
        "compute_resource="
        f"decision_total_ms:{runtime['decision_total_ms']:.3f} "
        f"avg:{runtime['decision_avg_ms']:.3f} "
        f"p50:{runtime['decision_p50_ms']:.3f} "
        f"p95:{runtime['decision_p95_ms']:.3f} "
        f"p99:{runtime['decision_p99_ms']:.3f} "
        f"max:{runtime['decision_max_ms']:.3f} "
        f"competition_budget_ms:{runtime['competition_budget_ms']:.3f} "
        f"remaining_min_ms:{runtime['competition_remaining_min_ms']:.3f} "
        f"remaining_last_ms:{runtime['competition_remaining_last_ms']:.3f} "
        f"exhausted_samples:{runtime['competition_exhausted_samples']}"
    )
    print(
        "search_work="
        f"proxy_only_turns:{runtime['proxy_only_turns']} "
        f"candidates_total:{runtime['candidate_total']} "
        f"candidates_avg:{runtime['candidate_avg']:.3f} "
        f"refined_total:{runtime['refined_total']} "
        f"refined_avg:{runtime['refined_avg']:.3f} "
        f"exact_transitions:{runtime['exact_transitions']}"
    )
    print(
        "behaviour="
        f"prey_turns:{runtime['prey_turns']} "
        f"virus_turns:{runtime['virus_turns']} "
        f"resource_turns:{runtime['resource_turns']} "
        f"escape_turns:{runtime['escape_turns']} "
        f"split_turns:{runtime['split_turns']} "
        f"direction_reversals:{runtime['direction_reversals']}/"
        f"{runtime['direction_pairs']} "
        f"observed_round_gaps:{runtime['observed_round_gaps']} "
        f"respawn_like_events:{runtime['respawn_like_events']}"
    )
    print(f"depths={dict(sorted(depths.items()))}")
    print(f"stop_reasons={dict(sorted(stop_reasons.items()))}")
    print(
        "root_candidates="
        f"generated:{generated} proxy_evaluated:{proxy_evaluated} "
        f"proxy_rate:{ratio(proxy_evaluated, generated):.6f} "
        f"exact_evaluated:{evaluated} "
        f"exact_rate:{ratio(evaluated, generated):.6f}"
    )

    print("phase_ms (exclusive, additive)")
    for name, elapsed in phase_ms.most_common():
        print(f"  {name}={elapsed:.3f}")

    print("operation_inclusive_ms (nested, do not add)")
    for name, elapsed in inclusive_ms.most_common():
        print(f"  {name}={elapsed:.3f}")

    print("caches")
    for name, (hit_key, miss_key, _call_key) in CACHE_SPECS.items():
        hits = counts[hit_key]
        misses = counts[miss_key]
        lookups = hits + misses
        print(
            f"  {name}=lookups:{lookups} hits:{hits} misses:{misses} "
            f"hit_rate:{ratio(hits, lookups):.6f}"
        )

    print("candidate_conservation")
    if counts["replay_base_raw"]:
        print(f"  replay_base_raw={counts['replay_base_raw']}")
    for prefix in ("base_candidate", "replay_candidate", "fallback_candidate"):
        raw = counts[f"{prefix}_raw"]
        unique = counts[f"{prefix}_unique"]
        zero = counts[f"{prefix}_zero_drops"]
        duplicates = counts[f"{prefix}_duplicate_drops"]
        print(
            f"  {prefix}=raw:{raw} unique:{unique} "
            f"zero:{zero} duplicates:{duplicates}"
        )

    if counts["audit_transitions"]:
        print(
            "audit="
            f"transitions:{counts['audit_transitions']} "
            f"fatal:{counts['audit_fatal_candidates']}"
        )

    print("value_sums")
    for name, value in value_sums.most_common():
        print(
            f"  {name}={value:.6f} "
            f"nonzero:{counts[f'{name}_nonzero']}/"
            f"{counts[f'{name}_samples']}"
        )

if __name__ == "__main__":
    main()
