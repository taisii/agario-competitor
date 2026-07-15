from __future__ import annotations

"""Compare semantic_potential with replay_dominance on identical observations.

The comparison is deliberately offline. Production semantic_potential never
constructs replay_dominance or pays for its proxy. Besides action agreement, the
report scores both actions with replay_dominance's shared cheap proxy so a large
angular difference is not mistaken for evidence that imitation would be better.
"""

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import sys
from time import perf_counter
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
BOTS = ROOT / "bots"
sys.path.insert(0, str(BOTS))
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_expected_responses import (  # noqa: E402
    ReplayFrame,
    _circle_visible,
    _mass_center,
    _point_visible,
    _view_center,
    _vision_size,
    extract_frames,
)
from lib.interface.events.moves.move_player import MovePlayer  # noqa: E402
from lib.models.blob_model import BlobModel, VisibleBlobModel  # noqa: E402
from lib.models.food_model import FoodModel  # noqa: E402
from lib.models.penguin_model import DirectionModel  # noqa: E402
from lib.models.virus_model import VirusModel  # noqa: E402
from strategies.base import StrategyContext, StrategyDecision  # noqa: E402
from strategies.features import can_consume_virus, can_eat_player_blob  # noqa: E402
from strategies.receding_horizon import Action, ReplayDominanceStrategy  # noqa: E402
from strategies.semantic_potential import SemanticPotentialStrategy  # noqa: E402


ADOPTION_ANGLE_DEGREES = 30.0
ADOPTION_PROXY_REGRET = 0.05


@dataclass(frozen=True, slots=True)
class ComparisonSample:
    round_number: int
    player_mass: float
    blob_count: int
    angle_degrees: float
    split_agreement: bool
    semantic_direction: tuple[float, float]
    replay_direction: tuple[float, float]
    semantic_split: bool
    replay_split: bool
    semantic_reason: str
    replay_reason: str
    semantic_elapsed_ms: float
    replay_elapsed_ms: float
    semantic_proxy_value: float
    replay_proxy_value: float
    proxy_regret: float
    strategic_focus: str
    predator_visible: bool
    wall_clearance: float
    current_safety_margin: float | None
    selected_safety_margin: float | None

    @property
    def adoption_candidate(self) -> bool:
        return self.strategic_focus != "background" and (
            (
                self.angle_degrees > ADOPTION_ANGLE_DEGREES
                and self.proxy_regret > ADOPTION_PROXY_REGRET
            )
            or (not self.split_agreement and self.proxy_regret > ADOPTION_PROXY_REGRET)
        )


def _context(
    frame: ReplayFrame,
    *,
    player_id: int,
    max_rounds: int,
) -> StrategyContext | None:
    player = frame.players.get(player_id)
    if player is None or not player.alive or not player.blobs:
        return None

    own_blobs = {
        blob_id: BlobModel(
            blob_id=blob_id,
            pos=blob.pos,
            radius=blob.radius,
            merge_cooldown=blob.merge_cooldown,
        )
        for blob_id, blob in enumerate(player.blobs)
    }
    center = _mass_center(player.blobs)
    vision_size = _vision_size(player, frame.base_vision_size)
    view_center = _view_center(player, frame.arena_size, vision_size)

    visible_food = [
        FoodModel(food_id=food.source_id, pos=food.pos)
        for food in frame.foods
        if _point_visible(view_center, vision_size, food)
    ]
    visible_viruses = [
        VirusModel(
            virus_id=virus.source_id,
            pos=virus.pos,
            radius=virus.radius,
        )
        for virus in frame.viruses
        if _circle_visible(view_center, vision_size, virus)
    ]
    visible_blobs = [
        VisibleBlobModel(
            player_id=other.player_id,
            team_id=other.team_id,
            blob_id=blob_id,
            pos=blob.pos,
            radius=blob.radius,
            merge_cooldown=blob.merge_cooldown,
        )
        for other in frame.players.values()
        if other.player_id != player_id and other.alive
        for blob_id, blob in enumerate(other.blobs)
        if _circle_visible(view_center, vision_size, blob)
    ]
    rankings = [
        other.player_id
        for other in sorted(
            (item for item in frame.players.values() if item.alive),
            key=lambda item: (
                -sum(blob.mass for blob in item.blobs),
                item.player_id,
            ),
        )
    ]
    mass = sum(blob.radius * blob.radius for blob in own_blobs.values())
    state = SimpleNamespace(
        me=SimpleNamespace(
            player_id=player_id,
            blobs=own_blobs,
            x=center[0],
            y=center[1],
            radius=math.sqrt(mass),
            alive=True,
        ),
        map=SimpleNamespace(size=frame.arena_size),
        max_rounds=max_rounds,
        round=frame.round_number,
        rankings=rankings,
        view_center=view_center,
        vision_size=vision_size,
        visible_food=visible_food,
        visible_viruses=visible_viruses,
        visible_blobs=visible_blobs,
    )
    update = {
        index: MovePlayer(
            player_id=other_id,
            direction=DirectionModel(x=command.direction[0], y=command.direction[1]),
            split=command.split,
        )
        for index, (other_id, command) in enumerate(frame.previous_commands.items())
    }
    return StrategyContext(
        game=SimpleNamespace(state=state),
        query=SimpleNamespace(update=update),
    )


def _angle_degrees(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    first_length = math.hypot(*first)
    second_length = math.hypot(*second)
    if first_length <= 1.0e-9 or second_length <= 1.0e-9:
        return 0.0 if first_length <= 1.0e-9 and second_length <= 1.0e-9 else 180.0
    cosine = (first[0] * second[0] + first[1] * second[1]) / (
        first_length * second_length
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _proxy_values(
    scorer: ReplayDominanceStrategy,
    context: StrategyContext,
    semantic: StrategyDecision,
    replay: StrategyDecision,
) -> tuple[float, float]:
    scorer._begin_replay_turn(context.game.state)
    turn = scorer._prepare_turn(context)
    if turn is None:
        return (0.0, 0.0)
    analysis = scorer._proxy_analysis(
        node=turn.node,
        foods=turn.foods,
        viruses=turn.viruses,
        arena_size=turn.arena_size,
    )

    def value(decision: StrategyDecision) -> float:
        return scorer._approximate_action_value(
            node=turn.node,
            action=Action(
                direction=decision.direction,
                split=decision.split,
                reason=decision.reason,
            ),
            foods=turn.foods,
            viruses=turn.viruses,
            arena_size=turn.arena_size,
            proxy_analysis=analysis,
        )

    return value(semantic), value(replay)


def _strategic_focus(context: StrategyContext) -> str:
    state = context.game.state
    own = tuple(state.me.blobs.values())
    enemies = tuple(state.visible_blobs)
    has_threat = any(
        can_eat_player_blob(enemy.radius, blob.radius)
        for blob in own
        for enemy in enemies
    )
    has_prey = any(
        can_eat_player_blob(blob.radius, enemy.radius)
        for blob in own
        for enemy in enemies
    )
    has_virus = any(
        can_consume_virus(blob.radius, virus.radius)
        for blob in own
        for virus in state.visible_viruses
    )
    if has_threat:
        return "threat"
    if has_prey and has_virus:
        return "prey_virus"
    if has_prey:
        return "prey"
    if has_virus:
        return "virus"
    return "background"


def compare(
    replay_path: Path,
    *,
    player_id: int,
    every_n: int = 1,
    max_samples: int | None = None,
) -> tuple[list[ComparisonSample], dict[str, object]]:
    started = json.loads(replay_path.read_text(encoding="utf-8"))[0]
    max_rounds = int(started.get("max_rounds", 1400))
    semantic = SemanticPotentialStrategy()
    replay = ReplayDominanceStrategy()
    scorer = ReplayDominanceStrategy()
    samples: list[ComparisonSample] = []

    for frame in extract_frames(replay_path):
        context = _context(frame, player_id=player_id, max_rounds=max_rounds)
        if context is None:
            continue

        started_at = perf_counter()
        semantic_decision = semantic.choose(context)
        semantic_elapsed_ms = (perf_counter() - started_at) * 1000.0
        started_at = perf_counter()
        replay_decision = replay.choose(context)
        replay_elapsed_ms = (perf_counter() - started_at) * 1000.0

        if frame.round_number % every_n:
            continue
        semantic_proxy, replay_proxy = _proxy_values(
            scorer,
            context,
            semantic_decision,
            replay_decision,
        )
        own = tuple(context.game.state.me.blobs.values())
        wall_clearance = min(
            min(
                blob.pos[0] - blob.radius,
                blob.pos[1] - blob.radius,
                frame.arena_size - blob.radius - blob.pos[0],
                frame.arena_size - blob.radius - blob.pos[1],
            )
            for blob in own
        )
        samples.append(
            ComparisonSample(
                round_number=frame.round_number,
                player_mass=sum(blob.radius * blob.radius for blob in own),
                blob_count=len(own),
                angle_degrees=_angle_degrees(
                    semantic_decision.direction,
                    replay_decision.direction,
                ),
                split_agreement=semantic_decision.split == replay_decision.split,
                semantic_direction=semantic_decision.direction,
                replay_direction=replay_decision.direction,
                semantic_split=semantic_decision.split,
                replay_split=replay_decision.split,
                semantic_reason=semantic_decision.reason,
                replay_reason=replay_decision.reason,
                semantic_elapsed_ms=semantic_elapsed_ms,
                replay_elapsed_ms=replay_elapsed_ms,
                semantic_proxy_value=semantic_proxy,
                replay_proxy_value=replay_proxy,
                proxy_regret=replay_proxy - semantic_proxy,
                strategic_focus=_strategic_focus(context),
                predator_visible=bool(
                    semantic_decision.diagnostics.get("current_safety_margin")
                    is not None
                ),
                wall_clearance=wall_clearance,
                current_safety_margin=semantic_decision.diagnostics.get(
                    "current_safety_margin"
                ),
                selected_safety_margin=semantic_decision.diagnostics.get(
                    "selected_safety_margin"
                ),
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples, summarise(samples)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def summarise(samples: list[ComparisonSample]) -> dict[str, object]:
    if not samples:
        return {"sample_count": 0, "recommended_differences": []}

    grouped: dict[tuple[str, str, str], list[ComparisonSample]] = defaultdict(list)
    for sample in samples:
        grouped[
            (sample.strategic_focus, sample.semantic_reason, sample.replay_reason)
        ].append(sample)
    recommendations = []
    for (focus, semantic_reason, replay_reason), rows in grouped.items():
        adoption_rows = [row for row in rows if row.adoption_candidate]
        if not adoption_rows:
            continue
        recommendations.append(
            {
                "strategic_focus": focus,
                "semantic_reason": semantic_reason,
                "replay_reason": replay_reason,
                "samples": len(rows),
                "adoption_candidates": len(adoption_rows),
                "mean_proxy_regret": statistics.fmean(
                    row.proxy_regret for row in adoption_rows
                ),
                "mean_angle_degrees": statistics.fmean(
                    row.angle_degrees for row in adoption_rows
                ),
            }
        )
    recommendations.sort(
        key=lambda row: (
            -row["adoption_candidates"],
            -row["mean_proxy_regret"],
        )
    )

    angles = [sample.angle_degrees for sample in samples]
    regrets = [sample.proxy_regret for sample in samples]
    semantic_times = [sample.semantic_elapsed_ms for sample in samples]
    replay_times = [sample.replay_elapsed_ms for sample in samples]
    focus_summary = {}
    for focus in ("threat", "prey_virus", "prey", "virus", "background"):
        rows = [sample for sample in samples if sample.strategic_focus == focus]
        if not rows:
            continue
        focus_summary[focus] = {
            "samples": len(rows),
            "direction_within_30_degrees_rate": sum(
                row.angle_degrees <= ADOPTION_ANGLE_DEGREES for row in rows
            )
            / len(rows),
            "mean_proxy_regret": statistics.fmean(row.proxy_regret for row in rows),
            "adoption_candidates": sum(row.adoption_candidate for row in rows),
        }
    return {
        "sample_count": len(samples),
        "direction_within_30_degrees_rate": sum(
            angle <= ADOPTION_ANGLE_DEGREES for angle in angles
        )
        / len(samples),
        "mean_angle_degrees": statistics.fmean(angles),
        "split_agreement_rate": sum(sample.split_agreement for sample in samples)
        / len(samples),
        "replay_proxy_preferred_rate": sum(regret > 0.0 for regret in regrets)
        / len(samples),
        "mean_proxy_regret": statistics.fmean(regrets),
        "p90_proxy_regret": _percentile(regrets, 0.90),
        "adoption_candidate_rate": sum(sample.adoption_candidate for sample in samples)
        / len(samples),
        "semantic_elapsed_ms": {
            "median": statistics.median(semantic_times),
            "p95": _percentile(semantic_times, 0.95),
            "max": max(semantic_times),
        },
        "replay_elapsed_ms": {
            "median": statistics.median(replay_times),
            "p95": _percentile(replay_times, 0.95),
            "max": max(replay_times),
        },
        "by_strategic_focus": focus_summary,
        "recommended_differences": recommendations[:12],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path, help="Local output/game.json replay")
    parser.add_argument("--player", type=int, default=0)
    parser.add_argument("--every-n", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.every_n < 1:
        raise SystemExit("--every-n must be positive")
    samples, summary = compare(
        args.replay.resolve(),
        player_id=args.player,
        every_n=args.every_n,
        max_samples=args.max_samples,
    )
    report = {
        "replay": str(args.replay.resolve()),
        "player_id": args.player,
        "thresholds": {
            "angle_degrees": ADOPTION_ANGLE_DEGREES,
            "proxy_regret": ADOPTION_PROXY_REGRET,
        },
        "summary": summary,
        "samples": [
            {**asdict(sample), "adoption_candidate": sample.adoption_candidate}
            for sample in samples
        ],
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
