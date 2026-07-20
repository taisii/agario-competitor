from __future__ import annotations

"""Convert a secure lead into mass with verified one-step contact captures."""

from dataclasses import replace
import math
import os

from strategies.base import StrategyContext, StrategyDecision
from strategies.semantic_potential import (
    SAFETY_RESERVE,
    _best_capture_target,
    _direction_or_fallback,
    _minimum_threat_margin,
    _observed_enemy_directions,
    _project_one_step_outcome,
)


MIN_LEADING_MASS = 20.0
MIN_CAPTURE_MASS = 1.0
MIN_CAPTURE_SHARE = 0.05
MAX_DIRECTION_CHANGE_DEGREES = 15.0


class SnowballCaptureLayer:
    """Take only already-secured prey when the bot has capital to compound.

    The base policy remains responsible for ordinary growth and survival. This
    layer is deliberately asymmetric: it activates only while ranked first and
    changes the submitted heading by at most 15 degrees, and requires the exact
    non-split one-step transition to consume meaningful enemy mass without
    losing any own fragment. It never creates additional fragments.
    """

    def __init__(self) -> None:
        self._enabled = _environment_enabled("SNOWBALL_CAPTURE_ENABLED", True)
        self._enemy_positions: dict[tuple[int, int], tuple[float, float]] = {}
        self._last_round: int | None = None

    def adjust(
        self,
        context: StrategyContext,
        decision: StrategyDecision,
    ) -> StrategyDecision:
        state = context.game.state
        round_number = int(getattr(state, "round", 0))
        if self._last_round is not None and round_number <= self._last_round:
            self._enemy_positions = {}
        self._last_round = round_number

        own = tuple(state.me.blobs.values())
        enemies = tuple(state.visible_blobs)
        previous_directions = _observed_enemy_directions(
            enemies,
            previous_positions=self._enemy_positions,
        )
        self._enemy_positions = {
            (int(enemy.player_id), enemy.blob_id): enemy.pos for enemy in enemies
        }
        diagnostics = dict(decision.diagnostics)
        diagnostics.update(
            {
                "snowball_capture_enabled": self._enabled,
                "snowball_capture_offered": False,
                "snowball_capture_intervened": False,
            }
        )

        rankings = tuple(getattr(state, "rankings", ()))
        player_id = int(state.me.player_id)
        total_mass = sum(blob.radius * blob.radius for blob in own)
        eligible = (
            self._enabled
            and not decision.split
            and rankings
            and rankings[0] == player_id
            and total_mass >= MIN_LEADING_MASS
            and bool(own)
            and bool(enemies)
        )
        if not eligible:
            return replace(decision, diagnostics=diagnostics)

        arena_size = float(state.map.size)
        route = _best_capture_target(
            own=own,
            enemies=enemies,
            previous_directions=previous_directions,
            arena_size=arena_size,
        )
        if route is None:
            return replace(decision, diagnostics=diagnostics)
        capture_direction = _direction_or_fallback(
            route.hunter.pos,
            route.target_pos,
            decision.direction,
        )
        direction_change = _angle_degrees(
            decision.direction,
            capture_direction,
        )
        if direction_change > MAX_DIRECTION_CHANGE_DEGREES:
            return replace(decision, diagnostics=diagnostics)

        target_id = f"{route.enemy.player_id}:{route.enemy.blob_id}"
        outcome = _project_one_step_outcome(
            own=own,
            direction=capture_direction,
            split=False,
            foods=tuple(state.visible_food),
            viruses=tuple(state.visible_viruses),
            enemies=enemies,
            arena_size=arena_size,
            target_id=target_id,
            target_pos=route.target_pos,
        )
        captured_mass = outcome.enemy_mass_gained
        capture_share = captured_mass / total_mass if total_mass > 0.0 else 0.0
        safety_margin = _minimum_threat_margin(outcome.own, outcome.enemies)
        offered = (
            captured_mass >= MIN_CAPTURE_MASS
            and capture_share >= MIN_CAPTURE_SHARE
            and outcome.own_mass_lost <= 1.0e-9
            and safety_margin >= SAFETY_RESERVE
        )
        diagnostics.update(
            {
                "snowball_capture_offered": offered,
                "snowball_capture_captured_mass": captured_mass,
                "snowball_capture_capture_share": capture_share,
                "snowball_capture_post_margin": safety_margin,
                "snowball_capture_direction_change_degrees": direction_change,
            }
        )
        if not offered:
            return replace(decision, diagnostics=diagnostics)

        secured = dict(diagnostics.get("secured_one_step_mass", {}))
        secured["enemy"] = captured_mass
        secured["lost"] = outcome.own_mass_lost
        diagnostics["secured_one_step_mass"] = secured
        diagnostics["snowball_capture_intervened"] = True
        return replace(
            decision,
            direction=capture_direction,
            split=False,
            target_kind="prey",
            target_id=target_id,
            reason="snowball_contact_capture",
            diagnostics=diagnostics,
        )


def _environment_enabled(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _angle_degrees(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    first_length = math.hypot(*first)
    second_length = math.hypot(*second)
    if first_length <= 1.0e-9 or second_length <= 1.0e-9:
        return math.inf
    cosine = (first[0] * second[0] + first[1] * second[1]) / (
        first_length * second_length
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
