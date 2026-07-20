from __future__ import annotations

"""A narrow post-fragmentation safety layer for preserving valuable mass."""

from dataclasses import dataclass, replace
import math
import os

from lib.models.blob_model import BlobModel, VisibleBlobModel
from strategies.base import StrategyContext, StrategyDecision
from strategies.features import can_eat_player_blob, normalise, player_speed
from strategies.semantic_potential import (
    _horizon_attack_reach,
    _project_action_blobs,
    _project_blobs,
    _wall_escape_direction,
)


EPSILON = 1.0e-9
DEFAULT_RECOVERY_ROUNDS = 10
DEFAULT_RISK_HORIZON = 3.0
DEFAULT_MAX_OVERRIDE_DEGREES = 30.0
DEFAULT_MIN_SAVED_MASS_FRACTION = 0.05
MIN_PROTECTED_FRAGMENT_MASS = 1.0


@dataclass(frozen=True, slots=True)
class AssetRisk:
    valuable_exposed_mass: float
    exposed_mass: float
    weighted_danger: float
    minimum_margin: float


class AssetPreservationLayer:
    """Override only high-risk routes shortly after fragmentation.

    A new fragment topology is observable even though the public query does not
    explicitly say whether a virus or split created it.  For the next few
    rounds this layer compares the base route with a small set of aggregate
    escape routes under worst-case predator closure.  It never replaces the
    normal resource policy outside that recovery window.
    """

    def __init__(self) -> None:
        self._enabled = _environment_enabled("ASSET_PRESERVATION_ENABLED", True)
        self._recovery_rounds = _environment_nonnegative_int(
            "ASSET_PRESERVATION_RECOVERY_ROUNDS",
            DEFAULT_RECOVERY_ROUNDS,
        )
        self._risk_horizon = _environment_nonnegative_float(
            "ASSET_PRESERVATION_RISK_HORIZON",
            DEFAULT_RISK_HORIZON,
        )
        self._max_override_angle = math.radians(
            _environment_nonnegative_float(
                "ASSET_PRESERVATION_MAX_OVERRIDE_DEGREES",
                DEFAULT_MAX_OVERRIDE_DEGREES,
            )
        )
        self._min_saved_mass_fraction = _environment_nonnegative_float(
            "ASSET_PRESERVATION_MIN_SAVED_MASS_FRACTION",
            DEFAULT_MIN_SAVED_MASS_FRACTION,
        )
        self._remaining = 0
        self._intervened_in_window = False
        self._previous_blob_count: int | None = None
        self._previous_decision_split = False
        self._last_round: int | None = None
        self._player_id: int | None = None

    def adjust(
        self,
        context: StrategyContext,
        decision: StrategyDecision,
    ) -> StrategyDecision:
        state = context.game.state
        round_number = int(state.round)
        player_id = int(state.me.player_id)
        if (
            self._last_round is not None
            and (round_number < self._last_round or player_id != self._player_id)
        ):
            self._remaining = 0
            self._intervened_in_window = False
            self._previous_blob_count = None
            self._previous_decision_split = False
        own = tuple(state.me.blobs.values())
        enemies = tuple(state.visible_blobs)
        current_count = len(own)
        topology_expanded = (
            self._previous_blob_count is not None
            and current_count > self._previous_blob_count
        )
        trigger = None
        if topology_expanded:
            trigger = "own_split" if self._previous_decision_split else "virus"
            self._remaining = self._recovery_rounds
            self._intervened_in_window = False

        diagnostics = dict(decision.diagnostics)
        diagnostics.update(
            {
                "asset_preservation_enabled": self._enabled,
                "asset_preservation_trigger": trigger,
                "asset_preservation_remaining": self._remaining,
                "asset_preservation_window_intervened": (
                    self._intervened_in_window
                ),
                "asset_preservation_intervened": False,
            }
        )
        adjusted = replace(decision, diagnostics=diagnostics)
        if (
            self._enabled
            and len(own) >= 2
            and enemies
        ):
            adjusted = self._protect_assets(
                context=context,
                decision=decision,
                own=own,
                enemies=enemies,
                diagnostics=diagnostics,
            )
            if adjusted.diagnostics.get("asset_preservation_intervened") is True:
                self._intervened_in_window = True

        if self._remaining > 0:
            self._remaining -= 1
        self._previous_blob_count = current_count
        self._previous_decision_split = adjusted.split
        self._last_round = round_number
        self._player_id = player_id
        return adjusted

    def _protect_assets(
        self,
        *,
        context: StrategyContext,
        decision: StrategyDecision,
        own: tuple[BlobModel, ...],
        enemies: tuple[VisibleBlobModel, ...],
        diagnostics: dict[str, object],
    ) -> StrategyDecision:
        arena_size = float(context.game.state.map.size)
        base_risk = _asset_risk(
            own,
            enemies,
            direction=decision.direction,
            split=decision.split,
            arena_size=arena_size,
            horizon=self._risk_horizon,
        )
        selected_direction = normalise(decision.direction)
        selected_risk = _asset_risk(
            own,
            enemies,
            direction=decision.direction,
            split=False,
            arena_size=arena_size,
            horizon=self._risk_horizon,
        )
        total_mass = sum(blob.radius * blob.radius for blob in own)
        saved_valuable_mass = max(
            0.0,
            base_risk.valuable_exposed_mass
            - selected_risk.valuable_exposed_mass,
        )
        saved_mass_fraction = saved_valuable_mass / max(total_mass, EPSILON)
        secured = decision.diagnostics.get("secured_one_step_mass")
        secured_mass = 0.0
        if isinstance(secured, dict):
            secured_mass = sum(
                float(value)
                for key in ("enemy", "virus", "food")
                if isinstance((value := secured.get(key)), int | float)
                and math.isfinite(float(value))
                and float(value) > 0.0
            )
        reduces_valuable_exposure = (
            decision.split
            and saved_valuable_mass > EPSILON
            and saved_mass_fraction + EPSILON
            >= self._min_saved_mass_fraction
            and saved_valuable_mass > secured_mass + EPSILON
        )
        suppress_split = reduces_valuable_exposure and decision.split
        intervened = reduces_valuable_exposure
        diagnostics.update(
            {
                "asset_preservation_candidate_count": 1,
                "asset_preservation_base_valuable_exposed_mass": (
                    base_risk.valuable_exposed_mass
                ),
                "asset_preservation_selected_valuable_exposed_mass": (
                    selected_risk.valuable_exposed_mass
                ),
                "asset_preservation_base_exposed_mass": base_risk.exposed_mass,
                "asset_preservation_selected_exposed_mass": (
                    selected_risk.exposed_mass
                ),
                "asset_preservation_base_danger": base_risk.weighted_danger,
                "asset_preservation_selected_danger": selected_risk.weighted_danger,
                "asset_preservation_saved_valuable_mass": saved_valuable_mass,
                "asset_preservation_saved_mass_fraction": saved_mass_fraction,
                "asset_preservation_min_saved_mass_fraction": (
                    self._min_saved_mass_fraction
                ),
                "asset_preservation_secured_mass": secured_mass,
                "asset_preservation_override_degrees": _angle_degrees(
                    decision.direction,
                    selected_direction,
                ),
                "asset_preservation_reduces_valuable_exposure": (
                    reduces_valuable_exposure
                ),
                "asset_preservation_suppressed_split": suppress_split,
                "asset_preservation_intervened": intervened,
            }
        )
        if not intervened:
            return replace(decision, diagnostics=diagnostics)
        return replace(
            decision,
            split=False,
            reason="post_fragment_split_veto",
            diagnostics=diagnostics,
        )


def _recovery_directions(
    own: tuple[BlobModel, ...],
    enemies: tuple[VisibleBlobModel, ...],
    *,
    base_direction: tuple[float, float],
    arena_size: float,
    horizon: float,
    max_override_angle: float,
) -> tuple[tuple[float, float], ...]:
    total_mass = sum(blob.radius * blob.radius for blob in own)
    center = (
        sum(blob.pos[0] * blob.radius * blob.radius for blob in own) / total_mass,
        sum(blob.pos[1] * blob.radius * blob.radius for blob in own) / total_mass,
    )
    escape_x = 0.0
    escape_y = 0.0
    per_player: dict[int, tuple[float, float, float]] = {}
    for blob in own:
        mass = blob.radius * blob.radius
        for enemy in enemies:
            if not can_eat_player_blob(enemy.radius, blob.radius):
                continue
            margin = math.dist(blob.pos, enemy.pos) - _horizon_attack_reach(
                enemy.radius,
                blob.radius,
                horizon,
            )
            awareness = player_speed(blob.radius) * horizon
            if margin >= awareness:
                continue
            away = normalise(
                (blob.pos[0] - enemy.pos[0], blob.pos[1] - enemy.pos[1])
            )
            severity = 1.0 + max(0.0, awareness - margin) / max(awareness, EPSILON)
            weight = mass * severity
            escape_x += away[0] * weight
            escape_y += away[1] * weight
            px, py, prior = per_player.get(int(enemy.player_id), (0.0, 0.0, 0.0))
            per_player[int(enemy.player_id)] = (
                px + away[0] * weight,
                py + away[1] * weight,
                prior + weight,
            )

    aggregate = normalise((escape_x, escape_y))
    raw = [normalise(base_direction)]
    if aggregate != (0.0, 0.0):
        raw.extend(
            (
                aggregate,
                (-aggregate[1], aggregate[0]),
                (aggregate[1], -aggregate[0]),
            )
        )
        wall = _wall_escape_direction(own, aggregate, arena_size)
        if wall != (0.0, 0.0):
            raw.append(wall)
    for x, y, _ in sorted(
        per_player.values(),
        key=lambda item: item[2],
        reverse=True,
    )[:3]:
        raw.append(normalise((x, y)))
    raw.append(normalise((arena_size / 2.0 - center[0], arena_size / 2.0 - center[1])))

    base = normalise(base_direction)
    unique: list[tuple[float, float]] = []
    for direction in raw:
        if direction == (0.0, 0.0):
            continue
        direction = _rotate_toward(base, direction, max_override_angle)
        if any(_dot(direction, prior) > 1.0 - 1.0e-6 for prior in unique):
            continue
        unique.append(direction)
    return tuple(unique)


def _asset_risk(
    own: tuple[BlobModel, ...],
    enemies: tuple[VisibleBlobModel, ...],
    *,
    direction: tuple[float, float],
    split: bool = False,
    arena_size: float,
    horizon: float,
) -> AssetRisk:
    if split:
        first_step = _project_action_blobs(
            own,
            direction,
            split=True,
            arena_size=arena_size,
        )
        projected = _project_blobs(
            first_step,
            direction,
            max(0.0, horizon - 1.0),
            arena_size,
        )
    else:
        projected = _project_blobs(own, direction, horizon, arena_size)
    valuable_exposed_mass = 0.0
    exposed_mass = 0.0
    weighted_danger = 0.0
    minimum_margin = math.inf
    for blob in projected:
        mass = blob.radius * blob.radius
        margins = tuple(
            math.dist(blob.pos, enemy.pos)
            - _horizon_attack_reach(enemy.radius, blob.radius, horizon)
            for enemy in enemies
            if can_eat_player_blob(enemy.radius, blob.radius)
        )
        if not margins:
            continue
        margin = min(margins)
        minimum_margin = min(minimum_margin, margin)
        if margin <= 0.0:
            exposed_mass += mass
            if mass + EPSILON >= MIN_PROTECTED_FRAGMENT_MASS:
                valuable_exposed_mass += mass
            weighted_danger += mass * (1.0 - margin)
        else:
            weighted_danger += mass / (1.0 + margin) ** 2
    return AssetRisk(
        valuable_exposed_mass=valuable_exposed_mass,
        exposed_mass=exposed_mass,
        weighted_danger=weighted_danger,
        minimum_margin=minimum_margin,
    )


def _dot(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1]


def _rotate_toward(
    source: tuple[float, float],
    target: tuple[float, float],
    max_angle: float,
) -> tuple[float, float]:
    source = normalise(source)
    target = normalise(target)
    if source == (0.0, 0.0):
        return target
    if target == (0.0, 0.0) or max_angle <= 0.0:
        return source
    cross = source[0] * target[1] - source[1] * target[0]
    dot = max(-1.0, min(1.0, _dot(source, target)))
    bounded = max(-max_angle, min(max_angle, math.atan2(cross, dot)))
    cosine = math.cos(bounded)
    sine = math.sin(bounded)
    return normalise(
        (
            source[0] * cosine - source[1] * sine,
            source[0] * sine + source[1] * cosine,
        )
    )


def _angle_degrees(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    first = normalise(first)
    second = normalise(second)
    return math.degrees(
        math.acos(max(-1.0, min(1.0, _dot(first, second))))
    )


def _environment_enabled(name: str, default: bool) -> bool:
    fallback = "1" if default else "0"
    return os.environ.get(name, fallback).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _environment_nonnegative_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _environment_nonnegative_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) and value >= 0.0 else default
