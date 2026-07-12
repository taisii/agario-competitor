from __future__ import annotations

"""Pure implementations of public engine rules.

The strategy simulators use different state models, so this module deliberately
operates on numbers and caller-provided accessors instead of owning another
blob model.  Ordering and geometry therefore have one authoritative
implementation without coupling policy code to a particular planner.
"""

import math
from typing import Callable, Iterable, TypeVar


T = TypeVar("T")


def movement_speed(
    radius: float,
    *,
    base_speed: float,
    radius_factor: float,
    minimum_speed: float,
) -> float:
    return max(minimum_speed, base_speed / (1.0 + radius * radius_factor))


def decayed_radius(
    radius: float,
    *,
    decay_rate: float,
    minimum_radius: float,
) -> float:
    mass = radius * radius
    minimum_mass = minimum_radius * minimum_radius
    if mass <= minimum_mass:
        return radius
    return math.sqrt(max(minimum_mass, mass * (1.0 - decay_rate)))


def decayed_mass_after_turns(
    mass: float,
    turns: int,
    *,
    decay_rate: float,
    minimum_radius: float,
) -> float:
    """Apply the engine's per-turn mass decay for a projected horizon."""

    minimum_mass = minimum_radius * minimum_radius
    if mass <= minimum_mass:
        return mass
    return max(minimum_mass, mass * (1.0 - decay_rate) ** max(0, turns))


def can_consume_virus(
    blob_radius: float,
    virus_radius: float,
    *,
    eat_size_ratio: float,
) -> bool:
    return blob_radius * blob_radius > virus_radius * virus_radius * eat_size_ratio


def circle_intersects_square(
    *,
    circle_x: float,
    circle_y: float,
    circle_radius: float,
    square_center_x: float,
    square_center_y: float,
    square_size: float,
) -> bool:
    """Match ``GameState.is_circle_in_vision`` including corner geometry."""

    half_size = square_size / 2.0
    dx_outside = max(abs(circle_x - square_center_x) - half_size, 0.0)
    dy_outside = max(abs(circle_y - square_center_y) - half_size, 0.0)
    return dx_outside * dx_outside + dy_outside * dy_outside <= circle_radius * circle_radius


def select_largest_first(
    candidates: Iterable[T],
    *,
    radius: Callable[[T], float],
    player_id: Callable[[T], int],
    blob_id: Callable[[T], int],
) -> T | None:
    """Select a collision winner using the engine's deterministic ordering."""

    return min(
        candidates,
        key=lambda candidate: (
            -radius(candidate),
            player_id(candidate),
            blob_id(candidate),
        ),
        default=None,
    )


def virus_replacement_positions(
    *,
    center_x: float,
    center_y: float,
    piece_radius: float,
    piece_count: int,
    overlap_epsilon: float,
) -> tuple[tuple[float, float], ...]:
    cols = math.ceil(math.sqrt(piece_count))
    rows = math.ceil(piece_count / cols)
    spacing = piece_radius * 2.0 + overlap_epsilon
    x_offset = (cols - 1) * spacing / 2.0
    y_offset = (rows - 1) * spacing / 2.0
    return tuple(
        (
            center_x + (index % cols) * spacing - x_offset,
            center_y + (index // cols) * spacing - y_offset,
        )
        for index in range(piece_count)
    )
