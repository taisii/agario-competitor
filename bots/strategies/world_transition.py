from __future__ import annotations

"""Typed commands for exact world-transition evaluation.

The ordinary search has a deliberately pessimistic, per-fragment opponent
motion model.  That model is useful for ranking but is not a legal player
action.  Expected-outcome evaluation accepts only ``CompleteJointCommand`` so
one direction and split flag is shared by every fragment of each live player.
"""

from dataclasses import dataclass, field
import math
from typing import Generic, TypeVar

from strategies.features import normalise


StateT = TypeVar("StateT")


def _normalised_weights(
    values: tuple[float, ...],
    weights: tuple[float, ...],
) -> tuple[float, ...]:
    if not values:
        raise ValueError("weighted distribution requires at least one sample")
    if len(weights) != len(values):
        raise ValueError("scenario weights must match samples")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("scenario samples must be finite")
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("scenario weights must be finite and non-negative")
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        raise ValueError("scenario weights must have positive total mass")
    return tuple(weight / weight_sum for weight in weights)


def _lower_tail_mean_from_normalised(
    values: tuple[float, ...],
    normalised_weights: tuple[float, ...],
    *,
    tail_fraction: float = 0.2,
) -> float:
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail fraction must be in (0, 1]")
    tail_remaining = tail_fraction
    tail_total = 0.0
    for value, weight in sorted(zip(values, normalised_weights, strict=True)):
        portion = min(weight, tail_remaining)
        tail_total += value * portion
        tail_remaining -= portion
        if tail_remaining <= 1e-12:
            break
    return tail_total / tail_fraction


@dataclass(frozen=True, slots=True)
class PlayerCommand:
    direction: tuple[float, float]
    split: bool = False

    @property
    def unit(self) -> tuple[float, float]:
        return normalise(self.direction)


@dataclass(frozen=True, slots=True)
class CompleteJointCommand:
    """Exactly one legal command for every live player in a world state."""

    commands: tuple[tuple[int, PlayerCommand], ...]

    def __post_init__(self) -> None:
        player_ids = tuple(player_id for player_id, _ in self.commands)
        if len(set(player_ids)) != len(player_ids):
            raise ValueError("joint command cannot contain duplicate players")
        canonical = tuple(sorted(self.commands))
        if self.commands != canonical:
            object.__setattr__(self, "commands", canonical)

    @classmethod
    def build(
        cls,
        *,
        live_player_ids: set[int] | frozenset[int],
        commands: dict[int, PlayerCommand],
    ) -> CompleteJointCommand:
        expected = set(live_player_ids)
        actual = set(commands)
        if actual != expected:
            missing = tuple(sorted(expected - actual))
            unexpected = tuple(sorted(actual - expected))
            raise ValueError(
                "joint command must cover every live player exactly once "
                f"(missing={missing}, unexpected={unexpected})"
            )
        return cls(tuple(sorted(commands.items())))

    def for_player(self, player_id: int) -> PlayerCommand:
        for candidate_id, command in self.commands:
            if candidate_id == player_id:
                return command
        raise KeyError(player_id)

    @property
    def player_ids(self) -> frozenset[int]:
        return frozenset(player_id for player_id, _ in self.commands)


@dataclass(frozen=True, slots=True)
class JointPhysicalTransition(Generic[StateT]):
    """Physical result exposed to policy sampling without hazard semantics."""

    state: StateT
    final_own_mass: float
    dead: bool
    movement_efficiency: float


@dataclass(frozen=True, slots=True)
class ExpectedOutcomeStats:
    """Distribution summary for one root under sampled legal responses."""

    samples: tuple[float, ...]
    mean_mass: float
    death_rate: float
    cvar20_mass: float

    @classmethod
    def from_samples(
        cls,
        samples: tuple[float, ...],
        weights: tuple[float, ...] | None = None,
    ) -> ExpectedOutcomeStats:
        if weights is None:
            if not samples:
                raise ValueError("expected outcome requires at least one sample")
            weights = (1.0 / len(samples),) * len(samples)
        normalised = _normalised_weights(samples, weights)
        return cls(
            samples=samples,
            mean_mass=sum(
                value * weight
                for value, weight in zip(samples, normalised, strict=True)
            ),
            death_rate=sum(
                weight
                for value, weight in zip(samples, normalised, strict=True)
                if value <= 0.0
            ),
            cvar20_mass=_lower_tail_mean_from_normalised(samples, normalised),
        )


@dataclass(frozen=True, slots=True)
class ExpectedEvidence:
    """Policy evidence; intentionally not a universal safety certificate."""

    scenario_ids: tuple[int, ...]
    scenario_weights: tuple[float, ...]
    base: ExpectedOutcomeStats
    tactical: ExpectedOutcomeStats
    base_gain_positive_probability: float = 0.0
    tactical_gain_positive_probability: float = 0.0
    heldout_model_error: float = math.inf
    minimum_gain: float = 0.0
    minimum_paired_delta_cvar20: float = 0.0
    paired_delta_cvar20: float = field(init=False)

    def __post_init__(self) -> None:
        count = len(self.scenario_ids)
        if count == 0:
            raise ValueError("expected evidence requires at least one scenario")
        if len(set(self.scenario_ids)) != count:
            raise ValueError("expected evidence scenario IDs must be unique")
        if len(self.base.samples) != count or len(self.tactical.samples) != count:
            raise ValueError("expected evidence samples must align with scenario IDs")
        normalised = _normalised_weights(self.base.samples, self.scenario_weights)
        if any(not math.isfinite(value) for value in self.tactical.samples):
            raise ValueError("scenario samples must be finite")
        for probability in (
            self.base_gain_positive_probability,
            self.tactical_gain_positive_probability,
        ):
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("gain probabilities must be finite and in [0, 1]")
        if math.isnan(self.heldout_model_error) or self.heldout_model_error < 0.0:
            raise ValueError("held-out model error must be non-negative")
        if not math.isfinite(self.minimum_gain) or self.minimum_gain < 0.0:
            raise ValueError("minimum gain must be finite and non-negative")
        if (
            not math.isfinite(self.minimum_paired_delta_cvar20)
            or self.minimum_paired_delta_cvar20 < 0.0
        ):
            raise ValueError("paired delta CVaR margin must be finite and non-negative")
        deltas = tuple(
            tactical - base
            for base, tactical in zip(
                self.base.samples,
                self.tactical.samples,
                strict=True,
            )
        )
        object.__setattr__(
            self,
            "paired_delta_cvar20",
            _lower_tail_mean_from_normalised(deltas, normalised),
        )

    @property
    def mean_delta(self) -> float:
        return self.tactical.mean_mass - self.base.mean_mass

    @property
    def calibrated(self) -> bool:
        return math.isfinite(self.heldout_model_error)

    @property
    def paired_death_nonworse(self) -> bool:
        return all(
            int(tactical <= 0.0) <= int(base <= 0.0)
            for base, tactical in zip(
                self.base.samples,
                self.tactical.samples,
                strict=True,
            )
        )

    @property
    def paired_survival_improvement(self) -> bool:
        return self.paired_death_nonworse and any(
            base <= 0.0 < tactical
            for base, tactical in zip(
                self.base.samples,
                self.tactical.samples,
                strict=True,
            )
        )

    @property
    def supports_override(self) -> bool:
        return (
            self.calibrated
            and self.mean_delta > self.heldout_model_error + self.minimum_gain
            and self.paired_death_nonworse
            and self.tactical.cvar20_mass >= self.base.cvar20_mass
            and self.paired_delta_cvar20 >= self.minimum_paired_delta_cvar20
        )
