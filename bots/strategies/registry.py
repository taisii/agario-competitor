from __future__ import annotations

import importlib
import os
import random
import time
from dataclasses import dataclass
from typing import Callable

from strategies.base import Strategy


@dataclass(frozen=True)
class SubmissionBundleSpec:
    """Repository-local inputs required to build one standalone submission."""

    strategy_class: str
    source_modules: tuple[str, ...]
    local_only_classes: frozenset[str] = frozenset()


@dataclass(frozen=True)
class StrategySpec:
    """One discoverable policy and the implementation used to construct it.

    Keeping import paths as data prevents the default bot from importing every
    experimental planner at startup.  The same catalog also records the role of
    each policy so benchmark and submission tooling do not need to infer it from
    file names.
    """

    name: str
    factory_path: str
    category: str
    factory_argument: int | None = None
    submission: SubmissionBundleSpec | None = None

    def create(self) -> Strategy:
        module_name, separator, attribute_name = self.factory_path.partition(":")
        if not separator:
            raise RuntimeError(f"Invalid strategy factory path: {self.factory_path!r}")
        module = importlib.import_module(module_name)
        factory: Callable[..., Strategy] = getattr(module, attribute_name)
        strategy = (
            factory()
            if self.factory_argument is None
            else factory(self.factory_argument)
        )
        if strategy.name != self.name:
            raise RuntimeError(
                f"Strategy catalog name {self.name!r} does not match "
                f"{self.factory_path} name {strategy.name!r}"
            )
        return strategy


def _spec(
    name: str,
    factory_path: str,
    category: str,
    *,
    factory_argument: int | None = None,
    submission: SubmissionBundleSpec | None = None,
) -> StrategySpec:
    return StrategySpec(
        name=name,
        factory_path=factory_path,
        category=category,
        factory_argument=factory_argument,
        submission=submission,
    )


_COMMON_SUBMISSION_MODULES = (
    "bots/strategies/base.py",
    "bots/simulation/rules.py",
    "bots/strategies/world_transition.py",
)


def _submission(
    strategy_class: str,
    *source_modules: str,
    local_only_classes: frozenset[str] = frozenset(),
    after_features_modules: tuple[str, ...] = (),
) -> SubmissionBundleSpec:
    return SubmissionBundleSpec(
        strategy_class=strategy_class,
        source_modules=(
            *_COMMON_SUBMISSION_MODULES,
            *source_modules,
            "bots/strategies/features.py",
            *after_features_modules,
        ),
        local_only_classes=local_only_classes,
    )


_BUILT_IN_SPECS = (
    _spec("food_greedy", "strategies.greedy:FoodGreedyStrategy", "baseline"),
    _spec(
        "event_driven_static_search",
        "strategies.event_driven:EventDrivenStaticSearchStrategy",
        "search",
        submission=_submission(
            "EventDrivenStaticSearchStrategy",
            "bots/strategies/greedy.py",
            "bots/strategies/receding_horizon.py",
            "bots/strategies/event_driven.py",
            local_only_classes=frozenset(),
        ),
    ),
    _spec(
        "expected_final_mass",
        "strategies.expected_final_mass:ExpectedFinalMassStrategy",
        "search",
        submission=_submission(
            "ExpectedFinalMassStrategy",
            "bots/strategies/randomness.py",
            "bots/strategies/replay_imitation.py",
            "bots/strategies/replay_profiles.py",
            "bots/strategies/receding_horizon.py",
            "bots/strategies/expected_final_mass.py",
        ),
    ),
    _spec(
        "local_tactical_search",
        "strategies.local_tactical_search:LocalTacticalSearchStrategy",
        "search",
        submission=_submission(
            "LocalTacticalSearchStrategy",
            "bots/strategies/randomness.py",
            "bots/strategies/replay_imitation.py",
            "bots/strategies/replay_profiles.py",
            "bots/strategies/receding_horizon.py",
            "bots/strategies/expected_final_mass.py",
            "bots/strategies/local_tactical_search.py",
        ),
    ),
    _spec(
        "outcome_teacher_hybrid",
        "strategies.outcome_teacher_hybrid:OutcomeTeacherHybridStrategy",
        "search",
        submission=_submission(
            "OutcomeTeacherHybridStrategy",
            "bots/strategies/receding_horizon.py",
            "bots/strategies/semantic_potential.py",
            "bots/strategies/outcome_teacher_hybrid.py",
        ),
    ),
    _spec(
        "potential_field_hunter",
        "strategies.potential_field:PotentialFieldHunterStrategy",
        "potential_field",
    ),
    _spec(
        "potential_tactical_hybrid",
        "strategies.potential_tactical_hybrid:PotentialTacticalHybridStrategy",
        "search",
        submission=_submission(
            "PotentialTacticalHybridStrategy",
            "bots/strategies/randomness.py",
            "bots/strategies/replay_imitation.py",
            "bots/strategies/replay_profiles.py",
            "bots/strategies/receding_horizon.py",
            "bots/strategies/expected_final_mass.py",
            "bots/strategies/local_tactical_search.py",
            after_features_modules=(
                "bots/strategies/potential_field.py",
                "bots/strategies/potential_tactical_hybrid.py",
            ),
        ),
    ),
    _spec(
        "potential_field_virus_farmer",
        "strategies.virus_farming:PotentialFieldVirusFarmerStrategy",
        "potential_field",
    ),
    _spec(
        "replay_dominance",
        "strategies.receding_horizon:ReplayDominanceStrategy",
        "search",
        submission=_submission(
            "ReplayDominanceStrategy",
            "bots/strategies/receding_horizon.py",
        ),
    ),
    _spec(
        "replay_distilled",
        "strategies.replay_distilled:ReplayDistilledStrategy",
        "potential_field",
        submission=_submission(
            "ReplayDistilledStrategy",
            "bots/strategies/semantic_potential.py",
            "bots/strategies/replay_distilled.py",
        ),
    ),
    _spec(
        "semantic_potential",
        "strategies.semantic_potential:SemanticPotentialStrategy",
        "potential_field",
        submission=_submission(
            "SemanticPotentialStrategy",
            "bots/strategies/semantic_potential.py",
        ),
    ),
    _spec(
        "semantic_lookahead",
        "strategies.semantic_potential:SemanticLookaheadStrategy",
        "search",
        submission=_submission(
            "SemanticLookaheadStrategy",
            "bots/strategies/semantic_potential.py",
        ),
    ),
    _spec(
        "static_retained_growth",
        "strategies.retained_growth:StaticRetainedGrowthStrategy",
        "potential_field",
        submission=_submission(
            "StaticRetainedGrowthStrategy",
            "bots/strategies/retained_growth.py",
        ),
    ),
    _spec(
        "static_option_growth",
        "strategies.virus_farming:StaticOptionGrowthStrategy",
        "potential_field",
        submission=_submission(
            "StaticOptionGrowthStrategy",
            "bots/strategies/greedy.py",
            "bots/strategies/potential_field.py",
            "bots/strategies/receding_horizon.py",
            "bots/strategies/virus_farming.py",
            local_only_classes=frozenset({"ReplayDominanceStrategy"}),
        ),
    ),
    _spec("survival_greedy", "strategies.greedy:SurvivalGreedyStrategy", "baseline"),
    _spec(
        "threat_aware_receding_horizon",
        "strategies.receding_horizon:ThreatAwareRecedingHorizonStrategy",
        "search",
        submission=_submission(
            "ThreatAwareRecedingHorizonStrategy",
            "bots/strategies/receding_horizon.py",
            local_only_classes=frozenset({"ReplayDominanceStrategy"}),
        ),
    ),
    _spec(
        "virus_hunter",
        "strategies.virus_farming:VirusHunterStrategy",
        "potential_field",
        submission=_submission(
            "VirusHunterStrategy",
            "bots/strategies/greedy.py",
            "bots/strategies/receding_horizon.py",
            "bots/strategies/virus_farming.py",
            local_only_classes=frozenset(
                {"PotentialFieldVirusFarmerStrategy", "ReplayDominanceStrategy"}
            ),
        ),
    ),
)

STRATEGY_SPECS: dict[str, StrategySpec] = {spec.name: spec for spec in _BUILT_IN_SPECS}

# Old commands remain valid, but aliases are intentionally excluded from the
# public strategy list so every behavior appears exactly once.
LEGACY_STRATEGY_ALIASES = {
    "champion": "threat_aware_receding_horizon",
    "p3_virus_farmer": "potential_field_virus_farmer",
    "potential_hunter": "potential_field_hunter",
}

DEFAULT_RANDOM_OPPONENT_STRATEGIES = (
    "semantic_lookahead",
    "semantic_potential",
    "replay_dominance",
    "threat_aware_receding_horizon",
    "event_driven_static_search",
    "static_retained_growth",
)


class RandomOpponentStrategy:
    """Choose one reproducible opponent after the player slot is known."""

    name = "random_opponent"

    def __init__(
        self,
        candidates: tuple[str, ...],
        *,
        base_seed: int,
        trial: int,
    ) -> None:
        if not candidates:
            raise ValueError("RandomOpponentStrategy requires at least one candidate")
        self._candidates = candidates
        self._base_seed = base_seed
        self._trial = trial
        self._selected: Strategy | None = None

    def _select_name(self, player_id: int) -> str:
        # Shuffle once per paired trial, then cycle by slot. With the standard
        # candidate-at-slot-zero layout, opponent slots 1..7 cover all six
        # policies; only the duplicated policy and placement vary.
        seed = (
            self._base_seed
            ^ ((self._trial + 1) * 0x9E3779B1)
        )
        shuffled = list(self._candidates)
        random.Random(seed).shuffle(shuffled)
        return shuffled[(player_id - 1) % len(shuffled)]

    def choose(self, context):
        if self._selected is None:
            selected_name = self._select_name(context.game.state.me.player_id)
            self._selected = create_strategy(selected_name)
            self.name = f"random_opponent:{selected_name}"
        return self._selected.choose(context)


def available_strategy_names(*, category: str | None = None) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name, spec in STRATEGY_SPECS.items()
            if category is None or spec.category == category
        )
    )


def submission_strategy_names() -> tuple[str, ...]:
    return tuple(
        sorted(
            spec.name for spec in STRATEGY_SPECS.values() if spec.submission is not None
        )
    )


def submission_strategy_spec(name: str) -> StrategySpec:
    """Return a catalog entry that carries standalone bundle metadata."""

    try:
        spec = STRATEGY_SPECS[name]
    except KeyError as exc:
        available = ", ".join(submission_strategy_names())
        raise ValueError(
            f"Unsupported submission strategy {name!r}. Available: {available}"
        ) from exc
    if spec.submission is None:
        available = ", ".join(submission_strategy_names())
        raise ValueError(
            f"Unsupported submission strategy {name!r}. Available: {available}"
        )
    return spec


def create_strategy(name: str) -> Strategy:
    if name == "random_opponent":
        return create_random_opponent_strategy()
    canonical_name = LEGACY_STRATEGY_ALIASES.get(name, name)
    try:
        spec = STRATEGY_SPECS[canonical_name]
    except KeyError as exc:
        available = ", ".join((*available_strategy_names(), "random_opponent"))
        raise ValueError(f"Unknown strategy {name!r}. Available: {available}") from exc
    return spec.create()


def create_random_opponent_strategy() -> Strategy:
    raw_candidates = os.environ.get("BOT_RANDOM_STRATEGIES")
    if raw_candidates:
        candidates = tuple(
            LEGACY_STRATEGY_ALIASES.get(item.strip(), item.strip())
            for item in raw_candidates.split(",")
            if item.strip()
        )
    else:
        candidates = DEFAULT_RANDOM_OPPONENT_STRATEGIES

    invalid = [candidate for candidate in candidates if candidate not in STRATEGY_SPECS]
    if invalid:
        available = ", ".join(available_strategy_names())
        raise ValueError(
            f"Invalid BOT_RANDOM_STRATEGIES entries {invalid}. Available: {available}"
        )

    seed = int(os.environ.get("BOT_RANDOM_SEED", os.getpid() ^ time.time_ns()))
    trial = int(os.environ.get("BOT_BENCHMARK_TRIAL", "0"))
    return RandomOpponentStrategy(candidates, base_seed=seed, trial=trial)
