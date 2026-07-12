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
    "bots/strategies/features.py",
)


def _submission(
    strategy_class: str,
    *source_modules: str,
    local_only_classes: frozenset[str] = frozenset(),
) -> SubmissionBundleSpec:
    return SubmissionBundleSpec(
        strategy_class=strategy_class,
        source_modules=(*_COMMON_SUBMISSION_MODULES, *source_modules),
        local_only_classes=local_only_classes,
    )


_BUILT_IN_SPECS = (
    _spec("food_greedy", "strategies.greedy:FoodGreedyStrategy", "baseline"),
    _spec(
        "potential_field_hunter",
        "strategies.potential_field:PotentialFieldHunterStrategy",
        "potential_field",
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

REPLAY_TEAM_IDS = (
    1, 2, 3, 4, 5, 6, 9, 10, 12, 13, 14, 15, 16, 17, 21, 22, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 34, 35, 38, 39, 44, 48, 49, 51, 53,
    55, 56, 58, 59, 63, 68, 75, 77,
)

CUSTOM_REPLAY_TEAM_IDS = frozenset(
    {1, 4, 13, 16, 22, 25, 29, 31, 35, 39, 44, 51, 53, 56, 68}
)
PROFILED_OPPONENT_TEAM_IDS = frozenset({2, 6, 27, 30, 38, 75})

_REPLAY_SPECS = tuple(
    _spec(
        f"replay_team_{team_id}",
        (
            f"strategies.replay_team_{team_id}:ReplayTeam{team_id}Strategy"
            if team_id in CUSTOM_REPLAY_TEAM_IDS
            else (
                "strategies.replay_opponent_policies:create_profiled_opponent_strategy"
                if team_id in PROFILED_OPPONENT_TEAM_IDS
                else "strategies.replay_imitation:create_profiled_replay_strategy"
            )
        ),
        "replay_opponent",
        factory_argument=None if team_id in CUSTOM_REPLAY_TEAM_IDS else team_id,
    )
    for team_id in REPLAY_TEAM_IDS
)

STRATEGY_SPECS: dict[str, StrategySpec] = {
    spec.name: spec for spec in (*_BUILT_IN_SPECS, *_REPLAY_SPECS)
}

# Old commands remain valid, but aliases are intentionally excluded from the
# public strategy list so every behavior appears exactly once.
LEGACY_STRATEGY_ALIASES = {
    "champion": "threat_aware_receding_horizon",
    "p3_virus_farmer": "potential_field_virus_farmer",
    "potential_hunter": "potential_field_hunter",
}

DEFAULT_RANDOM_OPPONENT_STRATEGIES = (
    "food_greedy",
    "survival_greedy",
    "potential_field_hunter",
    "potential_field_virus_farmer",
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
        seed = (
            self._base_seed
            ^ ((self._trial + 1) * 0x9E3779B1)
            ^ ((player_id + 1) * 0x85EBCA77)
        )
        return random.Random(seed).choice(self._candidates)

    def choose(self, context):
        if self._selected is None:
            selected_name = self._select_name(context.game.state.me.player_id)
            self._selected = create_strategy(selected_name)
            self.name = f"random_opponent:{selected_name}"
        return self._selected.choose(context)


def select_replay_team_id(
    *,
    player_id: int,
    base_seed: int,
    trial: int,
    team_ids: tuple[int, ...] = REPLAY_TEAM_IDS,
) -> int:
    """Select a replay opponent reproducibly for one benchmark slot."""

    if not team_ids:
        raise ValueError("Random replay opponent requires at least one team")
    seed = base_seed ^ ((trial + 1) * 0x9E3779B1) ^ ((player_id + 1) * 0x85EBCA77)
    return random.Random(seed).choice(team_ids)


class RandomReplayOpponentStrategy:
    """Lazily choose a replay clone after the engine assigns a player slot."""

    name = "random_replay_opponent"

    def __init__(
        self,
        *,
        base_seed: int,
        trial: int,
        on_selected: Callable[[str], None] | None = None,
    ) -> None:
        self._base_seed = base_seed
        self._trial = trial
        self._on_selected = on_selected
        self._selected: Strategy | None = None

    def choose(self, context):
        if self._selected is None:
            team_id = select_replay_team_id(
                player_id=int(context.game.state.me.player_id),
                base_seed=self._base_seed,
                trial=self._trial,
            )
            self._selected = create_strategy(f"replay_team_{team_id}")
            self.name = self._selected.name
            if self._on_selected is not None:
                self._on_selected(self.name)
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


def create_random_replay_opponent_strategy(
    *,
    on_selected: Callable[[str], None] | None = None,
) -> Strategy:
    seed = int(os.environ.get("BOT_RANDOM_SEED", os.getpid() ^ time.time_ns()))
    trial = int(os.environ.get("BOT_BENCHMARK_TRIAL", "0"))
    return RandomReplayOpponentStrategy(
        base_seed=seed,
        trial=trial,
        on_selected=on_selected,
    )
