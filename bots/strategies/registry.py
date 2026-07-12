from __future__ import annotations

import importlib
import os
import random
import time

from strategies.base import Strategy
from strategies.beam_search import (
    BeamHunterStrategy,
    BeamRlBalancedStrategy,
    BeamRlFarmerStrategy,
    BeamRlHunterStrategy,
    BeamRlOpportunistStrategy,
    BeamRlSurvivalStrategy,
    BeamRlTunedStrategy,
    BeamRlValueStrategy,
    BeamSurvivalStrategy,
)
from strategies.greedy import FoodGreedyStrategy, SurvivalGreedyStrategy
from strategies.potential_field import PotentialFieldHunterStrategy
from strategies.replay_profiles import PROFILES as REPLAY_PROFILES
from strategies.receding_horizon import (
    ReplayDominanceStrategy,
    ThreatAwareRecedingHorizonStrategy,
    VirusFarmingRecedingHorizonStrategy,
)
from strategies.unified_deterministic import UnifiedDeterministicStrategy
from strategies.virus_farming import (
    PotentialFieldVirusFarmerStrategy,
    VirusHunterStrategy,
)


STRATEGY_FACTORIES = {
    BeamHunterStrategy.name: BeamHunterStrategy,
    BeamRlBalancedStrategy.name: BeamRlBalancedStrategy,
    BeamRlFarmerStrategy.name: BeamRlFarmerStrategy,
    BeamRlHunterStrategy.name: BeamRlHunterStrategy,
    BeamRlOpportunistStrategy.name: BeamRlOpportunistStrategy,
    BeamRlSurvivalStrategy.name: BeamRlSurvivalStrategy,
    BeamRlTunedStrategy.name: BeamRlTunedStrategy,
    BeamRlValueStrategy.name: BeamRlValueStrategy,
    BeamSurvivalStrategy.name: BeamSurvivalStrategy,
    FoodGreedyStrategy.name: FoodGreedyStrategy,
    PotentialFieldHunterStrategy.name: PotentialFieldHunterStrategy,
    PotentialFieldVirusFarmerStrategy.name: PotentialFieldVirusFarmerStrategy,
    ReplayDominanceStrategy.name: ReplayDominanceStrategy,
    SurvivalGreedyStrategy.name: SurvivalGreedyStrategy,
    ThreatAwareRecedingHorizonStrategy.name: ThreatAwareRecedingHorizonStrategy,
    VirusHunterStrategy.name: VirusHunterStrategy,
    VirusFarmingRecedingHorizonStrategy.name: VirusFarmingRecedingHorizonStrategy,
    UnifiedDeterministicStrategy.name: UnifiedDeterministicStrategy,
}


def _create_replay_team_strategy(team_id: int) -> Strategy:
    module = importlib.import_module(f"strategies.replay_team_{team_id}")
    strategy_type = getattr(module, f"ReplayTeam{team_id}Strategy")
    return strategy_type()


# Keep all 42 replay opponents selectable through BOT_STRATEGY without a
# brittle block of imports. Their dedicated simulator entries still import the
# concrete classes directly, so one missing team module fails locally and
# clearly rather than silently falling back to a different behavior.
STRATEGY_FACTORIES.update(
    {
        f"replay_team_{team_id}": (
            lambda team_id=team_id: _create_replay_team_strategy(team_id)
        )
        for team_id in REPLAY_PROFILES
    }
)

# Old commands remain valid, but aliases are intentionally excluded from the
# public strategy list so each behavior appears exactly once.
LEGACY_STRATEGY_ALIASES = {
    "candidate_submission": VirusFarmingRecedingHorizonStrategy.name,
    "champion": ThreatAwareRecedingHorizonStrategy.name,
    "p3_virus_farmer": PotentialFieldVirusFarmerStrategy.name,
    "potential_hunter": PotentialFieldHunterStrategy.name,
}

DEFAULT_RANDOM_OPPONENT_STRATEGIES = (
    FoodGreedyStrategy.name,
    SurvivalGreedyStrategy.name,
    BeamSurvivalStrategy.name,
    PotentialFieldHunterStrategy.name,
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
            self._selected = STRATEGY_FACTORIES[selected_name]()
            self.name = f"random_opponent:{selected_name}"
        return self._selected.choose(context)


def available_strategy_names() -> tuple[str, ...]:
    return tuple(sorted(STRATEGY_FACTORIES))


def create_strategy(name: str) -> Strategy:
    if name == "random_opponent":
        return create_random_opponent_strategy()
    canonical_name = LEGACY_STRATEGY_ALIASES.get(name, name)
    try:
        return STRATEGY_FACTORIES[canonical_name]()
    except KeyError as exc:
        available = ", ".join((*available_strategy_names(), "random_opponent"))
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}") from exc


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

    invalid = [candidate for candidate in candidates if candidate not in STRATEGY_FACTORIES]
    if invalid:
        available = ", ".join(available_strategy_names())
        raise ValueError(f"Invalid BOT_RANDOM_STRATEGIES entries {invalid}. Available: {available}")

    seed = int(os.environ.get("BOT_RANDOM_SEED", os.getpid() ^ time.time_ns()))
    trial = int(os.environ.get("BOT_BENCHMARK_TRIAL", "0"))
    return RandomOpponentStrategy(candidates, base_seed=seed, trial=trial)
