from __future__ import annotations

import os
import random
import time

from strategies.base import Strategy
from strategies.beam_hunter import BeamHunterStrategy
from strategies.beam_rl_profiles import (
    BeamRlBalancedStrategy,
    BeamRlFarmerStrategy,
    BeamRlHunterStrategy,
    BeamRlOpportunistStrategy,
    BeamRlSurvivalStrategy,
    BeamRlTunedStrategy,
    BeamRlValueStrategy,
)
from strategies.beam_survival import BeamSurvivalStrategy
from strategies.champion import ChampionStrategy
from strategies.food_greedy import FoodGreedyStrategy
from strategies.potential_hunter import PotentialHunterStrategy
from strategies.survival_greedy import SurvivalGreedyStrategy


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
    ChampionStrategy.name: ChampionStrategy,
    FoodGreedyStrategy.name: FoodGreedyStrategy,
    PotentialHunterStrategy.name: PotentialHunterStrategy,
    SurvivalGreedyStrategy.name: SurvivalGreedyStrategy,
}

DEFAULT_RANDOM_OPPONENT_STRATEGIES = (
    FoodGreedyStrategy.name,
    SurvivalGreedyStrategy.name,
    BeamSurvivalStrategy.name,
    PotentialHunterStrategy.name,
)


def available_strategy_names() -> tuple[str, ...]:
    return tuple(sorted(STRATEGY_FACTORIES))


def create_strategy(name: str) -> Strategy:
    if name == "random_opponent":
        return create_random_opponent_strategy()
    try:
        return STRATEGY_FACTORIES[name]()
    except KeyError as exc:
        available = ", ".join((*available_strategy_names(), "random_opponent"))
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}") from exc


def create_random_opponent_strategy() -> Strategy:
    raw_candidates = os.environ.get("BOT_RANDOM_STRATEGIES")
    if raw_candidates:
        candidates = tuple(item.strip() for item in raw_candidates.split(",") if item.strip())
    else:
        candidates = DEFAULT_RANDOM_OPPONENT_STRATEGIES

    invalid = [candidate for candidate in candidates if candidate not in STRATEGY_FACTORIES]
    if invalid:
        available = ", ".join(available_strategy_names())
        raise ValueError(f"Invalid BOT_RANDOM_STRATEGIES entries {invalid}. Available: {available}")

    seed = int(os.environ.get("BOT_RANDOM_SEED", os.getpid() ^ time.time_ns()))
    selected_name = random.Random(seed).choice(candidates)
    selected = STRATEGY_FACTORIES[selected_name]()
    selected.name = f"random_opponent:{selected_name}"
    return selected
