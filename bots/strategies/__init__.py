"""Strategy contracts and lazily loaded public policy implementations."""

from __future__ import annotations

import importlib

from strategies.base import Strategy, StrategyContext, StrategyDecision


_LAZY_EXPORTS = {
    "FoodGreedyStrategy": "strategies.greedy",
    "EventDrivenStaticSearchStrategy": "strategies.event_driven",
    "PotentialFieldHunterStrategy": "strategies.potential_field",
    "ReplayDominanceStrategy": "strategies.receding_horizon",
    "SemanticLookaheadStrategy": "strategies.semantic_potential",
    "SemanticPotentialStrategy": "strategies.semantic_potential",
    "StaticOptionGrowthStrategy": "strategies.virus_farming",
    "StaticRetainedGrowthStrategy": "strategies.retained_growth",
    "SurvivalGreedyStrategy": "strategies.greedy",
    "ThreatAwareRecedingHorizonStrategy": "strategies.receding_horizon",
    "VirusHunterStrategy": "strategies.virus_farming",
}


def __getattr__(name: str):
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "FoodGreedyStrategy",
    "EventDrivenStaticSearchStrategy",
    "PotentialFieldHunterStrategy",
    "ReplayDominanceStrategy",
    "SemanticLookaheadStrategy",
    "SemanticPotentialStrategy",
    "StaticOptionGrowthStrategy",
    "StaticRetainedGrowthStrategy",
    "Strategy",
    "StrategyContext",
    "StrategyDecision",
    "SurvivalGreedyStrategy",
    "ThreatAwareRecedingHorizonStrategy",
    "VirusHunterStrategy",
]
