"""Strategy contracts and lazily loaded public policy implementations."""

from __future__ import annotations

import importlib

from strategies.base import Strategy, StrategyContext, StrategyDecision


_LAZY_EXPORTS = {
    "EventDrivenStaticSearchStrategy": "strategies.event_driven",
    "ReplayDominanceStrategy": "strategies.receding_horizon",
    "SemanticLookaheadStrategy": "strategies.semantic_potential",
    "SemanticPotentialStrategy": "strategies.semantic_potential",
    "StaticRetainedGrowthStrategy": "strategies.retained_growth",
    "ThreatAwareRecedingHorizonStrategy": "strategies.receding_horizon",
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
    "EventDrivenStaticSearchStrategy",
    "ReplayDominanceStrategy",
    "SemanticLookaheadStrategy",
    "SemanticPotentialStrategy",
    "StaticRetainedGrowthStrategy",
    "Strategy",
    "StrategyContext",
    "StrategyDecision",
    "ThreatAwareRecedingHorizonStrategy",
]
