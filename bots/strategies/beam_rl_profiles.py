from __future__ import annotations

import os
from dataclasses import replace

from bot_core import (
    BeamSearchPlanner,
    StrategyConfig,
    config_from_json_env,
    extract_world,
    log_decision_if_requested,
    profile_config,
)
from strategies.base import StrategyContext, StrategyDecision


class BeamRlProfileStrategy:
    profile_name = "balanced"
    name = "beam_rl_balanced"

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = _submission_safe_config(config or profile_config(self.profile_name))
        self.planner = BeamSearchPlanner(self.config)

    def choose(self, context: StrategyContext) -> StrategyDecision:
        raw_world = extract_world(context.game)
        world = _limited_world(raw_world)
        action = self.planner.choose_action(world)
        log_decision_if_requested(world, action, self.config)
        return StrategyDecision(
            direction=(action.dx, action.dy),
            split=action.split,
            target_kind="beam_rl",
            reason=action.label or self.config.name,
            diagnostics={
                "profile": self.config.name,
                "horizon": self.config.horizon,
                "beam_width": self.config.beam_width,
                "action_bins": self.config.action_bins,
                "allow_split": self.config.allow_split,
                "world_blob_count": world.blob_count,
                "world_enemy_count": len(world.enemies),
                "world_food_count": len(world.food),
                "raw_enemy_count": len(raw_world.enemies),
                "raw_food_count": len(raw_world.food),
            },
        )


class BeamRlBalancedStrategy(BeamRlProfileStrategy):
    profile_name = "balanced"
    name = "beam_rl_balanced"


class BeamRlSurvivalStrategy(BeamRlProfileStrategy):
    profile_name = "survival"
    name = "beam_rl_survival"


class BeamRlFarmerStrategy(BeamRlProfileStrategy):
    profile_name = "farmer"
    name = "beam_rl_farmer"


class BeamRlHunterStrategy(BeamRlProfileStrategy):
    profile_name = "hunter"
    name = "beam_rl_hunter"


class BeamRlOpportunistStrategy(BeamRlProfileStrategy):
    profile_name = "opportunist"
    name = "beam_rl_opportunist"


class BeamRlValueStrategy(BeamRlProfileStrategy):
    name = "beam_rl_value"

    def __init__(self) -> None:
        config = replace(profile_config("balanced"), name="rl_value", weight_learned_value=1.0)
        super().__init__(config=config)


class BeamRlTunedStrategy(BeamRlProfileStrategy):
    name = "beam_rl_tuned"

    def __init__(self) -> None:
        default = replace(profile_config("balanced"), name="tuned")
        super().__init__(config=config_from_json_env(default))


def _submission_safe_config(config: StrategyConfig) -> StrategyConfig:
    horizon = int(os.environ.get("BOT_BEAM_RL_HORIZON", min(config.horizon, 3)))
    beam_width = int(os.environ.get("BOT_BEAM_RL_WIDTH", min(config.beam_width, 3)))
    action_bins = int(os.environ.get("BOT_BEAM_RL_ACTION_BINS", min(config.action_bins, 10)))
    max_extra = int(os.environ.get("BOT_BEAM_RL_MAX_EXTRA", min(config.max_extra_candidates, 6)))
    return replace(
        config,
        horizon=max(1, horizon),
        beam_width=max(1, beam_width),
        action_bins=max(4, action_bins),
        max_extra_candidates=max(0, max_extra),
    )


def _limited_world(world):
    center = world.center
    max_food = int(os.environ.get("BOT_BEAM_RL_MAX_FOOD", "24"))
    max_enemies = int(os.environ.get("BOT_BEAM_RL_MAX_ENEMIES", "12"))
    max_viruses = int(os.environ.get("BOT_BEAM_RL_MAX_VIRUSES", "6"))

    food = tuple(
        sorted(
            world.food,
            key=lambda item: (item.pos.x - center.x) ** 2 + (item.pos.y - center.y) ** 2,
        )[:max_food]
    )
    enemies = tuple(
        sorted(
            world.enemies,
            key=lambda blob: (blob.pos.x - center.x) ** 2 + (blob.pos.y - center.y) ** 2,
        )[:max_enemies]
    )
    viruses = tuple(
        sorted(
            world.viruses,
            key=lambda virus: (virus.pos.x - center.x) ** 2 + (virus.pos.y - center.y) ** 2,
        )[:max_viruses]
    )
    return replace(world, food=food, enemies=enemies, viruses=viruses)
