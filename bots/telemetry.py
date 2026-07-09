from __future__ import annotations

import json
import os
import time
from pathlib import Path

from helper.game import Game
from strategies.base import StrategyDecision
from strategies.features import extract_visible_features


class MetricsLogger:
    def __init__(self) -> None:
        # Per-turn JSONL writes count against the engine's cumulative response
        # budget. Keep diagnostics opt-in so the default bot is submission-safe.
        self.enabled = os.environ.get("BOT_METRICS_ENABLED", "0") != "0"
        self.every_n = max(1, int(os.environ.get("BOT_METRICS_EVERY_N", "1")))
        self.path = Path(os.environ.get("BOT_METRICS_LOG", "bot_metrics.jsonl"))
        self._handle = None
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", buffering=1)

    def record(
        self,
        *,
        game: Game,
        strategy_name: str,
        decision: StrategyDecision,
        elapsed_ms: float,
    ) -> None:
        if not self.enabled or self._handle is None:
            return
        round_number = getattr(game.state, "round", -1)
        if round_number % self.every_n != 0:
            return

        features = extract_visible_features(game)
        own_blob_radii = [blob.radius for blob in features.own_blobs]
        rankings = list(getattr(game.state, "rankings", []))
        player_id = game.state.me.player_id

        payload = {
            "timestamp": time.time(),
            "strategy": strategy_name,
            "round": round_number,
            "rank_position": rankings.index(player_id) + 1
            if player_id in rankings
            else None,
            "rankings": rankings,
            "alive": game.state.me.alive,
            "my_player_id": player_id,
            "my_radius": game.state.me.radius,
            "my_mass": game.state.me.radius * game.state.me.radius,
            "my_blob_count": len(own_blob_radii),
            "largest_my_blob_radius": max(own_blob_radii) if own_blob_radii else None,
            "visible_food_count": len(game.state.visible_food),
            "visible_blob_count": len(game.state.visible_blobs),
            "visible_virus_count": len(game.state.visible_viruses),
            "predator_count": len(features.predators),
            "prey_count": len(features.prey),
            "neutral_count": len(features.neutral),
            "nearest_food_distance": features.nearest_food_distance,
            "nearest_predator_distance": features.nearest_predator.distance
            if features.nearest_predator
            else None,
            "nearest_predator_margin": features.nearest_predator.danger_margin
            if features.nearest_predator
            else None,
            "nearest_prey_distance": features.nearest_prey.distance
            if features.nearest_prey
            else None,
            "nearest_virus_distance": features.nearest_virus_distance,
            "decision_direction_x": decision.direction[0],
            "decision_direction_y": decision.direction[1],
            "decision_split": decision.split,
            "decision_target_kind": decision.target_kind,
            "decision_target_id": decision.target_id,
            "decision_reason": decision.reason,
            "decision_score": decision.score,
            "decision_diagnostics": decision.diagnostics,
            "decision_elapsed_ms": elapsed_ms,
        }
        self._handle.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
