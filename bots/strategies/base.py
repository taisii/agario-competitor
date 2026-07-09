from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from helper.game import Game
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.penguin_model import DirectionModel


@dataclass(frozen=True)
class StrategyContext:
    game: Game
    query: QueryMovePlayer


@dataclass(frozen=True)
class StrategyDecision:
    direction: tuple[float, float]
    split: bool = False
    target_kind: str = "none"
    target_id: str | None = None
    reason: str = ""
    score: float | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_move(self, player_id: int) -> MovePlayer:
        dx, dy = self.direction
        return MovePlayer(
            player_id=player_id,
            direction=DirectionModel(x=dx, y=dy),
            split=self.split,
        )


class Strategy(Protocol):
    name: str

    def choose(self, context: StrategyContext) -> StrategyDecision:
        raise NotImplementedError
