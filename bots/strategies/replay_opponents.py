"""Catalog and factories for official replay-derived opponents.

Replay clones are benchmark fixtures, not candidate bot strategies.  Keeping
them in their own catalog prevents the policy selector from presenting dozens
of opponent implementations as interchangeable product strategies.
"""

from __future__ import annotations

import importlib
import os
import random
import time
from dataclasses import dataclass
from typing import Callable

from strategies.base import Strategy


@dataclass(frozen=True)
class ReplayOpponentSpec:
    """One reproducible policy fitted to an official opponent team."""

    team_id: int
    factory_path: str
    factory_argument: int | None = None

    @property
    def name(self) -> str:
        return f"replay_team_{self.team_id}"

    def create(self) -> Strategy:
        module_name, separator, attribute_name = self.factory_path.partition(":")
        if not separator:
            raise RuntimeError(f"Invalid opponent factory path: {self.factory_path!r}")
        module = importlib.import_module(module_name)
        factory: Callable[..., Strategy] = getattr(module, attribute_name)
        opponent = (
            factory()
            if self.factory_argument is None
            else factory(self.factory_argument)
        )
        if opponent.name != self.name:
            raise RuntimeError(
                f"Replay opponent catalog name {self.name!r} does not match "
                f"{self.factory_path} name {opponent.name!r}"
            )
        return opponent


OBSERVED_REPLAY_TEAM_IDS = (
    1, 2, 3, 4, 5, 6, 9, 10, 12, 13, 14, 15, 16, 17, 21, 22, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 34, 35, 38, 39, 44, 48, 49, 51, 53,
    55, 56, 58, 59, 63, 68, 75, 77,
)

CUSTOM_REPLAY_TEAM_IDS = frozenset(
    {1, 4, 9, 13, 16, 22, 25, 29, 31, 35, 39, 44, 51, 53, 56, 68}
)
PROFILED_OPPONENT_TEAM_IDS = frozenset({2, 6, 27, 30, 38, 75})

# A high-pressure local panel, selected from 29 completed Submission #4 and
# Submission #4 Extra official matches.  Each team has at least three matches,
# mean final rank <= 4, top-three rate >= 1/3, mean kills >= 4, and mean split
# count >= 1.  The selection excludes food-only and passive opponents even
# when a small sample happens to give them a favourable placement.
REPLAY_TEAM_IDS = (21,)

# Teams that pass the official-results gate before clone runtime measurements.
# The evaluator tests these candidates and promotes only the measured subset
# represented by REPLAY_TEAM_IDS.
REPLAY_STRENGTH_CANDIDATE_TEAM_IDS = (
    1, 3, 4, 9, 10, 21, 24, 31, 35, 49, 59
)


def _spec(team_id: int) -> ReplayOpponentSpec:
    if team_id in CUSTOM_REPLAY_TEAM_IDS:
        return ReplayOpponentSpec(
            team_id,
            f"strategies.replay_team_{team_id}:ReplayTeam{team_id}Strategy",
        )
    if team_id in PROFILED_OPPONENT_TEAM_IDS:
        return ReplayOpponentSpec(
            team_id,
            "strategies.replay_opponent_policies:create_profiled_opponent_strategy",
            factory_argument=team_id,
        )
    return ReplayOpponentSpec(
        team_id,
        "strategies.replay_imitation:create_profiled_replay_strategy",
        factory_argument=team_id,
    )


_OBSERVED_REPLAY_OPPONENT_SPECS = {
    team_id: _spec(team_id) for team_id in OBSERVED_REPLAY_TEAM_IDS
}
REPLAY_OPPONENT_SPECS = {
    team_id: _OBSERVED_REPLAY_OPPONENT_SPECS[team_id]
    for team_id in REPLAY_TEAM_IDS
}


def replay_opponent_name(team_id: int) -> str:
    """Return the stable entry name for a known official opponent."""

    try:
        return REPLAY_OPPONENT_SPECS[team_id].name
    except KeyError as exc:
        available = ", ".join(str(value) for value in REPLAY_TEAM_IDS)
        raise ValueError(f"Unknown replay opponent team {team_id!r}. Available: {available}") from exc


def create_replay_opponent(team_id: int) -> Strategy:
    """Construct one official replay opponent by stable team ID."""

    try:
        return REPLAY_OPPONENT_SPECS[team_id].create()
    except KeyError as exc:
        available = ", ".join(str(value) for value in REPLAY_TEAM_IDS)
        raise ValueError(f"Unknown replay opponent team {team_id!r}. Available: {available}") from exc


def create_replay_candidate(team_id: int) -> Strategy:
    """Construct an archived or active replay clone for offline evaluation only."""

    try:
        return _OBSERVED_REPLAY_OPPONENT_SPECS[team_id].create()
    except KeyError as exc:
        available = ", ".join(str(value) for value in OBSERVED_REPLAY_TEAM_IDS)
        raise ValueError(
            f"Unknown replay candidate team {team_id!r}. Available: {available}"
        ) from exc


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


class RandomReplayOpponent:
    """Lazily choose one replay clone after the engine assigns a player slot."""

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
            self._selected = create_replay_opponent(team_id)
            self.name = self._selected.name
            if self._on_selected is not None:
                self._on_selected(self.name)
        return self._selected.choose(context)


def create_random_replay_opponent(
    *,
    on_selected: Callable[[str], None] | None = None,
) -> Strategy:
    """Construct a reproducibly seeded replay-opponent sampler."""

    seed = int(os.environ.get("BOT_RANDOM_SEED", os.getpid() ^ time.time_ns()))
    trial = int(os.environ.get("BOT_BENCHMARK_TRIAL", "0"))
    return RandomReplayOpponent(
        base_seed=seed,
        trial=trial,
        on_selected=on_selected,
    )
