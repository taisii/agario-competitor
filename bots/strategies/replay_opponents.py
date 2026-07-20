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
    1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 14, 15, 16, 17, 18, 21, 22, 24,
    26, 28, 29, 30, 31, 33, 34, 35, 37, 38, 39, 44, 47, 49, 51, 53,
    55, 58, 59, 63, 64, 68, 72, 78, 79, 80, 81, 85, 87, 88, 89, 90, 94,
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

# Random sparring prioritizes coverage over validation status.  Every opponent seen
# in the official replay cohort remains eligible; custom policies are used
# where available and the immutable fitted profile is the fallback.
RANDOM_REPLAY_TEAM_IDS = OBSERVED_REPLAY_TEAM_IDS

# Exact seven-opponent cohorts observed alongside team 73 in the latest 30
# successful Submission #53 matches.  Each row is ordered by original player
# slot after removing team 73.  The cloud fixture selects one complete row per
# local trial instead of sampling seven unrelated teams with replacement.
OFFICIAL_CLOUD_LINEUPS = (
    (53, 22, 35, 49, 2, 24, 64),
    (7, 85, 17, 2, 72, 1, 24),
    (31, 94, 22, 33, 72, 49, 10),
    (64, 85, 37, 90, 9, 30, 59),
    (38, 31, 24, 72, 1, 59, 4),
    (90, 9, 88, 30, 53, 33, 47),
    (30, 94, 88, 37, 17, 1, 2),
    (1, 30, 72, 24, 59, 22, 35),
    (24, 47, 1, 64, 33, 17, 9),
    (85, 35, 14, 72, 30, 22, 1),
    (87, 10, 49, 24, 64, 37, 59),
    (49, 47, 37, 85, 35, 90, 64),
    (89, 94, 22, 7, 59, 10, 53),
    (89, 59, 31, 22, 33, 10, 94),
    (33, 4, 35, 9, 17, 94, 22),
    (89, 4, 88, 37, 64, 9, 14),
    (89, 59, 47, 4, 94, 31, 35),
    (89, 87, 37, 24, 64, 35, 94),
    (89, 31, 49, 90, 64, 85, 53),
    (89, 47, 90, 88, 59, 87, 94),
    (89, 33, 90, 49, 38, 30, 59),
    (89, 10, 30, 24, 47, 22, 87),
    (89, 9, 31, 2, 94, 87, 49),
    (89, 59, 38, 9, 2, 37, 49),
    (89, 53, 64, 88, 90, 22, 10),
    (4, 31, 24, 87, 10, 22, 37),
    (17, 4, 15, 49, 31, 64, 53),
    (49, 85, 7, 10, 47, 38, 89),
    (10, 47, 14, 59, 24, 30, 49),
    (15, 72, 9, 85, 4, 47, 33),
)

# These four opponents inflicted 461.992 / 738.688 (62.54%) of team 73's
# player-to-player mass loss in the latest 30 successful Submission #53
# replays.  Their fitted replay profiles all fail the held-out imitation gate,
# so the exact-lineup fixture materially understates cloud pressure.  The
# pressure fixture retains the observed cohorts but substitutes a validated
# strong policy for these empirically dangerous, poorly reproduced teams.
OFFICIAL_CLOUD_HIGH_DAMAGE_TEAM_IDS = frozenset({1, 24, 35, 85})

# Use every team that inflicted at least 20.0 mass on team 73 in the cohort for
# the calibrated pressure fixture.  This yields 2.67 strong opponents per
# observed lineup on average: materially harder than the failed imitation
# fixture, while preserving weaker/non-attacking population members.  The
# broader 5.0-mass tier (3.70 strong opponents/lineup) overproduced loss mass.
OFFICIAL_CLOUD_PRESSURE_TEAM_IDS = frozenset(
    {1, 9, 14, 15, 24, 35, 37, 38, 47, 49, 85, 88}
)

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
    team_ids: tuple[int, ...] = RANDOM_REPLAY_TEAM_IDS,
) -> int:
    """Select a replay opponent reproducibly for one benchmark slot."""

    if not team_ids:
        raise ValueError("Random replay opponent requires at least one team")
    seed = base_seed ^ ((trial + 1) * 0x9E3779B1) ^ ((player_id + 1) * 0x85EBCA77)
    return random.Random(seed).choice(team_ids)


def select_cloud_replay_team_id(
    *,
    player_id: int,
    target_slot: int,
    base_seed: int,
    trial: int,
    lineups: tuple[tuple[int, ...], ...] = OFFICIAL_CLOUD_LINEUPS,
) -> int:
    """Select one member of an empirically observed seven-team cohort.

    Every opponent process receives the same seed and trial.  They therefore
    choose the same official cohort, then map its seven distinct teams onto the
    seven local opponent slots without replacement.  The small deterministic
    shuffle avoids coupling one official team to one engine player slot.
    """

    if not 0 <= target_slot < 8:
        raise ValueError(f"target_slot must be in [0, 7], found {target_slot}")
    if player_id == target_slot:
        raise ValueError("cloud replay opponent cannot occupy the target slot")
    if not 0 <= player_id < 8:
        raise ValueError(f"player_id must be in [0, 7], found {player_id}")
    if not lineups or any(len(lineup) != 7 for lineup in lineups):
        raise ValueError("cloud lineups must contain seven teams each")

    seed = base_seed ^ ((trial + 1) * 0x9E3779B1)
    rng = random.Random(seed)
    lineup = list(lineups[rng.randrange(len(lineups))])
    rng.shuffle(lineup)
    opponent_slots = tuple(slot for slot in range(8) if slot != target_slot)
    return lineup[opponent_slots.index(player_id)]


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
            self._selected = create_replay_candidate(team_id)
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


class CloudReplayOpponent:
    """Lazily construct the clone assigned by one observed cloud cohort."""

    name = "cloud_replay_opponent"

    def __init__(
        self,
        *,
        target_slot: int,
        base_seed: int,
        trial: int,
        candidate_factory: Callable[[int], Strategy] | None = None,
        on_selected: Callable[[str], None] | None = None,
    ) -> None:
        self._target_slot = target_slot
        self._base_seed = base_seed
        self._trial = trial
        self._candidate_factory = candidate_factory
        self._on_selected = on_selected
        self._selected: Strategy | None = None

    def choose(self, context):
        if self._selected is None:
            team_id = select_cloud_replay_team_id(
                player_id=int(context.game.state.me.player_id),
                target_slot=self._target_slot,
                base_seed=self._base_seed,
                trial=self._trial,
            )
            factory = self._candidate_factory or create_replay_candidate
            self._selected = factory(team_id)
            self.name = self._selected.name
            if self._on_selected is not None:
                self._on_selected(self.name)
        return self._selected.choose(context)


def create_cloud_replay_opponent(
    *,
    target_slot: int,
    on_selected: Callable[[str], None] | None = None,
) -> Strategy:
    """Construct a reproducible opponent from an observed cloud cohort."""

    seed = int(os.environ.get("BOT_RANDOM_SEED", "0"))
    trial = int(os.environ.get("BOT_BENCHMARK_TRIAL", "0"))
    return CloudReplayOpponent(
        target_slot=target_slot,
        base_seed=seed,
        trial=trial,
        on_selected=on_selected,
    )


def create_cloud_pressure_opponent(
    *,
    target_slot: int,
    on_selected: Callable[[str], None] | None = None,
) -> Strategy:
    """Construct an official-cohort fixture calibrated for cloud damage.

    Replay-derived policies remain preferable when their behaviour is usable.
    For the small empirically dominant damage set whose imitation profiles fail
    validation, ReplayDominance supplies the missing pursuit/split pressure.
    This is an evaluation fixture only; it is never a submission strategy.
    """

    from strategies.receding_horizon import ReplayDominanceStrategy

    def candidate_factory(team_id: int) -> Strategy:
        if team_id in OFFICIAL_CLOUD_PRESSURE_TEAM_IDS:
            return ReplayDominanceStrategy()
        return create_replay_candidate(team_id)

    seed = int(os.environ.get("BOT_RANDOM_SEED", "0"))
    trial = int(os.environ.get("BOT_BENCHMARK_TRIAL", "0"))
    return CloudReplayOpponent(
        target_slot=target_slot,
        base_seed=seed,
        trial=trial,
        candidate_factory=candidate_factory,
        on_selected=on_selected,
    )
