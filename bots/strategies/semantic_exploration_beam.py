from __future__ import annotations

"""Semantic potential with continuously scored movement-split candidates.

The production semantic strategy only splits toward visible prey.  This
experimental policy keeps the same bounded scorer and adds split variants of
the current route and nearest-virus route.  They are ordinary candidates: no
mass, round, or rank threshold forces a mode switch.
"""

from dataclasses import replace
import math
import os

from lib.config.arena import MAX_BLOB_COUNT
from lib.config.player import (
    SAME_PLAYER_OVERLAP_EPSILON,
    SPLIT_EJECT_SPEED,
    SPLIT_MIN_MASS,
)
from lib.models.blob_model import BlobModel
from lib.models.virus_model import VirusModel
from strategies.base import StrategyContext
from strategies.features import can_consume_virus, normalise, player_speed
from strategies.semantic_potential import (
    DirectionCandidate,
    PotentialScore,
    SQRT2,
    SemanticPotentialStrategy,
    _project_action_blobs,
)


EXPLORATION_UNSEEN_SCALE = 12.0
EXPLORATION_MASS_SCALE = 24.0
EXPLORATION_PAYBACK_TURNS = 24.0
REFERENCE_PLAYER_COUNT = 8
RIVAL_EVIDENCE_DECAY_TURNS = 240.0


class SemanticExplorationBeamStrategy(SemanticPotentialStrategy):
    """Compare a tiny movement-split beam with every existing semantic action."""

    name = "semantic_exploration_beam"

    def __init__(self) -> None:
        super().__init__()
        self._unseen_enemy_turns = 0
        self._exploration_pressure = 0.0
        self._rival_mass_lower_bound = 0.0
        self._last_evidence_round: int | None = None
        self._exploration_weight = float(
            os.environ.get("BOT_EXPLORATION_SPLIT_WEIGHT", "0.05")
        )
        self._fragmentation_cost_weight = float(
            os.environ.get("BOT_EXPLORATION_SPLIT_COST", "0.5")
        )

    def choose(self, context: StrategyContext):
        state = context.game.state
        own = tuple(state.me.blobs.values())
        if state.visible_blobs:
            self._unseen_enemy_turns = 0
        else:
            self._unseen_enemy_turns += 1

        rankings = tuple(int(player_id) for player_id in state.rankings)
        try:
            rank_index = rankings.index(int(state.me.player_id))
        except ValueError:
            rank_index = len(rankings)
        rank_signal = 1.0 - min(
            1.0,
            rank_index / max(1, len(rankings) - 1),
        )
        unseen_signal = 1.0 - math.exp(
            -self._unseen_enemy_turns / EXPLORATION_UNSEEN_SCALE
        )
        own_mass = sum(blob.radius * blob.radius for blob in own)
        current_round = int(state.round)
        elapsed = (
            0
            if self._last_evidence_round is None
            else max(0, current_round - self._last_evidence_round)
        )
        self._last_evidence_round = current_round
        self._rival_mass_lower_bound *= math.exp(
            -elapsed / RIVAL_EVIDENCE_DECAY_TURNS
        )

        # Rankings expose only order, not masses.  When we are not first, our
        # own mass is nevertheless a proven lower bound for at least one rival.
        # Visible fragments provide another (possibly partial) lower bound.
        # Retaining and smoothly decaying that evidence prevents a newly first
        # player from assuming that splitting its largest blob in half is safe.
        visible_player_masses: dict[int, float] = {}
        for enemy in state.visible_blobs:
            player_id = int(enemy.player_id)
            visible_player_masses[player_id] = (
                visible_player_masses.get(player_id, 0.0)
                + enemy.radius * enemy.radius
            )
        visible_rival_bound = max(visible_player_masses.values(), default=0.0)
        ranking_rival_bound = own_mass if rank_index > 0 else 0.0
        self._rival_mass_lower_bound = max(
            self._rival_mass_lower_bound,
            visible_rival_bound,
            ranking_rival_bound,
        )
        mass_signal = own_mass / (own_mass + EXPLORATION_MASS_SCALE)
        remaining_turns = max(0.0, float(state.max_rounds) - float(state.round))
        payback_signal = 1.0 - math.exp(
            -remaining_turns / EXPLORATION_PAYBACK_TURNS
        )
        opponent_signal = min(
            1.0,
            max(0, len(rankings) - 1) / (REFERENCE_PLAYER_COUNT - 1),
        )
        self._exploration_pressure = (
            rank_signal
            * unseen_signal
            * mass_signal
            * payback_signal
            * opponent_signal
        )
        return super().choose(context)

    def _candidates(self, **kwargs) -> tuple[DirectionCandidate, ...]:
        base = super()._candidates(**kwargs)
        own: tuple[BlobModel, ...] = kwargs["own"]
        enemies = kwargs["enemies"]
        split_viruses = self._split_virus_candidates(
            own=own,
            nearest_viruses=kwargs["nearest_viruses"],
            arena_size=kwargs["arena_size"],
        )
        if (
            enemies
            or len(own) >= MAX_BLOB_COUNT
            or not any(
                blob.radius * blob.radius >= SPLIT_MIN_MASS for blob in own
            )
        ):
            return (*base, *split_viruses)

        # An explicit virus-harvest split already represents the useful
        # movement split in this direction and carries a concrete mass target.
        # Do not add a duplicate blind-exploration action beside it.
        if split_viruses:
            return (*base, *split_viruses)

        routes = tuple(candidate for candidate in base if not candidate.split)
        route = next(
            (candidate for candidate in routes if candidate.family == "nearest_virus"),
            None,
        )
        if route is None:
            route = next(
                (
                    candidate
                    for candidate in routes
                    if candidate.family in {"continue", "boundary_recovery"}
                ),
                routes[0] if routes else None,
            )
        if route is None:
            return base
        return (
            *base,
            DirectionCandidate(
                family=f"explore_split_{route.family}",
                direction=route.direction,
                target_kind="exploration",
                split=True,
                split_depth=1,
            ),
        )

    def _split_virus_candidates(
        self,
        *,
        own: tuple[BlobModel, ...],
        nearest_viruses: tuple[tuple[VirusModel, BlobModel], ...],
        arena_size: float,
    ) -> tuple[DirectionCandidate, ...]:
        """Add at most two executable split routes toward visible viruses.

        Expected-final-mass wins often use a split to shorten virus contact,
        then harvest the resulting visible virus chain.  The base semantic
        strategy can score that chain but previously never proposed the split
        whenever an enemy was visible.  Projecting just the two existing virus
        routes keeps the candidate set bounded and rejects splits whose
        children are too small to consume the target.
        """

        if len(own) >= MAX_BLOB_COUNT or not any(
            blob.radius * blob.radius >= SPLIT_MIN_MASS for blob in own
        ):
            return ()

        candidates: list[DirectionCandidate] = []
        for rank, (virus, source) in enumerate(nearest_viruses[:2]):
            direction = normalise(
                (virus.pos[0] - source.pos[0], virus.pos[1] - source.pos[1])
            )
            if direction == (0.0, 0.0):
                direction = self._last_direction
            projected = _project_action_blobs(
                own,
                direction,
                split=True,
                arena_size=arena_size,
            )
            consumers = tuple(
                blob
                for blob in projected
                if can_consume_virus(blob.radius, virus.radius)
            )
            if not consumers:
                continue
            contact_turns = min(
                1.0
                + max(0.0, math.dist(blob.pos, virus.pos) - blob.radius)
                / max(player_speed(blob.radius), 1.0e-9)
                for blob in consumers
            )
            candidates.append(
                DirectionCandidate(
                    family=(
                        "split_nearest_virus"
                        if rank == 0
                        else "split_second_virus"
                    ),
                    direction=direction,
                    target_kind="virus",
                    target_id=str(virus.virus_id),
                    target_pos=virus.pos,
                    split=True,
                    split_depth=1,
                    contact_turns=contact_turns,
                )
            )
        return tuple(candidates)

    def _score_candidate(self, **kwargs) -> PotentialScore:
        score = super()._score_candidate(**kwargs)
        candidate: DirectionCandidate = kwargs["candidate"]
        if not candidate.family.startswith("explore_split_"):
            return score

        mobility, anchor_loss, post_split_anchor = _split_mobility_profile(
            kwargs["own"]
        )
        mobility_value = (
            self._exploration_weight
            * self._exploration_pressure
            * mobility
        )
        fragmentation_cost = (
            self._fragmentation_cost_weight
            * anchor_loss
            * (1.0 - 0.65 * self._exploration_pressure)
        )
        rival_exposure_cost = max(
            0.0,
            self._rival_mass_lower_bound - post_split_anchor,
        )
        exploration_value = (
            mobility_value - fragmentation_cost - rival_exposure_cost
        )
        return replace(
            score,
            total=score.total + exploration_value,
            intent=score.intent + exploration_value,
        )


def _split_mobility_profile(
    own: tuple[BlobModel, ...],
) -> tuple[float, float, float]:
    """Estimate split mobility analytically without another physics projection."""

    total_mass = sum(blob.radius * blob.radius for blob in own)
    if total_mass <= 0.0:
        return (0.0, 0.0, 0.0)
    remaining_slots = MAX_BLOB_COUNT - len(own)
    mobility = 0.0
    post_split_masses: list[float] = []
    for blob in own:
        mass = blob.radius * blob.radius
        if remaining_slots <= 0 or mass < SPLIT_MIN_MASS:
            post_split_masses.append(mass)
            continue
        remaining_slots -= 1
        child_radius = blob.radius / SQRT2
        child_speed = player_speed(child_radius)
        normal_speed = player_speed(blob.radius)
        launch = (
            2.0 * child_radius
            + SAME_PLAYER_OVERLAP_EPSILON
            + child_speed
            + SPLIT_EJECT_SPEED
        )
        centroid_advance = 0.5 * (child_speed + launch)
        mobility += mass * max(
            0.0,
            centroid_advance - normal_speed + child_speed - normal_speed,
        )
        post_split_masses.extend((mass * 0.5, mass * 0.5))
    anchor_loss = max(blob.radius * blob.radius for blob in own) - max(
        post_split_masses
    )
    return (
        mobility,
        max(0.0, anchor_loss),
        max(post_split_masses),
    )
