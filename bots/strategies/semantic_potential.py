from __future__ import annotations

"""Choose among a fixed set of meaningful directions using a potential field.

The strategy deliberately does not search arbitrary angles or opponent trees.
It creates a bounded set of semantic candidates, projects each candidate
through the same cheap field, and chooses the best safe result. Candidate generation
answers *where could we intentionally go?* while the field answers *which of
those intentions fits the wider visible position?*
"""

from dataclasses import dataclass, replace
import math
import os

from lib.config.arena import ARENA_SIZE, MAX_BLOB_COUNT
from lib.config.player import (
    EAT_SIZE_RATIO,
    FOOD_RADIUS,
    MASS_DECAY_RATE,
    SAME_PLAYER_OVERLAP_EPSILON,
    SPLIT_EJECT_SPEED,
    SPLIT_MIN_MASS,
    STARTING_RADIUS,
)
from lib.models.blob_model import BlobModel, VisibleBlobModel
from lib.models.food_model import FoodModel
from lib.models.virus_model import VirusModel
from strategies.base import StrategyContext, StrategyDecision
from strategies.features import (
    can_consume_virus,
    can_eat_player_blob,
    normalise,
    player_speed,
    squared_distance,
)


SQRT2 = math.sqrt(2.0)
MACRO_HORIZON = 5.0
WALL_INFLUENCE = 8.0
TARGET_MATCH_DISTANCE = 0.75
TARGET_SWITCH_MARGIN = 0.01
REFERENCE_VIRUS_RADIUS = 1.5
MULTI_SPLIT_MIN_MASS = 16.0
MAX_CAPTURE_SPLIT_DEPTH = 3
SAFETY_RESERVE = 3.0
SAFETY_MARGIN_SLACK = 0.35
DIRECTIONAL_HORIZON = 4.0
DIRECTIONAL_HALF_ANGLE = math.radians(32.0)
DIRECTIONAL_COSINE_LIMIT = math.cos(DIRECTIONAL_HALF_ANGLE)
FUTURE_MASS_DISCOUNT = 0.70
MAX_DIRECTIONAL_FOODS = 8
BOUNDARY_STALL_PROGRESS_RATIO = 0.10
EPSILON = 1.0e-9


@dataclass(frozen=True, slots=True)
class DirectionCandidate:
    """One movement intention; the family remains visible after scoring."""

    family: str
    direction: tuple[float, float]
    target_kind: str
    target_id: str | None = None
    target_pos: tuple[float, float] | None = None
    one_step_target_pos: tuple[float, float] | None = None
    split: bool = False
    split_depth: int = 0
    capture_mass: float = 0.0
    contact_turns: float | None = None


@dataclass(frozen=True, slots=True)
class PotentialScore:
    total: float
    catastrophic: bool
    safety_margin: float
    threat: float
    food: float
    prey: float
    virus: float
    wall: float
    corridor: float
    inertia: float
    intent: float
    secured_enemy_mass: float
    secured_virus_mass: float
    secured_food_mass: float
    own_mass_lost: float
    exact_prey_uplift: float


@dataclass(frozen=True, slots=True)
class DirectionalPotential:
    """Three-turn resource value inside one candidate's reachable fan."""

    food: float
    prey: float
    virus: float

    @property
    def total(self) -> float:
        return self.food + self.prey + self.virus


@dataclass(frozen=True, slots=True)
class OneStepOutcome:
    """Visible one-turn result in the engine's resource/eating order.

    When consecutive observations reveal the selected target's motion, its
    one-step center is projected too. Own split, movement, decay, virus
    contact, food contact, and cross-player eating follow the engine order.
    """

    own: tuple[BlobModel, ...]
    enemies: tuple[VisibleBlobModel, ...]
    foods: tuple[FoodModel, ...]
    viruses: tuple[VirusModel, ...]
    enemy_mass_gained: float
    virus_mass_gained: float
    food_mass_gained: float
    own_mass_lost: float


@dataclass(slots=True)
class _LocalBlob:
    owner_id: int
    blob_id: int
    pos: tuple[float, float]
    radius: float
    team_id: int
    is_own: bool


@dataclass(frozen=True, slots=True)
class TargetMemory:
    """A stationary resource plan tracked by geometry, never transient IDs."""

    kind: str
    pos: tuple[float, float]


@dataclass(frozen=True, slots=True)
class CaptureRoute:
    enemy: VisibleBlobModel
    hunter: BlobModel
    target_pos: tuple[float, float]
    one_step_target_pos: tuple[float, float]
    intercepting: bool
    turns_to_contact: float


@dataclass(frozen=True, slots=True)
class SplitCaptureRoute:
    depth: int
    direction: tuple[float, float]
    target_pos: tuple[float, float]
    one_step_target_pos: tuple[float, float]


class SemanticPotentialStrategy:
    """Rank a bounded semantic action set with a visible-world potential.

    The safety decision is lexicographic: whenever at least one candidate is
    outside every predator's one-step reachable set, catastrophic candidates
    are excluded before utility is compared.  This prevents a rich food field
    from numerically paying for an immediately lost fragment.
    """

    name = "semantic_potential"

    def __init__(self) -> None:
        self._last_direction = (1.0, 0.0)
        self._target_memory: TargetMemory | None = None
        self._enemy_positions: dict[tuple[int, int], tuple[float, float]] = {}
        self._safety_reserve = _environment_nonnegative_float(
            "SEMANTIC_SAFETY_RESERVE",
            SAFETY_RESERVE,
        )
        self._safety_margin_slack = _environment_nonnegative_float(
            "SEMANTIC_SAFETY_MARGIN_SLACK",
            SAFETY_MARGIN_SLACK,
        )
        self._exact_prey_outcome = _environment_enabled(
            "SEMANTIC_EXACT_PREY_OUTCOME",
            default=True,
        )
        self._exact_prey_min_mass = _environment_nonnegative_float(
            "SEMANTIC_EXACT_PREY_MIN_MASS",
            20.0,
        )
        self._exact_prey_ahead_only = _environment_enabled(
            "SEMANTIC_EXACT_PREY_AHEAD_ONLY",
            default=True,
        )
        self._exact_prey_requires_safety_reserve = _environment_enabled(
            "SEMANTIC_EXACT_PREY_REQUIRES_SAFETY_RESERVE",
            default=True,
        )
        self._exact_prey_max_turn_degrees = min(
            180.0,
            _environment_nonnegative_float(
                "SEMANTIC_EXACT_PREY_MAX_TURN_DEGREES",
                90.0,
            ),
        )
        self._exact_prey_preserve_split_choice = _environment_enabled(
            "SEMANTIC_EXACT_PREY_PRESERVE_SPLIT_CHOICE",
            default=True,
        )
        self._exact_prey_outcome_active = False

    def choose(self, context: StrategyContext) -> StrategyDecision:
        state = context.game.state
        own = tuple(state.me.blobs.values())
        if not own:
            return StrategyDecision(
                direction=self._last_direction,
                reason="dead_fallback",
            )

        arena_size = float(state.map.size or ARENA_SIZE)
        all_foods = tuple(state.visible_food)
        all_viruses = tuple(state.visible_viruses)
        enemies = tuple(state.visible_blobs)
        current_safety_margin = _minimum_threat_margin(own, enemies)
        rankings = tuple(getattr(state, "rankings", ()))
        is_rank_one = bool(
            rankings
            and int(rankings[0]) == int(state.me.player_id)
        )
        total_mass = sum(blob.radius * blob.radius for blob in own)
        self._exact_prey_outcome_active = (
            self._exact_prey_outcome
            and total_mass + EPSILON >= self._exact_prey_min_mass
            and (not self._exact_prey_ahead_only or is_rank_one)
            and (
                not self._exact_prey_requires_safety_reserve
                or current_safety_margin + EPSILON >= self._safety_reserve
            )
        )
        # A point in a corner can be visible yet physically unreachable: blob
        # centres are clamped to [radius, arena-radius], while eating requires
        # the point to lie inside the blob.  Routing such a point makes a blob
        # push into the hard boundary forever.  Keep the raw collections for
        # exact one-step split events, but exclude impossible stationary routes
        # from target memory and directional potential.
        foods = _reachable_stationary_resources(
            own,
            all_foods,
            arena_size=arena_size,
        )
        viruses = _reachable_stationary_resources(
            own,
            all_viruses,
            arena_size=arena_size,
        )
        previous_directions = _observed_enemy_directions(
            enemies,
            previous_positions=self._enemy_positions,
        )
        self._enemy_positions = {
            (int(enemy.player_id), enemy.blob_id): enemy.pos for enemy in enemies
        }
        self._target_memory = _refresh_target_memory(
            self._target_memory,
            foods=foods,
            viruses=viruses,
        )
        mass_phase = _mass_phase(own, viruses)
        escape_direction = _escape_direction(
            own,
            enemies,
        )
        routing_blob_ids = frozenset(
            blob.blob_id for blob in _routing_blobs(own, limit=4)
        )
        nearest_foods = _nearest_items_with_sources(
            own,
            foods,
            limit=MAX_DIRECTIONAL_FOODS,
        )
        nearest_viruses = _nearest_items_with_sources(
            own,
            viruses,
            limit=2,
        )
        # Food is individually tiny and the strategy already has two concrete
        # food targets.  A bounded nearby sample retains the density signal
        # without rescanning every visible pellet for every candidate.
        directional_foods = tuple(item for item, _ in nearest_foods)
        candidates = self._candidates(
            own=own,
            foods=foods,
            viruses=viruses,
            enemies=enemies,
            arena_size=arena_size,
            previous_directions=previous_directions,
            escape_direction=escape_direction,
            nearest_foods=nearest_foods[:2],
            nearest_viruses=nearest_viruses,
            target_memory=self._target_memory,
            fallback_direction=self._last_direction,
        )
        scored = tuple(
            (
                candidate,
                self._score_candidate(
                    candidate=candidate,
                    own=own,
                    foods=directional_foods,
                    all_foods=all_foods,
                    viruses=viruses,
                    all_viruses=all_viruses,
                    enemies=enemies,
                    arena_size=arena_size,
                    routing_blob_ids=routing_blob_ids,
                    escape_direction=escape_direction,
                    previous_directions=previous_directions,
                    previous_own_direction=self._last_direction,
                ),
            )
            for candidate in candidates
        )
        scored = self._refine_scored_candidates(
            scored=scored,
            own=own,
            all_foods=all_foods,
            all_viruses=all_viruses,
            enemies=enemies,
            arena_size=arena_size,
            previous_directions=previous_directions,
        )
        selected, score, safe = self._select_scored_candidate(
            scored,
            current_safety_margin=current_safety_margin,
        )
        selected, score, baseline_split, split_choice_preserved = (
            self._preserve_exact_prey_split_choice(
                scored,
                selected=selected,
                score=score,
                current_safety_margin=current_safety_margin,
            )
        )
        self._last_direction = selected.direction
        if (
            selected.target_kind in {"food", "virus"}
            and selected.target_pos is not None
        ):
            self._target_memory = TargetMemory(
                kind=selected.target_kind,
                pos=selected.target_pos,
            )
        elif selected.target_kind in {"escape", "wall", "prey"}:
            self._target_memory = None

        diagnostics = {
            "candidate_count": len(candidates),
            "safe_candidate_count": len(safe),
            "unreachable_stationary_resources": (
                len(all_foods)
                + len(all_viruses)
                - len(foods)
                - len(viruses)
            ),
            "current_safety_margin": _finite_or_none(current_safety_margin),
            "selected_safety_margin": _finite_or_none(score.safety_margin),
            "safety_reserve": self._safety_reserve,
            "safety_margin_slack": self._safety_margin_slack,
            "exact_prey_outcome_enabled": self._exact_prey_outcome,
            "exact_prey_outcome_active": self._exact_prey_outcome_active,
            "exact_prey_min_mass": self._exact_prey_min_mass,
            "exact_prey_ahead_only": self._exact_prey_ahead_only,
            "exact_prey_requires_safety_reserve": (
                self._exact_prey_requires_safety_reserve
            ),
            "exact_prey_max_turn_degrees": self._exact_prey_max_turn_degrees,
            "exact_prey_preserve_split_choice": (
                self._exact_prey_preserve_split_choice
            ),
            "exact_prey_baseline_split": baseline_split,
            "exact_prey_split_choice_preserved": split_choice_preserved,
            "exact_prey_selected_uplift": score.exact_prey_uplift,
            "mass_phase": mass_phase,
            "split_depth": selected.split_depth,
            "selected_contact_turns": selected.contact_turns,
            "candidate_scores": {
                candidate.family: round(candidate_score.total, 6)
                for candidate, candidate_score in scored
            },
            "selected_components": {
                "threat": score.threat,
                "food": score.food,
                "prey": score.prey,
                "virus": score.virus,
                "wall": score.wall,
                "corridor": score.corridor,
                "inertia": score.inertia,
                "intent": score.intent,
            },
            "secured_one_step_mass": {
                "enemy": score.secured_enemy_mass,
                "virus": score.secured_virus_mass,
                "food": score.secured_food_mass,
                "lost": score.own_mass_lost,
            },
        }
        diagnostics.update(self._decision_diagnostics(selected))
        return StrategyDecision(
            direction=selected.direction,
            split=selected.split,
            target_kind=selected.target_kind,
            target_id=selected.target_id,
            reason=selected.family,
            score=score.total,
            diagnostics=diagnostics,
        )

    def _select_scored_candidate(
        self,
        scored: tuple[tuple[DirectionCandidate, PotentialScore], ...],
        *,
        current_safety_margin: float,
    ) -> tuple[
        DirectionCandidate,
        PotentialScore,
        tuple[tuple[DirectionCandidate, PotentialScore], ...],
    ]:
        safe = tuple(item for item in scored if not item[1].catastrophic)
        if safe:
            if current_safety_margin < self._safety_reserve:
                non_split_safe = (
                    tuple(item for item in safe if not item[0].split) or safe
                )
                best_margin = max(item[1].safety_margin for item in non_split_safe)
                selection_pool = tuple(
                    item
                    for item in non_split_safe
                    if item[1].safety_margin
                    >= best_margin - self._safety_margin_slack
                )
            else:
                reserve_safe = tuple(
                    item
                    for item in safe
                    if item[1].safety_margin >= self._safety_reserve
                )
                selection_pool = reserve_safe or safe
            selected, score = max(
                selection_pool,
                key=lambda item: item[1].total,
            )
            continuing = next(
                (item for item in selection_pool if item[0].family == "continue"),
                None,
            )
            if (
                self._target_memory is not None
                and continuing is not None
                and selected.family != "continue"
                and selected.target_pos != continuing[0].target_pos
                and score.total < continuing[1].total + TARGET_SWITCH_MARGIN
            ):
                selected, score = continuing
            return selected, score, safe

        # Being inside every one-step attack envelope does not make the
        # envelopes equivalent. Maximise retained safety first and use
        # strategic utility only to break an equal-risk tie. Never create
        # additional vulnerable fragments while every option is already
        # inside a predator envelope.
        survival_pool = tuple(item for item in scored if not item[0].split) or scored
        selected, score = max(
            survival_pool,
            key=lambda item: (item[1].threat, item[1].total),
        )
        return selected, score, safe

    def _preserve_exact_prey_split_choice(
        self,
        scored: tuple[tuple[DirectionCandidate, PotentialScore], ...],
        *,
        selected: DirectionCandidate,
        score: PotentialScore,
        current_safety_margin: float,
    ) -> tuple[DirectionCandidate, PotentialScore, bool | None, bool]:
        if (
            not self._exact_prey_preserve_split_choice
            or not any(item[1].exact_prey_uplift > 0.0 for item in scored)
        ):
            return selected, score, None, False

        baseline_scored = tuple(
            (
                candidate,
                replace(
                    candidate_score,
                    total=(
                        candidate_score.total - candidate_score.exact_prey_uplift
                    ),
                    intent=(
                        candidate_score.intent - candidate_score.exact_prey_uplift
                    ),
                ),
            )
            for candidate, candidate_score in scored
        )
        baseline_selected, _, _ = self._select_scored_candidate(
            baseline_scored,
            current_safety_margin=current_safety_margin,
        )
        if baseline_selected.split == selected.split:
            return selected, score, baseline_selected.split, False

        baseline_score = next(
            candidate_score
            for candidate, candidate_score in scored
            if candidate is baseline_selected
        )
        return baseline_selected, baseline_score, baseline_selected.split, True

    def _refine_scored_candidates(
        self,
        *,
        scored: tuple[tuple[DirectionCandidate, PotentialScore], ...],
        own: tuple[BlobModel, ...],
        all_foods: tuple[FoodModel, ...],
        all_viruses: tuple[VirusModel, ...],
        enemies: tuple[VisibleBlobModel, ...],
        arena_size: float,
        previous_directions: dict[tuple[int, int], tuple[float, float]],
    ) -> tuple[tuple[DirectionCandidate, PotentialScore], ...]:
        """Allow bounded planners to refine the submitted one-ply ranking."""

        return scored

    def _decision_diagnostics(
        self,
        selected: DirectionCandidate,
    ) -> dict[str, object]:
        return {}

    def _candidates(
        self,
        *,
        own: tuple[BlobModel, ...],
        foods: tuple[FoodModel, ...],
        viruses: tuple[VirusModel, ...],
        enemies: tuple[VisibleBlobModel, ...],
        arena_size: float,
        previous_directions: dict[tuple[int, int], tuple[float, float]],
        escape_direction: tuple[float, float],
        nearest_foods: tuple[tuple[FoodModel | VirusModel, BlobModel], ...],
        nearest_viruses: tuple[tuple[FoodModel | VirusModel, BlobModel], ...],
        target_memory: TargetMemory | None,
        fallback_direction: tuple[float, float],
    ) -> tuple[DirectionCandidate, ...]:
        continuation = _continuation_candidate(
            own=own,
            foods=foods,
            viruses=viruses,
            memory=target_memory,
            fallback=fallback_direction,
        )
        progress_ratio = _movement_progress_ratio(
            own,
            continuation.direction,
            arena_size=arena_size,
        )
        if progress_ratio < BOUNDARY_STALL_PROGRESS_RATIO:
            # A clipped command is not a meaningful continuation.  Replace it
            # with the direction the same command can actually realise (usually
            # a tangent), or with the arena interior when both axes are blocked.
            # This is intentionally event-gated at a hard stop; proximity to a
            # wall alone still has no cost or avoidance behaviour.
            candidates = [
                DirectionCandidate(
                    family="boundary_recovery",
                    direction=_boundary_recovery_direction(
                        own,
                        continuation.direction,
                        arena_size=arena_size,
                    ),
                    target_kind="boundary",
                )
            ]
        else:
            candidates = [continuation]

        for rank, (food, source) in enumerate(nearest_foods):
            candidates.append(
                DirectionCandidate(
                    family="nearest_food" if rank == 0 else "second_food",
                    direction=_direction_or_fallback(
                        source.pos,
                        food.pos,
                        fallback_direction,
                    ),
                    target_kind="food",
                    target_id=str(food.food_id),
                    target_pos=food.pos,
                )
            )

        for rank, (virus, source) in enumerate(nearest_viruses):
            candidates.append(
                DirectionCandidate(
                    family="nearest_virus" if rank == 0 else "second_virus",
                    direction=_direction_or_fallback(
                        source.pos,
                        virus.pos,
                        fallback_direction,
                    ),
                    target_kind="virus",
                    target_id=str(virus.virus_id),
                    target_pos=virus.pos,
                )
            )

        prey = _best_capture_target(
            own,
            enemies,
            previous_directions=previous_directions,
            arena_size=arena_size,
        )
        if prey is not None:
            candidates.append(
                DirectionCandidate(
                    family=(
                        "intercept_enemy" if prey.intercepting else "capture_enemy"
                    ),
                    direction=_direction_or_fallback(
                        prey.hunter.pos,
                        prey.target_pos,
                        fallback_direction,
                    ),
                    target_kind="prey",
                    target_id=f"{prey.enemy.player_id}:{prey.enemy.blob_id}",
                    target_pos=prey.target_pos,
                    one_step_target_pos=prey.one_step_target_pos,
                    contact_turns=prey.turns_to_contact,
                )
            )

        candidates.extend(
            _split_capture_candidates(
                own=own,
                enemies=enemies,
                fallback=fallback_direction,
                arena_size=arena_size,
                previous_directions=previous_directions,
            )
        )

        wall_direction = _wall_escape_direction(
            own,
            escape_direction,
            arena_size,
        )
        if wall_direction != (0.0, 0.0):
            candidates.append(
                DirectionCandidate(
                    family="wall_avoidance",
                    direction=wall_direction,
                    target_kind="wall",
                )
            )

        if escape_direction != (0.0, 0.0):
            candidates.append(
                DirectionCandidate(
                    family="escape",
                    direction=escape_direction,
                    target_kind="escape",
                )
            )

        # Each semantic slot is bounded independently. Missing world objects
        # remove their slot, and wall deflection exists only during an escape.
        return tuple(candidates)

    def _score_candidate(
        self,
        *,
        candidate: DirectionCandidate,
        own: tuple[BlobModel, ...],
        foods: tuple[FoodModel, ...],
        all_foods: tuple[FoodModel, ...],
        viruses: tuple[VirusModel, ...],
        all_viruses: tuple[VirusModel, ...],
        enemies: tuple[VisibleBlobModel, ...],
        arena_size: float,
        routing_blob_ids: frozenset[int],
        escape_direction: tuple[float, float],
        previous_directions: dict[tuple[int, int], tuple[float, float]],
        previous_own_direction: tuple[float, float],
    ) -> PotentialScore:
        outcome = None
        scoring_enemies = enemies
        scoring_foods = foods
        scoring_viruses = viruses
        exact_non_split_prey = (
            self._exact_prey_outcome_active
            and not candidate.split
            and candidate.target_kind == "prey"
            and candidate.contact_turns is not None
            and candidate.contact_turns <= 1.0 + EPSILON
            and _angle_degrees(candidate.direction, previous_own_direction)
            <= self._exact_prey_max_turn_degrees + EPSILON
        )
        if candidate.split or candidate.target_kind == "virus":
            outcome = _project_one_step_outcome(
                own=own,
                direction=candidate.direction,
                split=candidate.split,
                foods=all_foods,
                viruses=all_viruses,
                enemies=enemies,
                arena_size=arena_size,
                target_id=candidate.target_id,
                target_pos=candidate.one_step_target_pos,
            )
            one_step = outcome.own
            scoring_enemies = outcome.enemies
            scoring_foods = _reachable_stationary_resources(
                one_step,
                outcome.foods,
                arena_size=arena_size,
            )
            scoring_viruses = _reachable_stationary_resources(
                one_step,
                outcome.viruses,
                arena_size=arena_size,
            )
        elif exact_non_split_prey:
            # Keep the ordinary non-split score as the baseline and use the
            # exact transition only as monotone evidence that contact really
            # secures mass.  Replacing ``scoring_enemies`` with the projected
            # world removes the just-eaten prey from directional potential;
            # that can *lower* a non-split candidate and accidentally make an
            # otherwise rejected split win.  The feature is specifically a
            # non-split upside layer, so it must never demote the base route.
            outcome = _project_one_step_outcome(
                own=own,
                direction=candidate.direction,
                split=False,
                foods=all_foods,
                viruses=all_viruses,
                enemies=enemies,
                arena_size=arena_size,
                target_id=candidate.target_id,
                target_pos=candidate.one_step_target_pos,
            )
            one_step = _project_action_blobs(
                own,
                candidate.direction,
                split=False,
                arena_size=arena_size,
            )
        else:
            one_step = _project_action_blobs(
                own,
                candidate.direction,
                split=False,
                arena_size=arena_size,
            )
        routing_own = tuple(blob for blob in own if blob.blob_id in routing_blob_ids)
        directional = _directional_potential(
            own=(one_step if candidate.split else routing_own),
            direction=candidate.direction,
            foods=scoring_foods,
            viruses=scoring_viruses,
            enemies=scoring_enemies,
            previous_directions=previous_directions,
        )
        catastrophic, threat, safety_margin = _threat_potential(
            one_step,
            scoring_enemies,
        )
        food = directional.food
        prey = directional.prey
        virus = directional.virus
        wall = 0.0
        if escape_direction != (0.0, 0.0):
            projected = _project_blobs(
                one_step,
                candidate.direction,
                MACRO_HORIZON - 1.0,
                arena_size,
            )
            wall = _blocked_escape_wall_potential(
                projected,
                escape_direction,
                arena_size,
            )
        corridor = 0.0
        inertia = 0.002 * _dot(candidate.direction, previous_own_direction)
        intent = _target_mass_opportunity(
            candidate=candidate,
            own=one_step,
            foods=(scoring_foods if candidate.split else all_foods),
            viruses=scoring_viruses,
            enemies=scoring_enemies,
        )
        exact_prey_uplift = 0.0
        if outcome is not None:
            if exact_non_split_prey:
                # The route-level opportunity already credits this prey.  An
                # exact contact may raise that estimate, but never lower it or
                # add unrelated incidental food/virus collection twice.
                exact_prey_uplift = max(
                    0.0,
                    outcome.enemy_mass_gained - intent,
                )
                intent += exact_prey_uplift
            else:
                intent += (
                    outcome.enemy_mass_gained
                    + outcome.virus_mass_gained
                    + outcome.food_mass_gained
                    - outcome.own_mass_lost
                )
            if outcome.own_mass_lost > 0.0:
                catastrophic = True
                threat -= outcome.own_mass_lost
        total = threat + food + prey + virus + wall + corridor + inertia + intent
        return PotentialScore(
            total=total,
            catastrophic=catastrophic,
            safety_margin=safety_margin,
            threat=threat,
            food=food,
            prey=prey,
            virus=virus,
            wall=wall,
            corridor=corridor,
            inertia=inertia,
            intent=intent,
            secured_enemy_mass=(0.0 if outcome is None else outcome.enemy_mass_gained),
            secured_virus_mass=(0.0 if outcome is None else outcome.virus_mass_gained),
            secured_food_mass=(0.0 if outcome is None else outcome.food_mass_gained),
            own_mass_lost=(0.0 if outcome is None else outcome.own_mass_lost),
            exact_prey_uplift=exact_prey_uplift,
        )




def _environment_enabled(name: str, *, default: bool = True) -> bool:
    fallback = "1" if default else "0"
    return os.environ.get(name, fallback).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }



def _environment_nonnegative_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) and value >= 0.0 else default



def _nearest_items_with_sources(
    own: tuple[BlobModel, ...],
    items: tuple[FoodModel, ...] | tuple[VirusModel, ...],
    *,
    limit: int,
) -> tuple[tuple[FoodModel | VirusModel, BlobModel], ...]:
    rows: list[tuple[float, int, FoodModel | VirusModel, BlobModel]] = []
    for order, item in enumerate(items):
        source = min(
            own,
            key=lambda blob: (
                squared_distance(blob.pos, item.pos),
                blob.blob_id,
            ),
        )
        distance_squared = squared_distance(source.pos, item.pos)
        rows.append((distance_squared, order, item, source))
    rows.sort(key=lambda row: (row[0], row[1]))
    return tuple((item, source) for _, _, item, source in rows[:limit])


def _reachable_stationary_resources(
    own: tuple[BlobModel, ...],
    resources: tuple[FoodModel, ...] | tuple[VirusModel, ...],
    *,
    arena_size: float,
) -> tuple[FoodModel, ...] | tuple[VirusModel, ...]:
    """Keep points that at least one current blob can geometrically cover.

    The engine clamps a blob centre by its radius, then consumes a stationary
    resource only when the resource point is within that radius.  This exact
    feasibility test matters at corners, where the closest legal centre can be
    ``sqrt(2) * radius`` away and no amount of movement reaches the point.
    """

    return tuple(
        resource
        for resource in resources
        if any(
            squared_distance(
                resource.pos,
                _stationary_resource_contact(
                    blob,
                    resource.pos,
                    arena_size=arena_size,
                ),
            )
            <= blob.radius * blob.radius + EPSILON
            for blob in own
        )
    )


def _stationary_resource_contact(
    blob: BlobModel,
    resource_pos: tuple[float, float],
    *,
    arena_size: float,
) -> tuple[float, float]:
    """Closest legal centre to the point, used only for feasibility."""

    return (
        _clamp(resource_pos[0], blob.radius, arena_size - blob.radius),
        _clamp(resource_pos[1], blob.radius, arena_size - blob.radius),
    )


def _movement_progress_ratio(
    own: tuple[BlobModel, ...],
    direction: tuple[float, float],
    *,
    arena_size: float,
) -> float:
    """Return mass-weighted realised travel divided by requested travel."""

    total_mass = sum(blob.radius * blob.radius for blob in own)
    if total_mass <= EPSILON:
        return 0.0
    realised = sum(
        blob.radius
        * blob.radius
        * math.hypot(
            _clamp(
                blob.pos[0] + direction[0] * player_speed(blob.radius),
                blob.radius,
                arena_size - blob.radius,
            )
            - blob.pos[0],
            _clamp(
                blob.pos[1] + direction[1] * player_speed(blob.radius),
                blob.radius,
                arena_size - blob.radius,
            )
            - blob.pos[1],
        )
        / max(player_speed(blob.radius), EPSILON)
        for blob in own
    )
    return realised / total_mass


def _boundary_recovery_direction(
    own: tuple[BlobModel, ...],
    direction: tuple[float, float],
    *,
    arena_size: float,
) -> tuple[float, float]:
    """Turn a hard-clipped continuation into useful tangent/interior motion."""

    projected = _project_blobs(own, direction, 1.0, arena_size)
    total_mass = sum(blob.radius * blob.radius for blob in own)
    realised = (
        sum(
            blob.radius * blob.radius * (moved.pos[0] - blob.pos[0])
            for blob, moved in zip(own, projected, strict=True)
        )
        / total_mass,
        sum(
            blob.radius * blob.radius * (moved.pos[1] - blob.pos[1])
            for blob, moved in zip(own, projected, strict=True)
        )
        / total_mass,
    )
    tangent = normalise(realised)
    if tangent != (0.0, 0.0):
        return tangent

    center = (
        sum(blob.radius * blob.radius * blob.pos[0] for blob in own) / total_mass,
        sum(blob.radius * blob.radius * blob.pos[1] for blob in own) / total_mass,
    )
    return _direction_or_fallback(
        center,
        (arena_size / 2.0, arena_size / 2.0),
        (-direction[0], -direction[1]),
    )


def _observed_enemy_directions(
    enemies: tuple[VisibleBlobModel, ...],
    *,
    previous_positions: dict[tuple[int, int], tuple[float, float]],
) -> dict[tuple[int, int], tuple[float, float]]:
    """Infer public enemy motion from consecutive visible positions.

    A large displacement means the ID survived a respawn/virus topology change
    rather than a normal one-turn move, so it is not treated as velocity.
    """

    directions: dict[tuple[int, int], tuple[float, float]] = {}
    for enemy in enemies:
        key = (int(enemy.player_id), enemy.blob_id)
        previous = previous_positions.get(key)
        if previous is None:
            continue
        displacement = (
            enemy.pos[0] - previous[0],
            enemy.pos[1] - previous[1],
        )
        if math.hypot(*displacement) > 3.0:
            continue
        direction = normalise(displacement)
        if direction != (0.0, 0.0):
            directions[key] = direction
    return directions


def _intercept_point(
    *,
    hunter: BlobModel,
    enemy: VisibleBlobModel,
    previous_direction: tuple[float, float],
    arena_size: float,
) -> tuple[tuple[float, float], float] | None:
    """Lead prey only while the observed trajectory is actually closing.

    A faster enemy is still catchable when it crosses our route, while a
    slower hunter can waste indefinitely following an enemy that is moving
    directly away.  Radial closing speed distinguishes those scenes without
    assuming that the enemy keeps the same command for an entire long chase.
    """

    if previous_direction == (0.0, 0.0):
        return (enemy.pos, 0.0)
    relative = (
        enemy.pos[0] - hunter.pos[0],
        enemy.pos[1] - hunter.pos[1],
    )
    prey_speed = player_speed(enemy.radius)
    hunter_speed = player_speed(hunter.radius)
    velocity = (
        previous_direction[0] * prey_speed,
        previous_direction[1] * prey_speed,
    )
    distance = math.hypot(*relative)
    if distance <= hunter.radius:
        return (enemy.pos, 0.0)
    line_to_enemy = (relative[0] / distance, relative[1] / distance)
    radial_closing_speed = hunter_speed - _dot(velocity, line_to_enemy)
    if radial_closing_speed <= EPSILON:
        return None

    a = _dot(velocity, velocity) - hunter_speed * hunter_speed
    b = 2.0 * (_dot(relative, velocity) - hunter.radius * hunter_speed)
    c = _dot(relative, relative) - hunter.radius * hunter.radius
    roots: list[float] = []
    if abs(a) <= EPSILON:
        if abs(b) > EPSILON:
            roots.append(-c / b)
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0.0:
            root = math.sqrt(discriminant)
            roots.extend(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)))
    positive = [turns for turns in roots if turns > 0.0 and math.isfinite(turns)]
    gap = max(0.0, distance - hunter.radius)
    turns = (
        min(positive)
        if positive
        else gap / max(radial_closing_speed, EPSILON)
    )
    # Long constant-velocity predictions are brittle because opponents turn.
    # Aim only a few visible turns ahead while retaining the closing/no-closing
    # decision from the measured trajectory.
    lead_turns = min(turns, DIRECTIONAL_HORIZON)
    return (
        (
            _clamp(
                enemy.pos[0] + velocity[0] * lead_turns,
                enemy.radius,
                arena_size - enemy.radius,
            ),
            _clamp(
                enemy.pos[1] + velocity[1] * lead_turns,
                enemy.radius,
                arena_size - enemy.radius,
            ),
        ),
        turns,
    )


def _mass_phase(
    own: tuple[BlobModel, ...],
    viruses: tuple[VirusModel, ...],
) -> str:
    largest_mass = max(blob.radius * blob.radius for blob in own)
    reference_virus_radius = min(
        (virus.radius for virus in viruses),
        default=REFERENCE_VIRUS_RADIUS,
    )
    virus_threshold_mass = (
        reference_virus_radius * reference_virus_radius * EAT_SIZE_RATIO
    )
    mass_ratio = largest_mass / max(virus_threshold_mass, EPSILON)
    if mass_ratio < 1.0:
        return "growth"
    if mass_ratio < 4.0:
        return "mixed"
    return "hunter"


def _target_mass_opportunity(
    *,
    candidate: DirectionCandidate,
    own: tuple[BlobModel, ...],
    foods: tuple[FoodModel, ...],
    viruses: tuple[VirusModel, ...],
    enemies: tuple[VisibleBlobModel, ...],
) -> float:
    """Discount the candidate's concrete target mass by turns to contact."""

    if candidate.target_pos is None or not own:
        return 0.0
    target_mass = 0.0
    if candidate.target_kind == "food":
        if not any(str(food.food_id) == candidate.target_id for food in foods):
            return 0.0
        target_mass = FOOD_RADIUS * FOOD_RADIUS
    elif candidate.target_kind == "virus":
        target = next(
            (virus for virus in viruses if str(virus.virus_id) == candidate.target_id),
            None,
        )
        if target is None or not any(
            can_consume_virus(blob.radius, target.radius) for blob in own
        ):
            return 0.0
        target_mass = target.radius * target.radius
    elif candidate.target_kind == "prey":
        target = next(
            (
                enemy
                for enemy in enemies
                if f"{enemy.player_id}:{enemy.blob_id}" == candidate.target_id
            ),
            None,
        )
        if target is None:
            return 0.0
        if not candidate.split and not any(
            can_eat_player_blob(blob.radius, target.radius, radius_margin=1.03)
            for blob in own
        ):
            return 0.0
        target_mass = target.radius * target.radius
    else:
        return 0.0

    if candidate.contact_turns is not None:
        turns = max(0.0, candidate.contact_turns)
    elif candidate.split:
        turns = max(1.0, float(candidate.split_depth))
    else:
        source = min(
            own,
            key=lambda blob: squared_distance(blob.pos, candidate.target_pos),
        )
        gap = max(
            0.0,
            math.dist(source.pos, candidate.target_pos) - source.radius,
        )
        turns = gap / max(player_speed(source.radius), EPSILON)
    reliability = 1.0
    if candidate.target_kind == "prey" and turns > DIRECTIONAL_HORIZON:
        reliability = (DIRECTIONAL_HORIZON / turns) ** 2
    return target_mass * reliability / (1.0 + turns / DIRECTIONAL_HORIZON)


def _refresh_target_memory(
    memory: TargetMemory | None,
    *,
    foods: tuple[FoodModel, ...],
    viruses: tuple[VirusModel, ...],
) -> TargetMemory | None:
    if memory is None:
        return None
    resources: tuple[FoodModel, ...] | tuple[VirusModel, ...]
    resources = foods if memory.kind == "food" else viruses
    matched = min(
        resources,
        key=lambda resource: squared_distance(resource.pos, memory.pos),
        default=None,
    )
    if (
        matched is None
        or squared_distance(matched.pos, memory.pos)
        > TARGET_MATCH_DISTANCE * TARGET_MATCH_DISTANCE
    ):
        return None
    return TargetMemory(kind=memory.kind, pos=matched.pos)


def _continuation_candidate(
    *,
    own: tuple[BlobModel, ...],
    foods: tuple[FoodModel, ...],
    viruses: tuple[VirusModel, ...],
    memory: TargetMemory | None,
    fallback: tuple[float, float],
) -> DirectionCandidate:
    if memory is None:
        return DirectionCandidate(
            family="continue",
            direction=fallback,
            target_kind="momentum",
        )
    resources: tuple[FoodModel, ...] | tuple[VirusModel, ...]
    resources = foods if memory.kind == "food" else viruses
    if not resources:
        # A projected child can consume the resource selected at the root.
        # Continuing that intent must then degrade to momentum, not crash the
        # entire bot process while evaluating the next ply.
        return DirectionCandidate(
            family="continue",
            direction=fallback,
            target_kind="momentum",
        )
    target = min(
        resources,
        key=lambda resource: squared_distance(resource.pos, memory.pos),
    )
    source = min(own, key=lambda blob: squared_distance(blob.pos, target.pos))
    target_id = (
        str(target.food_id) if isinstance(target, FoodModel) else str(target.virus_id)
    )
    return DirectionCandidate(
        family="continue",
        direction=_direction_or_fallback(source.pos, target.pos, fallback),
        target_kind=memory.kind,
        target_id=target_id,
        target_pos=target.pos,
    )


def _best_capture_target(
    own: tuple[BlobModel, ...],
    enemies: tuple[VisibleBlobModel, ...],
    *,
    previous_directions: dict[tuple[int, int], tuple[float, float]],
    arena_size: float,
) -> CaptureRoute | None:
    best: tuple[float, CaptureRoute] | None = None
    for enemy in enemies:
        previous_direction = previous_directions.get(
            (int(enemy.player_id), enemy.blob_id),
            (0.0, 0.0),
        )
        for hunter in own:
            if not can_eat_player_blob(
                hunter.radius,
                enemy.radius,
                radius_margin=1.03,
            ):
                continue
            intercepting = previous_direction != (0.0, 0.0)
            if intercepting:
                intercept = _intercept_point(
                    hunter=hunter,
                    enemy=enemy,
                    previous_direction=previous_direction,
                    arena_size=arena_size,
                )
                if intercept is None:
                    continue
                target_pos, turns = intercept
            else:
                target_pos = enemy.pos
                gap = max(0.0, math.dist(hunter.pos, enemy.pos) - hunter.radius)
                turns = gap / max(player_speed(hunter.radius), EPSILON)
            value = _capture_expected_mass_value(
                enemy.radius * enemy.radius,
                turns,
            )
            route = CaptureRoute(
                enemy=enemy,
                hunter=hunter,
                target_pos=target_pos,
                one_step_target_pos=_project_enemy_position(
                    enemy,
                    velocity=(
                        previous_direction[0] * player_speed(enemy.radius),
                        previous_direction[1] * player_speed(enemy.radius),
                    ),
                    turns=1.0,
                    arena_size=arena_size,
                ),
                intercepting=intercepting,
                turns_to_contact=turns,
            )
            if best is None or value > best[0]:
                best = (value, route)
    return None if best is None else best[1]


def _split_capture_candidates(
    *,
    own: tuple[BlobModel, ...],
    enemies: tuple[VisibleBlobModel, ...],
    fallback: tuple[float, float],
    arena_size: float,
    previous_directions: dict[tuple[int, int], tuple[float, float]],
) -> tuple[DirectionCandidate, ...]:
    """Return at most one immediate and one staged fragment-sweep route.

    Normal movement treats an enemy player as one strategic group because a
    shared command can donate a small own fragment while chasing another.
    Split capture is different: its value is the edible subset intersected by
    the launch corridor. Candidate generation must therefore start from each
    fragment, while the one-step outcome and threat field decide whether the
    remaining fragments make that launch unsafe.
    """

    if len(own) >= MAX_BLOB_COUNT or not any(
        blob.radius * blob.radius >= SPLIT_MIN_MASS for blob in own
    ):
        return ()

    best_single: tuple[float, DirectionCandidate] | None = None
    best_multi: tuple[float, DirectionCandidate] | None = None
    for enemy in enemies:
        hunter = min(
            own,
            key=lambda blob: squared_distance(blob.pos, enemy.pos),
        )
        distance = math.dist(hunter.pos, enemy.pos)
        route = _split_capture_route(
            hunter=hunter,
            enemy=enemy,
            fallback=fallback,
            own_count=len(own),
            arena_size=arena_size,
            previous_direction=previous_directions.get(
                (int(enemy.player_id), enemy.blob_id),
                (0.0, 0.0),
            ),
        )
        if route is None:
            continue
        first_step = _project_action_blobs(
            own,
            route.direction,
            split=True,
            arena_size=arena_size,
        )
        projected_enemies = _enemies_with_target_position(
            enemies,
            target_id=f"{enemy.player_id}:{enemy.blob_id}",
            target_pos=route.one_step_target_pos,
        )
        corridor_mass = _geometric_capture_mass(first_step, projected_enemies)
        family = "split_capture" if route.depth == 1 else "multi_split_capture"
        candidate = DirectionCandidate(
            family=family,
            direction=route.direction,
            target_kind="prey",
            target_id=f"{enemy.player_id}:{enemy.blob_id}",
            target_pos=route.target_pos,
            one_step_target_pos=route.one_step_target_pos,
            split=True,
            split_depth=route.depth,
            capture_mass=corridor_mass,
            contact_turns=float(route.depth),
        )
        # Immediate routes are ranked by the fragment subset crossed this
        # turn. Staged routes have no one-step harvest yet, so retain the
        # concrete target's mass rather than crediting its whole player.
        candidate_mass = corridor_mass or enemy.radius * enemy.radius
        value = candidate_mass / (distance + 1.0)
        if route.depth == 1:
            if best_single is None or value > best_single[0]:
                best_single = (value, candidate)
        elif best_multi is None or value > best_multi[0]:
            best_multi = (value, candidate)
    return tuple(row[1] for row in (best_single, best_multi) if row is not None)


def _capture_expected_mass_value(
    mass: float,
    turns: float,
) -> float:
    reliability = 1.0
    if turns > DIRECTIONAL_HORIZON:
        reliability = (DIRECTIONAL_HORIZON / turns) ** 2
    return mass * reliability / (turns + 2.0)


def _split_capture_route(
    *,
    hunter: BlobModel,
    enemy: VisibleBlobModel,
    fallback: tuple[float, float],
    own_count: int,
    arena_size: float,
    previous_direction: tuple[float, float],
) -> SplitCaptureRoute | None:
    """Find the first split depth whose launched child overlaps the prey.

    With motion history, use the observed enemy trajectory instead of adding
    an omnidirectional worst-case escape distance. Without history the old
    conservative envelope remains appropriate because no escape direction is
    yet observable.
    """

    hunter_mass = hunter.radius * hunter.radius
    enemy_velocity = (
        previous_direction[0] * player_speed(enemy.radius),
        previous_direction[1] * player_speed(enemy.radius),
    )
    for depth in range(1, MAX_CAPTURE_SPLIT_DEPTH + 1):
        if own_count * 2**depth > MAX_BLOB_COUNT:
            break
        if hunter_mass / 2 ** (depth - 1) < SPLIT_MIN_MASS:
            break
        if depth > 1 and hunter_mass < MULTI_SPLIT_MIN_MASS:
            continue
        child_radius = hunter.radius / SQRT2**depth
        if not can_eat_player_blob(
            child_radius,
            enemy.radius,
            radius_margin=1.05,
        ):
            break
        target_pos = _project_enemy_position(
            enemy,
            velocity=enemy_velocity,
            turns=float(depth),
            arena_size=arena_size,
        )
        direction = _direction_or_fallback(hunter.pos, target_pos, fallback)
        endpoint = _leading_split_endpoint(
            hunter=hunter,
            direction=direction,
            depth=depth,
            arena_size=arena_size,
        )
        if previous_direction != (0.0, 0.0):
            reachable = math.dist(endpoint, target_pos) <= child_radius
        else:
            capture_reach = math.dist(hunter.pos, endpoint) + child_radius
            prey_escape = player_speed(enemy.radius) * depth
            reachable = (
                math.dist(hunter.pos, enemy.pos) + prey_escape <= capture_reach
            )
        if reachable:
            return SplitCaptureRoute(
                depth=depth,
                direction=direction,
                target_pos=target_pos,
                one_step_target_pos=_project_enemy_position(
                    enemy,
                    velocity=enemy_velocity,
                    turns=1.0,
                    arena_size=arena_size,
                ),
            )
    return None


def _project_enemy_position(
    enemy: VisibleBlobModel,
    *,
    velocity: tuple[float, float],
    turns: float,
    arena_size: float,
) -> tuple[float, float]:
    return (
        _clamp(
            enemy.pos[0] + velocity[0] * turns,
            enemy.radius,
            arena_size - enemy.radius,
        ),
        _clamp(
            enemy.pos[1] + velocity[1] * turns,
            enemy.radius,
            arena_size - enemy.radius,
        ),
    )


def _leading_split_endpoint(
    *,
    hunter: BlobModel,
    direction: tuple[float, float],
    depth: int,
    arena_size: float,
) -> tuple[float, float]:
    x, y = hunter.pos
    radius = hunter.radius
    for _ in range(depth):
        radius /= SQRT2
        launch = (
            2.0 * radius
            + SAME_PLAYER_OVERLAP_EPSILON
            + player_speed(radius)
            + SPLIT_EJECT_SPEED
        )
        x = _clamp(x + direction[0] * launch, radius, arena_size - radius)
        y = _clamp(y + direction[1] * launch, radius, arena_size - radius)
    return (x, y)


def _wall_escape_direction(
    own: tuple[BlobModel, ...],
    escape_direction: tuple[float, float],
    arena_size: float,
) -> tuple[float, float]:
    """Deflect an active escape along a wall instead of avoiding every edge.

    A wall has no strategic meaning without a nearby predator. When the direct
    away-vector points into a close wall, suppress only the blocked component.
    At a head-on wall, either tangent remains a useful escape candidate; choose
    the tangent with more one-step clearance.
    """

    if escape_direction == (0.0, 0.0):
        return (0.0, 0.0)

    total_mass = sum(blob.radius * blob.radius for blob in own)
    left = right = bottom = top = 0.0
    for blob in own:
        mass = blob.radius * blob.radius
        margin = blob.radius + WALL_INFLUENCE
        left += mass * _wall_pressure(blob.pos[0] - blob.radius, margin)
        right += mass * _wall_pressure(
            arena_size - blob.radius - blob.pos[0],
            margin,
        )
        bottom += mass * _wall_pressure(blob.pos[1] - blob.radius, margin)
        top += mass * _wall_pressure(
            arena_size - blob.radius - blob.pos[1],
            margin,
        )

    x, y = escape_direction
    if x < 0.0:
        x *= 1.0 - left / total_mass
    elif x > 0.0:
        x *= 1.0 - right / total_mass
    if y < 0.0:
        y *= 1.0 - bottom / total_mass
    elif y > 0.0:
        y *= 1.0 - top / total_mass

    retained_escape = math.hypot(x, y)
    deflected = normalise((x, y))
    if retained_escape < 0.25:
        tangent = (-escape_direction[1], escape_direction[0])
        alternatives = (tangent, (-tangent[0], -tangent[1]))
        deflected = max(
            alternatives,
            key=lambda direction: _projected_wall_clearance(
                own,
                direction,
                arena_size,
            ),
        )
    if _dot(deflected, escape_direction) >= 1.0 - 1.0e-6:
        return (0.0, 0.0)
    return deflected


def _projected_wall_clearance(
    own: tuple[BlobModel, ...],
    direction: tuple[float, float],
    arena_size: float,
) -> float:
    projected = _project_blobs(own, direction, 1.0, arena_size)
    return min(
        min(
            blob.pos[0] - blob.radius,
            blob.pos[1] - blob.radius,
            arena_size - blob.radius - blob.pos[0],
            arena_size - blob.radius - blob.pos[1],
        )
        for blob in projected
    )


def _wall_pressure(clearance: float, influence: float) -> float:
    if clearance >= influence:
        return 0.0
    return (influence - max(0.0, clearance)) / influence


def _escape_direction(
    own: tuple[BlobModel, ...],
    enemies: tuple[VisibleBlobModel, ...],
) -> tuple[float, float]:
    x = 0.0
    y = 0.0
    for blob in own:
        for enemy in enemies:
            if not can_eat_player_blob(enemy.radius, blob.radius):
                continue
            distance = math.dist(blob.pos, enemy.pos)
            reach = _one_step_attack_reach(enemy.radius, blob.radius)
            margin = distance - reach
            awareness_margin = player_speed(blob.radius) * 4.0
            if margin >= awareness_margin:
                continue
            away = normalise((blob.pos[0] - enemy.pos[0], blob.pos[1] - enemy.pos[1]))
            severity = (awareness_margin - margin) / max(
                awareness_margin,
                EPSILON,
            )
            weight = blob.radius * blob.radius * (0.25 + severity * severity)
            x += away[0] * weight
            y += away[1] * weight
    return normalise((x, y))


def _routing_blobs(
    own: tuple[BlobModel, ...],
    *,
    limit: int,
) -> tuple[BlobModel, ...]:
    """Keep spatial and size extremes for bounded resource-route scoring."""

    if len(own) <= limit:
        return own
    total_mass = sum(blob.radius * blob.radius for blob in own)
    center = (
        sum(blob.pos[0] * blob.radius * blob.radius for blob in own) / total_mass,
        sum(blob.pos[1] * blob.radius * blob.radius for blob in own) / total_mass,
    )
    ordered = (
        max(own, key=lambda blob: blob.radius),
        min(own, key=lambda blob: blob.radius),
        min(own, key=lambda blob: blob.pos[0]),
        max(own, key=lambda blob: squared_distance(blob.pos, center)),
    )
    unique: list[BlobModel] = []
    seen: set[int] = set()
    for blob in (*ordered, *own):
        if blob.blob_id in seen:
            continue
        unique.append(blob)
        seen.add(blob.blob_id)
        if len(unique) == limit:
            break
    return tuple(unique)


def _project_action_blobs(
    own: tuple[BlobModel, ...],
    direction: tuple[float, float],
    *,
    split: bool,
    arena_size: float,
) -> tuple[BlobModel, ...]:
    if not split:
        return _project_blobs(own, direction, 1.0, arena_size)

    projected: list[BlobModel] = []
    remaining_slots = MAX_BLOB_COUNT - len(own)
    for source_index, blob in enumerate(own):
        can_split = remaining_slots > 0 and blob.radius * blob.radius >= SPLIT_MIN_MASS
        if not can_split:
            moved = _project_blobs((blob,), direction, 1.0, arena_size)[0]
            projected.append(
                BlobModel(
                    blob_id=source_index * 2,
                    pos=moved.pos,
                    radius=moved.radius,
                    merge_cooldown=moved.merge_cooldown,
                )
            )
            continue

        remaining_slots -= 1
        radius = blob.radius / SQRT2
        speed = player_speed(radius)
        parent_pos = (
            _clamp(
                blob.pos[0] + direction[0] * speed,
                radius,
                arena_size - radius,
            ),
            _clamp(
                blob.pos[1] + direction[1] * speed,
                radius,
                arena_size - radius,
            ),
        )
        child_pos = _leading_split_endpoint(
            hunter=blob,
            direction=direction,
            depth=1,
            arena_size=arena_size,
        )
        projected.extend(
            (
                BlobModel(
                    blob_id=source_index * 2,
                    pos=parent_pos,
                    radius=radius,
                    merge_cooldown=blob.merge_cooldown,
                ),
                BlobModel(
                    blob_id=source_index * 2 + 1,
                    pos=child_pos,
                    radius=radius,
                    merge_cooldown=blob.merge_cooldown,
                ),
            )
        )
    return tuple(projected)


def _geometric_capture_mass(
    projected: tuple[BlobModel, ...],
    enemies: tuple[VisibleBlobModel, ...],
) -> float:
    """Cheap upper bound used only to rank the bounded split directions."""

    remaining = list(enemies)
    local = list(projected)
    captured = 0.0
    changed = True
    while changed:
        changed = False
        for eater_index in sorted(
            range(len(local)),
            key=lambda index: -local[index].radius,
        ):
            eater = local[eater_index]
            target = next(
                (
                    enemy
                    for enemy in remaining
                    if _can_local_blob_eat(
                        eater.pos,
                        eater.radius,
                        enemy.pos,
                        enemy.radius,
                    )
                ),
                None,
            )
            if target is None:
                continue
            target_mass = target.radius * target.radius
            captured += target_mass
            remaining.remove(target)
            local[eater_index] = BlobModel(
                blob_id=eater.blob_id,
                pos=eater.pos,
                radius=math.sqrt(eater.radius * eater.radius + target_mass),
                merge_cooldown=eater.merge_cooldown,
            )
            changed = True
            break
    return captured


def _enemies_with_target_position(
    enemies: tuple[VisibleBlobModel, ...],
    *,
    target_id: str | None,
    target_pos: tuple[float, float] | None,
) -> tuple[VisibleBlobModel, ...]:
    """Move only the observed split target for cheap candidate ranking."""

    if target_id is None or target_pos is None:
        return enemies
    return tuple(
        VisibleBlobModel(
            player_id=enemy.player_id,
            team_id=enemy.team_id,
            blob_id=enemy.blob_id,
            pos=(
                target_pos
                if f"{enemy.player_id}:{enemy.blob_id}" == target_id
                else enemy.pos
            ),
            radius=enemy.radius,
            merge_cooldown=enemy.merge_cooldown,
        )
        for enemy in enemies
    )


def _project_one_step_outcome(
    *,
    own: tuple[BlobModel, ...],
    direction: tuple[float, float],
    split: bool = True,
    foods: tuple[FoodModel, ...],
    viruses: tuple[VirusModel, ...],
    enemies: tuple[VisibleBlobModel, ...],
    arena_size: float,
    target_id: str | None = None,
    target_pos: tuple[float, float] | None = None,
) -> OneStepOutcome:
    """Project one command through the engine's one-turn event order.

    Exact projection is bounded to split, virus-target, and selected prey
    candidates.  The first two can change blob topology before cross-player
    eating; the prey route needs exact contact credit rather than a geometric
    opportunity estimate.  When a target has two consecutive observations,
    its next center is projected; every deterministic event after movement
    follows the engine: decay, viruses, food, then largest-first cross-player
    eating.
    """

    projected_own = _project_action_blobs(
        own,
        direction,
        split=split,
        arena_size=arena_size,
    )
    local: list[_LocalBlob] = [
        _LocalBlob(
            owner_id=-1,
            blob_id=blob.blob_id,
            pos=blob.pos,
            radius=blob.radius,
            team_id=-1,
            is_own=True,
        )
        for blob in projected_own
    ]
    projected_enemies = _enemies_with_target_position(
        enemies,
        target_id=target_id,
        target_pos=target_pos,
    )
    local.extend(
        _LocalBlob(
            owner_id=int(enemy.player_id),
            blob_id=enemy.blob_id,
            pos=enemy.pos,
            radius=enemy.radius,
            team_id=int(enemy.team_id),
            is_own=False,
        )
        for enemy in projected_enemies
    )

    minimum_mass = STARTING_RADIUS * STARTING_RADIUS
    for blob in local:
        mass = blob.radius * blob.radius
        if mass > minimum_mass:
            blob.radius = math.sqrt(max(minimum_mass, mass * (1.0 - MASS_DECAY_RATE)))

    virus_mass_gained = 0.0
    remaining_viruses: list[VirusModel] = []
    for virus in viruses:
        candidates = [
            blob
            for blob in local
            if squared_distance(blob.pos, virus.pos) <= blob.radius * blob.radius
            and blob.radius * blob.radius > virus.radius * virus.radius * EAT_SIZE_RATIO
        ]
        if not candidates:
            remaining_viruses.append(virus)
            continue
        eater = min(
            candidates,
            key=lambda blob: (-blob.radius, blob.owner_id, blob.blob_id),
        )
        if eater.is_own:
            virus_mass_gained += virus.radius * virus.radius
        _replace_local_blob_after_virus(
            local,
            eater=eater,
            virus=virus,
            arena_size=arena_size,
        )

    food_mass_gained = 0.0
    remaining_foods: list[FoodModel] = []
    for food in foods:
        candidates = [
            blob
            for blob in local
            if squared_distance(blob.pos, food.pos) <= blob.radius * blob.radius
        ]
        if not candidates:
            remaining_foods.append(food)
            continue
        eater = min(
            candidates,
            key=lambda blob: (-blob.radius, blob.owner_id, blob.blob_id),
        )
        eater.radius = math.sqrt(
            eater.radius * eater.radius + FOOD_RADIUS * FOOD_RADIUS
        )
        if eater.is_own:
            food_mass_gained += FOOD_RADIUS * FOOD_RADIUS

    enemy_mass_gained = 0.0
    own_mass_lost = 0.0
    living_by_identity = {id(blob): blob for blob in local}
    changed = True
    while changed:
        changed = False
        living = sorted(
            living_by_identity.values(),
            key=lambda blob: (-blob.radius, blob.owner_id, blob.blob_id),
        )
        for eater in living:
            if id(eater) not in living_by_identity:
                continue
            for target in living:
                if (
                    id(target) not in living_by_identity
                    or eater.owner_id == target.owner_id
                ):
                    continue
                if not _can_local_blob_eat(
                    eater.pos,
                    eater.radius,
                    target.pos,
                    target.radius,
                ):
                    continue
                target_mass = target.radius * target.radius
                eater.radius = math.sqrt(eater.radius * eater.radius + target_mass)
                del living_by_identity[id(target)]
                if eater.is_own and not target.is_own:
                    enemy_mass_gained += target_mass
                elif not eater.is_own and target.is_own:
                    own_mass_lost += target_mass
                changed = True
                break
            if changed:
                break

    local = list(living_by_identity.values())

    return OneStepOutcome(
        own=tuple(
            BlobModel(
                blob_id=blob.blob_id,
                pos=blob.pos,
                radius=blob.radius,
                merge_cooldown=0,
            )
            for blob in local
            if blob.is_own
        ),
        enemies=tuple(
            VisibleBlobModel(
                player_id=blob.owner_id,
                team_id=blob.team_id,
                blob_id=blob.blob_id,
                pos=blob.pos,
                radius=blob.radius,
                merge_cooldown=0,
            )
            for blob in local
            if not blob.is_own
        ),
        foods=tuple(remaining_foods),
        viruses=tuple(remaining_viruses),
        enemy_mass_gained=enemy_mass_gained,
        virus_mass_gained=virus_mass_gained,
        food_mass_gained=food_mass_gained,
        own_mass_lost=own_mass_lost,
    )


def _replace_local_blob_after_virus(
    local: list[_LocalBlob],
    *,
    eater: _LocalBlob,
    virus: VirusModel,
    arena_size: float,
) -> None:
    owner_blobs = [blob for blob in local if blob.owner_id == eater.owner_id]
    piece_count = max(1, MAX_BLOB_COUNT - len(owner_blobs) + 1)
    total_mass = eater.radius * eater.radius + virus.radius * virus.radius
    piece_radius = math.sqrt(total_mass / piece_count)
    cols = math.ceil(math.sqrt(piece_count))
    rows = math.ceil(piece_count / cols)
    spacing = piece_radius * 2.0 + SAME_PLAYER_OVERLAP_EPSILON
    x_offset = (cols - 1) * spacing / 2.0
    y_offset = (rows - 1) * spacing / 2.0
    next_blob_id = max(blob.blob_id for blob in owner_blobs) + 1

    replacements: list[_LocalBlob] = []
    for index in range(piece_count):
        row = index // cols
        col = index % cols
        replacements.append(
            _LocalBlob(
                owner_id=eater.owner_id,
                blob_id=(eater.blob_id if index == 0 else next_blob_id + index - 1),
                pos=(
                    _clamp(
                        eater.pos[0] + col * spacing - x_offset,
                        piece_radius,
                        arena_size - piece_radius,
                    ),
                    _clamp(
                        eater.pos[1] + row * spacing - y_offset,
                        piece_radius,
                        arena_size - piece_radius,
                    ),
                ),
                radius=piece_radius,
                team_id=eater.team_id,
                is_own=eater.is_own,
            )
        )
    local.remove(eater)
    local.extend(replacements)


def _can_local_blob_eat(
    eater_pos: tuple[float, float],
    eater_radius: float,
    target_pos: tuple[float, float],
    target_radius: float,
) -> bool:
    return (
        eater_radius * eater_radius >= target_radius * target_radius * EAT_SIZE_RATIO
        and squared_distance(eater_pos, target_pos) <= eater_radius * eater_radius
    )


def _project_blobs(
    own: tuple[BlobModel, ...],
    direction: tuple[float, float],
    turns: float,
    arena_size: float,
) -> tuple[BlobModel, ...]:
    projected: list[BlobModel] = []
    for blob in own:
        travel = player_speed(blob.radius) * turns
        projected.append(
            BlobModel(
                blob_id=blob.blob_id,
                pos=(
                    _clamp(
                        blob.pos[0] + direction[0] * travel,
                        blob.radius,
                        arena_size - blob.radius,
                    ),
                    _clamp(
                        blob.pos[1] + direction[1] * travel,
                        blob.radius,
                        arena_size - blob.radius,
                    ),
                ),
                radius=blob.radius,
                merge_cooldown=blob.merge_cooldown,
            )
        )
    return tuple(projected)


def _threat_potential(
    projected: tuple[BlobModel, ...],
    enemies: tuple[VisibleBlobModel, ...],
) -> tuple[bool, float, float]:
    catastrophic = False
    value = 0.0
    minimum_margin = math.inf
    for blob in projected:
        mass = blob.radius * blob.radius
        for enemy in enemies:
            if not can_eat_player_blob(enemy.radius, blob.radius):
                continue
            margin = math.dist(blob.pos, enemy.pos) - _one_step_attack_reach(
                enemy.radius,
                blob.radius,
            )
            minimum_margin = min(minimum_margin, margin)
            if margin <= 0.0:
                catastrophic = True
                value -= mass * (8.0 - margin)
            else:
                value -= mass * 2.4 / (1.0 + margin / 4.0) ** 2
    return catastrophic, value, minimum_margin


def _minimum_threat_margin(
    own: tuple[BlobModel, ...],
    enemies: tuple[VisibleBlobModel, ...],
) -> float:
    return min(
        (
            math.dist(blob.pos, enemy.pos)
            - _one_step_attack_reach(enemy.radius, blob.radius)
            for blob in own
            for enemy in enemies
            if can_eat_player_blob(enemy.radius, blob.radius)
        ),
        default=math.inf,
    )


def _directional_potential(
    *,
    own: tuple[BlobModel, ...],
    direction: tuple[float, float],
    foods: tuple[FoodModel, ...],
    viruses: tuple[VirusModel, ...],
    enemies: tuple[VisibleBlobModel, ...],
    previous_directions: dict[tuple[int, int], tuple[float, float]],
) -> DirectionalPotential:
    """Estimate three-turn mass available in the candidate's fan.

    Every source is expressed in mass. Source-specific multipliers are
    deliberately absent: one virus is worth ``radius²``, one enemy fragment
    its current ``radius²``, and one food ``FOOD_RADIUS²``. Reachability is the
    only discount here; safety and actual one-step losses are scored elsewhere.
    """

    source_speeds = {
        source.blob_id: player_speed(source.radius) for source in own
    }
    source_travels = {
        blob_id: speed * DIRECTIONAL_HORIZON
        for blob_id, speed in source_speeds.items()
    }
    enemy_speeds = {
        (int(enemy.player_id), enemy.blob_id): player_speed(enemy.radius)
        for enemy in enemies
    }

    food_value = sum(
        FOOD_RADIUS
        * FOOD_RADIUS
        * _best_fan_weight(
            own,
            food.pos,
            direction,
            source_travels=source_travels,
        )
        for food in foods
    )
    prey_value = sum(
        enemy.radius
        * enemy.radius
        * max(
            (
                _prey_fan_weight(
                    source,
                    enemy,
                    direction,
                    previous_directions.get(
                        (int(enemy.player_id), enemy.blob_id),
                        (0.0, 0.0),
                    ),
                    source_speed=source_speeds[source.blob_id],
                    enemy_speed=enemy_speeds[
                        (int(enemy.player_id), enemy.blob_id)
                    ],
                )
                for source in own
                if can_eat_player_blob(
                    source.radius,
                    enemy.radius,
                    radius_margin=1.03,
                )
            ),
            default=0.0,
        )
        for enemy in enemies
    )

    virus_value = sum(
        virus.radius
        * virus.radius
        * max(
            (
                _fan_weight(
                    source,
                    virus.pos,
                    direction,
                    travel=source_travels[source.blob_id],
                )
                for source in own
                if can_consume_virus(source.radius, virus.radius)
            ),
            default=0.0,
        )
        for virus in viruses
    )

    return DirectionalPotential(
        food=FUTURE_MASS_DISCOUNT * food_value,
        prey=FUTURE_MASS_DISCOUNT * prey_value,
        virus=FUTURE_MASS_DISCOUNT * virus_value,
    )


def _prey_fan_weight(
    source: BlobModel,
    enemy: VisibleBlobModel,
    direction: tuple[float, float],
    previous_direction: tuple[float, float],
    *,
    source_speed: float,
    enemy_speed: float,
) -> float:
    """Replace static reach with measured relative closing reach."""

    static_weight = _fan_weight(
        source,
        enemy.pos,
        direction,
        travel=source_speed * DIRECTIONAL_HORIZON,
    )
    if static_weight <= 0.0 or previous_direction == (0.0, 0.0):
        return static_weight
    delta = (
        enemy.pos[0] - source.pos[0],
        enemy.pos[1] - source.pos[1],
    )
    distance = math.hypot(*delta)
    if distance <= source.radius:
        return 1.0
    line_to_enemy = (delta[0] / distance, delta[1] / distance)
    hunter_radial_speed = source_speed * max(
        0.0,
        _dot(direction, line_to_enemy),
    )
    enemy_radial_speed = enemy_speed * _dot(
        previous_direction,
        line_to_enemy,
    )
    closing_speed = hunter_radial_speed - enemy_radial_speed
    gap = distance - source.radius
    relative_travel = closing_speed * DIRECTIONAL_HORIZON
    if relative_travel <= gap:
        return 0.0

    static_travel = source_speed * DIRECTIONAL_HORIZON
    static_radial = 1.0 - gap / max(static_travel, EPSILON)
    relative_radial = 1.0 - gap / relative_travel
    return min(1.0, static_weight * relative_radial / max(static_radial, EPSILON))


def _best_fan_weight(
    own: tuple[BlobModel, ...],
    target: tuple[float, float],
    direction: tuple[float, float],
    *,
    source_travels: dict[int, float],
) -> float:
    return max(
        (
            _fan_weight(
                blob,
                target,
                direction,
                travel=source_travels[blob.blob_id],
            )
            for blob in own
        ),
        default=0.0,
    )


def _fan_weight(
    source: BlobModel,
    target: tuple[float, float],
    direction: tuple[float, float],
    *,
    travel: float | None = None,
) -> float:
    dx = target[0] - source.pos[0]
    dy = target[1] - source.pos[1]
    distance = math.hypot(dx, dy)
    if distance <= source.radius:
        return 1.0
    if travel is None:
        travel = player_speed(source.radius) * DIRECTIONAL_HORIZON
    gap = distance - source.radius
    if gap >= travel:
        return 0.0
    alignment = (dx * direction[0] + dy * direction[1]) / distance
    if alignment <= DIRECTIONAL_COSINE_LIMIT:
        return 0.0
    angular = (alignment - DIRECTIONAL_COSINE_LIMIT) / (1.0 - DIRECTIONAL_COSINE_LIMIT)
    radial = 1.0 - gap / max(travel, EPSILON)
    return radial * (0.20 + 0.80 * angular)


def _blocked_escape_wall_potential(
    projected: tuple[BlobModel, ...],
    escape_direction: tuple[float, float],
    arena_size: float,
) -> float:
    """Price only a wall that blocks the current predator-away direction."""

    if escape_direction == (0.0, 0.0):
        return 0.0

    value = 0.0
    for blob in projected:
        blocked = 0.0
        x, y = escape_direction
        if x < 0.0:
            blocked += x * x / (max(0.0, blob.pos[0] - blob.radius) + 1.0) ** 2
        elif x > 0.0:
            blocked += (
                x * x / (max(0.0, arena_size - blob.radius - blob.pos[0]) + 1.0) ** 2
            )
        if y < 0.0:
            blocked += y * y / (max(0.0, blob.pos[1] - blob.radius) + 1.0) ** 2
        elif y > 0.0:
            blocked += (
                y * y / (max(0.0, arena_size - blob.radius - blob.pos[1]) + 1.0) ** 2
            )
        value -= blob.radius * blob.radius * 0.55 * blocked
    return value


def _one_step_attack_reach(predator_radius: float, prey_radius: float) -> float:
    return _horizon_attack_reach(predator_radius, prey_radius, 1.0)


def _horizon_attack_reach(
    predator_radius: float,
    prey_radius: float,
    turns: float,
) -> float:
    """Farthest prey-center reach after movement and an immediate split.

    The split child may travel for the same horizon after its launch.  Keeping
    movement inside this envelope prevents callers from adding a separate
    predator-speed term and accidentally counting the first turn twice.
    """

    turns = max(0.0, turns)
    reach = predator_radius + player_speed(predator_radius) * turns
    if predator_radius * predator_radius >= SPLIT_MIN_MASS and can_eat_player_blob(
        predator_radius / SQRT2,
        prey_radius,
    ):
        child_radius = predator_radius / SQRT2
        reach = max(
            reach,
            3.0 * child_radius
            + SPLIT_EJECT_SPEED
            + player_speed(child_radius) * turns,
        )
    return reach


def _direction_or_fallback(
    source: tuple[float, float],
    target: tuple[float, float],
    fallback: tuple[float, float],
) -> tuple[float, float]:
    direction = normalise((target[0] - source[0], target[1] - source[1]))
    return fallback if direction == (0.0, 0.0) else direction


def _dot(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1]


def _angle_degrees(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    left_unit = normalise(left)
    right_unit = normalise(right)
    if left_unit == (0.0, 0.0) or right_unit == (0.0, 0.0):
        return 180.0
    cosine = min(1.0, max(-1.0, _dot(left_unit, right_unit)))
    return math.degrees(math.acos(cosine))


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _clamp(value: float, lower: float, upper: float) -> float:
    if lower > upper:
        return (lower + upper) / 2.0
    return min(max(value, lower), upper)
