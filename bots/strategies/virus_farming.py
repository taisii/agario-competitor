from __future__ import annotations

import math
from dataclasses import dataclass

from lib.config.arena import MAX_BLOB_COUNT
from lib.config.player import (
    BASE_PLAYER_SPEED,
    EAT_SIZE_RATIO,
    MASS_DECAY_RATE,
    MERGE_ATTRACTION_SPEED,
    MIN_PLAYER_SPEED,
    PLAYER_SPEED_RADIUS_FACTOR,
    SAME_PLAYER_OVERLAP_EPSILON,
    SPLIT_EJECT_SPEED,
    SPLIT_MIN_MASS,
    STARTING_RADIUS,
)
from lib.models.blob_model import BlobModel, VisibleBlobModel
from lib.models.virus_model import VirusModel
from simulation.rules import (
    can_consume_virus,
    decayed_mass_after_turns,
    movement_speed,
    virus_replacement_positions,
)
from strategies.base import StrategyContext, StrategyDecision
from strategies.receding_horizon import ThreatAwareRecedingHorizonStrategy
from strategies.features import (
    can_eat_player_blob,
    normalise,
    vector_from_to,
    virus_center_clearance,
)
from strategies.greedy import SurvivalGreedyStrategy
from strategies.potential_field import PotentialFieldHunterStrategy


SQRT2 = math.sqrt(2.0)
POST_SPLIT_REACTION_TURNS = 3
POST_SPLIT_POSITION_UNCERTAINTY = 0.75
MIN_POST_SPLIT_SAFETY_MARGIN = 0.0
UNSAFE_VIRUS_AVOIDANCE_BUFFER = 1.0
MERGE_RISK_HORIZON = 8
MASS_PRESERVATION_TARGET = 40.0
COMPETITIVE_MASS_PRESERVATION_TARGET = 16.0


@dataclass(frozen=True)
class VirusPlan:
    virus: VirusModel
    hunter: BlobModel
    center_distance: float
    contact_distance: float
    turns_to_contact: int
    projected_mass_at_contact: float
    projected_piece_count: int
    projected_piece_radius: float
    post_split_predator_count: int
    post_split_safety_margin: float


@dataclass(frozen=True)
class VirusPlanSearch:
    plan: VirusPlan | None
    visible_virus_count: int
    currently_consumable_pairs: int
    decay_rejected_pairs: int
    post_split_rejected_pairs: int
    mass_target_rejected_pairs: int
    safest_rejected_margin: float | None
    unsafe_virus_ids: tuple[int, ...]
    minimum_required_radius: float | None

    @property
    def unavailable_reason(self) -> str:
        if self.visible_virus_count == 0:
            return "no_visible_virus"
        if self.currently_consumable_pairs == 0:
            return "insufficient_blob_mass"
        if self.decay_rejected_pairs == self.currently_consumable_pairs:
            return "mass_decays_before_contact"
        if self.mass_target_rejected_pairs:
            return "mass_target_preservation"
        if self.post_split_rejected_pairs:
            return "post_split_predator_risk"
        return "no_reachable_virus"


class VirusHunterStrategy:
    """Prioritise reachable viruses while retaining an emergency escape path.

    Consuming a virus is deliberately the primary objective of this policy.
    The engine resolves mass decay before virus collisions, so a target is only
    considered reachable when the chosen blob will still pass the strict virus
    mass threshold on the estimated contact turn.
    """

    name = "virus_hunter"

    def __init__(
        self,
        danger_margin: float = 3.0,
        *,
        use_receding_horizon_growth: bool = True,
    ) -> None:
        self._survival = SurvivalGreedyStrategy(danger_margin=danger_margin)
        # The paired benchmark showed that rank/progress multipliers reduced
        # mass without improving wins.  Virus-specific preservation below is
        # the safety mechanism; ordinary growth keeps one consistent value
        # function throughout the match.
        self._growth = ThreatAwareRecedingHorizonStrategy(
            endgame_adaptation=False
        )
        self._use_receding_horizon_growth = use_receding_horizon_growth
        self._last_direction = (1.0, 0.0)
        self._mass_target_reached = False
        self._mass_preservation_reason: str | None = None

    def choose(self, context: StrategyContext) -> StrategyDecision:
        state = context.game.state
        own_blobs = tuple(state.me.blobs.values())
        if not own_blobs:
            return StrategyDecision(direction=(1.0, 0.0), reason="dead_fallback")
        total_mass = sum(blob.radius * blob.radius for blob in own_blobs)
        progress = int(state.round) / max(1, int(state.max_rounds))
        try:
            rank_position = tuple(state.rankings).index(state.me.player_id) + 1
        except ValueError:
            rank_position = len(tuple(state.rankings)) + 1
        if total_mass <= STARTING_RADIUS * STARTING_RADIUS * 1.05:
            self._mass_target_reached = False
            self._mass_preservation_reason = None
        elif total_mass >= MASS_PRESERVATION_TARGET:
            self._mass_target_reached = True
            self._mass_preservation_reason = "mass_target"
        elif (
            progress >= 0.5
            and rank_position <= 2
            and total_mass >= COMPETITIVE_MASS_PRESERVATION_TARGET
        ):
            # Match histories repeatedly showed a top-two bot with mass 19-29
            # entering a virus, becoming sixteen edible fragments, and losing
            # almost everything before merge.  Latch preservation once the bot
            # is already competitive; food and safe prey remain available.
            self._mass_target_reached = True
            self._mass_preservation_reason = "competitive_position"

        # A nearby predator remains an emergency override. Search virus safety
        # first even in that state: an escape line can otherwise cross a virus
        # that turns the currently safe large blob into edible fragments.
        fallback = self._survival.choose(context)
        projected_consumers = self._projected_consumers(own_blobs)
        arena_size = float(state.map.size or 60.0)

        search = self._search_virus_plan(
            own_blobs=projected_consumers,
            viruses=tuple(state.visible_viruses),
            enemies=tuple(state.visible_blobs),
            arena_size=arena_size,
            preserve_mass=self._mass_target_reached,
        )
        search_diagnostics = self._search_diagnostics(search)
        if fallback.reason == "predator_near":
            emergency = self._with_mode(
                fallback,
                "emergency_escape",
                search_diagnostics,
            )
            return self._remember(self._avoid_unsafe_virus_collision(
                decision=emergency,
                consumers=projected_consumers,
                viruses=tuple(state.visible_viruses),
                arena_size=arena_size,
            ))

        plan = search.plan
        if plan is not None:
            return self._remember(StrategyDecision(
                direction=normalise(vector_from_to(plan.hunter.pos, plan.virus.pos)),
                target_kind="virus",
                target_id=str(plan.virus.virus_id),
                reason="reachable_virus",
                score=-float(plan.turns_to_contact),
                diagnostics={
                    "virus_hunter_mode": "pursue_virus",
                    "hunter_blob_id": plan.hunter.blob_id,
                    "virus_center_distance": plan.center_distance,
                    "virus_contact_distance": plan.contact_distance,
                    "turns_to_contact": plan.turns_to_contact,
                    "projected_mass_at_contact": plan.projected_mass_at_contact,
                    "projected_pieces_created": plan.projected_piece_count,
                    "projected_piece_radius": plan.projected_piece_radius,
                    "post_split_predator_count": plan.post_split_predator_count,
                    "post_split_safety_margin": self._finite_or_none(
                        plan.post_split_safety_margin
                    ),
                    "currently_consumable_pairs": search.currently_consumable_pairs,
                    "decay_rejected_pairs": search.decay_rejected_pairs,
                    "post_split_rejected_pairs": search.post_split_rejected_pairs,
                    "mass_target_rejected_pairs": search.mass_target_rejected_pairs,
                    "mass_target_latched": self._mass_target_reached,
                    "mass_preservation_reason": self._mass_preservation_reason,
                    "unsafe_virus_ids": list(search.unsafe_virus_ids),
                },
            ))

        # Small fragments cannot consume a virus. Grow them using the existing
        # safe prey/food policy until at least one fragment can reach a virus.
        growth = self._with_mode(
            self._suppress_preservation_split(self._growth_decision(context)),
            "grow_for_virus",
            search_diagnostics,
        )
        unsafe_ids = set(search.unsafe_virus_ids)
        return self._remember(self._avoid_unsafe_virus_collision(
            decision=growth,
            consumers=projected_consumers,
            viruses=tuple(
                virus
                for virus in state.visible_viruses
                if virus.virus_id in unsafe_ids
            ),
            arena_size=arena_size,
        ))

    def _search_virus_plan(
        self,
        *,
        own_blobs: tuple[BlobModel, ...],
        viruses: tuple[VirusModel, ...],
        enemies: tuple[VisibleBlobModel, ...],
        arena_size: float,
        preserve_mass: bool = False,
    ) -> VirusPlanSearch:
        plans: list[VirusPlan] = []
        currently_consumable_pairs = 0
        decay_rejected_pairs = 0
        post_split_rejected_pairs = 0
        mass_target_rejected_pairs = 0
        rejected_margins: list[float] = []
        unsafe_virus_ids: set[int] = set()
        for blob in own_blobs:
            for virus in viruses:
                if not can_consume_virus(
                    blob.radius,
                    virus.radius,
                    eat_size_ratio=EAT_SIZE_RATIO,
                ):
                    continue
                currently_consumable_pairs += 1

                center_distance = math.dist(blob.pos, virus.pos)
                contact_distance = max(
                    0.0,
                    center_distance - blob.radius,
                )
                turns_to_contact = max(
                    1,
                    math.ceil(contact_distance / self._speed(blob.radius)),
                    blob.merge_cooldown,
                )
                projected_mass = self._mass_after_decay(
                    blob.radius * blob.radius,
                    turns_to_contact,
                )
                if not can_consume_virus(
                    math.sqrt(projected_mass),
                    virus.radius,
                    eat_size_ratio=EAT_SIZE_RATIO,
                ):
                    decay_rejected_pairs += 1
                    continue

                piece_count = max(1, MAX_BLOB_COUNT - len(own_blobs) + 1)
                mass_after_consumption = (
                    projected_mass + virus.radius * virus.radius
                )
                piece_radius = math.sqrt(mass_after_consumption / piece_count)
                if preserve_mass and piece_count > 1:
                    mass_target_rejected_pairs += 1
                    unsafe_virus_ids.add(virus.virus_id)
                    continue
                safety_margin, predator_count = self._post_split_safety(
                    hunter=blob,
                    virus=virus,
                    piece_count=piece_count,
                    piece_radius=piece_radius,
                    turns_to_contact=turns_to_contact,
                    enemies=enemies,
                    arena_size=arena_size,
                )
                if safety_margin < MIN_POST_SPLIT_SAFETY_MARGIN:
                    post_split_rejected_pairs += 1
                    rejected_margins.append(safety_margin)
                    unsafe_virus_ids.add(virus.virus_id)
                    continue

                plans.append(
                    VirusPlan(
                        virus=virus,
                        hunter=blob,
                        center_distance=center_distance,
                        contact_distance=contact_distance,
                        turns_to_contact=turns_to_contact,
                        projected_mass_at_contact=projected_mass,
                        projected_piece_count=piece_count,
                        projected_piece_radius=piece_radius,
                        post_split_predator_count=predator_count,
                        post_split_safety_margin=safety_margin,
                    )
                )

        plan = (
            min(
                plans,
                key=lambda item: (
                    item.turns_to_contact,
                    -item.post_split_safety_margin,
                    item.contact_distance,
                    -(item.virus.radius * item.virus.radius),
                    item.virus.virus_id,
                    item.hunter.blob_id,
                ),
            )
            if plans
            else None
        )
        return VirusPlanSearch(
            plan=plan,
            visible_virus_count=len(viruses),
            currently_consumable_pairs=currently_consumable_pairs,
            decay_rejected_pairs=decay_rejected_pairs,
            post_split_rejected_pairs=post_split_rejected_pairs,
            mass_target_rejected_pairs=mass_target_rejected_pairs,
            safest_rejected_margin=(max(rejected_margins) if rejected_margins else None),
            unsafe_virus_ids=tuple(sorted(unsafe_virus_ids)),
            minimum_required_radius=(
                min(virus.radius for virus in viruses) * math.sqrt(EAT_SIZE_RATIO)
                if viruses
                else None
            ),
        )

    def _projected_consumers(
        self,
        own_blobs: tuple[BlobModel, ...],
    ) -> tuple[BlobModel, ...]:
        """Combine connected fragments that will merge within the risk horizon.

        Movement decrements cooldown, then attraction and touching-blob merges
        run before virus resolution. Predicting several turns ahead gives the
        group time to steer away before a virus center enters the aggregate
        radius; waiting until cooldown one is too late for a large fragment grid.
        """

        count = len(own_blobs)
        parents = list(range(count))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        attraction_closure = 2.0 * MERGE_ATTRACTION_SPEED + SAME_PLAYER_OVERLAP_EPSILON
        for left in range(count):
            if own_blobs[left].merge_cooldown > MERGE_RISK_HORIZON:
                continue
            for right in range(left + 1, count):
                if own_blobs[right].merge_cooldown > MERGE_RISK_HORIZON:
                    continue
                if math.dist(own_blobs[left].pos, own_blobs[right].pos) <= (
                    own_blobs[left].radius
                    + own_blobs[right].radius
                    + attraction_closure
                ):
                    union(left, right)

        components: dict[int, list[BlobModel]] = {}
        for index, blob in enumerate(own_blobs):
            components.setdefault(find(index), []).append(blob)

        consumers: list[BlobModel] = []
        for blobs in components.values():
            if len(blobs) == 1:
                blob = blobs[0]
                # Cooldown prevents same-player merging, not food, prey, or
                # virus consumption by this already-existing blob.
                consumers.append(
                    BlobModel(
                        blob_id=blob.blob_id,
                        pos=blob.pos,
                        radius=blob.radius,
                        merge_cooldown=0,
                    )
                )
                continue
            total_mass = sum(blob.radius * blob.radius for blob in blobs)
            consumers.append(
                BlobModel(
                    blob_id=min(blob.blob_id for blob in blobs),
                    pos=(
                        sum(blob.pos[0] * blob.radius * blob.radius for blob in blobs)
                        / total_mass,
                        sum(blob.pos[1] * blob.radius * blob.radius for blob in blobs)
                        / total_mass,
                    ),
                    radius=math.sqrt(total_mass),
                    merge_cooldown=max(blob.merge_cooldown for blob in blobs),
                )
            )
        return tuple(consumers)

    def _avoid_unsafe_virus_collision(
        self,
        *,
        decision: StrategyDecision,
        consumers: tuple[BlobModel, ...],
        viruses: tuple[VirusModel, ...],
        arena_size: float,
    ) -> StrategyDecision:
        if not viruses:
            return decision

        base_direction = normalise(decision.direction)
        base_clearance = self._next_virus_clearance(
            direction=base_direction,
            consumers=consumers,
            viruses=viruses,
            arena_size=arena_size,
        )
        if base_clearance >= UNSAFE_VIRUS_AVOIDANCE_BUFFER:
            return decision

        candidates = [base_direction]
        candidates.extend(
            (math.cos(index * math.pi / 8.0), math.sin(index * math.pi / 8.0))
            for index in range(16)
        )
        scored: list[tuple[float, float, tuple[float, float]]] = []
        seen: set[tuple[int, int]] = set()
        for candidate in candidates:
            candidate = normalise(candidate)
            key = (round(candidate[0] * 1000), round(candidate[1] * 1000))
            if key in seen:
                continue
            seen.add(key)
            clearance = self._next_virus_clearance(
                direction=candidate,
                consumers=consumers,
                viruses=viruses,
                arena_size=arena_size,
            )
            alignment = (
                candidate[0] * base_direction[0]
                + candidate[1] * base_direction[1]
            )
            scored.append((clearance, alignment, candidate))

        feasible = [
            item for item in scored
            if item[0] >= UNSAFE_VIRUS_AVOIDANCE_BUFFER
        ]
        if feasible:
            clearance, _alignment, direction = max(
                feasible,
                key=lambda item: (
                    item[1] + min(item[0], 4.0) * 0.35,
                    item[0],
                ),
            )
        else:
            clearance, _alignment, direction = max(
                scored,
                key=lambda item: (item[0], item[1]),
            )

        return StrategyDecision(
            direction=direction,
            target_kind="avoid_virus",
            reason="avoid_unsafe_virus",
            diagnostics={
                **decision.diagnostics,
                "virus_hunter_mode": "avoid_unsafe_virus",
                "interrupted_reason": decision.reason,
                "interrupted_target_kind": decision.target_kind,
                "unsafe_virus_ids": [virus.virus_id for virus in viruses],
                "fallback_collision_clearance": base_clearance,
                "selected_collision_clearance": clearance,
            },
        )

    def _next_virus_clearance(
        self,
        *,
        direction: tuple[float, float],
        consumers: tuple[BlobModel, ...],
        viruses: tuple[VirusModel, ...],
        arena_size: float,
    ) -> float:
        minimum = math.inf
        for blob in consumers:
            turns = max(1, blob.merge_cooldown)
            projected_mass = self._mass_after_decay(blob.radius * blob.radius, turns)
            projected_radius = math.sqrt(projected_mass)
            x = min(
                max(
                    projected_radius,
                    blob.pos[0] + direction[0] * self._speed(blob.radius) * turns,
                ),
                arena_size - projected_radius,
            )
            y = min(
                max(
                    projected_radius,
                    blob.pos[1] + direction[1] * self._speed(blob.radius) * turns,
                ),
                arena_size - projected_radius,
            )
            for virus in viruses:
                if not can_consume_virus(
                    math.sqrt(projected_mass),
                    virus.radius,
                    eat_size_ratio=EAT_SIZE_RATIO,
                ):
                    continue
                minimum = min(
                    minimum,
                    virus_center_clearance(
                        (x, y),
                        projected_radius,
                        virus.pos,
                    ),
                )
        return minimum

    def _search_diagnostics(self, search: VirusPlanSearch) -> dict[str, object]:
        return {
            "virus_unavailable_reason": search.unavailable_reason,
            "currently_consumable_pairs": search.currently_consumable_pairs,
            "decay_rejected_pairs": search.decay_rejected_pairs,
            "post_split_rejected_pairs": search.post_split_rejected_pairs,
            "mass_target_rejected_pairs": search.mass_target_rejected_pairs,
            "mass_target_latched": self._mass_target_reached,
            "mass_preservation_reason": self._mass_preservation_reason,
            "safest_rejected_margin": search.safest_rejected_margin,
            "unsafe_virus_ids": list(search.unsafe_virus_ids),
            "minimum_required_radius": search.minimum_required_radius,
        }

    def _growth_decision(self, context: StrategyContext) -> StrategyDecision:
        state = context.game.state
        if not self._use_receding_horizon_growth or not (
            hasattr(context.query, "update")
            and hasattr(state, "view_center")
            and hasattr(state, "vision_size")
        ):
            return self._survival.choose(context)
        self._growth.previous_direction = self._last_direction
        return self._growth.choose(context)

    def _suppress_preservation_split(
        self,
        decision: StrategyDecision,
    ) -> StrategyDecision:
        if not self._mass_target_reached or not decision.split:
            return decision
        return StrategyDecision(
            direction=decision.direction,
            split=False,
            target_kind=decision.target_kind,
            target_id=decision.target_id,
            reason=decision.reason,
            score=decision.score,
            diagnostics={
                **decision.diagnostics,
                "offensive_split_requested": True,
                "offensive_split_allowed": False,
                "split_suppressed_reason": "mass_preservation",
            },
        )

    def _remember(self, decision: StrategyDecision) -> StrategyDecision:
        direction = normalise(decision.direction)
        if direction != (0.0, 0.0):
            self._last_direction = direction
        return decision

    def _post_split_safety(
        self,
        *,
        hunter: BlobModel,
        virus: VirusModel,
        piece_count: int,
        piece_radius: float,
        turns_to_contact: int,
        enemies: tuple[VisibleBlobModel, ...],
        arena_size: float,
    ) -> tuple[float, int]:
        # At the blob cap, virus consumption grows one existing blob and does
        # not create a new vulnerable group. It is therefore never less safe
        # than the pre-consumption state handled by the emergency fallback.
        if piece_count == 1:
            return (math.inf, 0)

        contact_center = self._projected_contact_center(
            hunter=hunter,
            virus=virus,
            turns_to_contact=turns_to_contact,
            arena_size=arena_size,
        )
        spread = self._replacement_spread(piece_radius, piece_count)
        minimum_margin = math.inf
        predator_count = 0
        for enemy in enemies:
            if not can_eat_player_blob(enemy.radius, piece_radius):
                continue
            predator_count += 1
            normal_reach = enemy.radius + self._speed(enemy.radius) * (
                turns_to_contact + POST_SPLIT_REACTION_TURNS
            )
            attack_reach = normal_reach
            if (
                enemy.radius * enemy.radius >= SPLIT_MIN_MASS
                and can_eat_player_blob(enemy.radius / SQRT2, piece_radius)
            ):
                split_reach = (
                    self._speed(enemy.radius) * turns_to_contact
                    + self._split_attack_reach(enemy.radius)
                    + POST_SPLIT_POSITION_UNCERTAINTY
                )
                attack_reach = max(attack_reach, split_reach)

            margin = (
                math.dist(enemy.pos, contact_center)
                - spread
                - attack_reach
                - POST_SPLIT_POSITION_UNCERTAINTY
            )
            minimum_margin = min(minimum_margin, margin)
        return (minimum_margin, predator_count)

    def _projected_contact_center(
        self,
        *,
        hunter: BlobModel,
        virus: VirusModel,
        turns_to_contact: int,
        arena_size: float,
    ) -> tuple[float, float]:
        direction = normalise(vector_from_to(hunter.pos, virus.pos))
        travel = min(
            math.dist(hunter.pos, virus.pos),
            self._speed(hunter.radius) * turns_to_contact,
        )
        x = hunter.pos[0] + direction[0] * travel
        y = hunter.pos[1] + direction[1] * travel
        return (
            min(max(hunter.radius, x), arena_size - hunter.radius),
            min(max(hunter.radius, y), arena_size - hunter.radius),
        )

    def _replacement_spread(self, piece_radius: float, piece_count: int) -> float:
        positions = virus_replacement_positions(
            center_x=0.0,
            center_y=0.0,
            piece_radius=piece_radius,
            piece_count=piece_count,
            overlap_epsilon=SAME_PLAYER_OVERLAP_EPSILON,
        )
        return max((math.hypot(x, y) for x, y in positions), default=0.0)

    def _split_attack_reach(self, radius: float) -> float:
        child_radius = radius / SQRT2
        return (
            3.0 * child_radius
            + SPLIT_EJECT_SPEED
            + self._speed(child_radius)
        )

    def _mass_after_decay(self, mass: float, turns: int) -> float:
        return decayed_mass_after_turns(
            mass,
            turns,
            decay_rate=MASS_DECAY_RATE,
            minimum_radius=STARTING_RADIUS,
        )

    def _speed(self, radius: float) -> float:
        return movement_speed(
            radius,
            base_speed=BASE_PLAYER_SPEED,
            radius_factor=PLAYER_SPEED_RADIUS_FACTOR,
            minimum_speed=MIN_PLAYER_SPEED,
        )

    def _finite_or_none(self, value: float) -> float | None:
        return value if math.isfinite(value) else None

    def _with_mode(
        self,
        decision: StrategyDecision,
        mode: str,
        diagnostics: dict[str, object] | None = None,
    ) -> StrategyDecision:
        return StrategyDecision(
            direction=decision.direction,
            split=decision.split,
            target_kind=decision.target_kind,
            target_id=decision.target_id,
            reason=decision.reason,
            score=decision.score,
            diagnostics={
                **decision.diagnostics,
                "virus_hunter_mode": mode,
                **(diagnostics or {}),
            },
        )

class PotentialFieldVirusFarmerStrategy:
    """Aggressive potential growth with safety-gated virus farming.

    PotentialHunter supplies prey/food pressure, while VirusHunter owns virus
    decisions, projected fragment safety, and mass preservation.
    """

    name = "potential_field_virus_farmer"

    def __init__(self, danger_margin: float = 3.0) -> None:
        # This wrapper supplies its own growth action. Disable VirusHunter's
        # heavier receding-horizon fallback so the discarded result does not consume
        # the engine's cumulative response budget.
        self._virus_policy = VirusHunterStrategy(
            danger_margin=danger_margin,
            use_receding_horizon_growth=False,
        )
        self._growth_policy = PotentialFieldHunterStrategy()

    def choose(self, context: StrategyContext) -> StrategyDecision:
        state = context.game.state
        own_blobs = tuple(state.me.blobs.values())
        virus_decision = self._virus_policy.choose(context)
        virus_mode = str(
            virus_decision.diagnostics.get("virus_hunter_mode")
            or "virus_fallback"
        )

        if not own_blobs:
            return self._with_mode(virus_decision, "dead_fallback")

        # Safe pursuit, emergency escape, and unsafe-virus collision avoidance
        # are authoritative. Potential growth may only replace the explicit
        # grow_for_virus fallback.
        if virus_mode != "grow_for_virus":
            return self._with_mode(virus_decision, virus_mode)

        growth = self._growth_policy.choose(context)
        total_mass = sum(blob.radius * blob.radius for blob in own_blobs)
        unavailable_reason = virus_decision.diagnostics.get(
            "virus_unavailable_reason"
        )

        split_suppressed_reason: str | None = None
        allow_split = bool(growth.split)
        if growth.split and len(own_blobs) != 1:
            allow_split = False
            split_suppressed_reason = "fragmented"
        elif growth.split and bool(
            virus_decision.diagnostics.get("mass_target_latched")
        ):
            allow_split = False
            split_suppressed_reason = "mass_target_preservation"
        elif growth.split and unavailable_reason != "no_visible_virus":
            # Preserve the individually capable blob while a visible virus is
            # temporarily too risky or just below the mass threshold.
            allow_split = False
            split_suppressed_reason = "visible_virus_growth"

        return StrategyDecision(
            direction=growth.direction,
            split=allow_split,
            target_kind=growth.target_kind,
            target_id=growth.target_id,
            reason=growth.reason,
            score=growth.score,
            diagnostics={
                **growth.diagnostics,
                **virus_decision.diagnostics,
                "potential_field_virus_farmer_mode": "potential_growth",
                "growth_reason": growth.reason,
                "growth_target_kind": growth.target_kind,
                "total_mass": total_mass,
                "offensive_split_requested": growth.split,
                "offensive_split_allowed": allow_split,
                "split_suppressed_reason": split_suppressed_reason,
            },
        )

    def _with_mode(
        self,
        decision: StrategyDecision,
        mode: str,
    ) -> StrategyDecision:
        return StrategyDecision(
            direction=decision.direction,
            split=decision.split,
            target_kind=decision.target_kind,
            target_id=decision.target_id,
            reason=decision.reason,
            score=decision.score,
            diagnostics={
                **decision.diagnostics,
                "potential_field_virus_farmer_mode": mode,
            },
        )
