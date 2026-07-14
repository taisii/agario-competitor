from __future__ import annotations

"""Expected-final-mass search seeded by strong official-replay policies."""

from dataclasses import replace
import math

from lib.config.arena import MAX_BLOB_COUNT
from lib.config.player import FOOD_RADIUS, STARTING_RADIUS
from strategies.base import StrategyContext, StrategyDecision
from strategies.features import can_eat_player_blob, normalise, player_speed
from strategies.receding_horizon import (
    Action,
    ReplayDominanceStrategy,
    _can_split_eat,
    _rotate,
    _split_chain_attack_reach,
)
from strategies.replay_imitation import (
    observation_from_context,
    observation_regime,
    predict_direction,
    predict_split,
)
from strategies.replay_profiles import PROFILES


class ExpectedFinalMassStrategy(ReplayDominanceStrategy):
    """Maximise final mass while using strong replay policies as proposals.

    Public leaderboard names cannot be mapped safely to replay ``team_id``
    values.  The experts below are therefore selected from the saved official
    cohort by measured final mass and held-out imitation fidelity:

    * team 59 supplies the ordinary growth line;
    * team 49 supplies prey and split-capture proposals;
    * team 24 supplies virus-growth proposals;
    * team 9 supplies proposals whenever a predator is visible.

    Their move is always compared with the current search policy's best proxy
    candidate.  The replay action is not a mode switch or a forced command.
    """

    name = "expected_final_mass"

    # observation_regime uses predator=1, prey=2, edible-virus=4.  Team 49 is
    # selected separately only when its high-fidelity split rule fires; merely
    # seeing distant edible prey must not interrupt an ordinary growth line.
    _LEADER_TEAM_BY_REGIME = (59, 9, 59, 9, 24, 9, 24, 9)

    def __init__(
        self,
        depth: int | None = None,
        width: int | None = None,
        angular_samples: int | None = None,
    ) -> None:
        super().__init__(depth=depth, width=width, angular_samples=angular_samples)
        self._max_rounds = 1400
        self._leader_action: Action | None = None
        self._leader_team_id: int | None = None
        self._leader_last_split_round = -10_000
        self._scoreboard_leader_player_id: int | None = None

    def choose(self, context: StrategyContext) -> StrategyDecision:
        state = context.game.state
        self._max_rounds = max(1, int(state.max_rounds))
        self._scoreboard_leader_player_id = (
            int(state.rankings[0]) if state.rankings else None
        )

        observation = observation_from_context(context)
        regime = observation_regime(observation)
        team_id, leader_direction, leader_split, leader_split_score = (
            self._leader_proposal(observation, regime)
        )
        self._leader_team_id = team_id
        self._leader_action = Action(
            direction=leader_direction,
            split=leader_split,
            reason=f"leader_imitation_{team_id}",
        )

        decision = super().choose(context)
        if decision.split:
            self._leader_last_split_round = int(state.round)
        selected_dot = max(
            -1.0,
            min(
                1.0,
                decision.direction[0] * leader_direction[0]
                + decision.direction[1] * leader_direction[1],
            ),
        )
        diagnostics = dict(decision.diagnostics)
        diagnostics.update(
            objective="expected_final_mass",
            recovery_value=round(self._recovery_terminal_mass(), 4),
            leader_team_id=team_id,
            leader_regime=regime,
            leader_imitation_dot=round(selected_dot, 6),
            leader_imitation_selected=decision.reason == self._leader_action.reason,
            leader_split_suggested=leader_split,
            leader_split_score=round(leader_split_score, 6),
        )
        return replace(decision, diagnostics=diagnostics)

    def _leader_proposal(self, observation, regime: int):
        if regime & 2 and not regime & 1:
            prey_profile = PROFILES[49]
            prey_direction = predict_direction(
                prey_profile,
                observation,
                self.previous_direction,
            )
            prey_split, prey_split_score = predict_split(
                prey_profile,
                observation,
                self.previous_direction,
                prey_direction,
                self._leader_last_split_round,
            )
            if prey_split:
                return 49, prey_direction, prey_split, prey_split_score

        team_id = self._LEADER_TEAM_BY_REGIME[regime]
        profile = PROFILES[team_id]
        direction = predict_direction(
            profile,
            observation,
            self.previous_direction,
        )
        split, split_score = predict_split(
            profile,
            observation,
            self.previous_direction,
            direction,
            self._leader_last_split_round,
        )
        return team_id, direction, split, split_score

    def _candidate_actions(
        self,
        *,
        node,
        foods,
        food_targets,
        viruses,
        arena_size: float,
        first_step: bool,
        allow_split: bool = True,
        angle_offset: int = 0,
    ):
        actions = super()._candidate_actions(
            node=node,
            foods=foods,
            food_targets=food_targets,
            viruses=viruses,
            arena_size=arena_size,
            first_step=first_step,
            allow_split=allow_split,
            angle_offset=angle_offset,
        )
        if not first_step or self._leader_action is None:
            return actions
        leader_action = self._leader_action
        if leader_action.split and not allow_split:
            leader_action = replace(leader_action, split=False)
        urgent_wall_escapes = self._urgent_wall_escape_actions(
            node,
            arena_size,
        )
        virus_collision_escapes = self._dangerous_virus_escape_actions(
            node,
            viruses,
            arena_size,
        )
        leader_merge_escapes = self._imminent_leader_merge_escape_actions(
            node,
            arena_size,
        )
        if virus_collision_escapes:
            # A low-retention virus that will be touched next turn is a
            # collision hazard, not merely an unattractive growth target.
            # Official matches 19574 and 19599 consumed such a virus while the
            # selected action was labelled prey/imitation; place two verified
            # non-contact roots before the anytime cutoff.
            non_split_leader = replace(leader_action, split=False)
            return self._dedupe_actions(
                (*virus_collision_escapes, non_split_leader, *actions)
            )
        # The deterministic search budget evaluates at least two roots.  Put
        # wall tangents first only for the geometry seen in official loss
        # 19212: the radial escape was clamped into a corner and the predator
        # consumed us twenty rounds later.  In ordinary space, retain the
        # leader proposal beside the policy's highest-value native action.
        if urgent_wall_escapes:
            return self._dedupe_actions((*urgent_wall_escapes, leader_action, *actions))
        if leader_merge_escapes:
            # Attacking one visible piece creates smaller cells just before the
            # unseen remainder recombines.  Match 19229 ended four rounds after
            # that exact transition, so split proposals are physically
            # dominated until the leader has merged or left the danger zone.
            non_split_leader = replace(leader_action, split=False)
            non_split_actions = tuple(action for action in actions if not action.split)
            return self._dedupe_actions(
                (*leader_merge_escapes, non_split_leader, *non_split_actions)
            )
        return self._dedupe_actions((leader_action, *actions))

    def _dangerous_virus_escape_actions(
        self,
        node,
        viruses,
        arena_size: float,
    ) -> tuple[Action, ...]:
        """Return non-contact moves for an unsafe virus reachable next turn."""

        hazards: list[tuple[object, object, float]] = []
        for virus in viruses:
            if virus.virus_id in node.consumed_virus_ids:
                continue
            for origin in node.own_blobs:
                if not self._can_still_consume_virus_at_contact(origin, virus):
                    continue
                gap = max(0.0, math.dist(origin.pos, virus.pos) - origin.radius)
                if gap > player_speed(origin.radius):
                    continue
                retention = self._virus_retained_mass_fraction(
                    node,
                    origin,
                    virus,
                    arena_size,
                )
                if retention >= 0.75:
                    continue
                hazards.append((origin, virus, retention))

        if not hazards:
            return ()

        escape_x = 0.0
        escape_y = 0.0
        for origin, virus, retention in hazards:
            away = normalise((origin.x - virus.pos[0], origin.y - virus.pos[1]))
            weight = 1.0 - retention
            escape_x += away[0] * weight
            escape_y += away[1] * weight
        predator_escape = self._escape_vector(node)
        escape = normalise(
            (
                escape_x + predator_escape[0],
                escape_y + predator_escape[1],
            )
        )
        if escape == (0.0, 0.0):
            escape = normalise((escape_x, escape_y))
        if escape == (0.0, 0.0):
            return ()

        candidates = self._dedupe_actions(
            (
                Action(escape, reason="dangerous_virus_escape"),
                Action(
                    _rotate(escape, math.pi / 4),
                    reason="dangerous_virus_escape_tangent",
                ),
                Action(
                    _rotate(escape, -math.pi / 4),
                    reason="dangerous_virus_escape_tangent",
                ),
                Action(
                    normalise((escape_x, escape_y)),
                    reason="dangerous_virus_escape",
                ),
            )
        )

        def collision_count(action: Action) -> int:
            count = 0
            for origin, virus, _ in hazards:
                moved = self._move_own(origin, action.direction, arena_size)
                if math.dist(moved.pos, virus.pos) <= moved.radius:
                    count += 1
            return count

        ranked = sorted(
            candidates,
            key=lambda action: (
                collision_count(action),
                -self._movement_efficiency(
                    node.own_blobs,
                    action.direction,
                    arena_size,
                ),
            ),
        )
        safe = tuple(action for action in ranked if collision_count(action) == 0)
        return safe[:2]

    def _imminent_leader_merge_escape_actions(
        self,
        node,
        arena_size: float,
    ) -> tuple[Action, ...]:
        """Avoid the visible edge of a leader that is about to recombine.

        Match 19229 showed only two equal-sized pieces of the leading player
        inside our view.  They looked edible in isolation, so the old policy
        chased them and a nearby virus.  The other fourteen pieces recombined
        four rounds later and swept the pop.  The scoreboard proves the owner
        has at least our total mass, while equal virus fragments let us infer a
        conservative full 16-piece radius from the visible samples.
        """

        geometry = self._leader_merge_geometry(node)
        if geometry is None:
            return ()
        fragment_center, inferred_radius = geometry
        vulnerable_radius = min(blob.radius for blob in node.own_blobs)
        danger_radius = inferred_radius
        if _can_split_eat(inferred_radius, vulnerable_radius):
            danger_radius = max(
                danger_radius,
                _split_chain_attack_reach(inferred_radius, vulnerable_radius),
            )

        if math.dist(node.center, fragment_center) > danger_radius + 8.0:
            return ()
        escape = normalise(
            (
                node.center[0] - fragment_center[0],
                node.center[1] - fragment_center[1],
            )
        )
        if escape == (0.0, 0.0):
            return ()

        candidates = (
            Action(escape, reason="leader_merge_escape"),
            Action(
                _rotate(escape, math.pi / 4),
                reason="leader_merge_escape_tangent",
            ),
            Action(
                _rotate(escape, -math.pi / 4),
                reason="leader_merge_escape_tangent",
            ),
        )
        ranked = sorted(
            candidates,
            key=lambda action: (
                -(
                    0.75
                    * (
                        action.direction[0] * escape[0]
                        + action.direction[1] * escape[1]
                    )
                    + 0.25
                    * self._movement_efficiency(
                        node.own_blobs,
                        action.direction,
                        arena_size,
                    )
                )
            ),
        )
        return tuple(ranked[:2])

    def _leader_merge_geometry(
        self,
        node,
    ) -> tuple[tuple[float, float], float] | None:
        """Infer the imminent whole-player geometry from visible equal pieces."""

        leader_id = self._scoreboard_leader_player_id
        if leader_id is None or leader_id == self._own_player_id:
            return None
        fragments = tuple(
            enemy
            for enemy in node.enemies
            if enemy.player_id == leader_id
            and enemy.stale_rounds == 0
            and 0 < enemy.merge_cooldown <= 12
        )
        if len(fragments) < 2:
            return None
        fragment_masses = tuple(fragment.mass for fragment in fragments)
        if max(fragment_masses) > min(fragment_masses) * 1.15:
            return None
        visible_mass = sum(fragment_masses)
        sampled_piece_mass = visible_mass / len(fragments)
        inferred_leader_mass = max(
            node.total_mass * 1.1,
            sampled_piece_mass * MAX_BLOB_COUNT,
        )
        return (
            (
                sum(fragment.x * fragment.mass for fragment in fragments)
                / visible_mass,
                sum(fragment.y * fragment.mass for fragment in fragments)
                / visible_mass,
            ),
            math.sqrt(inferred_leader_mass),
        )

    def _leader_merge_survival(
        self,
        node,
        arena_size: float,
        safety_weight: float,
    ) -> tuple[float, float]:
        """Return retained mass/probability against an inferred leader merge."""

        geometry = self._leader_merge_geometry(node)
        if geometry is None:
            return node.total_mass, 1.0
        center, inferred_radius = geometry
        midpoint = (
            self.survival_midpoint_base + self.survival_midpoint_scale * safety_weight
        )
        temperature = max(self.survival_temperature, 0.1)
        safe_mass = 0.0
        probabilities: list[float] = []
        for own in node.own_blobs:
            danger_radius = inferred_radius
            if _can_split_eat(inferred_radius, own.radius):
                danger_radius = max(
                    danger_radius,
                    _split_chain_attack_reach(inferred_radius, own.radius),
                )
            virtual_enemy = replace(
                node.enemies[0],
                player_id=int(self._scoreboard_leader_player_id),
                blob_id=-10_000,
                x=center[0],
                y=center[1],
                radius=inferred_radius,
                stale_rounds=0,
            )
            margin = (
                math.dist(own.pos, center)
                - danger_radius
                - self._wall_trap_factor(own, virtual_enemy, arena_size) * 4.0
            )
            scaled = max(-40.0, min(40.0, (margin - midpoint) / temperature))
            probability = 1.0 / (1.0 + math.exp(-scaled))
            probabilities.append(probability)
            safe_mass += own.mass * probability
        return safe_mass, max(probabilities, default=0.0)

    def _urgent_wall_escape_actions(
        self,
        node,
        arena_size: float,
    ) -> tuple[Action, ...]:
        """Expose both viable wall tangents before an anytime cutoff.

        A direct predator potential points outside the arena when the prey is
        between a predator and a corner.  The base policy already creates wide
        tangents, but proxy sorting could place them beyond the two-transition
        budget.  This method detects the continuous wall/predator geometry and
        promotes both tangents as candidates; the exact expected-mass search
        still chooses between them rather than forcing either direction.
        """

        escape = self._escape_vector(node)
        if escape == (0.0, 0.0):
            return ()

        urgent = False
        for own in node.own_blobs:
            for enemy in self._risk_enemies(node.enemies):
                if not can_eat_player_blob(enemy.radius, own.radius):
                    continue
                danger_radius = enemy.radius
                if _can_split_eat(enemy.radius, own.radius):
                    danger_radius = max(
                        danger_radius,
                        _split_chain_attack_reach(enemy.radius, own.radius),
                    )
                distance = math.dist(own.pos, enemy.pos)
                if (
                    distance <= danger_radius + 12.0
                    and self._wall_trap_factor(own, enemy, arena_size) >= 0.25
                ):
                    urgent = True
                    break
            if urgent:
                break
        if not urgent:
            return ()

        candidates = (
            Action(
                _rotate(escape, math.pi / 2),
                reason="urgent_wall_tangent",
            ),
            Action(
                _rotate(escape, -math.pi / 2),
                reason="urgent_wall_tangent",
            ),
        )
        return tuple(
            sorted(
                candidates,
                key=lambda action: (
                    -self._movement_efficiency(
                        node.own_blobs,
                        action.direction,
                        arena_size,
                    )
                ),
            )
        )

    def _movement_efficiency(
        self,
        own_blobs,
        direction: tuple[float, float],
        arena_size: float,
    ) -> float:
        """Return mass-weighted useful speed after arena clamping."""

        return self._move_own_blobs(
            own_blobs,
            direction,
            arena_size,
            calculate_blocked=False,
            profile_counter="expected_final_mass_movement_moves",
        ).efficiency

    def _safety_weight(self, rank_position: int, progress: float) -> float:
        # Rank is not the competition objective.  Absolute mass and the
        # probability of retaining it already price risk in _search_utility.
        return 1.3

    def _recovery_terminal_mass(self) -> float:
        """Smooth replay-calibrated final mass after an elimination now.

        In the saved official cohort, eliminations with roughly 600 rounds
        remaining ended near 9 mass, while eliminations in the final 300 rounds
        ended near starting mass.  With more than 700 rounds remaining the
        average recovered to about 30.  A logistic curve represents that
        transition without round-specific branches.
        """

        remaining_rounds = max(0.0, self._max_rounds - self._current_round)
        starting_mass = STARTING_RADIUS * STARTING_RADIUS
        recoverable_growth = 29.0 / (1.0 + math.exp(-(remaining_rounds - 680.0) / 90.0))
        return starting_mass + recoverable_growth

    def _opportunity_weights(self) -> tuple[float, ...]:
        remaining_fraction = max(
            0.0,
            min(1.0, (self._max_rounds - self._current_round) / self._max_rounds),
        )
        # The nearest opportunity transfers one-for-one into retained mass.
        # Earlier in the match, preserve more of the visible chain because
        # strong replay winners repeatedly farmed after the first acquisition.
        return (
            1.0,
            0.35 + 0.35 * remaining_fraction,
            0.10 + 0.35 * remaining_fraction,
            0.25 * remaining_fraction,
            0.15 * remaining_fraction,
        )

    def _search_utility(
        self,
        node,
        *,
        foods,
        viruses,
        arena_size: float,
        safety_weight: float,
        hazard_summary=None,
    ) -> float:
        """Return expected terminal mass rather than a placement surrogate."""

        if not node.own_blobs:
            return -1_000_000.0

        hazard = hazard_summary or self._hazard_summary(
            node.own_blobs,
            node.enemies,
            safety_weight,
            arena_size,
        )
        merge_safe_mass, merge_continuation = self._leader_merge_survival(
            node,
            arena_size,
            safety_weight,
        )
        safe_mass = min(hazard.safe_mass, merge_safe_mass)
        continuation_probability = min(
            hazard.continuation_probability,
            merge_continuation,
        )
        leader_merge_visible = self._leader_merge_geometry(node) is not None

        opportunities: list[float] = []
        for food in foods:
            if food.food_id in node.eaten_food_ids:
                continue
            gap = min(
                max(0.0, math.dist(own.pos, food.pos) - own.radius)
                for own in node.own_blobs
            )
            opportunities.append(FOOD_RADIUS * FOOD_RADIUS * math.exp(-gap / 6.0))

        for virus in viruses:
            if virus.virus_id in node.consumed_virus_ids:
                continue
            expected_mass = self._virus_expected_mass(node, virus, arena_size)
            if expected_mass is not None:
                opportunities.append(expected_mass)

        for enemy in node.enemies:
            if enemy.stale_rounds:
                continue
            if (
                leader_merge_visible
                and enemy.player_id == self._scoreboard_leader_player_id
            ):
                continue
            expected_mass = self._prey_expected_mass(node, enemy, arena_size)
            if expected_mass > 0.0:
                opportunities.append(expected_mass)

        opportunities.sort(reverse=True)
        opportunity_mass = sum(
            value * weight
            for value, weight in zip(
                opportunities[:5],
                self._opportunity_weights(),
                strict=False,
            )
        )

        recovery_mass = self._recovery_terminal_mass()
        starting_mass = STARTING_RADIUS * STARTING_RADIUS
        survival_growth_baseline = max(0.0, recovery_mass - starting_mass)
        expected_terminal_mass = (
            safe_mass
            + continuation_probability * (survival_growth_baseline + opportunity_mass)
            + (1.0 - continuation_probability) * recovery_mass
        )
        return 100.0 * expected_terminal_mass
