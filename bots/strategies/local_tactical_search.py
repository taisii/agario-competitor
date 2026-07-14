from __future__ import annotations

"""Shallow, wide local game search for deliberate, aggressive movement.

The exact simulator remains responsible for the selected first move.  This
module adds a broad two-step local search in front of it.  Every root action is
compared against many immediate continuations, then the whole plan is rebuilt
next turn.  It deliberately avoids a third step because opponent motion and
the visible state change too quickly for that prediction to remain reliable.
"""

from dataclasses import dataclass, replace
import math
import os

from lib.config.player import EAT_SIZE_RATIO, FOOD_RADIUS, SPLIT_EJECT_SPEED
from strategies.base import StrategyContext, StrategyDecision
from strategies.expected_final_mass import ExpectedFinalMassStrategy
from strategies.features import can_eat_player_blob, normalise, player_speed
from strategies.receding_horizon import (
    Action,
    EnemyBlob,
    SearchNode,
    TAU,
    _rotate,
    _split_chain_attack_reach,
)


@dataclass(frozen=True)
class LocalTarget:
    """One nearby resource shared by every root rollout."""

    kind: str
    pos: tuple[float, float]
    value: float
    identity: tuple[int, int]


@dataclass(frozen=True, slots=True)
class LocalEnemyResponse:
    """Geometry that is constant across every local rollout state."""

    enemy: EnemyBlob
    predator: bool
    reach: float
    speed: float


class LocalTacticalSearchStrategy(ExpectedFinalMassStrategy):
    """Compare many two-step routes against locally rational responses."""

    name = "local_tactical_search"

    _FOOD_RANGE = 11.0
    _TACTICAL_RANGE = 30.0
    # The root field already folds all retained visible food into one vector.
    # The local rollout only needs the six closest points for potential shape.
    _LOCAL_FOOD_LIMIT = 6
    _LOCAL_ROOT_LIMIT = 6
    _TARGET_DIRECTION_LIMIT = 2
    _DEEP_DIRECTION_LIMIT = 3

    def __init__(
        self,
        depth: int | None = None,
        width: int | None = None,
        angular_samples: int | None = None,
    ) -> None:
        # One exact transition validates a wide root set; the local proxy adds
        # the second move.  Replanning next turn is more reliable than a third
        # prediction through rapidly changing opponents.
        super().__init__(
            depth=1 if depth is None else depth,
            width=6 if width is None else width,
            angular_samples=6 if angular_samples is None else angular_samples,
        )
        self.max_food = min(self.max_food, 16)
        self.max_enemies = min(self.max_enemies, 6)
        # The DP is the broad lookahead.  Preserve the submitted strategy's
        # exact-transition bank so it remains available during the endgame.
        self.transition_budget_scale = 2
        self.minimum_transition_budget = 1
        self.local_continuation_prior = float(
            os.environ.get("BOT_LOCAL_CONTINUATION_PRIOR", "0")
        )
        self._local_root_scores: tuple[tuple[Action, float], ...] = ()
        self._local_target_counts: dict[str, int] = {}
        self._local_response_evaluations = 0

    def choose(self, context: StrategyContext) -> StrategyDecision:
        self._local_root_scores = ()
        self._local_target_counts = {}
        self._local_response_evaluations = 0
        decision = super().choose(context)
        diagnostics = dict(decision.diagnostics)
        diagnostics.update(
            local_planner_horizon="2",
            local_target_counts=dict(self._local_target_counts),
            local_roots_ranked=len(self._local_root_scores),
            local_selected_rank=self._local_selected_rank(decision),
            local_top_two_gap=(
                round(
                    self._local_root_scores[0][1] - self._local_root_scores[1][1],
                    6,
                )
                if len(self._local_root_scores) >= 2
                else None
            ),
            local_response_evaluations=self._local_response_evaluations,
        )
        return replace(decision, diagnostics=diagnostics)

    def _local_selected_rank(self, decision: StrategyDecision) -> int | None:
        selected_key = self._action_key(
            Action(decision.direction, decision.split, decision.reason)
        )
        return next(
            (
                rank
                for rank, (action, _) in enumerate(self._local_root_scores, start=1)
                if self._action_key(action) == selected_key
            ),
            None,
        )

    def _turn_cost(
        self,
        previous: tuple[float, float],
        current: tuple[float, float],
    ) -> float:
        """Penalise reversal, not ordinary steering.

        A ninety-degree turn is free.  The cost starts only after movement has
        a component back toward the previous position and rises smoothly to a
        full U-turn.  Escape and capture can still win when reversing is the
        physically best move.
        """

        before = normalise(previous)
        after = normalise(current)
        dot = max(-1.0, min(1.0, before[0] * after[0] + before[1] * after[1]))
        return 2.0 * max(0.0, -dot) ** 2

    def _actions_per_node_limit(self, depth_index: int) -> int:
        return 2 if depth_index == 0 else 1

    def _uses_compute_time_bank(self) -> bool:
        """Use deterministic fixed work rather than wall-clock action choices."""

        return False

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
        if not first_step:
            return tuple(
                sorted(
                    actions,
                    key=lambda action: (
                        self._turn_cost(node.last_direction, action.direction),
                        self._action_key(action),
                    ),
                )
            )

        geometry = self._node_geometry(node)
        center = geometry.center
        local_enemies = self._local_enemies(node)
        future_escape = self._near_future_predator_escape(node, local_enemies)
        if future_escape != (0.0, 0.0):
            actions = (
                Action(future_escape, reason="future_predator_escape"),
                *actions,
            )

        # Replay evidence supports the density field specifically during the
        # fragile small phase.  Once established, retain the submitted policy's
        # prey/virus value ordering instead of continuing to chase pellets.
        use_small_phase_field = node.total_mass <= 4.0
        field_direction, local_food_targets = self._root_food_field_and_targets(
            center,
            foods,
            node.eaten_food_ids,
        )
        if field_direction != (0.0, 0.0):
            actions = (*actions, Action(field_direction, reason="local_food_field"))

        local_dp_started = self._profile_start()
        ranked = self._rank_roots_by_local_dp(
            node=node,
            actions=actions,
            foods=foods,
            viruses=viruses,
            arena_size=arena_size,
            local_enemies=local_enemies,
            local_food_targets=local_food_targets,
            use_food_field=use_small_phase_field,
        )
        self._record_profile("local_dp", local_dp_started)

        # Collision and wall/merge escapes are based on exact imminent
        # geometry.  Keep the two escape alternatives ahead of the approximate
        # DP so an anytime cutoff cannot turn a safety fix into a regression.
        mandatory = tuple(
            action
            for action in actions
            if action.reason.startswith(
                (
                    "dangerous_virus_escape",
                    "urgent_wall",
                    "leader_merge_escape",
                    "future_predator_escape",
                )
            )
        )[:2]
        return self._dedupe_actions((*mandatory, *ranked))

    def _step(
        self,
        *,
        node,
        action,
        foods,
        viruses,
        arena_size: float,
        first_step: bool,
        safety_weight: float,
        aggression: float,
    ):
        result = super()._step(
            node=node,
            action=action,
            foods=foods,
            viruses=viruses,
            arena_size=arena_size,
            first_step=first_step,
            safety_weight=safety_weight,
            aggression=aggression,
        )
        if first_step:
            future_pressure = self._near_future_predator_pressure(result.node)
            if future_pressure > 0.0:
                result = replace(
                    result,
                    node=replace(
                        result.node,
                        score=result.node.score - future_pressure,
                    ),
                )
        if not first_step or not self._local_root_scores:
            return result

        action_key = self._action_key(action)
        rank = next(
            (
                index
                for index, (candidate, _) in enumerate(self._local_root_scores)
                if self._action_key(candidate) == action_key
            ),
            len(self._local_root_scores) - 1,
        )
        # Rank normalisation is robust to the very different raw scales of a
        # food field, a virus pop, and a split capture.  Exact first-step
        # physics can still overturn the DP, but evaluating more roots no
        # longer erases the continuation value that put a root near the top.
        denominator = max(1, len(self._local_root_scores) - 1)
        continuation_prior = self.local_continuation_prior * (1.0 - rank / denominator)
        return replace(
            result,
            node=replace(
                result.node,
                score=result.node.score + continuation_prior,
            ),
        )

    def _rank_roots_by_local_dp(
        self,
        *,
        node: SearchNode,
        actions: tuple[Action, ...],
        foods,
        viruses,
        arena_size: float,
        local_enemies: tuple[EnemyBlob, ...] | None = None,
        local_food_targets: tuple[LocalTarget, ...] | None = None,
        use_food_field: bool = False,
    ) -> tuple[Action, ...]:
        geometry = self._node_geometry(node)
        center = geometry.center
        primary_radius = geometry.primary.radius
        targets = self._local_targets(
            node,
            foods,
            viruses,
            arena_size,
            center=center,
            food_targets=local_food_targets,
        )
        target_counts: dict[str, int] = {}
        for target in targets:
            target_counts[target.kind] = target_counts.get(target.kind, 0) + 1
        self._local_target_counts = target_counts

        if local_enemies is None:
            local_enemies = self._local_enemies(node)
        base_scores = {
            self._action_key(action): score for action, score in self._root_proxy_scores
        }

        def local_root_family(action: Action) -> int:
            reason = action.reason
            if reason.startswith(
                (
                    "dangerous_virus_escape",
                    "urgent_wall",
                    "leader_merge_escape",
                    "future_predator_escape",
                    "leader_imitation_9",
                )
            ):
                family = 0
            elif action.split or "prey" in reason or reason == "leader_imitation_49":
                family = 1
            elif "virus" in reason or reason == "leader_imitation_24":
                family = 2
            elif reason in ("local_food_field", "leader_imitation_59"):
                family = 3
            else:
                family = 4
            return family

        ranked_roots = sorted(
            actions,
            key=lambda action: (
                -base_scores.get(self._action_key(action), 0.0),
                self._action_key(action),
            ),
        )
        selected: list[Action] = []
        for family in range(4):
            representative = next(
                (
                    action
                    for action in ranked_roots
                    if local_root_family(action) == family
                ),
                None,
            )
            if representative is not None:
                selected.append(representative)
        selected.extend(
            action
            for action in ranked_roots
            if action not in selected
        )
        selected_actions = tuple(selected[: self._LOCAL_ROOT_LIMIT])
        unselected_actions = tuple(
            action for action in ranked_roots if action not in selected_actions
        )
        response_facts = self._local_enemy_responses(node, local_enemies)
        response_cache: dict[tuple[int, int, int], float] = {}
        potential_cache: dict[tuple[int, int], float] = {}
        before_potential = self._local_potential(
            center,
            primary_radius,
            targets,
            potential_cache,
        )

        scored: list[tuple[Action, float]] = []
        rollout_cache: dict[tuple[float, float, bool], float] = {}
        move_distance = player_speed(primary_radius)
        last_direction = normalise(node.last_direction)
        for action in selected_actions:
            unit = normalise(action.direction)
            rollout_key = (unit[0], unit[1], action.split)
            cached_rollout = rollout_cache.get(rollout_key)
            if cached_rollout is not None:
                base_value = base_scores.get(self._action_key(action), 0.0)
                scored.append((action, base_value + 100.0 * cached_rollout))
                continue
            first_pos = self._advance_local_position(
                center,
                unit,
                primary_radius,
                arena_size,
                split=action.split,
                unit_direction=True,
                move_distance=move_distance,
            )
            first_value = self._local_step_value(
                node=node,
                before=center,
                after=first_pos,
                previous=last_direction,
                direction=unit,
                depth=1,
                targets=targets,
                response_facts=response_facts,
                response_cache=response_cache,
                potential_cache=potential_cache,
                before_potential=before_potential,
            )
            rollout_value = first_value
            for direction in self._local_deeper_directions(
                first_pos,
                unit,
                targets,
                use_food_field=use_food_field,
            ):
                next_pos = self._advance_local_position(
                    first_pos,
                    direction,
                    primary_radius,
                    arena_size,
                    unit_direction=True,
                    move_distance=move_distance,
                )
                value = first_value + 0.72 * self._local_step_value(
                    node=node,
                    before=first_pos,
                    after=next_pos,
                    previous=unit,
                    direction=direction,
                    depth=2,
                    targets=targets,
                    response_facts=response_facts,
                    response_cache=response_cache,
                    potential_cache=potential_cache,
                )
                rollout_value = max(rollout_value, value)
            rollout_cache[rollout_key] = rollout_value
            base_value = base_scores.get(self._action_key(action), 0.0)
            scored.append((action, base_value + 100.0 * rollout_value))

        scored.sort(key=lambda item: (-item[1], self._action_key(item[0])))
        self._local_root_scores = tuple(scored)
        return (*tuple(action for action, _ in scored), *unselected_actions)

    def _local_targets(
        self,
        node: SearchNode,
        foods,
        viruses,
        arena_size: float,
        *,
        center: tuple[float, float] | None = None,
        food_targets: tuple[LocalTarget, ...] | None = None,
    ) -> tuple[LocalTarget, ...]:
        center = node.center if center is None else center
        if food_targets is None:
            _, food_targets = self._root_food_field_and_targets(
                center,
                foods,
                node.eaten_food_ids,
            )
        targets = list(food_targets)

        for virus in viruses:
            if virus.virus_id in node.consumed_virus_ids:
                continue
            if math.dist(center, virus.pos) > self._TACTICAL_RANGE:
                continue
            expected_mass = self._virus_expected_mass(node, virus, arena_size)
            if expected_mass is None or expected_mass <= 0.0:
                continue
            targets.append(
                LocalTarget(
                    kind="virus",
                    pos=virus.pos,
                    value=expected_mass,
                    identity=(1, int(virus.virus_id)),
                )
            )

        for enemy in node.enemies:
            if enemy.stale_rounds:
                continue
            if math.dist(center, enemy.pos) > self._TACTICAL_RANGE:
                continue
            expected_mass = self._prey_expected_mass(node, enemy, arena_size)
            if expected_mass <= 0.0:
                continue
            targets.append(
                LocalTarget(
                    kind="prey",
                    pos=enemy.pos,
                    value=expected_mass,
                    identity=(2 + enemy.player_id, enemy.blob_id),
                )
            )

        kind_priority = {"prey": 0, "virus": 1, "food": 2}
        return tuple(
            sorted(
                targets,
                key=lambda target: (
                    kind_priority[target.kind],
                    -target.value,
                    math.dist(center, target.pos),
                    target.identity,
                ),
            )
        )

    def _root_food_field_and_targets(
        self,
        center: tuple[float, float],
        foods,
        eaten_food_ids,
    ) -> tuple[tuple[float, float], tuple[LocalTarget, ...]]:
        """Compute the full food field and retained DP points in one scan."""

        field_x = 0.0
        field_y = 0.0
        food_rows: list[tuple[float, LocalTarget]] = []
        range_squared = self._FOOD_RANGE * self._FOOD_RANGE
        for food in foods:
            if food.food_id in eaten_food_ids:
                continue
            dx = food.pos[0] - center[0]
            dy = food.pos[1] - center[1]
            distance_squared = dx * dx + dy * dy
            if distance_squared > range_squared:
                continue
            distance = max(0.25, math.sqrt(distance_squared))
            weight = 1.0 / (distance * distance)
            field_x += dx / distance * weight
            field_y += dy / distance * weight
            food_rows.append(
                (
                    distance_squared,
                    LocalTarget(
                        kind="food",
                        pos=food.pos,
                        value=FOOD_RADIUS * FOOD_RADIUS,
                        identity=(0, int(food.food_id)),
                    ),
                )
            )
        food_rows.sort(key=lambda row: row[0])
        return (
            normalise((field_x, field_y)),
            tuple(target for _, target in food_rows[: self._LOCAL_FOOD_LIMIT]),
        )

    def _local_enemies(self, node: SearchNode) -> tuple[EnemyBlob, ...]:
        center = node.center
        nearby = sorted(node.enemies, key=lambda enemy: math.dist(center, enemy.pos))[
            :3
        ]
        relevant = list(nearby)
        for enemy in node.enemies:
            if enemy in relevant:
                continue
            threatens = any(
                can_eat_player_blob(enemy.radius, own.radius)
                and math.dist(enemy.pos, own.pos)
                <= _split_chain_attack_reach(enemy.radius, own.radius)
                + 3.0 * player_speed(enemy.radius)
                + 4.0
                for own in node.own_blobs
            )
            if threatens:
                relevant.append(enemy)
        return tuple(
            sorted(relevant, key=lambda enemy: (enemy.player_id, enemy.blob_id))[:6]
        )

    def _near_future_predator_escape(
        self,
        node: SearchNode,
        local_enemies: tuple[EnemyBlob, ...] | None = None,
    ) -> tuple[float, float]:
        """Escape larger neighbours that are only a few pellets from lethal."""

        own = node.primary
        escape_x = 0.0
        escape_y = 0.0
        enemies = (
            local_enemies if local_enemies is not None else self._local_enemies(node)
        )
        for enemy in enemies:
            if not self._is_near_future_predator(own, enemy):
                continue
            dx = own.x - enemy.x
            dy = own.y - enemy.y
            distance = max(0.25, math.hypot(dx, dy))
            weight = 1.0 / (distance * distance)
            escape_x += dx / distance * weight
            escape_y += dy / distance * weight
        return normalise((escape_x, escape_y))

    @staticmethod
    def _is_near_future_predator(own, enemy: EnemyBlob) -> bool:
        if enemy.radius <= own.radius:
            return False
        threshold_mass = EAT_SIZE_RATIO * own.mass
        food_to_lethal = max(0.0, threshold_mass - enemy.mass) / (
            FOOD_RADIUS * FOOD_RADIUS
        )
        return food_to_lethal <= 6.0 and math.dist(own.pos, enemy.pos) <= 6.0

    def _near_future_predator_pressure(self, node: SearchNode) -> float:
        own = node.primary
        pressure = 0.0
        for enemy in self._local_enemies(node):
            if not self._is_near_future_predator(own, enemy):
                continue
            distance = math.dist(own.pos, enemy.pos)
            threshold_mass = EAT_SIZE_RATIO * own.mass
            food_to_lethal = max(0.0, threshold_mass - enemy.mass) / (
                FOOD_RADIUS * FOOD_RADIUS
            )
            urgency = max(0.0, 1.0 - food_to_lethal / 6.0)
            proximity = max(0.0, 1.0 - distance / 6.0)
            containment = max(0.0, (enemy.radius - distance) / enemy.radius)
            pressure += (
                4.0 * node.total_mass * urgency * proximity * (1.0 + 2.0 * containment)
            )
        return pressure

    def _local_deeper_directions(
        self,
        pos: tuple[float, float],
        previous: tuple[float, float],
        targets: tuple[LocalTarget, ...],
        *,
        use_food_field: bool,
    ) -> tuple[tuple[float, float], ...]:
        """Build the next actions from the current DP state.

        Official winners repeatedly bend toward the next food rather than
        preserving a route chosen at the root.  Recomputing target directions
        here makes those turns purposeful while retaining the reversal-only
        control cost.
        """

        field_x = 0.0
        field_y = 0.0
        ranked_targets: list[
            tuple[
                int,
                float,
                float,
                tuple[int, int],
                tuple[float, float],
            ]
        ] = []
        kind_priority = {"prey": 0, "virus": 1, "food": 2}
        for target in targets:
            dx = target.pos[0] - pos[0]
            dy = target.pos[1] - pos[1]
            distance_squared = dx * dx + dy * dy
            raw_distance = math.sqrt(distance_squared)
            distance = max(0.25, raw_distance)
            target_direction = (
                (dx / raw_distance, dy / raw_distance)
                if raw_distance > 0.0
                else (0.0, 0.0)
            )
            ranked_targets.append(
                (
                    kind_priority[target.kind],
                    distance_squared,
                    -target.value,
                    target.identity,
                    target_direction,
                )
            )
            if target.kind == "food":
                weight = 1.0 / (distance * distance)
                field_x += dx / distance * weight
                field_y += dy / distance * weight
        food_field = normalise((field_x, field_y)) if use_food_field else (0.0, 0.0)
        ranked_targets.sort(key=lambda row: row[:4])
        directions = [previous]
        directions.extend(
            row[4] for row in ranked_targets[: self._TARGET_DIRECTION_LIMIT]
        )
        directions.extend(
            (
                food_field,
                _rotate(previous, math.pi / 2),
                _rotate(previous, -math.pi / 2),
                _rotate(previous, math.pi / 6),
                _rotate(previous, -math.pi / 6),
            )
        )
        result: list[tuple[float, float]] = []
        seen: set[int] = set()
        for direction in directions:
            if direction == (0.0, 0.0):
                continue
            sector = self._direction_sector(direction)
            if sector in seen:
                continue
            seen.add(sector)
            result.append(direction)
        return tuple(result[: self._DEEP_DIRECTION_LIMIT])

    def _food_field_direction(
        self,
        pos: tuple[float, float],
        targets: tuple[LocalTarget, ...],
    ) -> tuple[float, float]:
        """Gradient of the replay-fitted short-range food potential.

        Every food contributes once, so this is linear in visible resources
        rather than pairwise in map positions.  Ordinary turns are not
        smoothed here; the separate control cost only discourages movement
        with a component back toward the previous position.
        """

        field_x = 0.0
        field_y = 0.0
        for target in targets:
            if target.kind != "food":
                continue
            dx = target.pos[0] - pos[0]
            dy = target.pos[1] - pos[1]
            distance = max(0.25, math.hypot(dx, dy))
            weight = 1.0 / (distance * distance)
            field_x += dx / distance * weight
            field_y += dy / distance * weight
        field = normalise((field_x, field_y))
        return field

    def _local_step_value(
        self,
        *,
        node: SearchNode,
        before: tuple[float, float],
        after: tuple[float, float],
        previous: tuple[float, float],
        direction: tuple[float, float],
        depth: int,
        targets: tuple[LocalTarget, ...],
        response_facts: tuple[LocalEnemyResponse, ...],
        response_cache,
        potential_cache,
        before_potential: float | None = None,
    ) -> float:
        old_potential = (
            before_potential
            if before_potential is not None
            else self._local_potential(
                before,
                node.primary.radius,
                targets,
                potential_cache,
            )
        )
        new_potential = self._local_potential(
            after,
            node.primary.radius,
            targets,
            potential_cache,
        )
        response_value = self._smart_response_value(
            node,
            after,
            depth,
            response_facts,
            response_cache,
        )
        return (
            new_potential
            - old_potential
            + response_value
            - self._turn_cost_between_units(previous, direction)
        )

    @staticmethod
    def _turn_cost_between_units(
        previous: tuple[float, float],
        current: tuple[float, float],
    ) -> float:
        dot = max(
            -1.0,
            min(1.0, previous[0] * current[0] + previous[1] * current[1]),
        )
        return 2.0 * max(0.0, -dot) ** 2

    def _local_potential(
        self,
        pos: tuple[float, float],
        own_radius: float,
        targets: tuple[LocalTarget, ...],
        cache: dict[tuple[int, int], float],
    ) -> float:
        key = (round(pos[0] * 2.0), round(pos[1] * 2.0))
        cached = cache.get(key)
        if cached is not None:
            return cached
        value = 0.0
        for target in targets:
            gap = max(0.0, math.dist(pos, target.pos) - own_radius)
            if target.kind == "food":
                value += 0.22 * target.value * math.exp(-gap / 2.8)
            elif target.kind == "virus":
                value += 1.35 * target.value * math.exp(-gap / 10.0)
            else:
                value += 1.55 * target.value * math.exp(-gap / 9.0)
        cache[key] = value
        return value

    def _local_enemy_responses(
        self,
        node: SearchNode,
        enemies: tuple[EnemyBlob, ...],
    ) -> tuple[LocalEnemyResponse, ...]:
        """Hoist opponent facts shared by all Bellman-style local states."""

        own_radius = node.primary.radius
        facts = []
        for enemy in enemies:
            predator = can_eat_player_blob(enemy.radius, own_radius)
            prey = can_eat_player_blob(own_radius, enemy.radius)
            # Size-neutral neighbours have no term in the local response
            # value, so projecting their motion cannot affect a root score.
            if not predator and not prey:
                continue
            facts.append(
                LocalEnemyResponse(
                    enemy=enemy,
                    predator=predator,
                    reach=(
                        _split_chain_attack_reach(enemy.radius, own_radius)
                        if predator
                        else 0.0
                    ),
                    speed=player_speed(enemy.radius),
                )
            )
        return tuple(facts)

    def _smart_response_value(
        self,
        node: SearchNode,
        own_pos: tuple[float, float],
        depth: int,
        response_facts: tuple[LocalEnemyResponse, ...],
        cache: dict[tuple[int, int, int], float],
    ) -> float:
        key = (round(own_pos[0] * 2.0), round(own_pos[1] * 2.0), depth)
        cached = cache.get(key)
        if cached is not None:
            return cached

        own_radius = node.primary.radius
        total_mass = node.total_mass
        value = 0.0
        for fact in response_facts:
            enemy = fact.enemy
            travel = fact.speed * depth
            current_distance = math.hypot(
                own_pos[0] - enemy.x,
                own_pos[1] - enemy.y,
            )
            if fact.predator:
                # Straight toward us is exactly the minimum-distance response
                # among equal-length candidate moves.
                distance = abs(current_distance - travel)
                margin = distance - fact.reach
                if margin < 10.0:
                    value -= total_mass * (10.0 - margin) / 10.0
            else:
                # Straight away is exactly the maximum-distance prey response.
                distance = current_distance + travel
                gap = max(0.0, distance - own_radius)
                value += 0.6 * enemy.mass * math.exp(-gap / 8.0)
        cache[key] = value
        self._local_response_evaluations += len(response_facts)
        return value

    @staticmethod
    def _direction_sector(direction: tuple[float, float]) -> int:
        return int(round(math.atan2(direction[1], direction[0]) / TAU * 16.0)) % 16

    @staticmethod
    def _advance_local_position(
        pos: tuple[float, float],
        direction: tuple[float, float],
        radius: float,
        arena_size: float,
        *,
        split: bool = False,
        unit_direction: bool = False,
        move_distance: float | None = None,
    ) -> tuple[float, float]:
        unit = direction if unit_direction else normalise(direction)
        distance = (
            player_speed(radius) if move_distance is None else move_distance
        ) + (0.5 * SPLIT_EJECT_SPEED if split else 0.0)
        return (
            max(radius, min(arena_size - radius, pos[0] + unit[0] * distance)),
            max(radius, min(arena_size - radius, pos[1] + unit[1] * distance)),
        )


class LocalTacticalSearchReferenceStrategy(LocalTacticalSearchStrategy):
    """Correctness-first planner used as the optimisation oracle.

    This version deliberately retains the widest two-step action set and root
    validation.  It is not submission-safe: benchmark it only with the local
    engine timeout explicitly raised.  The production strategy must reproduce
    its important choices before any search-width reduction is accepted.
    """

    name = "local_tactical_search_reference"
    _LOCAL_STATE_LIMIT = 32
    _TARGET_DIRECTION_LIMIT = 12
    _DEEP_DIRECTION_LIMIT = 24

    def __init__(
        self,
        depth: int | None = None,
        width: int | None = None,
        angular_samples: int | None = None,
    ) -> None:
        super().__init__(
            depth=1 if depth is None else depth,
            width=32 if width is None else width,
            angular_samples=18 if angular_samples is None else angular_samples,
        )
        self.max_food = 24
        self.max_enemies = 10
        if "BOT_LOCAL_CONTINUATION_PRIOR" not in os.environ:
            self.local_continuation_prior = 4.0

    def _uses_compute_time_bank(self) -> bool:
        return False

    def _transition_budget(
        self,
        own_blob_count: int,
        enemy_count: int = 0,
    ) -> int:
        return 64

    def _actions_per_node_limit(self, depth_index: int) -> int:
        return 64
