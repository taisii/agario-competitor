from __future__ import annotations

"""Semantic policy with a bounded replay-teacher residual correction.

ReplayDominanceStrategy is used only by the offline distillation script. At
runtime this strategy evaluates one fixed 16-feature linear model, then turns
the semantic decision toward that prediction by a small bounded angle.
"""

from dataclasses import dataclass, replace
import math
import os

from lib.config.player import EAT_SIZE_RATIO
from strategies.asset_preservation import AssetPreservationLayer
from strategies.base import StrategyContext, StrategyDecision
from strategies.features import normalise
from strategies.semantic_potential import SemanticPotentialStrategy


REPLAY_BASE_FEATURE_COUNT = 15
REPLAY_TEACHER_SOURCE_MATCHES = (
    29835,
    29837,
    29839,
    29844,
    29846,
    29848,
    29851,
    29852,
    29854,
    29855,
    29856,
    29857,
    29858,
    29859,
    29860,
    29863,
    29868,
    29869,
    29870,
    29874,
    29875,
    29877,
    29878,
    29880,
    29881,
    29882,
    29887,
    29899,
    29900,
    29904,
    29905,
)

REPLAY_DIRECTION_WEIGHTS = (
    0.24538051181733736,
    -0.001402331718921498,
    -0.007737576683473285,
    0.15961109316439875,
    0.001388974005099115,
    -0.10040882039123859,
    0.17207000987717694,
    -0.09763905346031417,
    0.3454865038961096,
    0.2489313013940314,
    0.03824473287382844,
    -0.02659844923954518,
    0.36780469905569757,
    0.0013993011064784287,
    0.24353856980533245,
    -0.024298909696751674,
)

REPLAY_REGIME_DIRECTION_WEIGHTS = (
    (
        0.14607379318673538,
        -0.00357188724434127,
        0.005827183682603147,
        0.1367792045310645,
        -0.0007157246665640646,
        0.026754152981747485,
        0.014285210872505501,
        -0.0851223469759452,
        0.591132922638927,
        0.0,
        0.0,
        0.0,
        0.0,
        0.01753385021881769,
        0.0,
        0.005133165528105063,
    ),
    (
        0.26991343755632285,
        -0.03392337392948684,
        -0.0185131496159031,
        -0.11048901463892002,
        0.0169269968812004,
        -0.4684740162522696,
        0.6806319887507931,
        -0.08987726287609413,
        0.21871119558702282,
        0.0,
        0.0,
        -0.0671326078406401,
        0.5433531259386959,
        -0.05016322718355247,
        0.0,
        -0.0961187585573933,
    ),
    (
        0.06648609751306966,
        0.00788449823654352,
        -0.011548349457631724,
        0.05594264718215277,
        0.006238537976275045,
        0.04733945906192125,
        -0.058675179676333095,
        0.000702442535823534,
        0.12792238973313652,
        0.4549499527546033,
        0.19717935459436842,
        0.0,
        0.0,
        0.045957155838963974,
        0.0,
        0.03073468186586924,
    ),
    (
        0.1718776169448905,
        0.010996355514679111,
        -0.027957802500767983,
        -0.11199300573523854,
        -0.027078629092385464,
        -0.34523663057755216,
        0.4682402080906482,
        -0.05956286642951294,
        0.17276853496288003,
        0.03827611327451614,
        0.1657459838372322,
        0.1310346519739391,
        0.4745215290826366,
        0.036881899855086185,
        0.0,
        -0.027304214425158867,
    ),
    (
        0.28720766767570954,
        0.05883300560148809,
        0.005068330685978223,
        0.06447746187534817,
        0.01575486494422642,
        -0.1687175708539692,
        0.18269158100155836,
        -0.01206295288759845,
        0.1498168801713857,
        0.0,
        0.0,
        0.0,
        0.0,
        0.9360072154829111,
        0.44459028371187265,
        0.0,
    ),
    (
        0.18987050790395488,
        -0.030985903663138498,
        -0.12534757104478741,
        -0.04871389055185611,
        -0.03021080949045292,
        -0.027015894251002925,
        0.08104037721444624,
        -0.04452184221042525,
        0.11839624675638416,
        0.0,
        0.0,
        0.24571463210283365,
        0.3087390306195449,
        -0.20575228499832937,
        0.31410240512911425,
        0.0,
    ),
    (
        0.2735815591084654,
        0.0006492589768460609,
        -0.03873025940898409,
        0.10920119327247496,
        -0.04680433121140073,
        -0.5612313744909915,
        0.5662887855895851,
        -0.0014149087103031682,
        0.07175838493238217,
        -0.0019471842016412291,
        0.17513081318410612,
        0.0,
        0.0,
        0.15980083576497348,
        0.3269581772778925,
        0.0,
    ),
    (
        0.22662900377381232,
        0.05060588389374343,
        -0.018924795661298593,
        0.02778126772977374,
        -0.03547990107941047,
        -0.8443264331754818,
        0.8819023342805054,
        0.08073067009936141,
        -0.07020982446878961,
        -0.0797019966651491,
        0.1936983715238543,
        -0.14192357417367946,
        0.6453738924695248,
        0.5324073321486968,
        0.19362898245717325,
        0.0,
    ),
)

DEFAULT_MAX_TEACHER_CORRECTION_DEGREES = 2.5


class ReplayDistilledStrategy(SemanticPotentialStrategy):
    """Replay-teacher residual constrained by the semantic safety policy."""

    name = "replay_distilled"

    def __init__(self) -> None:
        super().__init__()
        self._distilled_previous_direction = (0.0, 0.0)
        self._max_teacher_correction = math.radians(
            max(
                0.0,
                _environment_float(
                    "REPLAY_DISTILLED_MAX_CORRECTION_DEGREES",
                    DEFAULT_MAX_TEACHER_CORRECTION_DEGREES,
                ),
            )
        )
        self._correction_safety_margin = _environment_optional_nonnegative_float(
            "REPLAY_DISTILLED_CORRECTION_SAFETY_MARGIN"
        )
        self._asset_preservation = AssetPreservationLayer()

    def choose(self, context: StrategyContext) -> StrategyDecision:
        semantic = super().choose(context)
        base_features, regime = _replay_feature_vectors(
            context,
            self._distilled_previous_direction,
        )
        features = (semantic.direction,) + base_features
        teacher_direction = _weighted_direction(
            REPLAY_REGIME_DIRECTION_WEIGHTS[regime],
            features,
            fallback=semantic.direction,
        )
        safety_margins = tuple(
            float(value)
            for value in (
                semantic.diagnostics.get("current_safety_margin"),
                semantic.diagnostics.get("selected_safety_margin"),
            )
            if isinstance(value, int | float) and math.isfinite(float(value))
        )
        correction_suppressed = (
            self._correction_safety_margin is not None
            and safety_margins
            and min(safety_margins) < self._correction_safety_margin
        )
        corrected = (
            semantic.direction
            if correction_suppressed
            else _rotate_toward(
                semantic.direction,
                teacher_direction,
                self._max_teacher_correction,
            )
        )
        correction_degrees = _angle_degrees(semantic.direction, corrected)
        diagnostics = dict(semantic.diagnostics)
        diagnostics.update(
            {
                "replay_teacher_regime": regime,
                "replay_teacher_correction_degrees": correction_degrees,
                "replay_teacher_correction_suppressed": correction_suppressed,
                "replay_teacher_correction_safety_margin": (
                    self._correction_safety_margin
                ),
                "replay_teacher_source_matches": len(
                    REPLAY_TEACHER_SOURCE_MATCHES
                ),
            }
        )
        distilled = replace(
            semantic,
            direction=corrected,
            reason=(
                "replay_teacher_residual"
                if correction_degrees > 1e-6
                else semantic.reason
            ),
            diagnostics=diagnostics,
        )
        final = self._asset_preservation.adjust(context, distilled)
        self._distilled_previous_direction = final.direction
        self._last_direction = final.direction
        return final


def _weighted_direction(
    weights: tuple[float, ...],
    features: tuple[tuple[float, float], ...],
    *,
    fallback: tuple[float, float],
) -> tuple[float, float]:
    if len(weights) != len(features):
        raise ValueError("replay teacher weights do not match feature vectors")
    direction = normalise(
        (
            sum(weight * vector[0] for weight, vector in zip(weights, features)),
            sum(weight * vector[1] for weight, vector in zip(weights, features)),
        )
    )
    return direction if direction != (0.0, 0.0) else normalise(fallback)


def _rotate_toward(
    source: tuple[float, float],
    target: tuple[float, float],
    max_angle: float,
) -> tuple[float, float]:
    source = normalise(source)
    target = normalise(target)
    if source == (0.0, 0.0):
        return target
    if target == (0.0, 0.0) or max_angle <= 0.0:
        return source
    cross = source[0] * target[1] - source[1] * target[0]
    dot = max(-1.0, min(1.0, source[0] * target[0] + source[1] * target[1]))
    delta = math.atan2(cross, dot)
    bounded = max(-max_angle, min(max_angle, delta))
    cosine = math.cos(bounded)
    sine = math.sin(bounded)
    return normalise(
        (
            source[0] * cosine - source[1] * sine,
            source[0] * sine + source[1] * cosine,
        )
    )


def _angle_degrees(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    first = normalise(first)
    second = normalise(second)
    dot = max(-1.0, min(1.0, first[0] * second[0] + first[1] * second[1]))
    return math.degrees(math.acos(dot))


def _environment_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def _environment_optional_nonnegative_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) and value >= 0.0 else None


@dataclass(frozen=True, slots=True)
class _ReplayFeatureEntity:
    x: float
    y: float
    radius: float = 0.0
    merge_cooldown: int = 0


@dataclass(frozen=True, slots=True)
class _ReplayEntitySummary:
    nearest_vector: tuple[float, float]
    field_vector: tuple[float, float]


def _replay_feature_vectors(
    context: StrategyContext,
    previous_direction: tuple[float, float],
) -> tuple[tuple[tuple[float, float], ...], int]:
    state = context.game.state
    own = tuple(
        _ReplayFeatureEntity(
            x=blob.pos[0],
            y=blob.pos[1],
            radius=blob.radius,
            merge_cooldown=blob.merge_cooldown,
        )
        for blob in state.me.blobs.values()
    )
    visible = tuple(
        _ReplayFeatureEntity(
            x=blob.pos[0],
            y=blob.pos[1],
            radius=blob.radius,
            merge_cooldown=blob.merge_cooldown,
        )
        for blob in state.visible_blobs
    )
    foods = tuple(
        _ReplayFeatureEntity(x=food.pos[0], y=food.pos[1])
        for food in state.visible_food
    )
    viruses = tuple(
        _ReplayFeatureEntity(
            x=virus.pos[0],
            y=virus.pos[1],
            radius=virus.radius,
        )
        for virus in state.visible_viruses
    )
    own_masses = tuple(blob.radius * blob.radius for blob in own)
    total_mass = sum(own_masses)
    smallest_own_mass = min(own_masses, default=math.inf)
    largest_own_mass = max(own_masses, default=0.0)
    center = (
        (
            sum(blob.x * mass for blob, mass in zip(own, own_masses))
            / total_mass,
            sum(blob.y * mass for blob, mass in zip(own, own_masses))
            / total_mass,
        )
        if total_mass > 1e-9
        else (30.0, 30.0)
    )

    predators: list[_ReplayFeatureEntity] = []
    prey: list[_ReplayFeatureEntity] = []
    neutral: list[_ReplayFeatureEntity] = []
    for entity in visible:
        entity_mass = entity.radius * entity.radius
        if entity_mass >= smallest_own_mass * EAT_SIZE_RATIO:
            predators.append(entity)
        elif largest_own_mass >= entity_mass * EAT_SIZE_RATIO:
            prey.append(entity)
        else:
            neutral.append(entity)

    edible_viruses: list[_ReplayFeatureEntity] = []
    dangerous_viruses: list[_ReplayFeatureEntity] = []
    for virus in viruses:
        target = (
            edible_viruses
            if largest_own_mass
            > virus.radius * virus.radius * EAT_SIZE_RATIO
            else dangerous_viruses
        )
        target.append(virus)

    food = _summarize_replay_entities(center, foods)
    prey_summary = _summarize_replay_entities(center, prey)
    predator_summary = _summarize_replay_entities(
        center,
        predators,
        away=True,
    )
    neutral_summary = _summarize_replay_entities(
        center,
        neutral,
        away=True,
    )
    edible_virus_summary = _summarize_replay_entities(
        center,
        edible_viruses,
    )
    dangerous_virus_summary = _summarize_replay_entities(
        center,
        dangerous_viruses,
        away=True,
    )

    arena_size = float(state.map.size)
    left = max(center[0], 0.15)
    right = max(arena_size - center[0], 0.15)
    bottom = max(center[1], 0.15)
    top = max(arena_size - center[1], 0.15)
    wall = _replay_unit(
        (
            1.0 / left - 1.0 / right,
            1.0 / bottom - 1.0 / top,
        )
    )
    previous = _replay_unit(previous_direction)
    features = (
        (1.0, 0.0),
        (0.0, 1.0),
        previous,
        (-previous[1], previous[0]),
        _replay_unit(
            (
                arena_size / 2.0 - center[0],
                arena_size / 2.0 - center[1],
            )
        ),
        wall,
        food.nearest_vector,
        food.field_vector,
        prey_summary.nearest_vector,
        prey_summary.field_vector,
        predator_summary.nearest_vector,
        predator_summary.field_vector,
        neutral_summary.nearest_vector,
        edible_virus_summary.nearest_vector,
        dangerous_virus_summary.nearest_vector,
    )
    regime = (
        (1 if predators else 0)
        | (2 if prey else 0)
        | (4 if edible_viruses else 0)
    )
    return features, regime


def _summarize_replay_entities(
    origin: tuple[float, float],
    entities: tuple[_ReplayFeatureEntity, ...] | list[_ReplayFeatureEntity],
    *,
    away: bool = False,
) -> _ReplayEntitySummary:
    nearest: _ReplayFeatureEntity | None = None
    nearest_distance_squared = math.inf
    field_x = 0.0
    field_y = 0.0
    sign = -1.0 if away else 1.0
    for entity in entities:
        dx = entity.x - origin[0]
        dy = entity.y - origin[1]
        distance_squared = dx * dx + dy * dy
        if distance_squared < nearest_distance_squared:
            nearest = entity
            nearest_distance_squared = distance_squared
        if distance_squared <= 1e-9:
            continue
        scale = sign / (distance_squared + 0.25)
        field_x += dx * scale
        field_y += dy * scale
    nearest_vector = (
        _replay_unit(
            (
                sign * (nearest.x - origin[0]),
                sign * (nearest.y - origin[1]),
            )
        )
        if nearest is not None
        else (0.0, 0.0)
    )
    return _ReplayEntitySummary(
        nearest_vector=nearest_vector,
        field_vector=_replay_unit((field_x, field_y)),
    )


def _replay_unit(
    vector: tuple[float, float],
) -> tuple[float, float]:
    magnitude = math.hypot(*vector)
    if magnitude <= 1e-9 or not math.isfinite(magnitude):
        return (0.0, 0.0)
    return (vector[0] / magnitude, vector[1] / magnitude)


assert len(REPLAY_DIRECTION_WEIGHTS) == 1 + REPLAY_BASE_FEATURE_COUNT
assert len(REPLAY_REGIME_DIRECTION_WEIGHTS) == 8
assert all(
    len(weights) == 1 + REPLAY_BASE_FEATURE_COUNT
    for weights in REPLAY_REGIME_DIRECTION_WEIGHTS
)
