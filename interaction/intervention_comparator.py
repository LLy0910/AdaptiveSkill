import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


# =========================================================
# PROJECT PATH
#
# Allows this file to be run directly:
#
# python interaction/intervention_comparator.py
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# =========================================================
# PROJECT IMPORTS
# =========================================================

from task.task_constraints import (
    TaskConstraintResult,
    TaskConstraintEvaluator,
    STATE_VALID,
    STATE_RISK,
    STATE_VIOLATION,
    STATE_TRACKING_UNRELIABLE,
)


# =========================================================
# DECISION LABELS
# =========================================================

DECISION_STAY_QUIET = "STAY_QUIET"
DECISION_INTERVENE = "INTERVENE"
DECISION_PAUSE = "PAUSE"


# =========================================================
# COMPARISON RESULT
# =========================================================

@dataclass
class InterventionComparisonResult:
    """
    Comparison between:

    1. Internal reference-similarity baseline
    2. Task-aware intervention policy

    IMPORTANT:
    The reference-similarity baseline is an internal
    AdaptiveSkill comparison condition.

    It is NOT claimed to represent the state of the art
    in Learning from Demonstration.
    """

    reference_distance: float

    reference_policy_decision: str
    reference_policy_reason: str

    task_policy_decision: str
    task_policy_reason: str

    unnecessary_intervention_avoided: bool
    task_relevant_intervention: bool

    comparison_label: str


# =========================================================
# GEOMETRY
# =========================================================

def point_to_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    """
    Compute shortest Euclidean distance
    from point P to line segment AB.
    """

    ab_x = bx - ax
    ab_y = by - ay

    ap_x = px - ax
    ap_y = py - ay

    ab_len_sq = (
        ab_x * ab_x
        +
        ab_y * ab_y
    )

    # Degenerate segment.
    if ab_len_sq < 1e-12:
        return math.hypot(
            px - ax,
            py - ay,
        )

    t = (
        ap_x * ab_x
        +
        ap_y * ab_y
    ) / ab_len_sq

    t = max(
        0.0,
        min(
            1.0,
            t,
        )
    )

    closest_x = (
        ax
        +
        t * ab_x
    )

    closest_y = (
        ay
        +
        t * ab_y
    )

    return math.hypot(
        px - closest_x,
        py - closest_y,
    )


def point_to_polyline_distance(
    x: float,
    y: float,
    points: List[Tuple[float, float]],
) -> float:
    """
    Compute shortest distance from a point
    to an expert-reference polyline.
    """

    if len(points) == 0:
        return float("inf")

    if len(points) == 1:
        return math.hypot(
            x - points[0][0],
            y - points[0][1],
        )

    best_distance = float(
        "inf"
    )

    for index in range(
        len(points) - 1
    ):
        ax, ay = points[
            index
        ]

        bx, by = points[
            index + 1
        ]

        current_distance = (
            point_to_segment_distance(
                x,
                y,
                ax,
                ay,
                bx,
                by,
            )
        )

        best_distance = min(
            best_distance,
            current_distance,
        )

    return best_distance


# =========================================================
# INTERNAL REFERENCE-SIMILARITY BASELINE
# =========================================================

class ReferenceSimilarityPolicy:
    """
    Internal AdaptiveSkill comparison baseline.

    Simple rule:

        sufficiently far from expert reference
        ->
        intervene

    This intentionally represents the conceptual limitation
    of treating reference deviation as intervention need.

    It is NOT presented as a state-of-the-art LfD method.
    """

    def __init__(
        self,
        intervention_threshold: float = 0.30,
    ):
        self.intervention_threshold = float(
            intervention_threshold
        )


    def decide(
        self,
        reference_distance: float,
        tracking_reliable: bool = True,
    ):
        # -------------------------------------------------
        # Tracking unavailable
        # -------------------------------------------------

        if not tracking_reliable:
            return (
                DECISION_PAUSE,
                "Tracking is unreliable.",
            )

        # -------------------------------------------------
        # Reference deviation
        # -------------------------------------------------

        if (
            reference_distance
            >
            self.intervention_threshold
        ):
            return (
                DECISION_INTERVENE,
                (
                    "Deviation from the expert reference "
                    "exceeds the internal baseline threshold."
                ),
            )

        # -------------------------------------------------
        # Near reference
        # -------------------------------------------------

        return (
            DECISION_STAY_QUIET,
            (
                "Movement remains within the "
                "expert-reference corridor."
            ),
        )


# =========================================================
# TASK-AWARE INTERVENTION POLICY
# =========================================================

class TaskAwareInterventionPolicy:
    """
    Minimal task-aware intervention policy.

    VALID:
        stay quiet

    RISK:
        intervene

    VIOLATION:
        intervene

    TRACKING_UNRELIABLE:
        pause judgement

    Key conceptual distinction:

        reference deviation
        !=
        task-relevant intervention need
    """

    def decide(
        self,
        task_result: TaskConstraintResult,
    ):
        # -------------------------------------------------
        # Tracking unreliable
        # -------------------------------------------------

        if (
            task_result.state
            ==
            STATE_TRACKING_UNRELIABLE
        ):
            return (
                DECISION_PAUSE,
                (
                    "Tracking is unreliable. "
                    "Task judgement is paused."
                ),
            )

        # -------------------------------------------------
        # Actual task violation
        # -------------------------------------------------

        if (
            task_result.state
            ==
            STATE_VIOLATION
        ):
            return (
                DECISION_INTERVENE,
                (
                    "A task constraint has been violated."
                ),
            )

        # -------------------------------------------------
        # Task-relevant risk
        # -------------------------------------------------

        if (
            task_result.state
            ==
            STATE_RISK
        ):
            return (
                DECISION_INTERVENE,
                (
                    "The movement is approaching "
                    "a task-relevant constraint."
                ),
            )

        # -------------------------------------------------
        # Task remains valid
        # -------------------------------------------------

        if (
            task_result.state
            ==
            STATE_VALID
        ):
            return (
                DECISION_STAY_QUIET,
                (
                    "The movement remains "
                    "task-valid so far."
                ),
            )

        # Defensive fallback.
        return (
            DECISION_PAUSE,
            "Unknown task state.",
        )


# =========================================================
# INTERVENTION COMPARATOR
# =========================================================

class InterventionComparator:
    """
    Compare:

        A. Reference-similarity baseline
        B. Task-aware intervention policy

    The purpose is NOT to prove superiority.

    The purpose is to expose controlled counterexamples
    where reference similarity and intervention need
    disagree.
    """

    def __init__(
        self,
        expert_reference_points,
        reference_threshold=0.30,
    ):
        self.expert_reference_points = [
            (
                float(x),
                float(y),
            )
            for x, y
            in expert_reference_points
        ]

        self.reference_policy = (
            ReferenceSimilarityPolicy(
                intervention_threshold=
                    reference_threshold
            )
        )

        self.task_policy = (
            TaskAwareInterventionPolicy()
        )


    def compare(
        self,
        x,
        y,
        task_result,
    ):
        # =================================================
        # REFERENCE DISTANCE
        # =================================================

        reference_distance = (
            point_to_polyline_distance(
                float(x),
                float(y),
                self.expert_reference_points,
            )
        )

        # =================================================
        # REFERENCE-SIMILARITY POLICY
        # =================================================

        (
            reference_decision,
            reference_reason,
        ) = self.reference_policy.decide(
            reference_distance=
                reference_distance,

            tracking_reliable=
                task_result.tracking_reliable,
        )

        # =================================================
        # TASK-AWARE POLICY
        # =================================================

        (
            task_decision,
            task_reason,
        ) = self.task_policy.decide(
            task_result
        )

        # =================================================
        # INTERPRETATION
        # =================================================

        # -------------------------------------------------
        # Reference policy wants intervention,
        # but task remains valid.
        #
        # This is the central counterexample.
        # -------------------------------------------------

        unnecessary_intervention_avoided = (

            reference_decision
            ==
            DECISION_INTERVENE

            and

            task_decision
            ==
            DECISION_STAY_QUIET
        )

        # -------------------------------------------------
        # Task-aware policy intervenes because
        # an actual task constraint is at risk.
        # -------------------------------------------------

        task_relevant_intervention = (

            task_decision
            ==
            DECISION_INTERVENE

            and

            task_result.state
            in (
                STATE_RISK,
                STATE_VIOLATION,
            )
        )

        # =================================================
        # COMPARISON LABEL
        # =================================================

        if unnecessary_intervention_avoided:

            comparison_label = (
                "UNNECESSARY INTERVENTION AVOIDED"
            )

        elif task_relevant_intervention:

            comparison_label = (
                "TASK-RELEVANT INTERVENTION"
            )

        elif (
            reference_decision
            ==
            DECISION_PAUSE

            or

            task_decision
            ==
            DECISION_PAUSE
        ):

            comparison_label = (
                "JUDGEMENT PAUSED"
            )

        else:

            comparison_label = (
                "POLICIES AGREE"
            )

        # =================================================
        # RETURN
        # =================================================

        return InterventionComparisonResult(

            reference_distance=
                reference_distance,

            reference_policy_decision=
                reference_decision,

            reference_policy_reason=
                reference_reason,

            task_policy_decision=
                task_decision,

            task_policy_reason=
                task_reason,

            unnecessary_intervention_avoided=
                unnecessary_intervention_avoided,

            task_relevant_intervention=
                task_relevant_intervention,

            comparison_label=
                comparison_label,
        )


# =========================================================
# SELF TEST
# =========================================================

if __name__ == "__main__":

    evaluator = (
        TaskConstraintEvaluator()
    )


    # =====================================================
    # INTERNAL EXPERT REFERENCE
    #
    # Expert demonstration takes the UPPER route.
    #
    # This allows us to construct the key counterexample:
    #
    # lower route
    #     =
    # far from expert reference
    #
    # but
    #
    # lower route
    #     =
    # still task-valid
    #
    # Therefore:
    #
    # reference deviation
    #     !=
    # intervention need
    # =====================================================

    expert_reference = [

        (
            0.0,
            0.0,
        ),

        (
            0.70,
            -0.75,
        ),

        (
            1.10,
            -0.90,
        ),

        (
            2.20,
            -0.90,
        ),

        (
            2.60,
            -0.60,
        ),

        (
            3.20,
            0.0,
        ),
    ]


    comparator = (
        InterventionComparator(
            expert_reference_points=
                expert_reference,

            reference_threshold=
                0.30,
        )
    )


    # =====================================================
    # TESTS
    # =====================================================

    tests = [

        # -------------------------------------------------
        # 1. Reference-like + task-valid
        #
        # Both policies should remain quiet.
        # -------------------------------------------------

        (
            "REFERENCE-LIKE VALID",

            {
                "x": 1.65,
                "y": -0.90,

                "orientation_relative_deg":
                    0.0,

                "tracking_missing_sec":
                    0.0,
            },

            DECISION_STAY_QUIET,

            DECISION_STAY_QUIET,

            "POLICIES AGREE",
        ),


        # -------------------------------------------------
        # 2. Alternative lower route
        #
        # Far from expert reference
        # but still task-valid.
        #
        # This is the most important test.
        # -------------------------------------------------

        (
            "ALTERNATIVE VALID LOWER ROUTE",

            {
                "x": 1.65,
                "y": 0.90,

                "orientation_relative_deg":
                    0.0,

                "tracking_missing_sec":
                    0.0,
            },

            DECISION_INTERVENE,

            DECISION_STAY_QUIET,

            "UNNECESSARY INTERVENTION AVOIDED",
        ),


        # -------------------------------------------------
        # 3. Obstacle risk
        #
        # Task-aware policy should intervene because
        # the user is actually approaching a constraint.
        #
        # We do not require a specific reference-policy
        # decision here because that decision depends
        # only on reference geometry.
        # -------------------------------------------------

        (
            "TASK RISK - OBSTACLE",

            {
                "x": 1.65,
                "y": -0.55,

                "orientation_relative_deg":
                    0.0,

                "tracking_missing_sec":
                    0.0,
            },

            None,

            DECISION_INTERVENE,

            "TASK-RELEVANT INTERVENTION",
        ),


        # -------------------------------------------------
        # 4. Orientation risk
        #
        # Spatially near expert reference,
        # therefore reference-only policy stays quiet.
        #
        # But cup orientation is unsafe,
        # therefore task-aware policy intervenes.
        #
        # This is the second important counterexample.
        # -------------------------------------------------

        (
            "TASK RISK - ORIENTATION",

            {
                "x": 0.70,
                "y": -0.75,

                "orientation_relative_deg":
                    12.0,

                "tracking_missing_sec":
                    0.0,
            },

            DECISION_STAY_QUIET,

            DECISION_INTERVENE,

            "TASK-RELEVANT INTERVENTION",
        ),


        # -------------------------------------------------
        # 5. Tracking loss
        #
        # Neither policy should judge the user.
        # -------------------------------------------------

        (
            "TRACKING UNRELIABLE",

            {
                "x": 1.00,
                "y": -0.80,

                "orientation_relative_deg":
                    0.0,

                "tracking_missing_sec":
                    0.50,
            },

            DECISION_PAUSE,

            DECISION_PAUSE,

            "JUDGEMENT PAUSED",
        ),
    ]


    passed = 0


    print()

    print(
        "======================================================"
    )

    print(
        "AdaptiveSkill - Intervention Comparator"
    )

    print(
        "======================================================"
    )


    # =====================================================
    # RUN TESTS
    # =====================================================

    for (
        name,
        kwargs,
        expected_reference,
        expected_task,
        expected_label,
    ) in tests:

        task_result = (
            evaluator.evaluate(
                **kwargs
            )
        )


        comparison = (
            comparator.compare(

                x=
                    kwargs["x"],

                y=
                    kwargs["y"],

                task_result=
                    task_result,
            )
        )


        # -------------------------------------------------
        # Check reference decision
        #
        # None means:
        # we intentionally do not enforce
        # one expected reference decision.
        # -------------------------------------------------

        reference_ok = (

            expected_reference is None

            or

            comparison.reference_policy_decision
            ==
            expected_reference
        )


        task_ok = (

            comparison.task_policy_decision

            ==

            expected_task
        )


        label_ok = (

            comparison.comparison_label

            ==

            expected_label
        )


        ok = (

            reference_ok

            and

            task_ok

            and

            label_ok
        )


        if ok:

            passed += 1


        # =================================================
        # PRINT RESULT
        # =================================================

        print()

        print(
            name
        )

        print(
            "-" * 54
        )

        print(
            "Task state:",
            task_result.state
        )

        print(
            "Task valid so far:",
            task_result.task_valid_so_far
        )

        print(
            "Reference distance:",
            round(
                comparison.reference_distance,
                3,
            )
        )

        print(
            "Reference policy:",
            comparison.reference_policy_decision
        )

        print(
            "Task-aware policy:",
            comparison.task_policy_decision
        )

        print(
            "Unnecessary intervention avoided:",
            comparison.unnecessary_intervention_avoided
        )

        print(
            "Task-relevant intervention:",
            comparison.task_relevant_intervention
        )

        print(
            "Result:",
            comparison.comparison_label
        )

        print(
            "PASS:",
            ok
        )


    # =====================================================
    # FINAL RESULT
    # =====================================================

    print()

    print(
        "======================================================"
    )

    print(
        "RESULT:",
        f"{passed}/{len(tests)} tests passed"
    )


    if (
        passed
        ==
        len(tests)
    ):

        print(
            "INTERVENTION COMPARATOR: PASSED"
        )

    else:

        print(
            "INTERVENTION COMPARATOR: FAILED"
        )


    print(
        "======================================================"
    )

    print()