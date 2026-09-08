import sys
from pathlib import Path

import numpy as np


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# =========================================================
# IMPORTS
# =========================================================

from rollout.simple_trajectory_learner import (
    TrajectoryDemonstration,
    StrategyAwareTrajectoryLearner,
    learn_naive_single_mean,
    evaluate_learned_trajectory,
    resample_trajectory,
)


# =========================================================
# SYNTHETIC DEMONSTRATIONS
#
# These are controlled engineering sanity checks.
# They are NOT user-study data.
# =========================================================

def dense_path(
    waypoints,
    num_points=160,
):

    return resample_trajectory(
        np.asarray(
            waypoints,
            dtype=float,
        ),
        num_points=
            num_points,
    )


def make_upper_demo(
    demo_id,
    offset=0.0,
):

    return TrajectoryDemonstration(

        demo_id=
            demo_id,

        strategy=
            "UPPER",

        points=
            dense_path(
                [
                    (
                        0.0,
                        0.0,
                        0.0,
                    ),
                    (
                        0.75,
                        -0.82
                        +
                        offset,
                        0.0,
                    ),
                    (
                        1.20,
                        -0.90
                        +
                        offset,
                        0.0,
                    ),
                    (
                        2.15,
                        -0.90
                        +
                        offset,
                        0.0,
                    ),
                    (
                        2.60,
                        -0.62
                        +
                        offset
                        *
                        0.5,
                        0.0,
                    ),
                    (
                        3.20,
                        0.0,
                        0.0,
                    ),
                ]
            ),
    )


def make_lower_demo(
    demo_id,
    offset=0.0,
):

    return TrajectoryDemonstration(

        demo_id=
            demo_id,

        strategy=
            "LOWER",

        points=
            dense_path(
                [
                    (
                        0.0,
                        0.0,
                        0.0,
                    ),
                    (
                        0.75,
                        0.82
                        +
                        offset,
                        0.0,
                    ),
                    (
                        1.20,
                        0.90
                        +
                        offset,
                        0.0,
                    ),
                    (
                        2.15,
                        0.90
                        +
                        offset,
                        0.0,
                    ),
                    (
                        2.60,
                        0.62
                        +
                        offset
                        *
                        0.5,
                        0.0,
                    ),
                    (
                        3.20,
                        0.0,
                        0.0,
                    ),
                ]
            ),
    )


# =========================================================
# TEST
# =========================================================

def main():

    print()

    print(
        "========================================================"
    )

    print(
        "AdaptiveSkill - Simple Trajectory Learner Test"
    )

    print(
        "========================================================"
    )


    upper_demos = [

        make_upper_demo(
            "UPPER_1",
            offset=
                -0.03,
        ),

        make_upper_demo(
            "UPPER_2",
            offset=
                0.00,
        ),

        make_upper_demo(
            "UPPER_3",
            offset=
                0.03,
        ),
    ]


    lower_demos = [

        make_lower_demo(
            "LOWER_1",
            offset=
                -0.03,
        ),

        make_lower_demo(
            "LOWER_2",
            offset=
                0.00,
        ),

        make_lower_demo(
            "LOWER_3",
            offset=
                0.03,
        ),
    ]


    all_demos = (
        upper_demos
        +
        lower_demos
    )


    learner = (
        StrategyAwareTrajectoryLearner()
    )


    prototypes = learner.fit(
        all_demos
    )


    strategies = (
        learner.available_strategies()
    )


    # =====================================================
    # TEST 1
    # Learner should preserve two strategy modes.
    # =====================================================

    test_1 = (

        strategies

        ==

        [
            "LOWER",
            "UPPER",
        ]
    )


    # =====================================================
    # TEST 2
    # Upper learned prototype should remain task-valid.
    # =====================================================

    upper_prototype = (
        learner.predict(
            "UPPER"
        )
    )


    upper_eval = (
        evaluate_learned_trajectory(
            upper_prototype.points
        )
    )


    test_2 = (

        upper_eval.goal_reached

        and

        upper_eval.task_valid

        and

        not upper_eval.violation_seen
    )


    # =====================================================
    # TEST 3
    # Lower learned prototype should remain task-valid.
    # =====================================================

    lower_prototype = (
        learner.predict(
            "LOWER"
        )
    )


    lower_eval = (
        evaluate_learned_trajectory(
            lower_prototype.points
        )
    )


    test_3 = (

        lower_eval.goal_reached

        and

        lower_eval.task_valid

        and

        not lower_eval.violation_seen
    )


    # =====================================================
    # TEST 4
    # If multiple distinct strategies exist, learner should
    # refuse to silently collapse them without a strategy.
    # =====================================================

    refused_ambiguous_prediction = False


    try:

        learner.predict()


    except ValueError:

        refused_ambiguous_prediction = (
            True
        )


    test_4 = (
        refused_ambiguous_prediction
    )


    # =====================================================
    # TEST 5
    # Transparent naive baseline:
    # average upper + lower routes into one mean path.
    #
    # In this obstacle geometry, that mean is expected to
    # pass through the central obstacle region.
    # =====================================================

    naive = (
        learn_naive_single_mean(
            all_demos
        )
    )


    naive_eval = (
        evaluate_learned_trajectory(
            naive.points
        )
    )


    test_5 = (
        naive_eval.violation_seen
    )


    final_pass = all(
        [
            test_1,
            test_2,
            test_3,
            test_4,
            test_5,
        ]
    )


    # =====================================================
    # PRINT
    # =====================================================

    print()

    print(
        "Strategies learned:",
        strategies
    )


    print()

    print(
        "1. Separate UPPER / LOWER modes:",
        test_1
    )


    print(
        "2. UPPER learned prototype task-valid:",
        test_2
    )


    print(
        "   Goal reached:",
        upper_eval.goal_reached
    )


    print(
        "   Risk seen:",
        upper_eval.risk_seen
    )


    print(
        "   Violation seen:",
        upper_eval.violation_seen
    )


    print()


    print(
        "3. LOWER learned prototype task-valid:",
        test_3
    )


    print(
        "   Goal reached:",
        lower_eval.goal_reached
    )


    print(
        "   Risk seen:",
        lower_eval.risk_seen
    )


    print(
        "   Violation seen:",
        lower_eval.violation_seen
    )


    print()


    print(
        "4. Ambiguous multi-mode prediction refused:",
        test_4
    )


    print()


    print(
        "5. Naive mixed-strategy mean violates task:",
        test_5
    )


    print(
        "   Goal reached:",
        naive_eval.goal_reached
    )


    print(
        "   Risk seen:",
        naive_eval.risk_seen
    )


    print(
        "   Violation seen:",
        naive_eval.violation_seen
    )


    print()

    print(
        "========================================================"
    )

    print(
        "PASS:",
        final_pass
    )


    if final_pass:

        print(
            "SIMPLE TRAJECTORY LEARNER: PASSED"
        )


    else:

        print(
            "SIMPLE TRAJECTORY LEARNER: FAILED"
        )


    print(
        "========================================================"
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "This is a synthetic engineering sanity check of a "
        "lightweight prototype learner. It is not evidence "
        "of user learning, physical robot performance, or "
        "general policy learning."
    )

    print()


if __name__ == "__main__":

    main()
