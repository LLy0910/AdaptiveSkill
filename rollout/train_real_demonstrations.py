import argparse
import csv
import json
import math
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
    StrategyAwareTrajectoryLearner,
    learn_naive_single_mean,
    load_runtime_demonstration,
    classify_route_strategy,
    evaluate_learned_trajectory,
)


# =========================================================
# PATHS
# =========================================================

RUNTIME_DIR = (
    PROJECT_ROOT
    / "data"
    / "runtime_trials"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "learner_validation"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "real_demonstration_learner_summary.json"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "real_demonstration_learner_summary.csv"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "task"
    / "task_config.json"
)


# =========================================================
# HELPERS
# =========================================================

def safe_int(
    value,
    default=0,
):

    try:
        return int(
            float(value)
        )

    except Exception:
        return default


def safe_float(
    value,
    default=float("nan"),
):

    try:
        return float(value)

    except Exception:
        return default


def load_rows(
    path,
):

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        return list(
            csv.DictReader(
                file
            )
        )


def inspect_runtime_trial(
    path,
):

    rows = load_rows(
        path
    )


    if len(rows) < 2:

        return {
            "usable": False,
            "reason": "too_short",
        }


    goal_reached = any(

        safe_int(
            row.get(
                "goal_reached",
                0,
            )
        )
        ==
        1

        or

        safe_int(
            row.get(
                "trial_goal_reached_seen",
                0,
            )
        )
        ==
        1

        for row
        in rows
    )


    risk_seen = any(

        str(
            row.get(
                "task_state",
                ""
            )
        )
        ==
        "RISK"

        for row
        in rows
    )


    violation_seen = any(

        str(
            row.get(
                "task_state",
                ""
            )
        )
        ==
        "VIOLATION"

        or

        safe_int(
            row.get(
                "trial_violation_occurred",
                0,
            )
        )
        ==
        1

        for row
        in rows
    )


    max_assistance_level = max(

        (
            safe_int(
                row.get(
                    "assistance_level",
                    0,
                )
            )

            for row
            in rows
        ),

        default=0,
    )


    trial_label = str(
        rows[0].get(
            "trial_label",
            ""
        )
    ).strip()


    trial_id = str(
        rows[0].get(
            "trial_id",
            path.stem,
        )
    )


    strict_safe = (

        goal_reached

        and

        not risk_seen

        and

        not violation_seen

        and

        max_assistance_level == 0
    )


    return {

        "usable":
            strict_safe,

        "reason":
            (
                "strict_safe"

                if strict_safe

                else
                "not_strict_safe"
            ),

        "trial_id":
            trial_id,

        "trial_label":
            trial_label,

        "goal_reached":
            goal_reached,

        "risk_seen":
            risk_seen,

        "violation_seen":
            violation_seen,

        "max_assistance_level":
            max_assistance_level,
    }


def evaluation_to_dict(
    evaluation,
):

    clearance = (
        evaluation.min_obstacle_clearance
    )


    if isinstance(
        clearance,
        float,
    ) and math.isnan(
        clearance
    ):

        clearance_output = None

    else:

        clearance_output = round(
            float(clearance),
            4,
        )


    return {

        "goal_reached":
            bool(
                evaluation.goal_reached
            ),

        "risk_seen":
            bool(
                evaluation.risk_seen
            ),

        "violation_seen":
            bool(
                evaluation.violation_seen
            ),

        "segment_violation_seen":
            bool(
                evaluation.segment_violation_seen
            ),

        "min_obstacle_clearance":
            clearance_output,

        "max_orientation_error_deg":
            round(
                float(
                    evaluation.max_orientation_error_deg
                ),
                3,
            ),

        "task_valid":
            bool(
                evaluation.task_valid
            ),

        "strict_safe":
            bool(
                evaluation.strict_safe
            ),
    }


# =========================================================
# DISCOVER STRICT-SAFE HUMAN DEMONSTRATIONS
# =========================================================

def discover_demonstrations(
    per_strategy,
):

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        task_config = json.load(
            file
        )


    candidates = sorted(

        RUNTIME_DIR.glob(
            "SAFE_ALTERNATIVE_*.csv"
        ),

        key=lambda path:
            path.stat().st_mtime,
    )


    selected = {
        "UPPER": [],
        "LOWER": [],
    }


    rejected = []


    for path in candidates:

        inspection = (
            inspect_runtime_trial(
                path
            )
        )


        if not inspection[
            "usable"
        ]:

            rejected.append(
                {
                    "path":
                        str(path),

                    **inspection,
                }
            )

            continue


        try:

            demonstration = (
                load_runtime_demonstration(

                    path,

                    require_goal=
                        True,

                    reject_violation=
                        True,
                )
            )


        except Exception as error:

            rejected.append(
                {
                    "path":
                        str(path),

                    "usable":
                        False,

                    "reason":
                        str(error),
                }
            )

            continue


        strategy = (
            classify_route_strategy(

                demonstration.points,

                task_config,
            )
        )


        if strategy not in selected:

            rejected.append(
                {
                    "path":
                        str(path),

                    **inspection,

                    "reason":
                        (
                            "strategy_not_upper_or_lower:"
                            +
                            strategy
                        ),
                }
            )

            continue


        demonstration.strategy = (
            strategy
        )


        if (
            len(
                selected[
                    strategy
                ]
            )
            <
            per_strategy
        ):

            selected[
                strategy
            ].append(
                demonstration
            )


    return (
        selected,
        rejected,
    )


# =========================================================
# SAVE SUMMARY
# =========================================================

def save_summary(
    summary,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(

            summary,

            file,

            indent=4,

            ensure_ascii=False,
        )


    rows = []


    for condition_name in (
        "UPPER_PROTOTYPE",
        "LOWER_PROTOTYPE",
        "NAIVE_SINGLE_MEAN",
    ):

        evaluation = (
            summary[
                "evaluations"
            ][
                condition_name
            ]
        )


        rows.append(
            {
                "condition":
                    condition_name,

                **evaluation,
            }
        )


    with open(
        SUMMARY_CSV,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=list(
                rows[0].keys()
            ),
        )


        writer.writeheader()

        writer.writerows(
            rows
        )


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Train the AdaptiveSkill lightweight "
            "strategy-aware trajectory learner from real "
            "SAFE_ALTERNATIVE runtime demonstrations."
        )
    )


    parser.add_argument(

        "--per-strategy",

        type=int,

        default=3,

        help=(
            "Number of strict-safe human demonstrations "
            "required for UPPER and LOWER."
        ),
    )


    args = parser.parse_args()


    per_strategy = max(
        1,
        int(
            args.per_strategy
        ),
    )


    print()

    print(
        "========================================================"
    )

    print(
        "AdaptiveSkill - Real Demonstration Learner"
    )

    print(
        "========================================================"
    )


    (
        selected,
        rejected,
    ) = discover_demonstrations(
        per_strategy
    )


    upper_count = len(
        selected[
            "UPPER"
        ]
    )


    lower_count = len(
        selected[
            "LOWER"
        ]
    )


    print()

    print(
        "Strict-safe demonstrations found:"
    )


    print(
        "UPPER:",
        f"{upper_count}/{per_strategy}"
    )


    print(
        "LOWER:",
        f"{lower_count}/{per_strategy}"
    )


    print()


    if (
        upper_count
        <
        per_strategy

        or

        lower_count
        <
        per_strategy
    ):

        print(
            "DATA COLLECTION INCOMPLETE"
        )

        print()


        if (
            upper_count
            <
            per_strategy
        ):

            print(
                "Need",
                per_strategy
                -
                upper_count,
                "more strict-safe UPPER demonstration(s)."
            )


        if (
            lower_count
            <
            per_strategy
        ):

            print(
                "Need",
                per_strategy
                -
                lower_count,
                "more strict-safe LOWER demonstration(s)."
            )


        print()

        print(
            "Collection rule:"
        )

        print(
            "Select SAFE_ALTERNATIVE in the main demo, "
            "keep the cup upright, stay outside the safety "
            "margin for the whole route, reach TARGET, and "
            "keep assistance at L0."
        )

        print()

        print(
            "Rejected/non-training SAFE_ALTERNATIVE logs:",
            len(
                rejected
            )
        )


        print()

        print(
            "No learner result has been claimed yet."
        )

        print(
            "========================================================"
        )

        return


    demonstrations = (

        selected[
            "UPPER"
        ]

        +

        selected[
            "LOWER"
        ]
    )


    # =====================================================
    # FIT STRATEGY-AWARE LEARNER
    # =====================================================

    learner = (
        StrategyAwareTrajectoryLearner()
    )


    learner.fit(
        demonstrations
    )


    upper_prototype = (
        learner.predict(
            "UPPER"
        )
    )


    lower_prototype = (
        learner.predict(
            "LOWER"
        )
    )


    # =====================================================
    # DOWNSTREAM BASELINE
    #
    # Intentionally ignores strategy modes.
    # =====================================================

    naive_prototype = (
        learn_naive_single_mean(
            demonstrations
        )
    )


    # =====================================================
    # EVALUATE WITH THE SAME TASK CONSTRAINTS
    # =====================================================

    upper_evaluation = (
        evaluate_learned_trajectory(
            upper_prototype.points
        )
    )


    lower_evaluation = (
        evaluate_learned_trajectory(
            lower_prototype.points
        )
    )


    naive_evaluation = (
        evaluate_learned_trajectory(
            naive_prototype.points
        )
    )


    upper_pass = (
        upper_evaluation.task_valid

        and

        not upper_evaluation.violation_seen
    )


    lower_pass = (
        lower_evaluation.task_valid

        and

        not lower_evaluation.violation_seen
    )


    strategy_preservation_pass = (

        upper_pass

        and

        lower_pass
    )


    naive_failure_observed = (
        naive_evaluation.violation_seen
    )


    overall_sanity_pass = (

        strategy_preservation_pass

        and

        naive_failure_observed
    )


    summary = {

        "study_type":
            "real-human-demonstration engineering sanity check",

        "claim_boundary":
            (
                "This evaluates simple learned trajectory "
                "prototypes in the virtual task. It is not "
                "a user study, physical robot experiment, "
                "or evidence of general robot-policy learning."
            ),

        "required_demonstrations_per_strategy":
            per_strategy,

        "selected_demonstrations": {

            "UPPER":
                [
                    demonstration.demo_id

                    for demonstration
                    in selected[
                        "UPPER"
                    ]
                ],

            "LOWER":
                [
                    demonstration.demo_id

                    for demonstration
                    in selected[
                        "LOWER"
                    ]
                ],
        },

        "evaluations": {

            "UPPER_PROTOTYPE":
                evaluation_to_dict(
                    upper_evaluation
                ),

            "LOWER_PROTOTYPE":
                evaluation_to_dict(
                    lower_evaluation
                ),

            "NAIVE_SINGLE_MEAN":
                evaluation_to_dict(
                    naive_evaluation
                ),
        },

        "checks": {

            "upper_prototype_task_valid":
                upper_pass,

            "lower_prototype_task_valid":
                lower_pass,

            "distinct_valid_strategies_preserved":
                strategy_preservation_pass,

            "naive_mixed_strategy_violation_observed":
                naive_failure_observed,

            "overall_sanity_pass":
                overall_sanity_pass,
        },
    }


    save_summary(
        summary
    )


    # =====================================================
    # PRINT
    # =====================================================

    print(
        "Selected UPPER demonstrations:"
    )

    for demonstration in selected[
        "UPPER"
    ]:

        print(
            "  -",
            demonstration.demo_id
        )


    print()

    print(
        "Selected LOWER demonstrations:"
    )

    for demonstration in selected[
        "LOWER"
    ]:

        print(
            "  -",
            demonstration.demo_id
        )


    print()

    print(
        "--------------------------------------------------------"
    )

    print(
        "DOWNSTREAM LEARNED TRAJECTORY EVALUATION"
    )

    print(
        "--------------------------------------------------------"
    )


    print()

    print(
        "UPPER prototype:"
    )

    print(
        "  Goal reached:",
        upper_evaluation.goal_reached
    )

    print(
        "  Risk seen:",
        upper_evaluation.risk_seen
    )

    print(
        "  Violation seen:",
        upper_evaluation.violation_seen
    )

    print(
        "  Task valid:",
        upper_evaluation.task_valid
    )


    print()

    print(
        "LOWER prototype:"
    )

    print(
        "  Goal reached:",
        lower_evaluation.goal_reached
    )

    print(
        "  Risk seen:",
        lower_evaluation.risk_seen
    )

    print(
        "  Violation seen:",
        lower_evaluation.violation_seen
    )

    print(
        "  Task valid:",
        lower_evaluation.task_valid
    )


    print()

    print(
        "NAIVE mixed-strategy mean:"
    )

    print(
        "  Goal reached:",
        naive_evaluation.goal_reached
    )

    print(
        "  Risk seen:",
        naive_evaluation.risk_seen
    )

    print(
        "  Violation seen:",
        naive_evaluation.violation_seen
    )

    print(
        "  Task valid:",
        naive_evaluation.task_valid
    )


    print()

    print(
        "--------------------------------------------------------"
    )

    print(
        "CHECKS"
    )

    print(
        "--------------------------------------------------------"
    )


    print(
        "UPPER learned prototype task-valid:",
        upper_pass
    )


    print(
        "LOWER learned prototype task-valid:",
        lower_pass
    )


    print(
        "Distinct valid strategies preserved:",
        strategy_preservation_pass
    )


    print(
        "Naive mixed-strategy violation observed:",
        naive_failure_observed
    )


    print()

    print(
        "========================================================"
    )

    print(
        "PASS:",
        overall_sanity_pass
    )


    if overall_sanity_pass:

        print(
            "REAL DEMONSTRATION LEARNER: PASSED"
        )


    else:

        print(
            "REAL DEMONSTRATION LEARNER: NEEDS REVIEW"
        )


    print(
        "========================================================"
    )


    print()

    print(
        "Summary JSON:"
    )

    print(
        SUMMARY_JSON
    )


    print()

    print(
        "Summary CSV:"
    )

    print(
        SUMMARY_CSV
    )


    print()

    print(
        "NOTE:"
    )

    print(
        summary[
            "claim_boundary"
        ]
    )

    print()


if __name__ == "__main__":

    main()
