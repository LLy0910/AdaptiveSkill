import csv
import json
import math
import sys
from pathlib import Path


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
# PROJECT IMPORTS
# =========================================================

from metrics.runtime_scenarios import (
    RuntimeScenarioCatalog,
)


# =========================================================
# PATHS
# =========================================================

RUNTIME_DIR = (
    PROJECT_ROOT
    / "data"
    / "runtime_trials"
)

OUTPUT_CSV = (
    RUNTIME_DIR
    / "scenario_validation.csv"
)


# =========================================================
# CURRENT ENGINEERING THRESHOLDS
#
# These mirror the current prototype controller.
#
# IMPORTANT:
# They are engineering defaults.
# They are NOT experimentally validated optimal values.
# =========================================================

BRIEF_RISK_MAX_SEC = 0.80

PERSISTENT_RISK_MIN_SEC = 0.80

VIOLATION_RECOVERY_MAX_SEC = 2.00

PERSISTENT_VIOLATION_MIN_SEC = 2.00


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
        return float(
            value
        )

    except Exception:
        return default


def safe_bool(
    value,
    default=False,
):

    if isinstance(
        value,
        bool,
    ):
        return value


    if value is None:
        return default


    text = str(
        value
    ).strip().lower()


    if text in (
        "1",
        "true",
        "yes",
        "y",
    ):
        return True


    if text in (
        "0",
        "false",
        "no",
        "n",
        "",
    ):
        return False


    return default


def load_csv(
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


def load_json(
    path,
):

    if not path.exists():
        return {}


    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# =========================================================
# SCENARIO IDENTIFICATION
# =========================================================

def identify_scenario(
    csv_path,
    rows,
    catalog,
):

    # -----------------------------------------------------
    # Preferred:
    # trial_label saved by RuntimeLogger.
    # -----------------------------------------------------

    if rows:

        trial_label = str(
            rows[0].get(
                "trial_label",
                "",
            )
        ).strip()


        scenario = (
            catalog.get_by_id(
                trial_label
            )
        )


        if scenario is not None:
            return scenario


    # -----------------------------------------------------
    # Fallback:
    # filename prefix.
    # -----------------------------------------------------

    stem = csv_path.stem


    for scenario in catalog.scenarios:

        scenario_id = (
            scenario[
                "id"
            ]
        )


        if stem.startswith(
            scenario_id
            +
            "_"
        ):
            return scenario


    return None


# =========================================================
# TRANSITIONS
# =========================================================

def get_level_transitions(
    rows,
):

    transitions = []

    previous_level = None


    for row in rows:

        level = safe_int(
            row.get(
                "assistance_level",
                0,
            )
        )


        if (
            previous_level is None
            or
            level != previous_level
        ):

            transitions.append(
                level
            )


        previous_level = (
            level
        )


    return transitions


def transition_string(
    levels,
):

    if not levels:
        return "NONE"


    return " -> ".join(

        f"L{level}"

        for level
        in levels
    )


def contains_ordered_levels(
    transitions,
    required_levels,
):

    if not required_levels:
        return True


    index = 0


    for level in transitions:

        if (
            level
            ==
            required_levels[index]
        ):

            index += 1


            if (
                index
                ==
                len(
                    required_levels
                )
            ):
                return True


    return False


def detected_level_after(
    transitions,
    earlier_level,
    later_level,
):

    seen_earlier = False


    for level in transitions:

        if level == earlier_level:

            seen_earlier = True


        elif (
            seen_earlier
            and
            level == later_level
        ):

            return True


    return False


# =========================================================
# TIME BY ASSISTANCE LEVEL
# =========================================================

def estimate_time_by_level(
    rows,
):

    result = {
        0: 0.0,
        1: 0.0,
        2: 0.0,
        3: 0.0,
    }


    if len(rows) < 2:
        return result


    for index in range(
        len(rows) - 1
    ):

        current = rows[index]

        following = rows[
            index + 1
        ]


        level = safe_int(
            current.get(
                "assistance_level",
                0,
            )
        )


        t1 = safe_float(
            current.get(
                "timestamp_sec",
                0.0,
            ),
            0.0,
        )


        t2 = safe_float(
            following.get(
                "timestamp_sec",
                t1,
            ),
            t1,
        )


        dt = max(
            0.0,
            t2 - t1,
        )


        if level in result:

            result[level] += dt


    return result


def first_timestamp_for_level(
    rows,
    target_level,
):

    for row in rows:

        level = safe_int(
            row.get(
                "assistance_level",
                0,
            )
        )


        if level == target_level:

            return safe_float(
                row.get(
                    "timestamp_sec",
                    0.0,
                ),
                0.0,
            )


    return None


# =========================================================
# BOOLEAN / VALUE SEARCH
# =========================================================

def any_true(
    rows,
    column,
):

    return any(

        safe_int(
            row.get(
                column,
                0,
            )
        )
        ==
        1

        for row in rows
    )


def any_value(
    rows,
    column,
    target,
):

    return any(

        str(
            row.get(
                column,
                ""
            )
        )
        ==
        str(
            target
        )

        for row in rows
    )


# =========================================================
# MAXIMUM LOGGED VALUES
# =========================================================

def max_logged_float(
    rows,
    column,
    default=0.0,
):

    values = []


    for row in rows:

        value = safe_float(
            row.get(
                column,
                float("nan"),
            )
        )


        if not math.isnan(
            value
        ):
            values.append(
                value
            )


    if not values:
        return default


    return max(
        values
    )


# =========================================================
# POLICY OBSERVATIONS
# =========================================================

def reference_intervened(
    rows,
):

    return any_value(
        rows,
        "reference_policy",
        "INTERVENE",
    )


def task_intervened(
    rows,
):

    return any_value(
        rows,
        "task_policy",
        "INTERVENE",
    )


def unnecessary_reference_intervention_avoided(
    rows,
):

    return any_value(
        rows,
        "comparison_label",
        "UNNECESSARY INTERVENTION AVOIDED",
    )


# =========================================================
# RECOVERY
# =========================================================

def detected_recovery(
    rows,
):

    previously_assisted = False


    for row in rows:

        level = safe_int(
            row.get(
                "assistance_level",
                0,
            )
        )


        state = str(
            row.get(
                "task_state",
                "",
            )
        )


        if level > 0:

            previously_assisted = True


        if (
            previously_assisted
            and
            level == 0
            and
            state == "VALID"
        ):

            return True


    return False


# =========================================================
# ORIENTATION
# =========================================================

def get_valid_orientation_errors(
    rows,
):

    values = []


    for row in rows:

        value = safe_float(
            row.get(
                "orientation_error_deg",
                float("nan"),
            )
        )


        if not math.isnan(
            value
        ):

            values.append(
                abs(
                    value
                )
            )


    return values


# =========================================================
# ANALYZE ONE TRIAL
# =========================================================

def analyze_trial(
    csv_path,
    scenario,
):

    rows = load_csv(
        csv_path
    )


    if not rows:
        return None


    # =====================================================
    # SUMMARY JSON
    # =====================================================

    summary_path = (
        csv_path.with_name(
            csv_path.stem
            +
            "_summary.json"
        )
    )


    summary = load_json(
        summary_path
    )


    # =====================================================
    # ASSISTANCE
    # =====================================================

    transitions = (
        get_level_transitions(
            rows
        )
    )


    time_by_level = (
        estimate_time_by_level(
            rows
        )
    )


    max_level = max(

        (
            safe_int(
                row.get(
                    "assistance_level",
                    0,
                )
            )

            for row in rows
        ),

        default=0,
    )


    recovery_needed = (
        max_level > 0
    )


    returned_to_quiet = (

        detected_recovery(
            rows
        )

        if recovery_needed

        else False
    )


    # =====================================================
    # TASK STATES
    # =====================================================

    saw_valid = any_value(
        rows,
        "task_state",
        "VALID",
    )


    saw_risk = any_value(
        rows,
        "task_state",
        "RISK",
    )


    saw_violation = any_value(
        rows,
        "task_state",
        "VIOLATION",
    )


    saw_obstacle_constraint = (
        any_value(
            rows,
            "dominant_constraint",
            "OBSTACLE",
        )
    )


    saw_orientation_constraint = (
        any_value(
            rows,
            "dominant_constraint",
            "ORIENTATION",
        )
    )


    # =====================================================
    # PRESENTATION MODES
    # =====================================================

    saw_minimal_direction = (
        any_value(
            rows,
            "presentation_mode",
            "MINIMAL_DIRECTION",
        )
    )


    saw_explicit_waypoint = (
        any_value(
            rows,
            "presentation_mode",
            "EXPLICIT_WAYPOINT",
        )
    )


    saw_minimal_orientation = (
        any_value(
            rows,
            "presentation_mode",
            "MINIMAL_ORIENTATION",
        )
    )


    saw_explicit_orientation = (
        any_value(
            rows,
            "presentation_mode",
            "EXPLICIT_ORIENTATION",
        )
    )


    saw_strong_waypoint = (
        any_value(
            rows,
            "presentation_mode",
            "STRONG_WAYPOINT",
        )
    )


    saw_strong_orientation = (
        any_value(
            rows,
            "presentation_mode",
            "STRONG_ORIENTATION",
        )
    )


    saw_recovery_fade = (
        any_value(
            rows,
            "presentation_mode",
            "RECOVERY_FADE",
        )
    )


    # =====================================================
    # DURATIONS
    # =====================================================

    max_risk_duration = (
        max_logged_float(
            rows,
            "risk_duration_sec",
            0.0,
        )
    )


    max_violation_duration = (
        max_logged_float(
            rows,
            "violation_duration_sec",
            0.0,
        )
    )


    max_recent_violation_episodes = (
        int(
            max_logged_float(
                rows,
                "recent_violation_episodes",
                0.0,
            )
        )
    )


    # =====================================================
    # POLICY
    # =====================================================

    ref_intervened = (
        reference_intervened(
            rows
        )
    )


    task_aware_intervened = (
        task_intervened(
            rows
        )
    )


    avoided_unnecessary_reference = (
        unnecessary_reference_intervention_avoided(
            rows
        )
    )


    # =====================================================
    # SAFETY
    # =====================================================

    segment_crossing = (
        any_true(
            rows,
            "segment_crossing_detected",
        )
    )


    violation_occurred = bool(
        summary.get(
            "constraint_violation_occurred",
            any_true(
                rows,
                "trial_violation_occurred",
            ),
        )
    )


    violation_episodes = int(
        summary.get(
            "constraint_violation_episodes",
            max_logged_float(
                rows,
                "trial_violation_episodes",
                0.0,
            ),
        )
    )


    # =====================================================
    # GOAL
    # =====================================================

    goal_reached = bool(
        summary.get(
            "goal_reached",
            any_true(
                rows,
                "trial_goal_reached_seen",
            ),
        )
    )


    final_outcome = (
        summary.get(
            "final_task_outcome"
        )
    )


    if final_outcome is None:

        final_outcome = str(
            rows[-1].get(
                "trial_task_outcome",
                "UNKNOWN",
            )
        )


    # =====================================================
    # ORIENTATION METRICS
    # =====================================================

    orientation_errors = (
        get_valid_orientation_errors(
            rows
        )
    )


    if orientation_errors:

        max_orientation_error = max(
            orientation_errors
        )

    else:

        max_orientation_error = (
            float("nan")
        )


    # =====================================================
    # RESULT
    # =====================================================

    result = {

        "trial_id":
            rows[0].get(
                "trial_id",
                csv_path.stem,
            ),

        "scenario_id":
            scenario[
                "id"
            ],

        "scenario_title":
            scenario[
                "title"
            ],

        "logged_frames":
            len(
                rows
            ),

        "duration_sec":
            round(
                safe_float(
                    rows[-1].get(
                        "timestamp_sec",
                        0.0,
                    ),
                    0.0,
                ),
                3,
            ),

        # -------------------------------------------------
        # Assistance
        # -------------------------------------------------

        "max_assistance_level":
            max_level,

        "assistance_transitions":
            transition_string(
                transitions
            ),

        "first_L1_sec":
            first_timestamp_for_level(
                rows,
                1,
            ),

        "first_L2_sec":
            first_timestamp_for_level(
                rows,
                2,
            ),

        "first_L3_sec":
            first_timestamp_for_level(
                rows,
                3,
            ),

        "time_L0_sec":
            round(
                time_by_level[0],
                3,
            ),

        "time_L1_sec":
            round(
                time_by_level[1],
                3,
            ),

        "time_L2_sec":
            round(
                time_by_level[2],
                3,
            ),

        "time_L3_sec":
            round(
                time_by_level[3],
                3,
            ),

        # -------------------------------------------------
        # Durations
        # -------------------------------------------------

        "max_risk_duration_sec":
            round(
                max_risk_duration,
                3,
            ),

        "max_violation_duration_sec":
            round(
                max_violation_duration,
                3,
            ),

        "max_recent_violation_episodes":
            max_recent_violation_episodes,

        # -------------------------------------------------
        # Task states
        # -------------------------------------------------

        "saw_VALID":
            saw_valid,

        "saw_RISK":
            saw_risk,

        "saw_VIOLATION":
            saw_violation,

        "saw_OBSTACLE_constraint":
            saw_obstacle_constraint,

        "saw_ORIENTATION_constraint":
            saw_orientation_constraint,

        # -------------------------------------------------
        # Presentation
        # -------------------------------------------------

        "saw_MINIMAL_DIRECTION":
            saw_minimal_direction,

        "saw_EXPLICIT_WAYPOINT":
            saw_explicit_waypoint,

        "saw_MINIMAL_ORIENTATION":
            saw_minimal_orientation,

        "saw_EXPLICIT_ORIENTATION":
            saw_explicit_orientation,

        "saw_STRONG_WAYPOINT":
            saw_strong_waypoint,

        "saw_STRONG_ORIENTATION":
            saw_strong_orientation,

        "saw_RECOVERY_FADE":
            saw_recovery_fade,

        # -------------------------------------------------
        # Comparator
        # -------------------------------------------------

        "reference_intervened":
            ref_intervened,

        "task_aware_intervened":
            task_aware_intervened,

        "unnecessary_reference_intervention_avoided":
            avoided_unnecessary_reference,

        # -------------------------------------------------
        # Safety/history
        # -------------------------------------------------

        "segment_crossing_detected":
            segment_crossing,

        "violation_occurred":
            violation_occurred,

        "violation_episodes":
            violation_episodes,

        # -------------------------------------------------
        # Recovery
        # -------------------------------------------------

        "recovery_needed":
            recovery_needed,

        "returned_to_quiet":
            returned_to_quiet,

        # -------------------------------------------------
        # Orientation
        # -------------------------------------------------

        "max_orientation_error_deg":
            (
                round(
                    max_orientation_error,
                    3,
                )

                if not math.isnan(
                    max_orientation_error
                )

                else "nan"
            ),

        # -------------------------------------------------
        # Outcome
        # -------------------------------------------------

        "goal_reached":
            goal_reached,

        "final_task_outcome":
            final_outcome,

        # -------------------------------------------------
        # Internal
        # -------------------------------------------------

        "_transitions":
            transitions,
    }


    return result


# =========================================================
# PROTOCOL CLASSIFICATION
#
# This is intentionally separated from system validation.
#
# INVALID_PROTOCOL:
#     operator did not create the intended scenario.
#
# ABORTED:
#     trial ended before reaching the target.
#
# Only protocol-valid completed trials are scored as
# PASS or FAIL.
# =========================================================

def classify_protocol(
    result,
    scenario,
):

    scenario_id = (
        scenario[
            "id"
        ]
    )


    reasons = []


    # =====================================================
    # ABORTED / INCOMPLETE
    # =====================================================

    if not result[
        "goal_reached"
    ]:

        reasons.append(
            "Trial ended before reaching the target."
        )

        return (
            "ABORTED",
            reasons,
        )


    # =====================================================
    # SAFE ALTERNATIVE
    # =====================================================

    if (
        scenario_id
        ==
        "SAFE_ALTERNATIVE"
    ):

        if result[
            "saw_RISK"
        ]:

            reasons.append(
                "The route entered a task-risk region."
            )


        if result[
            "violation_occurred"
        ]:

            reasons.append(
                "A task violation occurred."
            )


        if not result[
            "reference_intervened"
        ]:

            reasons.append(
                "The route did not deviate enough to trigger "
                "the reference-similarity baseline."
            )


    # =====================================================
    # BRIEF RISK
    # =====================================================

    elif (
        scenario_id
        ==
        "BRIEF_RISK"
    ):

        if not result[
            "saw_RISK"
        ]:

            reasons.append(
                "No task risk was created."
            )


        if not result[
            "saw_OBSTACLE_constraint"
        ]:

            reasons.append(
                "The intended obstacle-risk condition "
                "was not observed."
            )


        if result[
            "violation_occurred"
        ]:

            reasons.append(
                "The interaction entered a violation "
                "instead of remaining a brief risk."
            )


        if (
            result[
                "max_risk_duration_sec"
            ]
            >=
            BRIEF_RISK_MAX_SEC
        ):

            reasons.append(
                "Risk persisted for too long for the "
                "BRIEF_RISK protocol "
                f"({result['max_risk_duration_sec']} sec)."
            )


    # =====================================================
    # PERSISTENT RISK
    # =====================================================

    elif (
        scenario_id
        ==
        "PERSISTENT_RISK"
    ):

        if not result[
            "saw_RISK"
        ]:

            reasons.append(
                "No task risk was created."
            )


        if not result[
            "saw_OBSTACLE_constraint"
        ]:

            reasons.append(
                "The intended obstacle-risk condition "
                "was not observed."
            )


        if result[
            "violation_occurred"
        ]:

            reasons.append(
                "The interaction entered a violation "
                "instead of remaining in the risk zone."
            )


        if (
            result[
                "max_risk_duration_sec"
            ]
            <
            PERSISTENT_RISK_MIN_SEC
        ):

            reasons.append(
                "Risk did not persist long enough for the "
                "PERSISTENT_RISK protocol "
                f"({result['max_risk_duration_sec']} sec)."
            )


    # =====================================================
    # ORIENTATION RISK
    # =====================================================

    elif (
        scenario_id
        ==
        "ORIENTATION_RISK"
    ):

        if not result[
            "saw_RISK"
        ]:

            reasons.append(
                "No task risk was created."
            )


        if not result[
            "saw_ORIENTATION_constraint"
        ]:

            reasons.append(
                "The intended orientation-risk condition "
                "was not observed."
            )


        if result[
            "saw_OBSTACLE_constraint"
        ]:

            reasons.append(
                "Obstacle risk was introduced during the "
                "orientation-specific protocol."
            )


        if result[
            "violation_occurred"
        ]:

            reasons.append(
                "Orientation error became a violation "
                "instead of remaining in the risk range."
            )


        if (
            result[
                "max_risk_duration_sec"
            ]
            <
            PERSISTENT_RISK_MIN_SEC
        ):

            reasons.append(
                "Orientation risk did not persist long enough "
                f"({result['max_risk_duration_sec']} sec)."
            )


    # =====================================================
    # VIOLATION + RECOVERY
    # =====================================================

    elif (
        scenario_id
        ==
        "VIOLATION_RECOVERY"
    ):

        if not result[
            "violation_occurred"
        ]:

            reasons.append(
                "No task violation was created."
            )


        if (
            result[
                "violation_episodes"
            ]
            >
            1
        ):

            reasons.append(
                "More than one violation episode occurred; "
                "this protocol expects one brief violation."
            )


        if (
            result[
                "max_violation_duration_sec"
            ]
            >=
            VIOLATION_RECOVERY_MAX_SEC
        ):

            reasons.append(
                "Violation persisted too long and became "
                "a persistent-violation condition "
                f"({result['max_violation_duration_sec']} sec)."
            )


    # =====================================================
    # PERSISTENT VIOLATION
    # =====================================================

    elif (
        scenario_id
        ==
        "PERSISTENT_VIOLATION"
    ):

        if not result[
            "violation_occurred"
        ]:

            reasons.append(
                "No task violation was created."
            )


        if (
            result[
                "max_violation_duration_sec"
            ]
            <
            PERSISTENT_VIOLATION_MIN_SEC
        ):

            reasons.append(
                "Violation did not persist long enough "
                f"({result['max_violation_duration_sec']} sec)."
            )


    # =====================================================
    # FINAL PROTOCOL STATUS
    # =====================================================

    if reasons:

        return (
            "INVALID_PROTOCOL",
            reasons,
        )


    return (
        "VALID_PROTOCOL",
        [],
    )


# =========================================================
# VALIDATION CHECK HELPERS
# =========================================================

def add_check(
    checks,
    name,
    passed,
    expected,
    actual,
):

    checks.append(
        {
            "name":
                name,

            "passed":
                bool(
                    passed
                ),

            "expected":
                expected,

            "actual":
                actual,
        }
    )


# =========================================================
# SYSTEM VALIDATION
#
# IMPORTANT:
# This function is only applied after the protocol itself
# has been classified as valid.
# =========================================================

def validate_system_behavior(
    result,
    scenario,
):

    checks = []


    scenario_id = (
        scenario[
            "id"
        ]
    )


    transitions = (
        result[
            "_transitions"
        ]
    )


    # =====================================================
    # SAFE ALTERNATIVE
    # =====================================================

    if (
        scenario_id
        ==
        "SAFE_ALTERNATIVE"
    ):

        add_check(

            checks,

            "Task-aware policy stayed quiet",

            not result[
                "task_aware_intervened"
            ],

            "False",

            result[
                "task_aware_intervened"
            ],
        )


        add_check(

            checks,

            "Assistance remained at L0",

            result[
                "max_assistance_level"
            ]
            ==
            0,

            "L0 only",

            result[
                "assistance_transitions"
            ],
        )


        add_check(

            checks,

            "Reference/task-aware disagreement observed",

            (
                result[
                    "reference_intervened"
                ]
                and
                not result[
                    "task_aware_intervened"
                ]
            ),

            (
                "Reference INTERVENE / "
                "Task-aware STAY_QUIET"
            ),

            (
                f"reference="
                f"{result['reference_intervened']}, "
                f"task-aware="
                f"{result['task_aware_intervened']}"
            ),
        )


        add_check(

            checks,

            "Unnecessary reference intervention avoided",

            result[
                "unnecessary_reference_intervention_avoided"
            ],

            True,

            result[
                "unnecessary_reference_intervention_avoided"
            ],
        )


    # =====================================================
    # BRIEF RISK
    # =====================================================

    elif (
        scenario_id
        ==
        "BRIEF_RISK"
    ):

        add_check(

            checks,

            "L1 minimal assistance was reached",

            1 in transitions,

            "L1 observed",

            result[
                "assistance_transitions"
            ],
        )


        add_check(

            checks,

            "System did not over-escalate to L2/L3",

            result[
                "max_assistance_level"
            ]
            ==
            1,

            "Maximum L1",

            f"L{result['max_assistance_level']}",
        )


        add_check(

            checks,

            "Minimal directional cue was shown",

            result[
                "saw_MINIMAL_DIRECTION"
            ],

            "MINIMAL_DIRECTION",

            result[
                "saw_MINIMAL_DIRECTION"
            ],
        )


        add_check(

            checks,

            "Assistance returned to L0",

            detected_level_after(
                transitions,
                1,
                0,
            ),

            "L1 ... L0",

            result[
                "assistance_transitions"
            ],
        )


        add_check(

            checks,

            "Runtime recovery detected",

            result[
                "returned_to_quiet"
            ],

            True,

            result[
                "returned_to_quiet"
            ],
        )


    # =====================================================
    # PERSISTENT RISK
    # =====================================================

    elif (
        scenario_id
        ==
        "PERSISTENT_RISK"
    ):

        add_check(

            checks,

            "Progressive escalation L1 -> L2",

            contains_ordered_levels(
                transitions,
                [
                    1,
                    2,
                ],
            ),

            "L1 before L2",

            result[
                "assistance_transitions"
            ],
        )


        add_check(

            checks,

            "Explicit waypoint was shown",

            result[
                "saw_EXPLICIT_WAYPOINT"
            ],

            "EXPLICIT_WAYPOINT",

            result[
                "saw_EXPLICIT_WAYPOINT"
            ],
        )


        add_check(

            checks,

            "Strong assistance was not reached",

            result[
                "max_assistance_level"
            ]
            ==
            2,

            "Maximum L2",

            f"L{result['max_assistance_level']}",
        )


        add_check(

            checks,

            "Assistance recovered from L2 to L0",

            detected_level_after(
                transitions,
                2,
                0,
            ),

            "L2 ... L0",

            result[
                "assistance_transitions"
            ],
        )


        add_check(

            checks,

            "Runtime recovery detected",

            result[
                "returned_to_quiet"
            ],

            True,

            result[
                "returned_to_quiet"
            ],
        )


    # =====================================================
    # ORIENTATION RISK
    # =====================================================

    elif (
        scenario_id
        ==
        "ORIENTATION_RISK"
    ):

        add_check(

            checks,

            "Minimal orientation cue was shown",

            result[
                "saw_MINIMAL_ORIENTATION"
            ],

            "MINIMAL_ORIENTATION",

            result[
                "saw_MINIMAL_ORIENTATION"
            ],
        )


        add_check(

            checks,

            "Explicit orientation cue was shown",

            result[
                "saw_EXPLICIT_ORIENTATION"
            ],

            "EXPLICIT_ORIENTATION",

            result[
                "saw_EXPLICIT_ORIENTATION"
            ],
        )


        add_check(

            checks,

            "Orientation assistance escalated L1 -> L2",

            contains_ordered_levels(
                transitions,
                [
                    1,
                    2,
                ],
            ),

            "L1 before L2",

            result[
                "assistance_transitions"
            ],
        )


        orientation_error = (
            result[
                "max_orientation_error_deg"
            ]
        )


        orientation_logged = (

            orientation_error
            !=
            "nan"

            and

            float(
                orientation_error
            )
            >=
            8.0
        )


        add_check(

            checks,

            "Orientation error was logged",

            orientation_logged,

            ">= 8 degrees",

            orientation_error,
        )


        add_check(

            checks,

            "Assistance recovered from L2 to L0",

            detected_level_after(
                transitions,
                2,
                0,
            ),

            "L2 ... L0",

            result[
                "assistance_transitions"
            ],
        )


    # =====================================================
    # VIOLATION + RECOVERY
    # =====================================================

    elif (
        scenario_id
        ==
        "VIOLATION_RECOVERY"
    ):

        add_check(

            checks,

            "Explicit L2 correction was reached",

            2 in transitions,

            "L2 observed",

            result[
                "assistance_transitions"
            ],
        )


        add_check(

            checks,

            "System did not over-escalate to L3",

            result[
                "max_assistance_level"
            ]
            ==
            2,

            "Maximum L2",

            f"L{result['max_assistance_level']}",
        )


        add_check(

            checks,

            "Violation remained in final trial history",

            result[
                "final_task_outcome"
            ]
            ==
            "GOAL_REACHED_WITH_VIOLATION",

            "GOAL_REACHED_WITH_VIOLATION",

            result[
                "final_task_outcome"
            ],
        )


        add_check(

            checks,

            "Assistance recovered from L2 to L0",

            detected_level_after(
                transitions,
                2,
                0,
            ),

            "L2 ... L0",

            result[
                "assistance_transitions"
            ],
        )


        add_check(

            checks,

            "Runtime recovery detected",

            result[
                "returned_to_quiet"
            ],

            True,

            result[
                "returned_to_quiet"
            ],
        )


    # =====================================================
    # PERSISTENT VIOLATION
    # =====================================================

    elif (
        scenario_id
        ==
        "PERSISTENT_VIOLATION"
    ):

        add_check(

            checks,

            "Strong L3 assistance was reached",

            3 in transitions,

            "L3 observed",

            result[
                "assistance_transitions"
            ],
        )


        add_check(

            checks,

            "Strong presentation was shown",

            (
                result[
                    "saw_STRONG_WAYPOINT"
                ]
                or
                result[
                    "saw_STRONG_ORIENTATION"
                ]
            ),

            "STRONG_* presentation",

            (
                f"waypoint="
                f"{result['saw_STRONG_WAYPOINT']}, "
                f"orientation="
                f"{result['saw_STRONG_ORIENTATION']}"
            ),
        )


        add_check(

            checks,

            "Violation remained in final trial history",

            result[
                "final_task_outcome"
            ]
            ==
            "GOAL_REACHED_WITH_VIOLATION",

            "GOAL_REACHED_WITH_VIOLATION",

            result[
                "final_task_outcome"
            ],
        )


        add_check(

            checks,

            "Assistance recovered from L3 to L0",

            detected_level_after(
                transitions,
                3,
                0,
            ),

            "L3 ... L0",

            result[
                "assistance_transitions"
            ],
        )


        add_check(

            checks,

            "Runtime recovery detected",

            result[
                "returned_to_quiet"
            ],

            True,

            result[
                "returned_to_quiet"
            ],
        )


    return checks


# =========================================================
# FINAL TRIAL STATUS
# =========================================================

def evaluate_trial(
    result,
    scenario,
):

    (
        protocol_status,
        protocol_reasons,
    ) = classify_protocol(
        result,
        scenario,
    )


    # -----------------------------------------------------
    # ABORTED / INVALID:
    # Do not score system behavior.
    # -----------------------------------------------------

    if (
        protocol_status
        !=
        "VALID_PROTOCOL"
    ):

        return {

            "status":
                protocol_status,

            "protocol_reasons":
                protocol_reasons,

            "system_checks":
                [],

            "system_pass":
                None,
        }


    # -----------------------------------------------------
    # Protocol valid:
    # Now judge the system.
    # -----------------------------------------------------

    checks = (
        validate_system_behavior(
            result,
            scenario,
        )
    )


    system_pass = all(

        check[
            "passed"
        ]

        for check in checks
    )


    return {

        "status":
            (
                "PASS"
                if system_pass
                else "FAIL"
            ),

        "protocol_reasons":
            [],

        "system_checks":
            checks,

        "system_pass":
            system_pass,
    }


# =========================================================
# PRINT ONE TRIAL
# =========================================================

def print_trial(
    result,
    evaluation,
):

    print()

    print(
        "=" * 72
    )

    print(
        result[
            "trial_id"
        ]
    )

    print(
        "=" * 72
    )


    print(
        "Scenario:",
        result[
            "scenario_id"
        ]
    )


    print(
        "Title:",
        result[
            "scenario_title"
        ]
    )


    print(
        "Frames:",
        result[
            "logged_frames"
        ]
    )


    print(
        "Duration:",
        result[
            "duration_sec"
        ],
        "sec"
    )


    print()


    print(
        "Assistance:",
        result[
            "assistance_transitions"
        ]
    )


    print(
        "Max assistance:",
        f"L{result['max_assistance_level']}"
    )


    print(
        "Max risk duration:",
        result[
            "max_risk_duration_sec"
        ],
        "sec"
    )


    print(
        "Max violation duration:",
        result[
            "max_violation_duration_sec"
        ],
        "sec"
    )


    print(
        "Risk observed:",
        result[
            "saw_RISK"
        ]
    )


    print(
        "Violation observed:",
        result[
            "violation_occurred"
        ]
    )


    print(
        "Goal reached:",
        result[
            "goal_reached"
        ]
    )


    print(
        "Outcome:",
        result[
            "final_task_outcome"
        ]
    )


    if result[
        "recovery_needed"
    ]:

        print(
            "Returned to quiet:",
            result[
                "returned_to_quiet"
            ]
        )

    else:

        print(
            "Recovery needed: NO"
        )


    print()

    print(
        "TRIAL STATUS:",
        evaluation[
            "status"
        ]
    )


    # =====================================================
    # INVALID / ABORTED
    # =====================================================

    if (
        evaluation[
            "status"
        ]
        in
        (
            "INVALID_PROTOCOL",
            "ABORTED",
        )
    ):

        print()

        print(
            "PROTOCOL NOTES"
        )

        print(
            "-" * 72
        )


        for reason in evaluation[
            "protocol_reasons"
        ]:

            print(
                "[NOT SCORED]",
                reason
            )


        print()

        print(
            "SYSTEM VALIDATION: NOT SCORED"
        )


        return


    # =====================================================
    # PASS / FAIL
    # =====================================================

    print()

    print(
        "SYSTEM VALIDATION CHECKS"
    )

    print(
        "-" * 72
    )


    for check in evaluation[
        "system_checks"
    ]:

        marker = (
            "PASS"
            if check[
                "passed"
            ]
            else "FAIL"
        )


        print(
            f"[{marker}] "
            f"{check['name']}"
        )


        if not check[
            "passed"
        ]:

            print(
                "       Expected:",
                check[
                    "expected"
                ]
            )

            print(
                "       Actual:",
                check[
                    "actual"
                ]
            )


    print()

    print(
        "SYSTEM RESULT:",
        evaluation[
            "status"
        ]
    )


# =========================================================
# SAVE CSV
# =========================================================

def save_results(
    evaluated_trials,
):

    if not evaluated_trials:
        return


    output_rows = []


    for item in evaluated_trials:

        result = dict(
            item[
                "result"
            ]
        )


        result.pop(
            "_transitions",
            None,
        )


        evaluation = (
            item[
                "evaluation"
            ]
        )


        result[
            "trial_status"
        ] = (
            evaluation[
                "status"
            ]
        )


        result[
            "protocol_notes"
        ] = " | ".join(
            evaluation[
                "protocol_reasons"
            ]
        )


        result[
            "failed_system_checks"
        ] = " | ".join(

            check[
                "name"
            ]

            for check
            in evaluation[
                "system_checks"
            ]

            if not check[
                "passed"
            ]
        )


        output_rows.append(
            result
        )


    fields = list(
        output_rows[0].keys()
    )


    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=
                fields,
        )


        writer.writeheader()

        writer.writerows(
            output_rows
        )


# =========================================================
# COVERAGE SUMMARY
# =========================================================

def print_coverage(
    evaluated_trials,
    catalog,
):

    print()

    print(
        "=" * 84
    )

    print(
        "SCENARIO COVERAGE"
    )

    print(
        "=" * 84
    )


    scenario_items = {

        scenario[
            "id"
        ]: []

        for scenario
        in catalog.scenarios
    }


    for item in evaluated_trials:

        scenario_id = (
            item[
                "result"
            ][
                "scenario_id"
            ]
        )


        scenario_items[
            scenario_id
        ].append(
            item
        )


    covered = 0


    for scenario in catalog.scenarios:

        scenario_id = (
            scenario[
                "id"
            ]
        )


        items = (
            scenario_items[
                scenario_id
            ]
        )


        if not items:

            print(
                f"{scenario_id:<24} "
                f"NO DATA"
            )

            continue


        valid_items = [

            item

            for item in items

            if item[
                "evaluation"
            ][
                "status"
            ]
            in
            (
                "PASS",
                "FAIL",
            )
        ]


        pass_count = sum(

            1

            for item in valid_items

            if item[
                "evaluation"
            ][
                "status"
            ]
            ==
            "PASS"
        )


        fail_count = sum(

            1

            for item in valid_items

            if item[
                "evaluation"
            ][
                "status"
            ]
            ==
            "FAIL"
        )


        invalid_count = sum(

            1

            for item in items

            if item[
                "evaluation"
            ][
                "status"
            ]
            ==
            "INVALID_PROTOCOL"
        )


        aborted_count = sum(

            1

            for item in items

            if item[
                "evaluation"
            ][
                "status"
            ]
            ==
            "ABORTED"
        )


        if valid_items:

            covered += 1


        print(

            f"{scenario_id:<24} "

            f"VALID={len(valid_items)}  "

            f"PASS={pass_count}  "

            f"FAIL={fail_count}  "

            f"INVALID={invalid_count}  "

            f"ABORTED={aborted_count}"
        )


    # =====================================================
    # OVERALL COUNTS
    # =====================================================

    total = len(
        evaluated_trials
    )


    scored = sum(

        1

        for item in evaluated_trials

        if item[
            "evaluation"
        ][
            "status"
        ]
        in
        (
            "PASS",
            "FAIL",
        )
    )


    passed = sum(

        1

        for item in evaluated_trials

        if item[
            "evaluation"
        ][
            "status"
        ]
        ==
        "PASS"
    )


    failed = sum(

        1

        for item in evaluated_trials

        if item[
            "evaluation"
        ][
            "status"
        ]
        ==
        "FAIL"
    )


    invalid = sum(

        1

        for item in evaluated_trials

        if item[
            "evaluation"
        ][
            "status"
        ]
        ==
        "INVALID_PROTOCOL"
    )


    aborted = sum(

        1

        for item in evaluated_trials

        if item[
            "evaluation"
        ][
            "status"
        ]
        ==
        "ABORTED"
    )


    print()

    print(
        "Scenario coverage:",
        f"{covered}/{len(catalog.scenarios)}"
    )


    print(
        "Total recorded trials:",
        total
    )


    print(
        "Scored valid trials:",
        scored
    )


    print(
        "PASS:",
        passed
    )


    print(
        "FAIL:",
        failed
    )


    print(
        "INVALID_PROTOCOL:",
        invalid
    )


    print(
        "ABORTED:",
        aborted
    )


    print()

    print(
        "Validation CSV:"
    )


    print(
        OUTPUT_CSV
    )


    print()

    print(
        "NOTE:"
    )


    print(
        "PASS/FAIL is calculated only for completed "
        "protocol-valid trials."
    )


    print(
        "INVALID_PROTOCOL and ABORTED trials are retained "
        "for transparency but are not counted as "
        "system-validation failures."
    )


    print(
        "These are prototype engineering validation trials, "
        "not a user study."
    )


    print(
        "=" * 84
    )

    print()


# =========================================================
# MAIN
# =========================================================

def main():

    print()

    print(
        "========================================================"
    )

    print(
        "AdaptiveSkill - Scenario-Aware Runtime Validation"
    )

    print(
        "========================================================"
    )


    catalog = (
        RuntimeScenarioCatalog()
    )


    if not RUNTIME_DIR.exists():

        print(
            "Runtime directory not found:"
        )

        print(
            RUNTIME_DIR
        )

        return


    # =====================================================
    # DISCOVER SCENARIO-LABELLED TRIALS
    # =====================================================

    csv_files = []


    for path in RUNTIME_DIR.glob(
        "*.csv"
    ):

        if path.name in (
            "runtime_analysis.csv",
            "scenario_validation.csv",
        ):
            continue


        rows = load_csv(
            path
        )


        if not rows:
            continue


        scenario = identify_scenario(

            csv_path=
                path,

            rows=
                rows,

            catalog=
                catalog,
        )


        if scenario is None:
            continue


        csv_files.append(
            (
                path,
                scenario,
            )
        )


    csv_files.sort(

        key=lambda item:
            item[0]
            .stat()
            .st_mtime
    )


    if not csv_files:

        print()

        print(
            "No scenario-labelled runtime trials found."
        )

        return


    # =====================================================
    # ANALYZE
    # =====================================================

    evaluated_trials = []


    for (
        csv_path,
        scenario,
    ) in csv_files:

        result = analyze_trial(

            csv_path=
                csv_path,

            scenario=
                scenario,
        )


        if result is None:
            continue


        evaluation = (
            evaluate_trial(
                result,
                scenario,
            )
        )


        evaluated_trials.append(

            {
                "result":
                    result,

                "evaluation":
                    evaluation,
            }
        )


        print_trial(
            result,
            evaluation,
        )


    # =====================================================
    # SAVE / COVERAGE
    # =====================================================

    save_results(
        evaluated_trials
    )


    print_coverage(

        evaluated_trials=
            evaluated_trials,

        catalog=
            catalog,
    )


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":

    main()