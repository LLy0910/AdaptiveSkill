import csv
import json
import time
from datetime import datetime
from pathlib import Path


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "runtime_trials"
)


# =========================================================
# CSV COLUMNS
# =========================================================

CSV_FIELDS = [

    # -----------------------------------------------------
    # Trial / time
    # -----------------------------------------------------

    "trial_id",
    "trial_label",

    "timestamp_sec",

    # -----------------------------------------------------
    # User / virtual object state
    # -----------------------------------------------------

    "x",
    "y",
    "orientation_deg",

    # -----------------------------------------------------
    # Task constraints
    # -----------------------------------------------------

    "task_state",
    "dominant_constraint",

    "task_valid_so_far",
    "goal_reached",

    "obstacle_risk",
    "obstacle_violation",
    "obstacle_clearance",

    "orientation_error_deg",

    "segment_crossing_detected",

    # -----------------------------------------------------
    # Reference baseline vs task-aware policy
    # -----------------------------------------------------

    "reference_distance",

    "reference_policy",
    "task_policy",

    "comparison_label",

    # -----------------------------------------------------
    # Base corrective guidance
    # -----------------------------------------------------

    "base_guidance_type",
    "base_guidance_message",

    # -----------------------------------------------------
    # Adaptive scaffolding
    # -----------------------------------------------------

    "assistance_level",
    "assistance_label",
    "requested_level",

    "risk_duration_sec",
    "violation_duration_sec",

    "recent_violation_episodes",

    "assistance_reason",

    # -----------------------------------------------------
    # User-facing presentation
    # -----------------------------------------------------

    "presentation_mode",

    "show_direction_arrow",
    "show_waypoint",

    "show_orientation_ghost",

    "show_rotation_direction",
    "show_rotation_degrees",

    "show_strong_banner",
    "show_recovery_status",

    "assistance_paused",

    # -----------------------------------------------------
    # Sticky trial history
    # -----------------------------------------------------

    "trial_violation_occurred",
    "trial_violation_episodes",

    "trial_goal_reached_seen",

    "trial_task_outcome",
]


# =========================================================
# HELPERS
# =========================================================

def safe_get(
    obj,
    name,
    default=None,
):

    if obj is None:
        return default

    return getattr(
        obj,
        name,
        default,
    )


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


def bool_to_int(
    value,
):

    return (
        1
        if bool(value)
        else 0
    )


# =========================================================
# RUNTIME LOGGER
# =========================================================

class RuntimeLogger:
    """
    Logs one interactive prototype trial frame-by-frame.

    Purpose:
        provide runtime validation evidence that the
        implemented task-aware adaptive assistance policy
        behaves as designed during actual interaction.

    IMPORTANT:
        This is prototype runtime validation.

        It is NOT:
            - a user study
            - evidence of learning effectiveness
            - evidence of psychological state
            - evidence that thresholds are optimal
    """


    def __init__(
        self,
        output_dir=None,
    ):

        if output_dir is None:

            output_dir = (
                DEFAULT_OUTPUT_DIR
            )


        self.output_dir = Path(
            output_dir
        )


        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        self.reset()


    # =====================================================
    # RESET
    # =====================================================

    def reset(self):

        self.active = False

        self.trial_id = None

        self.trial_label = None

        self.start_time = None


        self.csv_path = None

        self.summary_path = None


        self.csv_file = None

        self.writer = None


        self.frame_count = 0


        self.max_assistance_level = 0


        self.assistance_transitions = []


        self.previous_assistance_level = None


        self.first_timestamp_by_level = {}


        self.task_state_counts = {}


        self.presentation_counts = {}


        self.segment_crossing_count = 0


    # =====================================================
    # START TRIAL
    # =====================================================

    def start_trial(
        self,
        trial_label="UNLABELED",
        timestamp_sec=None,
    ):

        # -------------------------------------------------
        # If an old trial is still open, close it safely.
        # -------------------------------------------------

        if self.active:

            self.finish_trial(
                timestamp_sec=
                    timestamp_sec
            )


        self.reset()


        if timestamp_sec is None:

            timestamp_sec = (
                time.perf_counter()
            )


        now_string = (
            datetime.now()
            .strftime(
                "%Y%m%d_%H%M%S_%f"
            )
        )


        safe_label = (

            str(
                trial_label
            )
            .strip()
            .replace(
                " ",
                "_"
            )
            .replace(
                "/",
                "_"
            )
            .replace(
                "\\",
                "_"
            )
        )


        if not safe_label:

            safe_label = (
                "UNLABELED"
            )


        self.trial_id = (

            safe_label

            +

            "_"

            +

            now_string
        )


        self.trial_label = str(
            trial_label
        )


        self.start_time = float(
            timestamp_sec
        )


        self.csv_path = (

            self.output_dir

            /

            (
                self.trial_id
                +
                ".csv"
            )
        )


        self.summary_path = (

            self.output_dir

            /

            (
                self.trial_id
                +
                "_summary.json"
            )
        )


        self.csv_file = open(

            self.csv_path,

            "w",

            newline="",

            encoding="utf-8",
        )


        self.writer = csv.DictWriter(

            self.csv_file,

            fieldnames=
                CSV_FIELDS,
        )


        self.writer.writeheader()


        self.active = True


        return (
            self.csv_path
        )


    # =====================================================
    # RELATIVE TIME
    # =====================================================

    def _relative_time(
        self,
        timestamp_sec,
    ):

        if self.start_time is None:

            return 0.0


        return max(

            0.0,

            float(
                timestamp_sec
            )

            -

            float(
                self.start_time
            )
        )


    # =====================================================
    # ASSISTANCE HISTORY
    # =====================================================

    def _update_assistance_history(
        self,
        relative_time,
        scaffolding_decision,
    ):

        level = int(
            safe_get(
                scaffolding_decision,
                "level",
                0,
            )
        )


        # -------------------------------------------------
        # Highest assistance used in trial.
        # -------------------------------------------------

        self.max_assistance_level = max(

            self.max_assistance_level,

            level,
        )


        # -------------------------------------------------
        # First appearance of each assistance level.
        # -------------------------------------------------

        if (
            level
            not in
            self.first_timestamp_by_level
        ):

            self.first_timestamp_by_level[
                level
            ] = round(
                relative_time,
                4,
            )


        # -------------------------------------------------
        # First frame.
        # -------------------------------------------------

        if (
            self.previous_assistance_level
            is None
        ):

            self.assistance_transitions.append(

                {
                    "timestamp_sec":
                        round(
                            relative_time,
                            4,
                        ),

                    "from_level":
                        None,

                    "to_level":
                        level,
                }
            )


        # -------------------------------------------------
        # Real transition.
        # -------------------------------------------------

        elif (
            level
            !=
            self.previous_assistance_level
        ):

            self.assistance_transitions.append(

                {
                    "timestamp_sec":
                        round(
                            relative_time,
                            4,
                        ),

                    "from_level":
                        int(
                            self.previous_assistance_level
                        ),

                    "to_level":
                        level,
                }
            )


        self.previous_assistance_level = (
            level
        )


    # =====================================================
    # COUNTS
    # =====================================================

    def _increment_count(
        self,
        dictionary,
        key,
    ):

        key = str(
            key
        )


        dictionary[key] = (

            dictionary.get(
                key,
                0,
            )

            +

            1
        )


    # =====================================================
    # LOG FRAME
    # =====================================================

    def log_frame(
        self,

        timestamp_sec,

        x,
        y,
        orientation_deg,

        task_result,

        comparison,

        base_guidance,

        scaffolding_decision,

        presentation,

        trial_summary=None,

        segment_crossing_detected=False,
    ):

        if not self.active:

            raise RuntimeError(
                "RuntimeLogger.log_frame() called "
                "before start_trial()."
            )


        relative_time = (
            self._relative_time(
                timestamp_sec
            )
        )


        # =================================================
        # ASSISTANCE HISTORY
        # =================================================

        self._update_assistance_history(

            relative_time=
                relative_time,

            scaffolding_decision=
                scaffolding_decision,
        )


        # =================================================
        # TASK / PRESENTATION
        # =================================================

        task_state = safe_get(

            task_result,

            "state",

            "UNKNOWN",
        )


        presentation_mode = safe_get(

            presentation,

            "display_mode",

            "NONE",
        )


        self._increment_count(

            self.task_state_counts,

            task_state,
        )


        self._increment_count(

            self.presentation_counts,

            presentation_mode,
        )


        if segment_crossing_detected:

            self.segment_crossing_count += 1


        # =================================================
        # CSV ROW
        # =================================================

        row = {

            # ------------------------------------------------
            # Trial
            # ------------------------------------------------

            "trial_id":
                self.trial_id,

            "trial_label":
                self.trial_label,

            "timestamp_sec":
                round(
                    relative_time,
                    6,
                ),


            # ------------------------------------------------
            # Current state
            # ------------------------------------------------

            "x":
                round(
                    safe_float(
                        x
                    ),
                    6,
                ),

            "y":
                round(
                    safe_float(
                        y
                    ),
                    6,
                ),

            "orientation_deg":
                round(
                    safe_float(
                        orientation_deg
                    ),
                    4,
                ),


            # ------------------------------------------------
            # Task constraints
            # ------------------------------------------------

            "task_state":
                task_state,

            "dominant_constraint":
                safe_get(
                    task_result,
                    "dominant_constraint",
                    "NONE",
                ),

            "task_valid_so_far":
                bool_to_int(
                    safe_get(
                        task_result,
                        "task_valid_so_far",
                        False,
                    )
                ),

            "goal_reached":
                bool_to_int(
                    safe_get(
                        task_result,
                        "goal_reached",
                        False,
                    )
                ),

            "obstacle_risk":
                bool_to_int(
                    safe_get(
                        task_result,
                        "obstacle_risk",
                        False,
                    )
                ),

            "obstacle_violation":
                bool_to_int(
                    safe_get(
                        task_result,
                        "obstacle_violation",
                        False,
                    )
                ),

            "obstacle_clearance":
                round(
                    safe_float(
                        safe_get(
                            task_result,
                            "obstacle_clearance",
                            float("nan"),
                        )
                    ),
                    6,
                ),

            # =================================================
            # FIXED FIELD
            #
            # Correct field on TaskConstraintResult:
            #
            # orientation_error_deg
            #
            # NOT:
            #
            # orientation_error
            # =================================================

            "orientation_error_deg":
                round(
                    safe_float(
                        safe_get(
                            task_result,
                            "orientation_error_deg",
                            float("nan"),
                        )
                    ),
                    4,
                ),

            "segment_crossing_detected":
                bool_to_int(
                    segment_crossing_detected
                ),


            # ------------------------------------------------
            # Baseline vs task-aware policy
            # ------------------------------------------------

            "reference_distance":
                round(
                    safe_float(
                        safe_get(
                            comparison,
                            "reference_distance",
                            float("nan"),
                        )
                    ),
                    6,
                ),

            "reference_policy":
                safe_get(
                    comparison,
                    "reference_policy_decision",
                    "UNKNOWN",
                ),

            "task_policy":
                safe_get(
                    comparison,
                    "task_policy_decision",
                    "UNKNOWN",
                ),

            "comparison_label":
                safe_get(
                    comparison,
                    "comparison_label",
                    "UNKNOWN",
                ),


            # ------------------------------------------------
            # Base task-aware guidance
            # ------------------------------------------------

            "base_guidance_type":
                safe_get(
                    base_guidance,
                    "guidance_type",
                    "NONE",
                ),

            "base_guidance_message":
                safe_get(
                    base_guidance,
                    "message",
                    "",
                ),


            # ------------------------------------------------
            # Adaptive scaffolding
            # ------------------------------------------------

            "assistance_level":
                int(
                    safe_get(
                        scaffolding_decision,
                        "level",
                        0,
                    )
                ),

            "assistance_label":
                safe_get(
                    scaffolding_decision,
                    "label",
                    "UNKNOWN",
                ),

            "requested_level":
                int(
                    safe_get(
                        scaffolding_decision,
                        "requested_level",
                        0,
                    )
                ),

            "risk_duration_sec":
                round(
                    safe_float(
                        safe_get(
                            scaffolding_decision,
                            "risk_duration_sec",
                            0.0,
                        )
                    ),
                    4,
                ),

            "violation_duration_sec":
                round(
                    safe_float(
                        safe_get(
                            scaffolding_decision,
                            "violation_duration_sec",
                            0.0,
                        )
                    ),
                    4,
                ),

            "recent_violation_episodes":
                int(
                    safe_get(
                        scaffolding_decision,
                        "recent_violation_episodes",
                        0,
                    )
                ),

            "assistance_reason":
                safe_get(
                    scaffolding_decision,
                    "reason",
                    "",
                ),


            # ------------------------------------------------
            # User-facing guidance presentation
            # ------------------------------------------------

            "presentation_mode":
                presentation_mode,

            "show_direction_arrow":
                bool_to_int(
                    safe_get(
                        presentation,
                        "show_direction_arrow",
                        False,
                    )
                ),

            "show_waypoint":
                bool_to_int(
                    safe_get(
                        presentation,
                        "show_waypoint",
                        False,
                    )
                ),

            "show_orientation_ghost":
                bool_to_int(
                    safe_get(
                        presentation,
                        "show_orientation_ghost",
                        False,
                    )
                ),

            "show_rotation_direction":
                bool_to_int(
                    safe_get(
                        presentation,
                        "show_rotation_direction",
                        False,
                    )
                ),

            "show_rotation_degrees":
                bool_to_int(
                    safe_get(
                        presentation,
                        "show_rotation_degrees",
                        False,
                    )
                ),

            "show_strong_banner":
                bool_to_int(
                    safe_get(
                        presentation,
                        "show_strong_banner",
                        False,
                    )
                ),

            "show_recovery_status":
                bool_to_int(
                    safe_get(
                        presentation,
                        "show_recovery_status",
                        False,
                    )
                ),

            "assistance_paused":
                bool_to_int(
                    safe_get(
                        presentation,
                        "paused",
                        False,
                    )
                ),


            # ------------------------------------------------
            # Sticky trial-level history
            # ------------------------------------------------

            "trial_violation_occurred":
                bool_to_int(
                    safe_get(
                        trial_summary,
                        "constraint_violation_occurred",
                        False,
                    )
                ),

            "trial_violation_episodes":
                int(
                    safe_get(
                        trial_summary,
                        "constraint_violation_episodes",
                        0,
                    )
                ),

            "trial_goal_reached_seen":
                bool_to_int(
                    safe_get(
                        trial_summary,
                        "goal_reached_seen",
                        False,
                    )
                ),

            "trial_task_outcome":
                safe_get(
                    trial_summary,
                    "task_outcome",
                    "IN_PROGRESS",
                ),
        }


        self.writer.writerow(
            row
        )


        self.frame_count += 1


        # -------------------------------------------------
        # Flush after each frame so data survives
        # accidental demo interruption.
        # -------------------------------------------------

        self.csv_file.flush()


    # =====================================================
    # FINISH TRIAL
    # =====================================================

    def finish_trial(
        self,
        timestamp_sec=None,
        final_trial_summary=None,
    ):

        if not self.active:

            return None


        if timestamp_sec is None:

            timestamp_sec = (
                time.perf_counter()
            )


        duration = (
            self._relative_time(
                timestamp_sec
            )
        )


        # =================================================
        # SUMMARY JSON
        # =================================================

        summary = {

            "trial_id":
                self.trial_id,

            "trial_label":
                self.trial_label,

            "duration_sec":
                round(
                    duration,
                    4,
                ),

            "logged_frames":
                int(
                    self.frame_count
                ),

            # ------------------------------------------------
            # Assistance
            # ------------------------------------------------

            "max_assistance_level":
                int(
                    self.max_assistance_level
                ),

            "first_timestamp_by_level":
                {

                    str(
                        key
                    ):
                        value

                    for key, value
                    in
                    self.first_timestamp_by_level.items()
                },

            "assistance_transitions":
                self.assistance_transitions,

            # ------------------------------------------------
            # Runtime state counts
            # ------------------------------------------------

            "task_state_counts":
                self.task_state_counts,

            "presentation_counts":
                self.presentation_counts,

            "segment_crossing_count":
                int(
                    self.segment_crossing_count
                ),

            # ------------------------------------------------
            # Final trial outcome
            # ------------------------------------------------

            "final_task_outcome":
                safe_get(
                    final_trial_summary,
                    "task_outcome",
                    None,
                ),

            "constraint_violation_occurred":
                bool(
                    safe_get(
                        final_trial_summary,
                        "constraint_violation_occurred",
                        False,
                    )
                ),

            "constraint_violation_episodes":
                int(
                    safe_get(
                        final_trial_summary,
                        "constraint_violation_episodes",
                        0,
                    )
                ),

            "goal_reached":
                bool(
                    safe_get(
                        final_trial_summary,
                        "goal_reached_seen",
                        False,
                    )
                ),

            "returned_to_quiet":
                bool(
                    safe_get(
                        final_trial_summary,
                        "returned_to_quiet",
                        False,
                    )
                ),
        }


        with open(

            self.summary_path,

            "w",

            encoding="utf-8",

        ) as file:

            json.dump(

                summary,

                file,

                indent=4,

                ensure_ascii=False,
            )


        if self.csv_file is not None:

            self.csv_file.flush()

            self.csv_file.close()


        csv_path = (
            self.csv_path
        )


        summary_path = (
            self.summary_path
        )


        self.active = False

        self.csv_file = None

        self.writer = None


        return {

            "csv_path":
                csv_path,

            "summary_path":
                summary_path,

            "summary":
                summary,
        }


# =========================================================
# SELF TEST
# =========================================================

if __name__ == "__main__":

    from types import SimpleNamespace


    print()

    print(
        "========================================================"
    )

    print(
        "AdaptiveSkill - Runtime Logger Test"
    )

    print(
        "========================================================"
    )


    logger = (
        RuntimeLogger()
    )


    # =====================================================
    # START TEST TRIAL
    # =====================================================

    logger.start_trial(

        trial_label=
            "LOGGER_TEST",

        timestamp_sec=
            100.0,
    )


    # =====================================================
    # SYNTHETIC TEST OBJECTS
    # =====================================================

    def make_task(
        state,
        violation=False,
    ):

        return SimpleNamespace(

            state=
                state,

            dominant_constraint=
                (
                    "OBSTACLE"

                    if state != "VALID"

                    else "NONE"
                ),

            task_valid_so_far=
                not violation,

            goal_reached=
                False,

            obstacle_risk=
                (
                    state
                    ==
                    "RISK"
                ),

            obstacle_violation=
                violation,

            obstacle_clearance=
                (
                    0.10

                    if state == "RISK"

                    else 0.0

                    if violation

                    else 0.50
                ),

            # =================================================
            # FIXED SELF-TEST ATTRIBUTE
            # =================================================

            orientation_error_deg=
                0.0,
        )


    def make_comparison(
        level,
    ):

        return SimpleNamespace(

            reference_distance=
                0.8,

            reference_policy_decision=
                "INTERVENE",

            task_policy_decision=
                (
                    "STAY_QUIET"

                    if level == 0

                    else "INTERVENE"
                ),

            comparison_label=
                (
                    "UNNECESSARY INTERVENTION AVOIDED"

                    if level == 0

                    else "TASK-RELEVANT INTERVENTION"
                ),
        )


    def make_guidance(
        guidance_type,
    ):

        return SimpleNamespace(

            guidance_type=
                guidance_type,

            message=
                "Synthetic logger test.",
        )


    def make_scaffolding(
        level,
        label,
        requested_level=None,
    ):

        if requested_level is None:

            requested_level = (
                level
            )


        return SimpleNamespace(

            level=
                level,

            label=
                label,

            requested_level=
                requested_level,

            risk_duration_sec=
                0.0,

            violation_duration_sec=
                0.0,

            recent_violation_episodes=
                0,

            reason=
                "Synthetic logger test.",
        )


    def make_presentation(
        mode,
        waypoint=False,
        strong=False,
        recovery=False,
    ):

        return SimpleNamespace(

            display_mode=
                mode,

            show_direction_arrow=
                (
                    mode
                    !=
                    "NONE"
                ),

            show_waypoint=
                waypoint,

            show_orientation_ghost=
                False,

            show_rotation_direction=
                False,

            show_rotation_degrees=
                False,

            show_strong_banner=
                strong,

            show_recovery_status=
                recovery,

            paused=
                False,
        )


    # =====================================================
    # SYNTHETIC SEQUENCE
    #
    # L0 -> L1 -> L2 -> L3 -> L2 -> L1 -> L0
    # =====================================================

    sequence = [

        (
            100.0,
            "VALID",
            0,
            "OBSERVE",
            "NONE",
        ),

        (
            100.5,
            "RISK",
            1,
            "MINIMAL CUE",
            "MINIMAL_DIRECTION",
        ),

        (
            101.4,
            "RISK",
            2,
            "EXPLICIT CORRECTION",
            "EXPLICIT_WAYPOINT",
        ),

        (
            102.0,
            "VIOLATION",
            2,
            "EXPLICIT CORRECTION",
            "EXPLICIT_WAYPOINT",
        ),

        (
            104.1,
            "VIOLATION",
            3,
            "STRONG ASSISTANCE",
            "STRONG_WAYPOINT",
        ),

        (
            104.8,
            "VALID",
            2,
            "EXPLICIT CORRECTION",
            "RECOVERY_FADE",
        ),

        (
            105.5,
            "VALID",
            1,
            "MINIMAL CUE",
            "RECOVERY_FADE",
        ),

        (
            106.2,
            "VALID",
            0,
            "OBSERVE",
            "NONE",
        ),
    ]


    sticky_violation = False

    violation_episodes = 0


    for index, item in enumerate(
        sequence
    ):

        (
            timestamp,
            state,
            level,
            label,
            mode,
        ) = item


        violation = (

            state
            ==
            "VIOLATION"
        )


        if (
            violation

            and

            not sticky_violation
        ):

            violation_episodes += 1


        if violation:

            sticky_violation = True


        task = make_task(

            state=
                state,

            violation=
                violation,
        )


        comparison = (
            make_comparison(
                level
            )
        )


        guidance = (
            make_guidance(

                "NONE"

                if level == 0

                else "DIRECTIONAL_NUDGE"
            )
        )


        scaffolding = (
            make_scaffolding(
                level,
                label,
            )
        )


        presentation = (
            make_presentation(

                mode=
                    mode,

                waypoint=
                    level >= 2,

                strong=
                    level == 3,

                recovery=
                    (
                        mode
                        ==
                        "RECOVERY_FADE"
                    ),
            )
        )


        trial_summary = (
            SimpleNamespace(

                constraint_violation_occurred=
                    sticky_violation,

                constraint_violation_episodes=
                    violation_episodes,

                goal_reached_seen=
                    False,

                task_outcome=
                    (
                        "VIOLATION_OCCURRED"

                        if sticky_violation

                        else "IN_PROGRESS"
                    ),
            )
        )


        logger.log_frame(

            timestamp_sec=
                timestamp,

            x=
                (
                    0.2
                    +
                    index
                    *
                    0.3
                ),

            y=
                0.8,

            orientation_deg=
                0.0,

            task_result=
                task,

            comparison=
                comparison,

            base_guidance=
                guidance,

            scaffolding_decision=
                scaffolding,

            presentation=
                presentation,

            trial_summary=
                trial_summary,

            segment_crossing_detected=
                False,
        )


    # =====================================================
    # FINAL TRIAL SUMMARY
    # =====================================================

    final_trial_summary = (
        SimpleNamespace(

            task_outcome=
                "VIOLATION_OCCURRED",

            constraint_violation_occurred=
                True,

            constraint_violation_episodes=
                1,

            goal_reached_seen=
                False,

            returned_to_quiet=
                True,
        )
    )


    result = (
        logger.finish_trial(

            timestamp_sec=
                106.2,

            final_trial_summary=
                final_trial_summary,
        )
    )


    summary = (
        result[
            "summary"
        ]
    )


    # =====================================================
    # EXPECTED TRANSITIONS
    # =====================================================

    expected_levels = [

        0,
        1,
        2,
        3,
        2,
        1,
        0,
    ]


    actual_levels = [

        transition[
            "to_level"
        ]

        for transition
        in
        summary[
            "assistance_transitions"
        ]
    ]


    # =====================================================
    # VALIDATION
    # =====================================================

    pass_frames = (

        summary[
            "logged_frames"
        ]
        ==
        8
    )


    pass_max_level = (

        summary[
            "max_assistance_level"
        ]
        ==
        3
    )


    pass_transitions = (

        actual_levels
        ==
        expected_levels
    )


    pass_violation = (

        summary[
            "constraint_violation_occurred"
        ]
        is True
    )


    pass_recovery = (

        summary[
            "returned_to_quiet"
        ]
        is True
    )


    final_pass = (

        pass_frames

        and

        pass_max_level

        and

        pass_transitions

        and

        pass_violation

        and

        pass_recovery
    )


    # =====================================================
    # PRINT RESULT
    # =====================================================

    print()

    print(
        "CSV:"
    )

    print(
        result[
            "csv_path"
        ]
    )


    print()

    print(
        "SUMMARY:"
    )

    print(
        result[
            "summary_path"
        ]
    )


    print()

    print(
        "Logged frames:",
        summary[
            "logged_frames"
        ]
    )


    print(
        "Max assistance level:",
        summary[
            "max_assistance_level"
        ]
    )


    print(
        "Expected transitions:",
        expected_levels
    )


    print(
        "Actual transitions:",
        actual_levels
    )


    print(
        "Violation occurred:",
        summary[
            "constraint_violation_occurred"
        ]
    )


    print(
        "Returned to quiet:",
        summary[
            "returned_to_quiet"
        ]
    )


    print()

    print(
        "Frames PASS:",
        pass_frames
    )


    print(
        "Max level PASS:",
        pass_max_level
    )


    print(
        "Transitions PASS:",
        pass_transitions
    )


    print(
        "Violation history PASS:",
        pass_violation
    )


    print(
        "Recovery PASS:",
        pass_recovery
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
            "RUNTIME LOGGER: PASSED"
        )


    else:

        print(
            "RUNTIME LOGGER: FAILED"
        )


    print(
        "========================================================"
    )

    print()