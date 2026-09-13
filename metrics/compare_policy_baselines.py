import csv
import json
import math
import sys
from collections import defaultdict
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
# IMPORTS
# =========================================================

from interaction.policy_baselines import (
    ReferenceOnlyPolicy,
    TaskAwareBinaryPolicy,
    DECISION_INTERVENE,
)


# =========================================================
# PATHS
# =========================================================

RUNTIME_DIR = (
    PROJECT_ROOT
    / "data"
    / "runtime_trials"
)

SCENARIO_VALIDATION_CSV = (
    RUNTIME_DIR
    / "scenario_validation.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "policy_comparison"
)

# Keep the earlier v1 outputs for transparency.
OUTPUT_CSV = (
    OUTPUT_DIR
    / "three_policy_comparison_v2.csv"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "three_policy_comparison_summary_v2.json"
)


# =========================================================
# PRESENTATION MODES
#
# IMPORTANT SEMANTIC FIX:
#
# assistance_level is the controller's internal scaffold
# state. It is NOT always equal to what the user sees.
#
# During RECOVERY_FADE the controller may still internally
# be at L2/L1 while stale corrective waypoint/ghost cues
# have already been removed. Therefore exposure must be
# measured from presentation_mode, not assistance_level.
# =========================================================

MINIMAL_MODES = {
    "MINIMAL_DIRECTION",
    "MINIMAL_ORIENTATION",
}

EXPLICIT_MODES = {
    "EXPLICIT_DIRECTION",
    "EXPLICIT_WAYPOINT",
    "EXPLICIT_ORIENTATION",
}

STRONG_MODES = {
    "STRONG_WAYPOINT",
    "STRONG_ORIENTATION",
}

RECOVERY_MODES = {
    "RECOVERY_FADE",
}

PAUSED_MODES = {
    "PAUSED",
}

NO_GUIDANCE_MODES = {
    "NONE",
    "",
}


# =========================================================
# HELPERS
# =========================================================

def safe_float(
    value,
    default=0.0,
):

    try:
        return float(value)

    except Exception:
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


def estimate_frame_durations(
    rows,
):

    if not rows:
        return []


    if len(rows) == 1:
        return [0.0]


    timestamps = [

        safe_float(
            row.get(
                "timestamp_sec",
                0.0,
            )
        )

        for row
        in rows
    ]


    durations = []


    for index in range(
        len(rows)
    ):

        if index < len(rows) - 1:

            dt = max(
                0.0,
                timestamps[
                    index + 1
                ]
                -
                timestamps[
                    index
                ],
            )

        else:

            # Do not invent duration after the final sample.
            dt = 0.0


        durations.append(
            dt
        )


    return durations


# =========================================================
# VALID PASS TRIAL INDEX
#
# Only trials already classified PASS by the predefined
# scenario-aware analyzer are compared here.
# =========================================================

def load_pass_trials():

    if not SCENARIO_VALIDATION_CSV.exists():

        raise FileNotFoundError(
            "scenario_validation.csv not found. "
            "Run metrics/analyze_runtime_trials.py first."
        )


    rows = load_csv(
        SCENARIO_VALIDATION_CSV
    )


    pass_trials = {}


    for row in rows:

        if (
            str(
                row.get(
                    "trial_status",
                    ""
                )
            )
            !=
            "PASS"
        ):

            continue


        trial_id = str(
            row.get(
                "trial_id",
                ""
            )
        ).strip()


        if not trial_id:
            continue


        pass_trials[
            trial_id
        ] = {

            "scenario_id":
                str(
                    row.get(
                        "scenario_id",
                        ""
                    )
                ),

            "trial_id":
                trial_id,
        }


    return pass_trials


# =========================================================
# PRESENTATION CLASSIFICATION
# =========================================================

def classify_adaptive_presentation(
    presentation_mode,
):

    mode = str(
        presentation_mode
    ).strip()


    if mode in MINIMAL_MODES:

        return "MINIMAL"


    if mode in EXPLICIT_MODES:

        return "EXPLICIT"


    if mode in STRONG_MODES:

        return "STRONG"


    if mode in RECOVERY_MODES:

        return "RECOVERY_FADE"


    if mode in PAUSED_MODES:

        return "PAUSED"


    return "NONE"


# =========================================================
# ANALYZE ONE OBSERVED TRACE
#
# All three policies are evaluated on the SAME recorded
# trace. This is an offline policy-exposure comparison.
#
# It is NOT a causal estimate of how the human would behave
# under another policy.
# =========================================================

def analyze_trial(
    csv_path,
    scenario_id,
):

    rows = load_csv(
        csv_path
    )


    if not rows:
        return None


    durations = (
        estimate_frame_durations(
            rows
        )
    )


    reference_policy = (
        ReferenceOnlyPolicy(
            reference_threshold=
                0.30
        )
    )


    binary_policy = (
        TaskAwareBinaryPolicy()
    )


    # =====================================================
    # TASK EXPOSURE
    # =====================================================

    total_time = 0.0

    task_valid_time = 0.0

    task_risk_time = 0.0

    task_violation_time = 0.0

    task_risk_or_violation_time = 0.0


    # =====================================================
    # A. REFERENCE-ONLY
    # =====================================================

    reference_intervention_time = 0.0

    reference_unnecessary_valid_time = 0.0

    reference_missed_task_risk_time = 0.0


    # =====================================================
    # B. TASK-AWARE BINARY
    # =====================================================

    binary_intervention_time = 0.0

    binary_full_guidance_time = 0.0


    # =====================================================
    # C. ADAPTIVESKILL
    #
    # USER-VISIBLE exposure is classified from
    # presentation_mode, not internal assistance_level.
    # =====================================================

    adaptive_minimal_time = 0.0

    adaptive_explicit_time = 0.0

    adaptive_strong_time = 0.0

    adaptive_corrective_time = 0.0

    adaptive_recovery_fade_time = 0.0

    adaptive_paused_time = 0.0

    adaptive_no_guidance_time = 0.0

    adaptive_any_visible_assistance_time = 0.0


    # =====================================================
    # LOOP
    # =====================================================

    for (
        row,
        dt,
    ) in zip(
        rows,
        durations,
    ):

        total_time += dt


        task_state = str(
            row.get(
                "task_state",
                ""
            )
        ).strip()


        reference_distance = safe_float(
            row.get(
                "reference_distance",
                0.0,
            ),
            0.0,
        )


        tracking_reliable = (

            task_state
            !=
            "TRACKING_UNRELIABLE"
        )


        # -------------------------------------------------
        # TASK EXPOSURE
        # -------------------------------------------------

        if task_state == "VALID":

            task_valid_time += dt


        elif task_state == "RISK":

            task_risk_time += dt

            task_risk_or_violation_time += dt


        elif task_state == "VIOLATION":

            task_violation_time += dt

            task_risk_or_violation_time += dt


        # -------------------------------------------------
        # A. REFERENCE-ONLY
        # -------------------------------------------------

        reference_decision = (
            reference_policy.decide(

                reference_distance=
                    reference_distance,

                tracking_reliable=
                    tracking_reliable,
            )
        )


        if (
            reference_decision.decision
            ==
            DECISION_INTERVENE
        ):

            reference_intervention_time += dt


            if task_state == "VALID":

                reference_unnecessary_valid_time += dt


        elif (
            task_state
            in
            (
                "RISK",
                "VIOLATION",
            )
        ):

            reference_missed_task_risk_time += dt


        # -------------------------------------------------
        # B. TASK-AWARE BINARY
        # -------------------------------------------------

        binary_decision = (
            binary_policy.decide(
                task_state
            )
        )


        if (
            binary_decision.decision
            ==
            DECISION_INTERVENE
        ):

            binary_intervention_time += dt

            # Binary comparator gives the same complete
            # correction whenever task risk/violation exists.
            binary_full_guidance_time += dt


        # -------------------------------------------------
        # C. ADAPTIVESKILL USER-VISIBLE PRESENTATION
        # -------------------------------------------------

        presentation_class = (
            classify_adaptive_presentation(
                row.get(
                    "presentation_mode",
                    ""
                )
            )
        )


        if (
            presentation_class
            ==
            "MINIMAL"
        ):

            adaptive_minimal_time += dt

            adaptive_any_visible_assistance_time += dt


        elif (
            presentation_class
            ==
            "EXPLICIT"
        ):

            adaptive_explicit_time += dt

            adaptive_corrective_time += dt

            adaptive_any_visible_assistance_time += dt


        elif (
            presentation_class
            ==
            "STRONG"
        ):

            adaptive_strong_time += dt

            adaptive_corrective_time += dt

            adaptive_any_visible_assistance_time += dt


        elif (
            presentation_class
            ==
            "RECOVERY_FADE"
        ):

            adaptive_recovery_fade_time += dt

            adaptive_any_visible_assistance_time += dt


        elif (
            presentation_class
            ==
            "PAUSED"
        ):

            adaptive_paused_time += dt


        else:

            adaptive_no_guidance_time += dt


    # =====================================================
    # COMPARISON METRICS
    # =====================================================

    corrective_reduction_vs_binary = (

        binary_full_guidance_time

        -
        adaptive_corrective_time
    )


    if (
        binary_full_guidance_time
        >
        1e-9
    ):

        corrective_reduction_ratio = (

            corrective_reduction_vs_binary

            /

            binary_full_guidance_time
        )

    else:

        corrective_reduction_ratio = (
            float(
                "nan"
            )
        )


    # -----------------------------------------------------
    # Minimal-cue substitution:
    # how much risk exposure was handled with a lighter cue
    # rather than explicit/strong correction?
    # -----------------------------------------------------

    if (
        task_risk_or_violation_time
        >
        1e-9
    ):

        minimal_share_of_risky_time = (

            adaptive_minimal_time

            /

            task_risk_or_violation_time
        )

    else:

        minimal_share_of_risky_time = (
            float(
                "nan"
            )
        )


    return {

        "trial_id":
            rows[0].get(
                "trial_id",
                csv_path.stem,
            ),

        "scenario_id":
            scenario_id,

        "duration_sec":
            round(
                total_time,
                4,
            ),

        # -------------------------------------------------
        # Task-state exposure
        # -------------------------------------------------

        "task_valid_time_sec":
            round(
                task_valid_time,
                4,
            ),

        "task_risk_time_sec":
            round(
                task_risk_time,
                4,
            ),

        "task_violation_time_sec":
            round(
                task_violation_time,
                4,
            ),

        "task_risk_or_violation_time_sec":
            round(
                task_risk_or_violation_time,
                4,
            ),

        # -------------------------------------------------
        # Reference-only
        # -------------------------------------------------

        "reference_intervention_time_sec":
            round(
                reference_intervention_time,
                4,
            ),

        "reference_unnecessary_valid_time_sec":
            round(
                reference_unnecessary_valid_time,
                4,
            ),

        "reference_missed_task_risk_time_sec":
            round(
                reference_missed_task_risk_time,
                4,
            ),

        # -------------------------------------------------
        # Task-aware binary
        # -------------------------------------------------

        "binary_intervention_time_sec":
            round(
                binary_intervention_time,
                4,
            ),

        "binary_full_guidance_time_sec":
            round(
                binary_full_guidance_time,
                4,
            ),

        # -------------------------------------------------
        # AdaptiveSkill USER-VISIBLE exposure
        # -------------------------------------------------

        "adaptive_minimal_cue_time_sec":
            round(
                adaptive_minimal_time,
                4,
            ),

        "adaptive_explicit_guidance_time_sec":
            round(
                adaptive_explicit_time,
                4,
            ),

        "adaptive_strong_guidance_time_sec":
            round(
                adaptive_strong_time,
                4,
            ),

        "adaptive_corrective_guidance_time_sec":
            round(
                adaptive_corrective_time,
                4,
            ),

        "adaptive_recovery_fade_time_sec":
            round(
                adaptive_recovery_fade_time,
                4,
            ),

        "adaptive_any_visible_assistance_time_sec":
            round(
                adaptive_any_visible_assistance_time,
                4,
            ),

        "adaptive_no_guidance_time_sec":
            round(
                adaptive_no_guidance_time,
                4,
            ),

        # -------------------------------------------------
        # Comparisons
        # -------------------------------------------------

        "adaptive_corrective_reduction_vs_binary_sec":
            round(
                corrective_reduction_vs_binary,
                4,
            ),

        "adaptive_corrective_reduction_vs_binary_ratio":
            (
                round(
                    corrective_reduction_ratio,
                    4,
                )

                if not math.isnan(
                    corrective_reduction_ratio
                )

                else ""
            ),

        "adaptive_minimal_share_of_risky_time":
            (
                round(
                    minimal_share_of_risky_time,
                    4,
                )

                if not math.isnan(
                    minimal_share_of_risky_time
                )

                else ""
            ),
    }


# =========================================================
# SAVE
# =========================================================

def save_results(
    results,
    summary,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=list(
                results[0].keys()
            ),
        )


        writer.writeheader()

        writer.writerows(
            results
        )


    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(

            summary,

            file,

            indent=4,

            ensure_ascii=False,
        )


# =========================================================
# MAIN
# =========================================================

def main():

    print()

    print(
        "========================================================"
    )

    print(
        "AdaptiveSkill - Three-Policy Offline Comparison v2"
    )

    print(
        "========================================================"
    )

    print()

    print(
        "Metric semantics:"
    )

    print(
        "Adaptive user-visible guidance is classified from "
        "presentation_mode, not internal assistance_level."
    )

    print(
        "RECOVERY_FADE is reported separately and is NOT "
        "counted as explicit/strong corrective guidance."
    )


    pass_trials = (
        load_pass_trials()
    )


    if not pass_trials:

        print(
            "No protocol-valid PASS trials were found."
        )

        return


    results = []


    for (
        trial_id,
        metadata,
    ) in pass_trials.items():

        csv_path = (
            RUNTIME_DIR
            /
            f"{trial_id}.csv"
        )


        if not csv_path.exists():

            print(
                "Skipping missing runtime CSV:",
                csv_path
            )

            continue


        result = (
            analyze_trial(

                csv_path=
                    csv_path,

                scenario_id=
                    metadata[
                        "scenario_id"
                    ],
            )
        )


        if result is not None:

            results.append(
                result
            )


    if not results:

        print(
            "No comparable runtime traces found."
        )

        return


    results.sort(
        key=lambda row:
            (
                row[
                    "scenario_id"
                ],
                row[
                    "trial_id"
                ],
            )
    )


    # =====================================================
    # PRINT PER TRIAL
    # =====================================================

    for row in results:

        print()

        print(
            "--------------------------------------------------------"
        )

        print(
            row[
                "scenario_id"
            ],
            "|",
            row[
                "trial_id"
            ]
        )

        print(
            "--------------------------------------------------------"
        )


        print(
            "Reference-only intervention:",
            row[
                "reference_intervention_time_sec"
            ],
            "sec"
        )


        print(
            "  unnecessary while task-valid:",
            row[
                "reference_unnecessary_valid_time_sec"
            ],
            "sec"
        )


        print(
            "  missed task risk:",
            row[
                "reference_missed_task_risk_time_sec"
            ],
            "sec"
        )


        print(
            "Task-aware binary full correction:",
            row[
                "binary_full_guidance_time_sec"
            ],
            "sec"
        )


        print(
            "AdaptiveSkill user-visible exposure:"
        )


        print(
            "  minimal cue:",
            row[
                "adaptive_minimal_cue_time_sec"
            ],
            "sec"
        )


        print(
            "  explicit guidance:",
            row[
                "adaptive_explicit_guidance_time_sec"
            ],
            "sec"
        )


        print(
            "  strong guidance:",
            row[
                "adaptive_strong_guidance_time_sec"
            ],
            "sec"
        )


        print(
            "  corrective total (explicit + strong):",
            row[
                "adaptive_corrective_guidance_time_sec"
            ],
            "sec"
        )


        print(
            "  recovery fade:",
            row[
                "adaptive_recovery_fade_time_sec"
            ],
            "sec"
        )


        ratio = row[
            "adaptive_corrective_reduction_vs_binary_ratio"
        ]


        if ratio != "":

            print(
                "Adaptive corrective-guidance reduction "
                "vs binary:",
                f"{float(ratio) * 100:.1f}%"
            )


    # =====================================================
    # AGGREGATE
    # =====================================================

    aggregate = defaultdict(
        float
    )


    aggregate_keys = (
        "duration_sec",
        "task_valid_time_sec",
        "task_risk_time_sec",
        "task_violation_time_sec",
        "task_risk_or_violation_time_sec",
        "reference_intervention_time_sec",
        "reference_unnecessary_valid_time_sec",
        "reference_missed_task_risk_time_sec",
        "binary_full_guidance_time_sec",
        "adaptive_minimal_cue_time_sec",
        "adaptive_explicit_guidance_time_sec",
        "adaptive_strong_guidance_time_sec",
        "adaptive_corrective_guidance_time_sec",
        "adaptive_recovery_fade_time_sec",
        "adaptive_any_visible_assistance_time_sec",
        "adaptive_no_guidance_time_sec",
    )


    for row in results:

        for key in aggregate_keys:

            aggregate[
                key
            ] += safe_float(
                row[
                    key
                ],
                0.0,
            )


    binary_full = aggregate[
        "binary_full_guidance_time_sec"
    ]


    adaptive_corrective = aggregate[
        "adaptive_corrective_guidance_time_sec"
    ]


    if binary_full > 1e-9:

        overall_reduction_ratio = (

            binary_full
            -
            adaptive_corrective

        ) / binary_full

    else:

        overall_reduction_ratio = (
            float(
                "nan"
            )
        )


    summary = {

        "analysis_type":
            "offline same-trace policy exposure comparison v2",

        "metric_semantics":
            {
                "adaptive_exposure_source":
                    "presentation_mode",

                "corrective_guidance":
                    (
                        "EXPLICIT_* plus STRONG_* "
                        "presentation modes"
                    ),

                "recovery_fade":
                    (
                        "reported separately and excluded "
                        "from corrective-guidance time"
                    ),
            },

        "claim_boundary":
            (
                "All policies are evaluated on traces recorded "
                "under the current AdaptiveSkill system. "
                "This comparison characterises policy exposure "
                "on the same observed trajectories; it does not "
                "estimate how a human would causally behave "
                "under a different policy."
            ),

        "trials_compared":
            len(
                results
            ),

        "aggregate": {

            key:
                round(
                    value,
                    4,
                )

            for key, value
            in aggregate.items()
        },

        "overall_adaptive_corrective_guidance_reduction_vs_binary_ratio":
            (
                round(
                    overall_reduction_ratio,
                    4,
                )

                if not math.isnan(
                    overall_reduction_ratio
                )

                else None
            ),
    }


    save_results(
        results,
        summary,
    )


    print()

    print(
        "========================================================"
    )

    print(
        "AGGREGATE"
    )

    print(
        "========================================================"
    )


    print(
        "Protocol-valid PASS traces compared:",
        len(
            results
        )
    )


    print(
        "Reference unnecessary-valid intervention time:",
        round(
            aggregate[
                "reference_unnecessary_valid_time_sec"
            ],
            3,
        ),
        "sec"
    )


    print(
        "Reference missed-task-risk time:",
        round(
            aggregate[
                "reference_missed_task_risk_time_sec"
            ],
            3,
        ),
        "sec"
    )


    print(
        "Task-aware binary full-correction time:",
        round(
            binary_full,
            3,
        ),
        "sec"
    )


    print(
        "Adaptive minimal-cue time:",
        round(
            aggregate[
                "adaptive_minimal_cue_time_sec"
            ],
            3,
        ),
        "sec"
    )


    print(
        "Adaptive corrective-guidance time:",
        round(
            adaptive_corrective,
            3,
        ),
        "sec"
    )


    print(
        "Adaptive recovery-fade time:",
        round(
            aggregate[
                "adaptive_recovery_fade_time_sec"
            ],
            3,
        ),
        "sec"
    )


    if not math.isnan(
        overall_reduction_ratio
    ):

        print(
            "Adaptive corrective-guidance reduction vs binary:",
            f"{overall_reduction_ratio * 100:.1f}%"
        )


    print()

    print(
        "Output CSV:"
    )

    print(
        OUTPUT_CSV
    )


    print()

    print(
        "Output JSON:"
    )

    print(
        OUTPUT_JSON
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

    print(
        "The baselines are transparent engineering "
        "comparators, not claims of literature-best "
        "performance."
    )

    print(
        "A lower corrective-guidance time is not by itself "
        "evidence of lower workload, better learning, or "
        "better user experience."
    )

    print(
        "========================================================"
    )

    print()


if __name__ == "__main__":

    main()
