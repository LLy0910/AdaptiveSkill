import csv
from pathlib import Path

import numpy as np

from online_motion_metrics import (
    load_reference,
    OnlineMotionMetrics
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRIAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "trials"
)


# =========================================================
# LOAD TRIAL
# =========================================================

def load_trial(path):

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        required = [
            "tracking",
            "x_norm",
            "y_norm",
            "angle_raw"
        ]

        if reader.fieldnames is None:
            return []

        missing = [
            column
            for column in required
            if column not in reader.fieldnames
        ]

        if missing:

            print(
                f"Skipping {path.name}: "
                f"missing {missing}"
            )

            return []


        for row in reader:

            try:

                tracking = int(
                    row["tracking"]
                )

            except Exception:
                continue


            if tracking != 1:
                continue


            try:

                rows.append({

                    "x_norm":
                        float(
                            row["x_norm"]
                        ),

                    "y_norm":
                        float(
                            row["y_norm"]
                        ),

                    "angle_raw":
                        float(
                            row["angle_raw"]
                        )
                })

            except Exception:
                continue


    return rows


# =========================================================
# CONDITION
# =========================================================

def infer_condition(path):

    name = path.name

    if name.startswith(
        "A_correct_"
    ):
        return "A - CORRECT"

    if name.startswith(
        "B_path_wrong_"
    ):
        return "B - PATH WRONG"

    if name.startswith(
        "C_angle_wrong_"
    ):
        return "C - ANGLE WRONG"

    return "UNKNOWN"


# =========================================================
# ANALYZE ONE TRIAL
# =========================================================

def analyze_trial(
    path,
    reference
):

    rows = load_trial(
        path
    )

    if len(rows) < 2:
        return None


    metrics = OnlineMotionMetrics(
        reference
    )


    angle_errors = []

    path_distances = []

    progress_values = []


    for row in rows:

        result = metrics.update(

            x_norm=
                row["x_norm"],

            y_norm=
                row["y_norm"],

            raw_angle=
                row["angle_raw"],

            tracking=True
        )


        angle_errors.append(
            result[
                "relative_angle_error"
            ]
        )

        path_distances.append(
            result[
                "path_distance"
            ]
        )

        progress_values.append(
            result[
                "progress"
            ]
        )


    monotonic = all(

        progress_values[i]
        >=
        progress_values[i - 1]

        for i in range(
            1,
            len(progress_values)
        )
    )


    return {

        "trial":
            path.stem,

        "condition":
            infer_condition(
                path
            ),

        "mean_path_distance":
            float(
                np.mean(
                    path_distances
                )
            ),

        "mean_online_angle_error":
            float(
                np.mean(
                    angle_errors
                )
            ),

        "p90_online_angle_error":
            float(
                np.percentile(
                    angle_errors,
                    90
                )
            ),

        "final_progress":
            float(
                progress_values[-1]
            ),

        "monotonic":
            monotonic
    }


# =========================================================
# MAIN
# =========================================================

reference = load_reference()


trial_files = []


trial_files.extend(
    TRIAL_DIR.glob(
        "A_correct_*.csv"
    )
)

trial_files.extend(
    TRIAL_DIR.glob(
        "B_path_wrong_*.csv"
    )
)

trial_files.extend(
    TRIAL_DIR.glob(
        "C_angle_wrong_*.csv"
    )
)


trial_files = sorted(
    trial_files,
    key=lambda path: path.name
)


print()
print(
    "=============================================="
)

print(
    "AdaptiveSkill - Online Metric Validation"
)

print(
    "=============================================="
)

print()

print(
    "Formal trials found:",
    len(trial_files)
)

print()


results = []


for path in trial_files:

    result = analyze_trial(
        path,
        reference
    )

    if result is None:
        continue

    results.append(
        result
    )


    print(
        result["trial"]
    )

    print(
        "  Condition:",
        result["condition"]
    )

    print(
        "  Mean Path Distance:",
        round(
            result[
                "mean_path_distance"
            ],
            4
        )
    )

    print(
        "  Mean Online Angle Error:",
        round(
            result[
                "mean_online_angle_error"
            ],
            2
        ),
        "deg"
    )

    print(
        "  P90 Online Angle Error:",
        round(
            result[
                "p90_online_angle_error"
            ],
            2
        ),
        "deg"
    )

    print(
        "  Final Progress:",
        round(
            result[
                "final_progress"
            ],
            3
        )
    )

    print(
        "  Monotonic:",
        result[
            "monotonic"
        ]
    )

    print()


# =========================================================
# GROUP ANALYSIS
# =========================================================

groups = {

    "A - CORRECT": [],

    "B - PATH WRONG": [],

    "C - ANGLE WRONG": []
}


for result in results:

    condition = (
        result["condition"]
    )

    if condition in groups:

        groups[
            condition
        ].append(
            result
        )


print()
print(
    "=============================================="
)

print(
    "GROUP ANALYSIS"
)

print(
    "=============================================="
)


for condition, rows in groups.items():

    print()
    print(condition)
    print("-" * 46)

    if len(rows) == 0:

        print(
            "No trials."
        )

        continue


    mean_path = float(
        np.mean(
            [
                row[
                    "mean_path_distance"
                ]
                for row in rows
            ]
        )
    )


    mean_angle = float(
        np.mean(
            [
                row[
                    "mean_online_angle_error"
                ]
                for row in rows
            ]
        )
    )


    mean_p90 = float(
        np.mean(
            [
                row[
                    "p90_online_angle_error"
                ]
                for row in rows
            ]
        )
    )


    mean_final_progress = float(
        np.mean(
            [
                row[
                    "final_progress"
                ]
                for row in rows
            ]
        )
    )


    print(
        "Trials:",
        len(rows)
    )

    print(
        "Mean Path Distance:",
        round(
            mean_path,
            4
        )
    )

    print(
        "Mean Online Angle Error:",
        round(
            mean_angle,
            2
        ),
        "deg"
    )

    print(
        "Mean P90 Angle Error:",
        round(
            mean_p90,
            2
        ),
        "deg"
    )

    print(
        "Mean Final Progress:",
        round(
            mean_final_progress,
            3
        )
    )


print()
print(
    "=============================================="
)

print(
    "TARGET PATTERN"
)

print(
    "=============================================="
)

print()

print(
    "A Correct:"
)

print(
    "  low path distance"
)

print(
    "  low online angle error"
)

print()

print(
    "B Path Wrong:"
)

print(
    "  higher path distance"
)

print()

print(
    "C Angle Wrong:"
)

print(
    "  path near A"
)

print(
    "  online angle error clearly higher than A"
)

print()