import csv
from pathlib import Path

import numpy as np


# =========================================================
# 1. PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "reference.csv"
)

TRIAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "trials"
)

OUTPUT_PATH = (
    TRIAL_DIR
    / "angle_analysis.csv"
)


# =========================================================
# 2. CHECK FILES
# =========================================================

if not REFERENCE_PATH.exists():

    print(
        "ERROR: reference.csv not found."
    )

    raise SystemExit


if not TRIAL_DIR.exists():

    print(
        "ERROR: data/trials folder not found."
    )

    raise SystemExit


# =========================================================
# 3. LOAD REFERENCE
# =========================================================

def load_reference():

    rows = []

    with open(
        REFERENCE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        required = [
            "timestamp",
            "angle_raw"
        ]

        if reader.fieldnames is None:

            print(
                "ERROR: reference.csv is empty."
            )

            raise SystemExit


        missing = [
            column
            for column in required
            if column not in reader.fieldnames
        ]


        if missing:

            print(
                "ERROR: reference.csv missing:",
                missing
            )

            raise SystemExit


        for row in reader:

            rows.append({

                "timestamp":
                    float(
                        row["timestamp"]
                    ),

                "angle_raw":
                    float(
                        row["angle_raw"]
                    )
            })


    if len(rows) < 2:

        print(
            "ERROR: reference too short."
        )

        raise SystemExit


    return rows


# =========================================================
# 4. LOAD ONE TRIAL
# =========================================================

def load_trial(
    path
):

    rows = []


    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)


        if reader.fieldnames is None:

            return []


        required = [
            "timestamp",
            "tracking",
            "angle_raw",
            "path_error"
        ]


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

                    "timestamp":
                        float(
                            row["timestamp"]
                        ),

                    "angle_raw":
                        float(
                            row["angle_raw"]
                        ),

                    "path_error":
                        float(
                            row["path_error"]
                        )
                })

            except Exception:

                continue


    return rows


# =========================================================
# 5. CONDITION
# =========================================================

def infer_condition(
    path
):

    name = path.name


    if name.startswith(
        "A_correct_"
    ):

        return (
            "A - CORRECT"
        )


    if name.startswith(
        "B_path_wrong_"
    ):

        return (
            "B - PATH WRONG"
        )


    if name.startswith(
        "C_angle_wrong_"
    ):

        return (
            "C - ANGLE WRONG"
        )


    return (
        "UNKNOWN"
    )


# =========================================================
# 6. NORMALIZED TIME PROGRESS
# =========================================================

def calculate_progress(
    timestamps
):
    """
    把一次完整动作的时间转换成 0~1。

    START = 0
    END   = 1

    这样两个人即使一个做 4 秒，
    一个做 6 秒，也可以按动作进度比较。
    """

    timestamps = np.asarray(
        timestamps,
        dtype=float
    )


    start = timestamps[0]

    end = timestamps[-1]


    duration = (
        end
        -
        start
    )


    if duration <= 1e-9:

        return np.linspace(
            0.0,
            1.0,
            len(timestamps)
        )


    return (
        timestamps
        -
        start
    ) / duration


# =========================================================
# 7. UNWRAP + RE-ZERO ANGLE
# =========================================================

def calculate_relative_rotation(
    raw_angles
):
    """
    不再直接比较绝对 raw angle。

    而是问：

    从这一次动作真正开始算起，
    到当前一共转了多少度？

    例如：

    START = -105°
    END   = -65°

    Relative rotation:
    0° → 40°
    """

    raw_angles = np.asarray(
        raw_angles,
        dtype=float
    )


    # 防止 +179 / -179 跳变
    unwrapped = np.rad2deg(
        np.unwrap(
            np.deg2rad(
                raw_angles
            )
        )
    )


    # ★关键：
    # 每个 Trial 自己第一帧重新归零
    #
    # 避免 Calibration 起始角度的细微误差
    # 污染整个 Trial。
    relative = (
        unwrapped
        -
        unwrapped[0]
    )


    return relative


# =========================================================
# 8. PREPARE REFERENCE ANGLE PROFILE
# =========================================================

def prepare_reference_profile(
    reference
):

    timestamps = np.array(
        [
            row["timestamp"]
            for row in reference
        ],
        dtype=float
    )


    raw_angles = np.array(
        [
            row["angle_raw"]
            for row in reference
        ],
        dtype=float
    )


    progress = (
        calculate_progress(
            timestamps
        )
    )


    relative_rotation = (
        calculate_relative_rotation(
            raw_angles
        )
    )


    return (
        progress,
        relative_rotation
    )


# =========================================================
# 9. ANALYZE ONE TRIAL
# =========================================================

def analyze_trial(
    path,
    reference_progress,
    reference_rotation
):

    trial = load_trial(
        path
    )


    if len(trial) < 2:

        return None


    timestamps = np.array(
        [
            row["timestamp"]
            for row in trial
        ],
        dtype=float
    )


    raw_angles = np.array(
        [
            row["angle_raw"]
            for row in trial
        ],
        dtype=float
    )


    path_errors = np.array(
        [
            row["path_error"]
            for row in trial
        ],
        dtype=float
    )


    # -----------------------------------------------------
    # Trial progress 0 → 1
    # -----------------------------------------------------

    trial_progress = (
        calculate_progress(
            timestamps
        )
    )


    # -----------------------------------------------------
    # Trial 自己真正的 relative rotation
    # -----------------------------------------------------

    learner_rotation = (
        calculate_relative_rotation(
            raw_angles
        )
    )


    # -----------------------------------------------------
    # 把 Reference angle profile
    # 插值到 learner 的 progress
    #
    # 例如 learner 现在 37%：
    # 就比较 Reference 37% 应该转多少。
    # -----------------------------------------------------

    reference_rotation_aligned = (
        np.interp(

            trial_progress,

            reference_progress,

            reference_rotation
        )
    )


    # -----------------------------------------------------
    # 新 ANGLE ERROR
    # -----------------------------------------------------

    angle_error = np.abs(

        learner_rotation

        -

        reference_rotation_aligned
    )


    # -----------------------------------------------------
    # 整段总旋转量
    # -----------------------------------------------------

    learner_total_rotation = float(
        learner_rotation[-1]
        -
        learner_rotation[0]
    )


    reference_total_rotation = float(
        reference_rotation[-1]
        -
        reference_rotation[0]
    )


    total_rotation_error = abs(

        learner_total_rotation

        -

        reference_total_rotation
    )


    # -----------------------------------------------------
    # Duration
    # -----------------------------------------------------

    duration = float(
        timestamps[-1]
        -
        timestamps[0]
    )


    condition = (
        infer_condition(
            path
        )
    )


    return {

        "trial_id":
            path.stem,

        "condition":
            condition,

        "duration_sec":
            round(
                duration,
                4
            ),

        "mean_path_error":
            round(
                float(
                    np.mean(
                        path_errors
                    )
                ),
                6
            ),

        "median_path_error":
            round(
                float(
                    np.median(
                        path_errors
                    )
                ),
                6
            ),

        "p90_path_error":
            round(
                float(
                    np.percentile(
                        path_errors,
                        90
                    )
                ),
                6
            ),

        # ★新版 angle metrics
        "progress_angle_mean":
            round(
                float(
                    np.mean(
                        angle_error
                    )
                ),
                4
            ),

        "progress_angle_median":
            round(
                float(
                    np.median(
                        angle_error
                    )
                ),
                4
            ),

        "progress_angle_p90":
            round(
                float(
                    np.percentile(
                        angle_error,
                        90
                    )
                ),
                4
            ),

        "reference_total_rotation_deg":
            round(
                reference_total_rotation,
                4
            ),

        "learner_total_rotation_deg":
            round(
                learner_total_rotation,
                4
            ),

        "total_rotation_error_deg":
            round(
                total_rotation_error,
                4
            )
    }


# =========================================================
# 10. GROUP STATISTICS
# =========================================================

def print_group_statistics(
    results
):

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
        "===================================================="
    )

    print(
        "GROUP ANALYSIS"
    )

    print(
        "===================================================="
    )


    for condition, rows in groups.items():


        print()
        print(
            condition
        )

        print(
            "-" * 52
        )


        if len(rows) == 0:

            print(
                "No trials."
            )

            continue


        mean_paths = np.array(
            [
                row[
                    "mean_path_error"
                ]
                for row in rows
            ],
            dtype=float
        )


        angle_means = np.array(
            [
                row[
                    "progress_angle_mean"
                ]
                for row in rows
            ],
            dtype=float
        )


        angle_p90s = np.array(
            [
                row[
                    "progress_angle_p90"
                ]
                for row in rows
            ],
            dtype=float
        )


        total_rotation_errors = np.array(
            [
                row[
                    "total_rotation_error_deg"
                ]
                for row in rows
            ],
            dtype=float
        )


        print(
            "Trials:",
            len(rows)
        )


        print(
            "Group Mean Path Error:",
            round(
                float(
                    np.mean(
                        mean_paths
                    )
                ),
                4
            )
        )


        print(
            "Group Mean Progress-Angle Error:",
            round(
                float(
                    np.mean(
                        angle_means
                    )
                ),
                2
            ),
            "deg"
        )


        print(
            "Group Mean P90 Angle Error:",
            round(
                float(
                    np.mean(
                        angle_p90s
                    )
                ),
                2
            ),
            "deg"
        )


        print(
            "Group Mean Total-Rotation Error:",
            round(
                float(
                    np.mean(
                        total_rotation_errors
                    )
                ),
                2
            ),
            "deg"
        )


# =========================================================
# 11. MAIN
# =========================================================

reference = (
    load_reference()
)


(
    reference_progress,
    reference_rotation
) = (
    prepare_reference_profile(
        reference
    )
)


print()
print(
    "Reference total rotation:"
)

print(
    round(
        float(
            reference_rotation[-1]
            -
            reference_rotation[0]
        ),
        2
    ),
    "deg"
)

print()


# =========================================================
# 12. FIND FORMAL TRIAL FILES
# =========================================================

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


if len(trial_files) == 0:

    print(
        "ERROR: No formal Trial CSV files found."
    )

    raise SystemExit


print(
    "Found formal trials:",
    len(trial_files)
)

print()


# =========================================================
# 13. ANALYZE
# =========================================================

results = []


for path in trial_files:

    result = (
        analyze_trial(

            path,

            reference_progress,

            reference_rotation
        )
    )


    if result is None:

        print(
            "Skipped:",
            path.name
        )

        continue


    results.append(
        result
    )


    print(
        result[
            "trial_id"
        ]
    )


    print(
        "  Condition:",
        result[
            "condition"
        ]
    )


    print(
        "  Path:",
        result[
            "mean_path_error"
        ]
    )


    print(
        "  New Angle Mean:",
        result[
            "progress_angle_mean"
        ],
        "deg"
    )


    print(
        "  New Angle P90:",
        result[
            "progress_angle_p90"
        ],
        "deg"
    )


    print(
        "  Learner Total Rotation:",
        result[
            "learner_total_rotation_deg"
        ],
        "deg"
    )


    print(
        "  Total Rotation Error:",
        result[
            "total_rotation_error_deg"
        ],
        "deg"
    )


    print()


# =========================================================
# 14. SAVE ANALYSIS
# =========================================================

if len(results) > 0:


    fieldnames = list(
        results[0].keys()
    )


    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )


    print()
    print(
        "Saved analysis:"
    )

    print(
        OUTPUT_PATH
    )


# =========================================================
# 15. GROUP RESULTS
# =========================================================

print_group_statistics(
    results
)


print()
print(
    "===================================================="
)

print(
    "Interpretation target:"
)

print()

print(
    "A Correct:"
)

print(
    "  low Path Error"
)

print(
    "  low Progress-Angle Error"
)

print()

print(
    "B Path Wrong:"
)

print(
    "  HIGH Path Error"
)

print(
    "  angle may remain near A"
)

print()

print(
    "C Angle Wrong:"
)

print(
    "  Path Error near A"
)

print(
    "  HIGH Progress-Angle Error"
)

print(
    "===================================================="
)

print()