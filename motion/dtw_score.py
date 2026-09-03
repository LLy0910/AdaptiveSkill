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
    / "dtw_analysis.csv"
)


# =========================================================
# 2. CHECK FILES
# =========================================================

if not REFERENCE_PATH.exists():
    print("ERROR: reference.csv not found.")
    raise SystemExit

if not TRIAL_DIR.exists():
    print("ERROR: data/trials folder not found.")
    raise SystemExit


# =========================================================
# 3. BASIC FUNCTIONS
# =========================================================

def calculate_relative_rotation(raw_angles):
    """
    把绝对 2D hand orientation
    转换成“相对于本次动作起点转了多少”。

    例如：

    Raw:
    -105, -100, -90, -70

    Relative:
    0, 5, 15, 35
    """

    raw_angles = np.asarray(
        raw_angles,
        dtype=float
    )

    if len(raw_angles) == 0:
        return np.array([])

    # 防止 179 / -179 跳变
    unwrapped = np.rad2deg(
        np.unwrap(
            np.deg2rad(
                raw_angles
            )
        )
    )

    return (
        unwrapped
        -
        unwrapped[0]
    )


def infer_condition(path):
    """
    根据 Trial 文件名判断实验条件。
    """

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
# 4. LOAD REFERENCE
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
            "x_norm",
            "y_norm",
            "angle_raw"
        ]

        if reader.fieldnames is None:
            print("ERROR: reference.csv is empty.")
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

    if len(rows) < 2:

        print(
            "ERROR: reference is too short."
        )

        raise SystemExit

    return rows


# =========================================================
# 5. LOAD ONE TRIAL
# =========================================================

def load_trial(path):

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
            "x_norm",
            "y_norm",
            "angle_raw"
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

            # DTW 只使用有效追踪帧
            if tracking != 1:
                continue

            try:

                rows.append({

                    "timestamp":
                        float(
                            row["timestamp"]
                        ),

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
# 6. GENERIC DTW
# =========================================================

def dtw_distance(
    sequence_a,
    sequence_b,
    local_cost_function
):
    """
    Dynamic Time Warping。

    返回：

    total_cost
    normalized_cost
    warping_path_length

    normalized_cost =
    总 DTW cost / 对齐路径长度

    这样不同 Trial 帧数不同，
    分数也更容易比较。
    """

    n = len(sequence_a)
    m = len(sequence_b)

    if n == 0 or m == 0:

        return (
            float("inf"),
            float("inf"),
            0
        )


    # -----------------------------------------------------
    # Cost matrix
    # -----------------------------------------------------

    cost = np.full(
        (
            n + 1,
            m + 1
        ),
        np.inf,
        dtype=float
    )


    path_length = np.zeros(
        (
            n + 1,
            m + 1
        ),
        dtype=np.int32
    )


    cost[0, 0] = 0.0


    # =====================================================
    # Dynamic Programming
    # =====================================================

    for i in range(
        1,
        n + 1
    ):

        for j in range(
            1,
            m + 1
        ):

            local_cost = (
                local_cost_function(
                    sequence_a[i - 1],
                    sequence_b[j - 1]
                )
            )


            candidates = [

                # 上
                (
                    cost[
                        i - 1,
                        j
                    ],
                    path_length[
                        i - 1,
                        j
                    ]
                ),

                # 左
                (
                    cost[
                        i,
                        j - 1
                    ],
                    path_length[
                        i,
                        j - 1
                    ]
                ),

                # 对角线
                (
                    cost[
                        i - 1,
                        j - 1
                    ],
                    path_length[
                        i - 1,
                        j - 1
                    ]
                )
            ]


            best_cost, best_length = min(
                candidates,
                key=lambda item: item[0]
            )


            cost[i, j] = (
                local_cost
                +
                best_cost
            )


            path_length[i, j] = (
                best_length
                +
                1
            )


    total_cost = float(
        cost[n, m]
    )


    final_path_length = int(
        path_length[n, m]
    )


    if final_path_length <= 0:

        normalized_cost = (
            float("inf")
        )

    else:

        normalized_cost = (
            total_cost
            /
            final_path_length
        )


    return (
        total_cost,
        normalized_cost,
        final_path_length
    )


# =========================================================
# 7. XY LOCAL COST
# =========================================================

def xy_local_cost(
    point_a,
    point_b
):

    dx = (
        point_a[0]
        -
        point_b[0]
    )

    dy = (
        point_a[1]
        -
        point_b[1]
    )

    return float(
        np.sqrt(
            dx * dx
            +
            dy * dy
        )
    )


# =========================================================
# 8. ANGLE LOCAL COST
# =========================================================

def angle_local_cost(
    angle_a,
    angle_b
):

    return abs(
        float(angle_a)
        -
        float(angle_b)
    )


# =========================================================
# 9. PREPARE REFERENCE
# =========================================================

def prepare_reference(reference):

    reference_xy = np.array(
        [
            [
                row["x_norm"],
                row["y_norm"]
            ]
            for row in reference
        ],
        dtype=float
    )


    reference_raw_angles = np.array(
        [
            row["angle_raw"]
            for row in reference
        ],
        dtype=float
    )


    reference_rotation = (
        calculate_relative_rotation(
            reference_raw_angles
        )
    )


    # -----------------------------------------------------
    # Reference path length
    #
    # 用于 Combined DTW 的 dimensionless normalization
    # -----------------------------------------------------

    path_steps = np.linalg.norm(
        np.diff(
            reference_xy,
            axis=0
        ),
        axis=1
    )


    reference_path_length = float(
        np.sum(
            path_steps
        )
    )


    reference_total_rotation = abs(
        float(
            reference_rotation[-1]
            -
            reference_rotation[0]
        )
    )


    # 避免除 0
    reference_path_scale = max(
        reference_path_length,
        0.1
    )


    reference_angle_scale = max(
        reference_total_rotation,
        5.0
    )


    return (
        reference_xy,
        reference_rotation,
        reference_path_scale,
        reference_angle_scale
    )


# =========================================================
# 10. COMBINED LOCAL COST
# =========================================================

def make_combined_local_cost(
    path_scale,
    angle_scale
):
    """
    Combined DTW 只作为探索指标。

    XY 与 Angle 单位完全不同：

    XY:
    palm-normalized units

    Angle:
    degrees

    所以先分别除以 Reference 自身尺度，
    再相加。

    不把它当最终理论评分，
    主要用于观察整体趋势。
    """

    def combined_cost(
        point_a,
        point_b
    ):

        # point:
        # [x, y, relative_angle]

        xy_cost = xy_local_cost(
            point_a[:2],
            point_b[:2]
        )


        angle_cost = abs(
            point_a[2]
            -
            point_b[2]
        )


        normalized_xy = (
            xy_cost
            /
            path_scale
        )


        normalized_angle = (
            angle_cost
            /
            angle_scale
        )


        return float(
            normalized_xy
            +
            normalized_angle
        )


    return combined_cost


# =========================================================
# 11. ANALYZE ONE TRIAL
# =========================================================

def analyze_trial(
    path,
    reference_xy,
    reference_rotation,
    reference_path_scale,
    reference_angle_scale
):

    trial = (
        load_trial(
            path
        )
    )


    if len(trial) < 2:
        return None


    # -----------------------------------------------------
    # Trial XY
    # -----------------------------------------------------

    trial_xy = np.array(
        [
            [
                row["x_norm"],
                row["y_norm"]
            ]
            for row in trial
        ],
        dtype=float
    )


    # -----------------------------------------------------
    # Trial relative rotation
    # -----------------------------------------------------

    trial_raw_angles = np.array(
        [
            row["angle_raw"]
            for row in trial
        ],
        dtype=float
    )


    trial_rotation = (
        calculate_relative_rotation(
            trial_raw_angles
        )
    )


    # =====================================================
    # XY DTW
    # =====================================================

    (
        xy_total,
        xy_normalized,
        xy_path_length
    ) = (
        dtw_distance(
            trial_xy,
            reference_xy,
            xy_local_cost
        )
    )


    # =====================================================
    # ANGLE DTW
    # =====================================================

    (
        angle_total,
        angle_normalized,
        angle_path_length
    ) = (
        dtw_distance(
            trial_rotation,
            reference_rotation,
            angle_local_cost
        )
    )


    # =====================================================
    # COMBINED DTW
    # =====================================================

    trial_combined = np.column_stack(
        (
            trial_xy,
            trial_rotation
        )
    )


    reference_combined = np.column_stack(
        (
            reference_xy,
            reference_rotation
        )
    )


    combined_cost_function = (
        make_combined_local_cost(
            reference_path_scale,
            reference_angle_scale
        )
    )


    (
        combined_total,
        combined_normalized,
        combined_path_length
    ) = (
        dtw_distance(
            trial_combined,
            reference_combined,
            combined_cost_function
        )
    )


    # =====================================================
    # OTHER DATA
    # =====================================================

    duration = (
        trial[-1]["timestamp"]
        -
        trial[0]["timestamp"]
    )


    learner_total_rotation = abs(
        float(
            trial_rotation[-1]
            -
            trial_rotation[0]
        )
    )


    reference_total_rotation = abs(
        float(
            reference_rotation[-1]
            -
            reference_rotation[0]
        )
    )


    total_rotation_error = abs(
        learner_total_rotation
        -
        reference_total_rotation
    )


    return {

        "trial_id":
            path.stem,

        "condition":
            infer_condition(
                path
            ),

        "duration_sec":
            round(
                float(duration),
                4
            ),

        "trial_frames":
            len(trial),

        # XY
        "xy_dtw_total":
            round(
                xy_total,
                6
            ),

        "xy_dtw_normalized":
            round(
                xy_normalized,
                6
            ),

        "xy_warp_path_length":
            xy_path_length,

        # Angle
        "angle_dtw_total":
            round(
                angle_total,
                6
            ),

        "angle_dtw_normalized":
            round(
                angle_normalized,
                6
            ),

        "angle_warp_path_length":
            angle_path_length,

        # Combined
        "combined_dtw_total":
            round(
                combined_total,
                6
            ),

        "combined_dtw_normalized":
            round(
                combined_normalized,
                6
            ),

        "combined_warp_path_length":
            combined_path_length,

        # Rotation diagnostics
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
# 12. GROUP STATISTICS
# =========================================================

def print_group_statistics(
    results
):

    groups = {

        "A - CORRECT":
            [],

        "B - PATH WRONG":
            [],

        "C - ANGLE WRONG":
            []
    }


    for row in results:

        condition = (
            row["condition"]
        )

        if condition in groups:

            groups[
                condition
            ].append(
                row
            )


    print()
    print(
        "======================================================"
    )

    print(
        "DTW GROUP ANALYSIS"
    )

    print(
        "======================================================"
    )


    for condition, rows in groups.items():


        print()
        print(
            condition
        )

        print(
            "-" * 54
        )


        if len(rows) == 0:

            print(
                "No trials."
            )

            continue


        xy_scores = np.array(
            [
                row[
                    "xy_dtw_normalized"
                ]
                for row in rows
            ],
            dtype=float
        )


        angle_scores = np.array(
            [
                row[
                    "angle_dtw_normalized"
                ]
                for row in rows
            ],
            dtype=float
        )


        combined_scores = np.array(
            [
                row[
                    "combined_dtw_normalized"
                ]
                for row in rows
            ],
            dtype=float
        )


        rotation_errors = np.array(
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
            "Mean XY DTW:",
            round(
                float(
                    np.mean(
                        xy_scores
                    )
                ),
                4
            )
        )


        print(
            "Mean Angle DTW:",
            round(
                float(
                    np.mean(
                        angle_scores
                    )
                ),
                2
            ),
            "deg"
        )


        print(
            "Mean Combined DTW:",
            round(
                float(
                    np.mean(
                        combined_scores
                    )
                ),
                4
            )
        )


        print(
            "Mean Total Rotation Error:",
            round(
                float(
                    np.mean(
                        rotation_errors
                    )
                ),
                2
            ),
            "deg"
        )


# =========================================================
# 13. MAIN
# =========================================================

reference = (
    load_reference()
)


(
    reference_xy,
    reference_rotation,
    reference_path_scale,
    reference_angle_scale
) = (
    prepare_reference(
        reference
    )
)


print()
print(
    "=========================================="
)

print(
    "AdaptiveSkill Day 3 - DTW Analysis"
)

print(
    "=========================================="
)

print()


print(
    "Reference frames:",
    len(reference)
)


print(
    "Reference path length:",
    round(
        reference_path_scale,
        4
    )
)


print(
    "Reference total rotation:",
    round(
        reference_angle_scale,
        2
    ),
    "deg"
)

print()


# =========================================================
# 14. FIND FORMAL TRIALS
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
# 15. RUN DTW
# =========================================================

results = []


for path in trial_files:


    result = (
        analyze_trial(

            path,

            reference_xy,

            reference_rotation,

            reference_path_scale,

            reference_angle_scale
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
        result["trial_id"]
    )


    print(
        "  Condition:",
        result["condition"]
    )


    print(
        "  XY DTW:",
        result[
            "xy_dtw_normalized"
        ]
    )


    print(
        "  Angle DTW:",
        result[
            "angle_dtw_normalized"
        ],
        "deg"
    )


    print(
        "  Combined DTW:",
        result[
            "combined_dtw_normalized"
        ]
    )


    print(
        "  Rotation Error:",
        result[
            "total_rotation_error_deg"
        ],
        "deg"
    )


    print()


# =========================================================
# 16. SAVE DTW ANALYSIS
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
        "DTW analysis saved:"
    )

    print(
        OUTPUT_PATH
    )


# =========================================================
# 17. GROUP RESULTS
# =========================================================

print_group_statistics(
    results
)


print()
print(
    "======================================================"
)

print(
    "EXPECTED PATTERN"
)

print(
    "======================================================"
)

print()

print(
    "A Correct:"
)

print(
    "  LOW XY DTW"
)

print(
    "  LOW Angle DTW"
)

print()


print(
    "B Path Wrong:"
)

print(
    "  HIGH XY DTW"
)

print(
    "  Angle DTW may remain closer to A"
)

print()


print(
    "C Angle Wrong:"
)

print(
    "  XY DTW near A"
)

print(
    "  HIGH Angle DTW"
)

print()

print(
    "Combined DTW is exploratory only."
)

print(
    "Do not use it as the final adaptive threshold yet."
)

print(
    "======================================================"
)

print()