import csv
from pathlib import Path

from online_motion_metrics import (
    load_reference,
    OnlineMotionMetrics
)

from assistance_policy import (
    AssistancePolicy,
    load_assistance_config
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
            "timestamp",
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
    reference,
    config
):

    rows = load_trial(
        path
    )


    if len(rows) < 2:
        return None


    metrics = OnlineMotionMetrics(
        reference
    )


    policy = AssistancePolicy(
        config
    )


    first_timestamp = (
        rows[0]["timestamp"]
    )


    max_level = 0

    level_counts = {
        0: 0,
        1: 0,
        2: 0,
        3: 0
    }


    first_level_1 = None
    first_level_2 = None
    first_level_3 = None


    final_result = None


    for row in rows:

        timestamp = (
            row["timestamp"]
            -
            first_timestamp
        )


        motion = metrics.update(

            x_norm=
                row["x_norm"],

            y_norm=
                row["y_norm"],

            raw_angle=
                row["angle_raw"],

            tracking=True
        )


        # ---------------------------------------------
        # Old formal trials do not contain hesitation
        # events from the new detector.
        #
        # For this replay we evaluate motion errors only.
        # ---------------------------------------------

        hesitation_result = {

            "state":
                "NORMAL",

            "event":
                False,

            "hesitating":
                False
        }


        final_result = policy.update(

            timestamp=
                timestamp,

            path_error=
                motion[
                    "path_distance"
                ],

            relative_angle_error=
                motion[
                    "relative_angle_error"
                ],

            progress=
                motion[
                    "progress"
                ],

            learner_rotation=
                motion[
                    "learner_relative_rotation"
                ],

            expected_rotation=
                motion[
                    "expected_relative_rotation"
                ],

            hesitation_result=
                hesitation_result,

            tracking=True
        )


        level = int(
            final_result["level"]
        )


        level_counts[
            level
        ] += 1


        max_level = max(
            max_level,
            level
        )


        if (
            level >= 1
            and
            first_level_1 is None
        ):

            first_level_1 = (
                timestamp
            )


        if (
            level >= 2
            and
            first_level_2 is None
        ):

            first_level_2 = (
                timestamp
            )


        if (
            level >= 3
            and
            first_level_3 is None
        ):

            first_level_3 = (
                timestamp
            )


    total_frames = sum(
        level_counts.values()
    )


    percentages = {}

    for level in range(4):

        percentages[
            level
        ] = (

            level_counts[level]

            /
            max(
                total_frames,
                1
            )

            *
            100.0
        )


    return {

        "trial":
            path.stem,

        "condition":
            infer_condition(
                path
            ),

        "max_level":
            max_level,

        "level_0_pct":
            percentages[0],

        "level_1_pct":
            percentages[1],

        "level_2_pct":
            percentages[2],

        "level_3_pct":
            percentages[3],

        "first_level_1":
            first_level_1,

        "first_level_2":
            first_level_2,

        "first_level_3":
            first_level_3,

        "final_reason":
            (
                final_result["reason"]
                if final_result
                else ""
            )
    }


# =========================================================
# MAIN
# =========================================================

reference = load_reference()

config = load_assistance_config()


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
    "================================================"
)

print(
    "AdaptiveSkill - Assistance Policy Replay"
)

print(
    "================================================"
)

print()

print(
    "Formal trials:",
    len(trial_files)
)

print()


results = []


for path in trial_files:

    result = analyze_trial(

        path,
        reference,
        config
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
        "  MAX LEVEL:",
        result["max_level"]
    )


    print(
        "  Time at levels:"
    )


    print(
        "    L0:",
        round(
            result[
                "level_0_pct"
            ],
            1
        ),
        "%"
    )


    print(
        "    L1:",
        round(
            result[
                "level_1_pct"
            ],
            1
        ),
        "%"
    )


    print(
        "    L2:",
        round(
            result[
                "level_2_pct"
            ],
            1
        ),
        "%"
    )


    print(
        "    L3:",
        round(
            result[
                "level_3_pct"
            ],
            1
        ),
        "%"
    )


    print(
        "  First L1:",
        (
            round(
                result[
                    "first_level_1"
                ],
                2
            )
            if
            result[
                "first_level_1"
            ]
            is not None
            else
            "-"
        )
    )


    print(
        "  First L2:",
        (
            round(
                result[
                    "first_level_2"
                ],
                2
            )
            if
            result[
                "first_level_2"
            ]
            is not None
            else
            "-"
        )
    )


    print(
        "  First L3:",
        (
            round(
                result[
                    "first_level_3"
                ],
                2
            )
            if
            result[
                "first_level_3"
            ]
            is not None
            else
            "-"
        )
    )


    print()


# =========================================================
# GROUP SUMMARY
# =========================================================

print()
print(
    "================================================"
)

print(
    "GROUP SUMMARY"
)

print(
    "================================================"
)


conditions = [

    "A - CORRECT",

    "B - PATH WRONG",

    "C - ANGLE WRONG"
]


for condition in conditions:

    group = [

        result

        for result in results

        if result[
            "condition"
        ]
        ==
        condition
    ]


    print()
    print(condition)
    print("-" * 48)


    if len(group) == 0:

        print(
            "No trials."
        )

        continue


    max_levels = [

        result[
            "max_level"
        ]

        for result in group
    ]


    print(
        "Trials:",
        len(group)
    )


    print(
        "Max levels:",
        max_levels
    )


    print(
        "Mean max level:",
        round(
            sum(
                max_levels
            )
            /
            len(
                max_levels
            ),
            2
        )
    )


print()
print(
    "================================================"
)

print(
    "DESIRED BEHAVIOR"
)

print(
    "================================================"
)

print()

print(
    "A Correct:"
)

print(
    "  mostly Level 0"
)

print(
    "  occasional Level 1 is acceptable"
)

print(
    "  should NOT normally reach Level 2/3"
)

print()

print(
    "B Path Wrong:"
)

print(
    "  should escalate above Correct"
)

print(
    "  sustained strong error may reach Level 2/3"
)

print()

print(
    "C Angle Wrong:"
)

print(
    "  should escalate above Correct"
)

print(
    "  sustained rotation error may reach Level 2/3"
)

print()