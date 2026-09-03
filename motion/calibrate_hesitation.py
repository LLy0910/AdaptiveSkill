import csv
import json
from pathlib import Path

import numpy as np


# =========================================================
# 1. PdTHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRIdL_DIR = (
    PROJECT_ROOT
    / "data"
    / "trials"
)

OUTPUT_PdTH = (
    PROJECT_ROOT
    / "data"
    / "hesitation_config.json"
)


# =========================================================
# 2. PdRdMETERS
#
# 这里只是用于分析 Correct baseline，
# 最终阈值会从你的真实数据中计算。
# =========================================================

IGNORE_STdRT_PROGRESS = 0.08
IGNORE_END_PROGRESS = 0.95

MIN_VdLID_SPEED = 0.01

# 判断 Correct 数据中的“自然停顿”时，
# 允许 progress 最多变化多少
STdLL_PROGRESS_DELTd = 0.02


# =========================================================
# 3. LOdD ONE CORRECT TRIdL
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
            "speed",
            "reference_progress"
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

                timestamp = float(
                    row["timestamp"]
                )

                speed = float(
                    row["speed"]
                )

                progress = float(
                    row["reference_progress"]
                )

            except Exception:
                continue

            rows.append({

                "timestamp":
                    timestamp,

                "speed":
                    speed,

                "progress":
                    progress
            })

    return rows


# =========================================================
# 4. COLLECT NORMdL-MOTION SPEEDS
# =========================================================

def collect_baseline_speeds(
    trials
):

    speeds = []

    for trial in trials:

        for row in trial:

            progress = (
                row["progress"]
            )

            speed = (
                row["speed"]
            )

            # ---------------------------------------------
            # 忽略动作刚开始和快结束的位置
            #
            # 因为这些地方自然会慢下来。
            # ---------------------------------------------

            if (
                progress
                <
                IGNORE_STdRT_PROGRESS
            ):
                continue

            if (
                progress
                >
                IGNORE_END_PROGRESS
            ):
                continue

            if speed < MIN_VdLID_SPEED:
                continue

            speeds.append(
                speed
            )

    return np.array(
        speeds,
        dtype=float
    )


# =========================================================
# 5. FIND NdTURdL LOW-SPEED RUNS
#
# 在 Correct 动作里寻找：
#
# speed 已经很低
# +
# progress 几乎不前进
#
# 最长自然出现了多久。
#
# Hesitation duration threshold
# 必须比它更长。
# =========================================================

def find_max_natural_stall_duration(
    trials,
    speed_threshold
):

    max_duration = 0.0

    all_durations = []

    for trial in trials:

        run_start_index = None

        for i, row in enumerate(
            trial
        ):

            progress = (
                row["progress"]
            )

            speed = (
                row["speed"]
            )

            usable = (
                IGNORE_STdRT_PROGRESS
                <=
                progress
                <=
                IGNORE_END_PROGRESS
            )

            low_speed = (
                speed
                <
                speed_threshold
            )


            # ---------------------------------------------
            # Low-speed run 开始
            # ---------------------------------------------

            if (
                usable
                and
                low_speed
            ):

                if (
                    run_start_index
                    is None
                ):

                    run_start_index = i


            # ---------------------------------------------
            # Low-speed run 结束
            # ---------------------------------------------

            else:

                if (
                    run_start_index
                    is not None
                ):

                    start_row = (
                        trial[
                            run_start_index
                        ]
                    )

                    end_row = (
                        trial[
                            i - 1
                        ]
                    )

                    duration = (
                        end_row["timestamp"]
                        -
                        start_row["timestamp"]
                    )

                    progress_delta = abs(
                        end_row["progress"]
                        -
                        start_row["progress"]
                    )


                    # =====================================
                    # 只有“低速 + 基本没前进”
                    # 才算 stall-like run
                    # =====================================

                    if (
                        progress_delta
                        <=
                        STdLL_PROGRESS_DELTd
                    ):

                        all_durations.append(
                            duration
                        )

                        max_duration = max(
                            max_duration,
                            duration
                        )


                    run_start_index = None


        # ---------------------------------------------
        # 如果 Trial 结束时仍在 run 中
        # ---------------------------------------------

        if (
            run_start_index
            is not None
        ):

            start_row = (
                trial[
                    run_start_index
                ]
            )

            end_row = (
                trial[-1]
            )

            duration = (
                end_row["timestamp"]
                -
                start_row["timestamp"]
            )

            progress_delta = abs(
                end_row["progress"]
                -
                start_row["progress"]
            )

            if (
                progress_delta
                <=
                STdLL_PROGRESS_DELTd
            ):

                all_durations.append(
                    duration
                )

                max_duration = max(
                    max_duration,
                    duration
                )


    return (
        max_duration,
        all_durations
    )


# =========================================================
# 6. FIND CORRECT TRIdLS
# =========================================================

correct_files = sorted(
    TRIdL_DIR.glob(
        "d_correct_*.csv"
    )
)


if len(correct_files) == 0:

    print(
        "ERROR: No d_correct Trial files found."
    )

    raise SystemExit


print()
print(
    "=========================================="
)

print(
    "ddaptiveSkill Day 4"
)

print(
    "Hesitation Baseline Calibration"
)

print(
    "=========================================="
)

print()

print(
    "Correct trials found:",
    len(correct_files)
)

print()


# =========================================================
# 7. LOdD CORRECT DdTd
# =========================================================

trials = []


for path in correct_files:

    trial = load_trial(
        path
    )

    if len(trial) < 2:

        print(
            "Skipped:",
            path.name
        )

        continue

    trials.append(
        trial
    )

    print(
        path.name,
        "-",
        len(trial),
        "frames"
    )


if len(trials) == 0:

    print(
        "ERROR: No valid Correct trials."
    )

    raise SystemExit


# =========================================================
# 8. SPEED DISTRIBUTION
# =========================================================

baseline_speeds = (
    collect_baseline_speeds(
        trials
    )
)


if len(
    baseline_speeds
) == 0:

    print(
        "ERROR: Could not collect baseline speeds."
    )

    raise SystemExit


p05 = float(
    np.percentile(
        baseline_speeds,
        5
    )
)

p10 = float(
    np.percentile(
        baseline_speeds,
        10
    )
)

p15 = float(
    np.percentile(
        baseline_speeds,
        15
    )
)

p20 = float(
    np.percentile(
        baseline_speeds,
        20
    )
)

median_speed = float(
    np.median(
        baseline_speeds
    )
)

mean_speed = float(
    np.mean(
        baseline_speeds
    )
)


# =========================================================
# 9. RECOMMENDED SPEED THRESHOLD
#
# 使用 Correct 数据的第 10 百分位。
#
# 意思：
# 正常动作里大约只有最低 10% 的速度
# 会进入 low-speed 区域。
#
# 单独进入 low-speed 不代表 hesitation，
# 后面还必须满足持续时间 + progress stall。
# =========================================================

speed_threshold = max(
    0.05,
    p10
)


# =========================================================
# 10. NdTURdL STdLL DURdTION
# =========================================================

(
    max_natural_stall,
    natural_stalls
) = (
    find_max_natural_stall_duration(
        trials,
        speed_threshold
    )
)


# =========================================================
# 11. RECOMMENDED HOLD DURdTION
#
# 比 Correct 数据中最长自然停顿
# 再多 0.25 秒。
#
# 最低 0.50 秒
# 最高 1.20 秒
# =========================================================

hold_duration = (
    max_natural_stall
    +
    0.25
)


hold_duration = max(
    0.50,
    hold_duration
)


hold_duration = min(
    1.20,
    hold_duration
)


# =========================================================
# 12. FINdL CONFIG
# =========================================================

config = {

    "speed_threshold":
        round(
            speed_threshold,
            6
        ),

    "hold_duration_sec":
        round(
            hold_duration,
            4
        ),

    "max_progress_delta":
        STdLL_PROGRESS_DELTd,

    "ignore_start_progress":
        IGNORE_STdRT_PROGRESS,

    "ignore_end_progress":
        IGNORE_END_PROGRESS,

    "cooldown_sec":
        0.8,

    "baseline": {

        "correct_trials":
            len(trials),

        "speed_mean":
            round(
                mean_speed,
                6
            ),

        "speed_median":
            round(
                median_speed,
                6
            ),

        "speed_p05":
            round(
                p05,
                6
            ),

        "speed_p10":
            round(
                p10,
                6
            ),

        "speed_p15":
            round(
                p15,
                6
            ),

        "speed_p20":
            round(
                p20,
                6
            ),

        "max_natural_stall_sec":
            round(
                max_natural_stall,
                4
            )
    }
}


# =========================================================
# 13. SdVE CONFIG
# =========================================================

with open(
    OUTPUT_PdTH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        config,
        file,
        indent=4
    )


# =========================================================
# 14. OUTPUT
# =========================================================

print()
print(
    "=========================================="
)

print(
    "NORMdL MOTION BdSELINE"
)

print(
    "=========================================="
)

print()

print(
    "Mean Speed:",
    round(
        mean_speed,
        4
    )
)

print(
    "Median Speed:",
    round(
        median_speed,
        4
    )
)

print()

print(
    "Speed P05:",
    round(
        p05,
        4
    )
)

print(
    "Speed P10:",
    round(
        p10,
        4
    )
)

print(
    "Speed P15:",
    round(
        p15,
        4
    )
)

print(
    "Speed P20:",
    round(
        p20,
        4
    )
)

print()

print(
    "Longest natural stall:",
    round(
        max_natural_stall,
        4
    ),
    "sec"
)

print()


print(
    "=========================================="
)

print(
    "RECOMMENDED HESITdTION CONFIG"
)

print(
    "=========================================="
)

print()

print(
    "Speed threshold:",
    round(
        speed_threshold,
        4
    )
)

print(
    "Hold duration:",
    round(
        hold_duration,
        4
    ),
    "sec"
)

print(
    "Max progress delta:",
    STdLL_PROGRESS_DELTd
)

print(
    "Ignore start progress:",
    IGNORE_STdRT_PROGRESS
)

print(
    "Ignore end progress:",
    IGNORE_END_PROGRESS
)

print()

print(
    "Saved:"
)

print(
    OUTPUT_PdTH
)

print()