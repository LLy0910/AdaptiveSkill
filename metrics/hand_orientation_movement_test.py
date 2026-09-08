import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# PATHS
# =========================================================

MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
TASK_CONFIG_PATH = PROJECT_ROOT / "task" / "task_config.json"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "hand_orientation_movement"
)


# =========================================================
# PROTOCOL
# =========================================================

NEUTRAL_CALIBRATION_SEC = 2.5
TARGET_TRIALS = 3
MAX_TRIAL_SEC = 8.0

MIN_NEUTRAL_SAMPLES = 30
MIN_VALID_TRIAL_SAMPLES = 20

CAMERA_INDEX = 0


# =========================================================
# CONFIG
# =========================================================

if not MODEL_PATH.exists():
    print("ERROR: hand_landmarker.task not found:")
    print(MODEL_PATH)
    raise SystemExit

if not TASK_CONFIG_PATH.exists():
    print("ERROR: task_config.json not found:")
    print(TASK_CONFIG_PATH)
    raise SystemExit

with open(TASK_CONFIG_PATH, "r", encoding="utf-8") as file:
    TASK_CONFIG = json.load(file)

TARGET_X = float(TASK_CONFIG["target"]["x"])
TARGET_RADIUS = float(TASK_CONFIG["target"]["radius"])

MILD_TOLERANCE_DEG = float(
    TASK_CONFIG["orientation"]["mild_tolerance_deg"]
)

STRONG_TOLERANCE_DEG = float(
    TASK_CONFIG["orientation"]["strong_tolerance_deg"]
)


# =========================================================
# MEDIAPIPE
# =========================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

OPTIONS = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


# =========================================================
# HELPERS
# =========================================================

def distance_xy(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx * dx + dy * dy)


def circular_angle_difference(angle1, angle2):
    return (angle1 - angle2 + 180.0) % 360.0 - 180.0


def calculate_hand_angle(landmarks):
    wrist = landmarks[0]
    middle_mcp = landmarks[9]

    dx = middle_mcp.x - wrist.x
    dy = middle_mcp.y - wrist.y

    return math.degrees(
        math.atan2(dy, dx)
    )


def calculate_palm_size(landmarks):
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]

    return distance_xy(
        index_mcp.x,
        index_mcp.y,
        pinky_mcp.x,
        pinky_mcp.y,
    )


def robust_unwrapped_median_angle(angles_deg):
    values = np.asarray(angles_deg, dtype=float)

    unwrapped = np.rad2deg(
        np.unwrap(
            np.deg2rad(values)
        )
    )

    median_value = float(
        np.median(unwrapped)
    )

    return (median_value + 180.0) % 360.0 - 180.0


def percentile(values, q):
    if not values:
        return float("nan")

    return float(
        np.percentile(
            np.asarray(values, dtype=float),
            q,
        )
    )


def round_or_none(value, digits=4):
    try:
        value = float(value)
    except Exception:
        return None

    if math.isnan(value):
        return None

    return round(value, digits)


# =========================================================
# SESSION PATHS
# =========================================================

SESSION_ID = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

SESSION_DIR = OUTPUT_ROOT / SESSION_ID

SESSION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# SUMMARY
# =========================================================

def summarize_rows(rows):
    valid_rows = [
        row
        for row in rows
        if int(row["hand_detected"]) == 1
        and row["relative_angle_deg"] != ""
    ]

    total_frames = len(rows)
    valid_frames = len(valid_rows)

    tracking_ratio = (
        valid_frames / total_frames
        if total_frames > 0
        else 0.0
    )

    abs_errors = [
        abs(float(row["relative_angle_deg"]))
        for row in valid_rows
    ]

    signed_errors = [
        float(row["relative_angle_deg"])
        for row in valid_rows
    ]

    x_norm_values = [
        float(row["x_norm"])
        for row in valid_rows
        if row["x_norm"] != ""
    ]

    y_norm_values = [
        float(row["y_norm"])
        for row in valid_rows
        if row["y_norm"] != ""
    ]

    speeds = [
        float(row["speed_norm_per_sec"])
        for row in valid_rows
        if row["speed_norm_per_sec"] != ""
    ]

    if abs_errors:
        mean_abs = float(np.mean(abs_errors))
        p90_abs = percentile(abs_errors, 90)
        p95_abs = percentile(abs_errors, 95)
        max_abs = float(np.max(abs_errors))

        mild_fraction = float(
            np.mean(
                np.asarray(abs_errors)
                > MILD_TOLERANCE_DEG
            )
        )

        strong_fraction = float(
            np.mean(
                np.asarray(abs_errors)
                > STRONG_TOLERANCE_DEG
            )
        )

        std_signed = float(
            np.std(
                np.asarray(signed_errors),
                ddof=0,
            )
        )
    else:
        mean_abs = float("nan")
        p90_abs = float("nan")
        p95_abs = float("nan")
        max_abs = float("nan")
        mild_fraction = float("nan")
        strong_fraction = float("nan")
        std_signed = float("nan")

    x_span = (
        float(np.max(x_norm_values) - np.min(x_norm_values))
        if x_norm_values
        else float("nan")
    )

    y_span = (
        float(np.max(y_norm_values) - np.min(y_norm_values))
        if y_norm_values
        else float("nan")
    )

    mean_speed = (
        float(np.mean(speeds))
        if speeds
        else float("nan")
    )

    p95_speed = (
        percentile(speeds, 95)
        if speeds
        else float("nan")
    )

    return {
        "total_frames": total_frames,
        "valid_frames": valid_frames,
        "tracking_ratio": round_or_none(
            tracking_ratio, 4
        ),
        "mean_abs_error_deg": round_or_none(
            mean_abs, 3
        ),
        "std_signed_error_deg": round_or_none(
            std_signed, 3
        ),
        "p90_abs_error_deg": round_or_none(
            p90_abs, 3
        ),
        "p95_abs_error_deg": round_or_none(
            p95_abs, 3
        ),
        "max_abs_error_deg": round_or_none(
            max_abs, 3
        ),
        "fraction_above_mild_tolerance": round_or_none(
            mild_fraction, 4
        ),
        "fraction_above_strong_tolerance": round_or_none(
            strong_fraction, 4
        ),
        "x_span_palm_widths": round_or_none(
            x_span, 3
        ),
        "y_span_palm_widths": round_or_none(
            y_span, 3
        ),
        "mean_speed_palm_widths_per_sec": round_or_none(
            mean_speed, 3
        ),
        "p95_speed_palm_widths_per_sec": round_or_none(
            p95_speed, 3
        ),
        "mild_tolerance_deg": MILD_TOLERANCE_DEG,
        "strong_tolerance_deg": STRONG_TOLERANCE_DEG,
    }


# =========================================================
# FILE OUTPUT
# =========================================================

def save_trial_csv(trial_index, rows):
    path = SESSION_DIR / f"trial_{trial_index:02d}.csv"

    fieldnames = [
        "trial_index",
        "timestamp_sec",
        "hand_detected",
        "x_norm",
        "y_norm",
        "raw_angle_deg",
        "relative_angle_deg",
        "abs_error_deg",
        "speed_norm_per_sec",
        "target_reached",
    ]

    with open(
        path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    return path


def save_session_summary(
    calibration,
    trial_summaries,
    pooled_summary,
):
    json_path = (
        SESSION_DIR
        / "session_summary.json"
    )

    csv_path = (
        SESSION_DIR
        / "trial_summary.csv"
    )

    payload = {
        "session_id": SESSION_ID,
        "protocol": {
            "neutral_calibration_sec":
                NEUTRAL_CALIBRATION_SEC,
            "target_trials":
                TARGET_TRIALS,
            "max_trial_sec":
                MAX_TRIAL_SEC,
            "movement_instruction":
                (
                    "Start from calibrated neutral pose "
                    "and move the hand naturally to the "
                    "right until x displacement reaches "
                    "approximately the current virtual "
                    "task target distance."
                ),
            "orientation_definition":
                (
                    "2D wrist-to-middle-MCP angle "
                    "relative to calibrated neutral pose"
                ),
        },
        "calibration": calibration,
        "task_thresholds": {
            "mild_tolerance_deg":
                MILD_TOLERANCE_DEG,
            "strong_tolerance_deg":
                STRONG_TOLERANCE_DEG,
        },
        "trials": trial_summaries,
        "pooled": pooled_summary,
        "claim_boundary": (
            "This is a technical dynamic hand-orientation "
            "probe under the current MediaPipe sensing "
            "pipeline. It is not a user study, not a full "
            "robot-teaching task, and does not establish "
            "optimal orientation thresholds."
        ),
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=4,
            ensure_ascii=False,
        )

    if trial_summaries:
        with open(
            csv_path,
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    trial_summaries[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                trial_summaries
            )

    return json_path, csv_path


# =========================================================
# UI
# =========================================================

def put_text(
    frame,
    text,
    x,
    y,
    scale=0.62,
    color=(255, 255, 255),
    thickness=2,
):
    cv2.putText(
        frame,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        color,
        int(thickness),
        cv2.LINE_AA,
    )


def draw_progress(
    frame,
    x_norm,
):
    height, width, _ = frame.shape

    left = 50
    right = width - 50
    y = height - 55

    cv2.line(
        frame,
        (left, y),
        (right, y),
        (180, 180, 180),
        3,
        cv2.LINE_AA,
    )

    progress = 0.0

    if x_norm is not None:
        progress = max(
            0.0,
            min(
                1.0,
                x_norm / TARGET_X,
            ),
        )

    marker_x = int(
        left
        +
        progress
        *
        (right - left)
    )

    cv2.circle(
        frame,
        (marker_x, y),
        8,
        (0, 220, 255),
        -1,
        cv2.LINE_AA,
    )

    put_text(
        frame,
        "START",
        left,
        y - 18,
        scale=0.48,
    )

    put_text(
        frame,
        "TARGET",
        right - 70,
        y - 18,
        scale=0.48,
    )


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW,
)

if not cap.isOpened():
    print("ERROR: Camera cannot be opened.")
    raise SystemExit


# =========================================================
# STATE
# =========================================================

stage = "IDLE"

calibrating = False
calibration_start_time = None
calibration_samples = []

calibration = None

trial_start_time = None
trial_rows = []

completed_trials = 0
trial_summaries = []
all_trial_rows = []

previous_x_norm = None
previous_y_norm = None
previous_time = None


# =========================================================
# CONSOLE INTRO
# =========================================================

print()
print("========================================================")
print("AdaptiveSkill - Dynamic Hand Orientation Test")
print("========================================================")
print()
print("Purpose:")
print(
    "Measure orientation variability while the hand "
    "actually moves through a task-scale reach."
)
print()
print("Protocol:")
print(
    "1. Put ONE hand in a comfortable upright pose "
    "near the left/centre of the camera view."
)
print(
    f"2. Press C and hold for "
    f"{NEUTRAL_CALIBRATION_SEC:.1f} sec."
)
print(
    "3. Return to the same start pose before each trial."
)
print(
    "4. Press SPACE and move naturally to the RIGHT "
    "until the progress marker reaches TARGET."
)
print(
    f"5. Complete {TARGET_TRIALS} trials."
)
print()
print(
    "Do not twist the wrist to chase the threshold. "
    "Keep the hand as naturally upright as you can."
)
print()
print(
    "Live display shows movement progress only, "
    "NOT live orientation error."
)
print()
print(
    "Current thresholds:",
    f"mild={MILD_TOLERANCE_DEG} deg,",
    f"strong={STRONG_TOLERANCE_DEG} deg"
)
print()
print(
    "Controls: C = calibrate | SPACE = trial | "
    "R = reset | Q = quit"
)
print("========================================================")
print()


# =========================================================
# MAIN LOOP
# =========================================================

with HandLandmarker.create_from_options(
    OPTIONS
) as landmarker:

    video_start_time = (
        time.perf_counter()
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Cannot read camera frame.")
            break

        frame = cv2.flip(
            frame,
            1,
        )

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=
                mp.ImageFormat.SRGB,
            data=
                rgb_frame,
        )

        timestamp_ms = int(
            (
                time.perf_counter()
                -
                video_start_time
            )
            *
            1000
        )

        result = (
            landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )
        )

        hand_detected = bool(
            result.hand_landmarks
        )

        raw_angle_deg = None
        x_raw = None
        y_raw = None
        palm_size = None

        x_norm = None
        y_norm = None
        relative_angle_deg = None

        now = time.perf_counter()

        if hand_detected:
            landmarks = result.hand_landmarks[0]

            index_tip = landmarks[8]

            x_raw = float(
                index_tip.x
            )

            y_raw = float(
                index_tip.y
            )

            raw_angle_deg = (
                calculate_hand_angle(
                    landmarks
                )
            )

            palm_size = (
                calculate_palm_size(
                    landmarks
                )
            )

            if calibration is not None:
                scale = calibration["palm_size"]

                if scale > 1e-6:
                    x_norm = (
                        x_raw
                        -
                        calibration["start_x"]
                    ) / scale

                    y_norm = (
                        y_raw
                        -
                        calibration["start_y"]
                    ) / scale

                relative_angle_deg = (
                    circular_angle_difference(
                        raw_angle_deg,
                        calibration["start_angle"],
                    )
                )

        # =================================================
        # CALIBRATION
        # =================================================

        if stage == "CALIBRATING":

            if hand_detected:
                calibration_samples.append(
                    {
                        "x":
                            x_raw,
                        "y":
                            y_raw,
                        "palm_size":
                            palm_size,
                        "angle":
                            raw_angle_deg,
                    }
                )

            elapsed = (
                now
                -
                calibration_start_time
            )

            if elapsed >= NEUTRAL_CALIBRATION_SEC:

                if (
                    len(calibration_samples)
                    <
                    MIN_NEUTRAL_SAMPLES
                ):
                    print()
                    print(
                        "Calibration FAILED: not enough "
                        "valid hand samples."
                    )
                    print(
                        "Press C and try again."
                    )
                    print()

                    calibration_samples = []
                    stage = "IDLE"

                else:
                    calibration = {
                        "start_x":
                            float(
                                np.median(
                                    [
                                        item["x"]
                                        for item
                                        in calibration_samples
                                    ]
                                )
                            ),
                        "start_y":
                            float(
                                np.median(
                                    [
                                        item["y"]
                                        for item
                                        in calibration_samples
                                    ]
                                )
                            ),
                        "palm_size":
                            float(
                                np.median(
                                    [
                                        item["palm_size"]
                                        for item
                                        in calibration_samples
                                    ]
                                )
                            ),
                        "start_angle":
                            robust_unwrapped_median_angle(
                                [
                                    item["angle"]
                                    for item
                                    in calibration_samples
                                ]
                            ),
                        "valid_samples":
                            len(
                                calibration_samples
                            ),
                    }

                    stage = "IDLE"

                    print()
                    print(
                        "Calibration complete."
                    )
                    print(
                        "Valid samples:",
                        calibration["valid_samples"]
                    )
                    print(
                        "Palm scale:",
                        round(
                            calibration["palm_size"],
                            5,
                        )
                    )
                    print(
                        "Neutral angle:",
                        round(
                            calibration["start_angle"],
                            3,
                        ),
                        "deg"
                    )
                    print()
                    print(
                        "Return to the start pose and "
                        "press SPACE for trial 1."
                    )
                    print()

        # =================================================
        # RECORDING
        # =================================================

        if stage == "RECORDING":

            elapsed = (
                now
                -
                trial_start_time
            )

            speed = ""

            if (
                hand_detected
                and
                x_norm is not None
                and
                y_norm is not None
            ):
                if (
                    previous_x_norm is not None
                    and
                    previous_y_norm is not None
                    and
                    previous_time is not None
                ):
                    dt = max(
                        1e-6,
                        now - previous_time,
                    )

                    speed_value = (
                        math.sqrt(
                            (
                                x_norm
                                -
                                previous_x_norm
                            )
                            ** 2
                            +
                            (
                                y_norm
                                -
                                previous_y_norm
                            )
                            ** 2
                        )
                        /
                        dt
                    )

                    speed = round(
                        speed_value,
                        6,
                    )

                previous_x_norm = x_norm
                previous_y_norm = y_norm
                previous_time = now

            target_reached = (
                hand_detected
                and
                x_norm is not None
                and
                x_norm
                >=
                (
                    TARGET_X
                    -
                    TARGET_RADIUS
                )
            )

            row = {
                "trial_index":
                    completed_trials + 1,
                "timestamp_sec":
                    round(
                        elapsed,
                        6,
                    ),
                "hand_detected":
                    1
                    if hand_detected
                    else 0,
                "x_norm":
                    (
                        round(x_norm, 6)
                        if x_norm is not None
                        else ""
                    ),
                "y_norm":
                    (
                        round(y_norm, 6)
                        if y_norm is not None
                        else ""
                    ),
                "raw_angle_deg":
                    (
                        round(
                            raw_angle_deg,
                            6,
                        )
                        if raw_angle_deg is not None
                        else ""
                    ),
                "relative_angle_deg":
                    (
                        round(
                            relative_angle_deg,
                            6,
                        )
                        if relative_angle_deg is not None
                        else ""
                    ),
                "abs_error_deg":
                    (
                        round(
                            abs(
                                relative_angle_deg
                            ),
                            6,
                        )
                        if relative_angle_deg is not None
                        else ""
                    ),
                "speed_norm_per_sec":
                    speed,
                "target_reached":
                    1
                    if target_reached
                    else 0,
            }

            trial_rows.append(row)

            should_finish = (
                target_reached
                or
                elapsed >= MAX_TRIAL_SEC
            )

            if should_finish:

                trial_number = (
                    completed_trials + 1
                )

                valid_sample_count = sum(
                    1
                    for item in trial_rows
                    if int(
                        item["hand_detected"]
                    )
                    ==
                    1
                    and
                    item["relative_angle_deg"]
                    !=
                    ""
                )

                csv_path = save_trial_csv(
                    trial_number,
                    trial_rows,
                )

                summary = summarize_rows(
                    trial_rows
                )

                summary["trial_index"] = (
                    trial_number
                )

                summary["target_reached"] = (
                    any(
                        int(
                            item["target_reached"]
                        )
                        ==
                        1
                        for item
                        in trial_rows
                    )
                )

                summary["csv_path"] = str(
                    csv_path
                )

                summary["valid_for_dynamic_probe"] = (
                    valid_sample_count
                    >=
                    MIN_VALID_TRIAL_SAMPLES
                    and
                    bool(
                        summary["target_reached"]
                    )
                )

                trial_summaries.append(
                    summary
                )

                all_trial_rows.extend(
                    trial_rows
                )

                completed_trials += 1

                print()
                print(
                    "--------------------------------------------------------"
                )
                print(
                    f"Trial {trial_number} complete"
                )
                print(
                    "--------------------------------------------------------"
                )
                print(
                    "Target reached:",
                    summary["target_reached"]
                )
                print(
                    "Tracking ratio:",
                    (
                        f"{summary['tracking_ratio'] * 100:.1f}%"
                        if summary["tracking_ratio"] is not None
                        else "N/A"
                    )
                )
                print(
                    "X span:",
                    summary["x_span_palm_widths"],
                    "palm widths"
                )
                print(
                    "Mean abs error:",
                    summary["mean_abs_error_deg"],
                    "deg"
                )
                print(
                    "P90 abs error:",
                    summary["p90_abs_error_deg"],
                    "deg"
                )
                print(
                    "P95 abs error:",
                    summary["p95_abs_error_deg"],
                    "deg"
                )
                print(
                    "Max abs error:",
                    summary["max_abs_error_deg"],
                    "deg"
                )
                print(
                    "Frames > mild threshold:",
                    (
                        f"{summary['fraction_above_mild_tolerance'] * 100:.1f}%"
                        if summary["fraction_above_mild_tolerance"] is not None
                        else "N/A"
                    )
                )
                print(
                    "Frames > strong threshold:",
                    (
                        f"{summary['fraction_above_strong_tolerance'] * 100:.1f}%"
                        if summary["fraction_above_strong_tolerance"] is not None
                        else "N/A"
                    )
                )
                print(
                    "Valid dynamic probe:",
                    summary["valid_for_dynamic_probe"]
                )
                print(
                    "--------------------------------------------------------"
                )
                print()

                trial_rows = []
                trial_start_time = None

                previous_x_norm = None
                previous_y_norm = None
                previous_time = None

                if completed_trials >= TARGET_TRIALS:
                    stage = "COMPLETE"

                    pooled_summary = summarize_rows(
                        all_trial_rows
                    )

                    (
                        json_path,
                        csv_summary_path,
                    ) = save_session_summary(
                        calibration,
                        trial_summaries,
                        pooled_summary,
                    )

                    print()
                    print(
                        "========================================================"
                    )
                    print(
                        "SESSION COMPLETE"
                    )
                    print(
                        "========================================================"
                    )
                    print()
                    print(
                        "POOLED DYNAMIC RESULTS"
                    )
                    print(
                        "Tracking ratio:",
                        (
                            f"{pooled_summary['tracking_ratio'] * 100:.1f}%"
                            if pooled_summary["tracking_ratio"] is not None
                            else "N/A"
                        )
                    )
                    print(
                        "Mean abs error:",
                        pooled_summary["mean_abs_error_deg"],
                        "deg"
                    )
                    print(
                        "P90 abs error:",
                        pooled_summary["p90_abs_error_deg"],
                        "deg"
                    )
                    print(
                        "P95 abs error:",
                        pooled_summary["p95_abs_error_deg"],
                        "deg"
                    )
                    print(
                        "Max abs error:",
                        pooled_summary["max_abs_error_deg"],
                        "deg"
                    )
                    print(
                        "Frames > mild threshold:",
                        (
                            f"{pooled_summary['fraction_above_mild_tolerance'] * 100:.1f}%"
                            if pooled_summary["fraction_above_mild_tolerance"] is not None
                            else "N/A"
                        )
                    )
                    print(
                        "Frames > strong threshold:",
                        (
                            f"{pooled_summary['fraction_above_strong_tolerance'] * 100:.1f}%"
                            if pooled_summary["fraction_above_strong_tolerance"] is not None
                            else "N/A"
                        )
                    )
                    print()
                    print(
                        "Summary JSON:"
                    )
                    print(json_path)
                    print()
                    print(
                        "Trial summary CSV:"
                    )
                    print(csv_summary_path)
                    print()
                    print(
                        "NOTE:"
                    )
                    print(
                        "This is a dynamic technical probe. "
                        "Do not treat these values as optimal "
                        "orientation thresholds."
                    )
                    print(
                        "========================================================"
                    )
                    print()

                else:
                    stage = "IDLE"

                    print(
                        "Return to the calibrated start pose."
                    )
                    print(
                        f"Press SPACE for trial "
                        f"{completed_trials + 1}."
                    )
                    print()

        # =================================================
        # UI
        # =================================================

        height, width, _ = frame.shape

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (width, 150),
            (25, 25, 25),
            -1,
        )

        cv2.addWeighted(
            overlay,
            0.78,
            frame,
            0.22,
            0.0,
            frame,
        )

        put_text(
            frame,
            "AdaptiveSkill - Dynamic Hand Orientation Test",
            20,
            34,
            scale=0.70,
        )

        hand_text = (
            "Hand: DETECTED"
            if hand_detected
            else
            "Hand: NOT DETECTED"
        )

        put_text(
            frame,
            hand_text,
            20,
            68,
            scale=0.56,
            color=(
                (100, 230, 100)
                if hand_detected
                else
                (80, 80, 240)
            ),
        )

        if stage == "CALIBRATING":
            status_text = (
                "CALIBRATING - hold naturally upright"
            )
        elif stage == "RECORDING":
            status_text = (
                f"TRIAL {completed_trials + 1}/{TARGET_TRIALS} "
                "- move RIGHT to TARGET"
            )
        elif stage == "COMPLETE":
            status_text = (
                "COMPLETE - press Q to close"
            )
        else:
            status_text = (
                "C = calibrate | SPACE = trial | "
                "R = reset | Q = quit"
            )

        put_text(
            frame,
            status_text,
            20,
            105,
            scale=0.54,
            color=(240, 220, 120),
        )

        if stage == "RECORDING":
            put_text(
                frame,
                "Keep hand naturally upright. Do not chase angle numbers.",
                20,
                136,
                scale=0.46,
            )

        if hand_detected:
            landmarks = result.hand_landmarks[0]

            wrist = landmarks[0]
            middle_mcp = landmarks[9]

            wrist_px = (
                int(wrist.x * width),
                int(wrist.y * height),
            )

            middle_px = (
                int(middle_mcp.x * width),
                int(middle_mcp.y * height),
            )

            cv2.line(
                frame,
                wrist_px,
                middle_px,
                (0, 220, 255),
                4,
                cv2.LINE_AA,
            )

            cv2.circle(
                frame,
                wrist_px,
                6,
                (255, 255, 255),
                -1,
            )

            cv2.circle(
                frame,
                middle_px,
                6,
                (0, 220, 255),
                -1,
            )

        if stage == "RECORDING":
            draw_progress(
                frame,
                x_norm,
            )

        cv2.imshow(
            "AdaptiveSkill - Dynamic Hand Orientation Test",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        # =================================================
        # CONTROLS
        # =================================================

        if key in (ord("q"), ord("Q")):
            break

        if key in (ord("r"), ord("R")):
            print()
            print(
                "Session reset."
            )
            print(
                "Press C to recalibrate."
            )
            print()

            stage = "IDLE"
            calibration = None
            calibration_start_time = None
            calibration_samples = []

            trial_start_time = None
            trial_rows = []

            completed_trials = 0
            trial_summaries = []
            all_trial_rows = []

            previous_x_norm = None
            previous_y_norm = None
            previous_time = None

            continue

        if key in (ord("c"), ord("C")):
            if stage == "IDLE":
                calibration_samples = []
                calibration_start_time = (
                    time.perf_counter()
                )
                calibration = None
                stage = "CALIBRATING"

                print()
                print(
                    "Calibration started."
                )
                print(
                    "Hold the hand naturally upright "
                    "in the start position."
                )
                print()

            continue

        if key == 32:
            if calibration is None:
                print()
                print(
                    "Calibration is not ready. Press C first."
                )
                print()
                continue

            if stage != "IDLE":
                continue

            if completed_trials >= TARGET_TRIALS:
                continue

            trial_rows = []
            trial_start_time = time.perf_counter()

            previous_x_norm = None
            previous_y_norm = None
            previous_time = None

            stage = "RECORDING"

            print()
            print(
                f"Trial {completed_trials + 1} started."
            )
            print(
                "Move the hand naturally to the RIGHT "
                "until TARGET."
            )
            print(
                "Keep it naturally upright; do not chase "
                "orientation numbers."
            )
            print()


# =========================================================
# CLEANUP
# =========================================================

cap.release()
cv2.destroyAllWindows()
