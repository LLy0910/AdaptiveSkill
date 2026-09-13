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
# IMPORTS
# =========================================================

from task.task_constraints import (
    TaskConstraintEvaluator,
    STATE_VALID,
    STATE_RISK,
    STATE_VIOLATION,
)


# =========================================================
# MEDIAPIPE TASK ALIASES
# =========================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# =========================================================
# PATHS
# =========================================================

MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
TASK_CONFIG_PATH = PROJECT_ROOT / "task" / "task_config.json"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "hand_orientation_route"
)


# =========================================================
# PROTOCOL
# =========================================================

NEUTRAL_CALIBRATION_SEC = 2.5
MAX_TRIAL_SEC = 12.0
MIN_NEUTRAL_SAMPLES = 30
MIN_VALID_TRIAL_SAMPLES = 20

CAMERA_INDEX = 0

# 3 upper + 3 lower
TRIAL_PROTOCOL = [
    "UPPER",
    "UPPER",
    "UPPER",
    "LOWER",
    "LOWER",
    "LOWER",
]


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


TASK_EVALUATOR = TaskConstraintEvaluator(
    TASK_CONFIG_PATH
)

TARGET_X = float(TASK_CONFIG["target"]["x"])
TARGET_Y = float(TASK_CONFIG["target"]["y"])
TARGET_RADIUS = float(TASK_CONFIG["target"]["radius"])

MILD_TOLERANCE_DEG = float(
    TASK_CONFIG["orientation"]["mild_tolerance_deg"]
)
STRONG_TOLERANCE_DEG = float(
    TASK_CONFIG["orientation"]["strong_tolerance_deg"]
)

OBSTACLE = TASK_CONFIG["obstacles"][0]
OBS_X_MIN = float(OBSTACLE["x_min"])
OBS_X_MAX = float(OBSTACLE["x_max"])
OBS_Y_MIN = float(OBSTACLE["y_min"])
OBS_Y_MAX = float(OBSTACLE["y_max"])
SAFETY_MARGIN = float(TASK_CONFIG["safety_margin"])

UPPER_SAFE_Y = OBS_Y_MIN - SAFETY_MARGIN
LOWER_SAFE_Y = OBS_Y_MAX + SAFETY_MARGIN


# =========================================================
# WORLD / UI
# =========================================================

WORLD_X_MIN = -0.4
WORLD_X_MAX = 3.7
WORLD_Y_MIN = -1.55
WORLD_Y_MAX = 1.55

PANEL_TOP = 120
PANEL_BOTTOM_MARGIN = 60


def world_to_screen(x, y, width, height):
    left = 55
    right = width - 55
    top = PANEL_TOP
    bottom = height - PANEL_BOTTOM_MARGIN

    x_ratio = (
        (float(x) - WORLD_X_MIN)
        /
        (WORLD_X_MAX - WORLD_X_MIN)
    )

    y_ratio = (
        (float(y) - WORLD_Y_MIN)
        /
        (WORLD_Y_MAX - WORLD_Y_MIN)
    )

    sx = int(
        left
        +
        x_ratio
        *
        (right - left)
    )

    sy = int(
        top
        +
        y_ratio
        *
        (bottom - top)
    )

    return sx, sy


# =========================================================
# ANGLE / HAND HELPERS
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
# SESSION
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
# ROUTE / SUMMARY
# =========================================================

def route_side_valid(rows, expected_side):
    corridor_y = []

    for row in rows:
        if int(row["hand_detected"]) != 1:
            continue

        if row["x_norm"] == "" or row["y_norm"] == "":
            continue

        x = float(row["x_norm"])
        y = float(row["y_norm"])

        if OBS_X_MIN <= x <= OBS_X_MAX:
            corridor_y.append(y)

    if not corridor_y:
        return False

    if expected_side == "UPPER":
        return max(corridor_y) < UPPER_SAFE_Y

    return min(corridor_y) > LOWER_SAFE_Y


def summarize_rows(rows, expected_side):
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

    if abs_errors:
        mean_abs = float(np.mean(abs_errors))
        p90_abs = percentile(abs_errors, 90)
        p95_abs = percentile(abs_errors, 95)
        max_abs = float(np.max(abs_errors))

        mild_fraction = float(
            np.mean(
                np.asarray(abs_errors)
                >
                MILD_TOLERANCE_DEG
            )
        )

        strong_fraction = float(
            np.mean(
                np.asarray(abs_errors)
                >
                STRONG_TOLERANCE_DEG
            )
        )
    else:
        mean_abs = float("nan")
        p90_abs = float("nan")
        p95_abs = float("nan")
        max_abs = float("nan")
        mild_fraction = float("nan")
        strong_fraction = float("nan")

    goal_reached = any(
        int(row["goal_reached"]) == 1
        for row in rows
    )

    obstacle_risk_seen = any(
        int(row["obstacle_risk"]) == 1
        for row in rows
    )

    obstacle_violation_seen = any(
        int(row["obstacle_violation"]) == 1
        for row in rows
    )

    orientation_risk_seen = any(
        int(row["orientation_risk"]) == 1
        for row in rows
    )

    orientation_violation_seen = any(
        int(row["orientation_violation"]) == 1
        for row in rows
    )

    expected_side_valid = route_side_valid(
        rows,
        expected_side,
    )

    route_protocol_valid = (
        goal_reached
        and
        expected_side_valid
        and
        not obstacle_violation_seen
    )

    return {
        "expected_side": expected_side,
        "total_frames": total_frames,
        "valid_frames": valid_frames,
        "tracking_ratio": round_or_none(
            tracking_ratio, 4
        ),
        "goal_reached": goal_reached,
        "expected_side_valid": expected_side_valid,
        "route_protocol_valid": route_protocol_valid,
        "obstacle_risk_seen": obstacle_risk_seen,
        "obstacle_violation_seen": obstacle_violation_seen,
        "orientation_risk_seen": orientation_risk_seen,
        "orientation_violation_seen": orientation_violation_seen,
        "mean_abs_error_deg": round_or_none(
            mean_abs, 3
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
        "mild_tolerance_deg": MILD_TOLERANCE_DEG,
        "strong_tolerance_deg": STRONG_TOLERANCE_DEG,
    }


def pooled_orientation_summary(rows):
    valid_rows = [
        row
        for row in rows
        if int(row["hand_detected"]) == 1
        and row["relative_angle_deg"] != ""
    ]

    abs_errors = [
        abs(float(row["relative_angle_deg"]))
        for row in valid_rows
    ]

    if not abs_errors:
        return {
            "tracking_ratio": 0.0,
            "mean_abs_error_deg": None,
            "p90_abs_error_deg": None,
            "p95_abs_error_deg": None,
            "max_abs_error_deg": None,
            "fraction_above_mild_tolerance": None,
            "fraction_above_strong_tolerance": None,
        }

    total_frames = len(rows)
    tracking_ratio = (
        len(valid_rows) / total_frames
        if total_frames > 0
        else 0.0
    )

    values = np.asarray(abs_errors)

    return {
        "tracking_ratio": round(
            tracking_ratio,
            4,
        ),
        "mean_abs_error_deg": round(
            float(np.mean(values)),
            3,
        ),
        "p90_abs_error_deg": round(
            float(np.percentile(values, 90)),
            3,
        ),
        "p95_abs_error_deg": round(
            float(np.percentile(values, 95)),
            3,
        ),
        "max_abs_error_deg": round(
            float(np.max(values)),
            3,
        ),
        "fraction_above_mild_tolerance": round(
            float(
                np.mean(
                    values > MILD_TOLERANCE_DEG
                )
            ),
            4,
        ),
        "fraction_above_strong_tolerance": round(
            float(
                np.mean(
                    values > STRONG_TOLERANCE_DEG
                )
            ),
            4,
        ),
    }


# =========================================================
# FILE OUTPUT
# =========================================================

def save_trial_csv(trial_index, expected_side, rows):
    path = (
        SESSION_DIR
        /
        f"trial_{trial_index:02d}_{expected_side}.csv"
    )

    fieldnames = [
        "trial_index",
        "expected_side",
        "timestamp_sec",
        "hand_detected",
        "x_norm",
        "y_norm",
        "raw_angle_deg",
        "relative_angle_deg",
        "abs_error_deg",
        "task_state",
        "dominant_constraint",
        "goal_reached",
        "obstacle_risk",
        "obstacle_violation",
        "orientation_risk",
        "orientation_violation",
        "orientation_error_deg",
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
    all_rows,
):
    json_path = (
        SESSION_DIR
        /
        "session_summary.json"
    )

    csv_path = (
        SESSION_DIR
        /
        "trial_summary.csv"
    )

    upper_rows = [
        row
        for row in all_rows
        if row["expected_side"] == "UPPER"
    ]

    lower_rows = [
        row
        for row in all_rows
        if row["expected_side"] == "LOWER"
    ]

    payload = {
        "session_id": SESSION_ID,
        "protocol": {
            "neutral_calibration_sec":
                NEUTRAL_CALIBRATION_SEC,
            "trial_protocol":
                TRIAL_PROTOCOL,
            "max_trial_sec":
                MAX_TRIAL_SEC,
            "orientation_definition":
                (
                    "2D wrist-to-middle-MCP angle "
                    "relative to calibrated neutral pose"
                ),
            "purpose":
                (
                    "Probe natural orientation variability "
                    "during upper/lower obstacle-avoidance "
                    "routes using the same normalized "
                    "hand-space geometry as the virtual task."
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
        "pooled_all": pooled_orientation_summary(
            all_rows
        ),
        "pooled_upper": pooled_orientation_summary(
            upper_rows
        ),
        "pooled_lower": pooled_orientation_summary(
            lower_rows
        ),
        "claim_boundary": (
            "This is a technical embodied-input probe, "
            "not a user study. It characterises how the "
            "current 2D hand-orientation proxy behaves "
            "during task-like upper/lower routes. It does "
            "not establish optimal thresholds or physical "
            "robot performance."
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

    return json_path, csv_path, payload


# =========================================================
# UI HELPERS
# =========================================================

def put_text(
    frame,
    text,
    x,
    y,
    scale=0.58,
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


def draw_task_world(
    frame,
    current_x,
    current_y,
    expected_side,
):
    height, width, _ = frame.shape

    # safety boundary
    safety_left, safety_top = world_to_screen(
        OBS_X_MIN,
        OBS_Y_MIN - SAFETY_MARGIN,
        width,
        height,
    )
    safety_right, safety_bottom = world_to_screen(
        OBS_X_MAX,
        OBS_Y_MAX + SAFETY_MARGIN,
        width,
        height,
    )

    cv2.rectangle(
        frame,
        (safety_left, safety_top),
        (safety_right, safety_bottom),
        (135, 135, 135),
        2,
        cv2.LINE_AA,
    )

    # obstacle
    obs_left, obs_top = world_to_screen(
        OBS_X_MIN,
        OBS_Y_MIN,
        width,
        height,
    )
    obs_right, obs_bottom = world_to_screen(
        OBS_X_MAX,
        OBS_Y_MAX,
        width,
        height,
    )

    cv2.rectangle(
        frame,
        (obs_left, obs_top),
        (obs_right, obs_bottom),
        (80, 80, 80),
        -1,
    )

    put_text(
        frame,
        "OBSTACLE",
        obs_left + 8,
        int((obs_top + obs_bottom) / 2),
        scale=0.43,
    )

    # start / target
    start = world_to_screen(
        0.0,
        0.0,
        width,
        height,
    )

    target = world_to_screen(
        TARGET_X,
        TARGET_Y,
        width,
        height,
    )

    cv2.circle(
        frame,
        start,
        11,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )

    cv2.circle(
        frame,
        target,
        18,
        (100, 220, 100),
        3,
        cv2.LINE_AA,
    )

    put_text(
        frame,
        "START",
        start[0] - 30,
        start[1] + 34,
        scale=0.42,
    )

    put_text(
        frame,
        "TARGET",
        target[0] - 35,
        target[1] + 42,
        scale=0.42,
    )

    # route side hint only
    hint_y = (
        OBS_Y_MIN
        -
        SAFETY_MARGIN
        -
        0.25
        if expected_side == "UPPER"
        else
        OBS_Y_MAX
        +
        SAFETY_MARGIN
        +
        0.25
    )

    hint_a = world_to_screen(
        0.9,
        hint_y,
        width,
        height,
    )

    hint_b = world_to_screen(
        2.35,
        hint_y,
        width,
        height,
    )

    cv2.line(
        frame,
        hint_a,
        hint_b,
        (170, 170, 170),
        2,
        cv2.LINE_AA,
    )

    put_text(
        frame,
        (
            "PASS ABOVE"
            if expected_side == "UPPER"
            else
            "PASS BELOW"
        ),
        hint_a[0],
        hint_a[1] - 12,
        scale=0.42,
        color=(210, 210, 210),
    )

    # current hand point
    if current_x is not None and current_y is not None:
        current = world_to_screen(
            current_x,
            current_y,
            width,
            height,
        )

        cv2.circle(
            frame,
            current,
            10,
            (0, 220, 255),
            -1,
            cv2.LINE_AA,
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

calibration_start_time = None
calibration_samples = []
calibration = None

trial_start_time = None
trial_rows = []
completed_trials = 0

trial_summaries = []
all_rows = []

previous_world_position = None


# =========================================================
# INTRO
# =========================================================

print()
print("========================================================")
print("AdaptiveSkill - Hand Obstacle-Route Orientation Test")
print("========================================================")
print()
print(
    "Purpose: test orientation variability during the "
    "actual upper/lower obstacle-avoidance movement pattern."
)
print()
print(
    "Protocol: 3 UPPER routes, then 3 LOWER routes."
)
print(
    "No live angle number is shown. Follow the route cue "
    "and keep the hand naturally upright."
)
print()
print(
    f"Current thresholds: mild={MILD_TOLERANCE_DEG} deg, "
    f"strong={STRONG_TOLERANCE_DEG} deg"
)
print()
print(
    "Controls: C = calibrate | SPACE = start next trial | "
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

    video_start_time = time.perf_counter()

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
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
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

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        hand_detected = bool(
            result.hand_landmarks
        )

        x_raw = None
        y_raw = None
        palm_size = None
        raw_angle_deg = None

        x_norm = None
        y_norm = None
        relative_angle_deg = None

        now = time.perf_counter()

        if hand_detected:
            landmarks = result.hand_landmarks[0]

            index_tip = landmarks[8]

            x_raw = float(index_tip.x)
            y_raw = float(index_tip.y)

            palm_size = calculate_palm_size(
                landmarks
            )

            raw_angle_deg = calculate_hand_angle(
                landmarks
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
                        "x": x_raw,
                        "y": y_raw,
                        "palm_size": palm_size,
                        "angle": raw_angle_deg,
                    }
                )

            elapsed = (
                now
                -
                calibration_start_time
            )

            if elapsed >= NEUTRAL_CALIBRATION_SEC:

                if len(calibration_samples) < MIN_NEUTRAL_SAMPLES:
                    print()
                    print(
                        "Calibration FAILED: not enough "
                        "valid samples."
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
                            len(calibration_samples),
                    }

                    stage = "IDLE"

                    print()
                    print("Calibration complete.")
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
                        "Return to START and press SPACE "
                        "for UPPER trial 1."
                    )
                    print()

        # =================================================
        # RECORDING
        # =================================================

        if stage == "RECORDING":

            expected_side = TRIAL_PROTOCOL[
                completed_trials
            ]

            elapsed = now - trial_start_time

            task_result = None
            point_result = None

            if (
                hand_detected
                and
                x_norm is not None
                and
                y_norm is not None
                and
                relative_angle_deg is not None
            ):
                point_result = TASK_EVALUATOR.evaluate(
                    x=x_norm,
                    y=y_norm,
                    orientation_relative_deg=
                        relative_angle_deg,
                    tracking_missing_sec=0.0,
                )

                if previous_world_position is not None:
                    task_result = TASK_EVALUATOR.evaluate_segment(
                        previous_x=
                            previous_world_position[0],
                        previous_y=
                            previous_world_position[1],
                        x=x_norm,
                        y=y_norm,
                        orientation_relative_deg=
                            relative_angle_deg,
                        tracking_missing_sec=0.0,
                    )
                else:
                    task_result = point_result

                previous_world_position = (
                    x_norm,
                    y_norm,
                )

            if task_result is not None:
                task_state = task_result.state

                dominant_constraint = str(
                    getattr(
                        task_result,
                        "dominant_constraint",
                        "",
                    )
                )

                # Goal is an endpoint property; obstacle safety
                # remains segment-aware.
                goal_reached = bool(
                    getattr(
                        point_result,
                        "goal_reached",
                        False,
                    )
                )

                obstacle_risk = bool(
                    getattr(
                        task_result,
                        "obstacle_risk",
                        False,
                    )
                )

                obstacle_violation = bool(
                    getattr(
                        task_result,
                        "obstacle_violation",
                        False,
                    )
                )

                orientation_risk = bool(
                    getattr(
                        task_result,
                        "orientation_risk",
                        False,
                    )
                )

                orientation_violation = bool(
                    getattr(
                        task_result,
                        "orientation_violation",
                        False,
                    )
                )

                orientation_error_deg = float(
                    getattr(
                        task_result,
                        "orientation_error_deg",
                        abs(
                            relative_angle_deg
                        ),
                    )
                )
            else:
                task_state = ""
                dominant_constraint = ""
                goal_reached = False
                obstacle_risk = False
                obstacle_violation = False
                orientation_risk = False
                orientation_violation = False
                orientation_error_deg = None

            row = {
                "trial_index":
                    completed_trials + 1,
                "expected_side":
                    expected_side,
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
                        round(
                            x_norm,
                            6,
                        )
                        if x_norm is not None
                        else ""
                    ),
                "y_norm":
                    (
                        round(
                            y_norm,
                            6,
                        )
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
                "task_state":
                    task_state,
                "dominant_constraint":
                    dominant_constraint,
                "goal_reached":
                    1
                    if goal_reached
                    else 0,
                "obstacle_risk":
                    1
                    if obstacle_risk
                    else 0,
                "obstacle_violation":
                    1
                    if obstacle_violation
                    else 0,
                "orientation_risk":
                    1
                    if orientation_risk
                    else 0,
                "orientation_violation":
                    1
                    if orientation_violation
                    else 0,
                "orientation_error_deg":
                    (
                        round(
                            orientation_error_deg,
                            6,
                        )
                        if orientation_error_deg is not None
                        else ""
                    ),
            }

            trial_rows.append(row)

            should_finish = (
                goal_reached
                or
                elapsed >= MAX_TRIAL_SEC
            )

            if should_finish:
                trial_number = completed_trials + 1

                csv_path = save_trial_csv(
                    trial_number,
                    expected_side,
                    trial_rows,
                )

                summary = summarize_rows(
                    trial_rows,
                    expected_side,
                )

                summary["trial_index"] = trial_number
                summary["csv_path"] = str(csv_path)

                valid_sample_count = sum(
                    1
                    for item in trial_rows
                    if int(item["hand_detected"]) == 1
                    and item["relative_angle_deg"] != ""
                )

                summary["enough_samples"] = (
                    valid_sample_count
                    >=
                    MIN_VALID_TRIAL_SAMPLES
                )

                trial_summaries.append(
                    summary
                )

                all_rows.extend(
                    trial_rows
                )

                completed_trials += 1

                print()
                print(
                    "--------------------------------------------------------"
                )
                print(
                    f"Trial {trial_number} complete | "
                    f"{expected_side}"
                )
                print(
                    "--------------------------------------------------------"
                )
                print(
                    "Goal reached:",
                    summary["goal_reached"]
                )
                print(
                    "Expected-side route valid:",
                    summary["expected_side_valid"]
                )
                print(
                    "Route protocol valid:",
                    summary["route_protocol_valid"]
                )
                print(
                    "Obstacle risk seen:",
                    summary["obstacle_risk_seen"]
                )
                print(
                    "Obstacle violation seen:",
                    summary["obstacle_violation_seen"]
                )
                print(
                    "Orientation risk seen:",
                    summary["orientation_risk_seen"]
                )
                print(
                    "Orientation violation seen:",
                    summary["orientation_violation_seen"]
                )
                print(
                    "Mean abs orientation error:",
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
                    "--------------------------------------------------------"
                )
                print()

                trial_rows = []
                trial_start_time = None
                previous_world_position = None

                if completed_trials >= len(TRIAL_PROTOCOL):
                    stage = "COMPLETE"

                    (
                        json_path,
                        csv_summary_path,
                        payload,
                    ) = save_session_summary(
                        calibration,
                        trial_summaries,
                        all_rows,
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

                    for label, pooled in (
                        (
                            "ALL ROUTES",
                            payload["pooled_all"],
                        ),
                        (
                            "UPPER",
                            payload["pooled_upper"],
                        ),
                        (
                            "LOWER",
                            payload["pooled_lower"],
                        ),
                    ):
                        print(label)
                        print(
                            "  Mean abs error:",
                            pooled["mean_abs_error_deg"],
                            "deg"
                        )
                        print(
                            "  P90:",
                            pooled["p90_abs_error_deg"],
                            "deg"
                        )
                        print(
                            "  P95:",
                            pooled["p95_abs_error_deg"],
                            "deg"
                        )
                        print(
                            "  Max:",
                            pooled["max_abs_error_deg"],
                            "deg"
                        )
                        print(
                            "  > mild:",
                            (
                                f"{pooled['fraction_above_mild_tolerance'] * 100:.1f}%"
                                if pooled["fraction_above_mild_tolerance"] is not None
                                else "N/A"
                            )
                        )
                        print(
                            "  > strong:",
                            (
                                f"{pooled['fraction_above_strong_tolerance'] * 100:.1f}%"
                                if pooled["fraction_above_strong_tolerance"] is not None
                                else "N/A"
                            )
                        )
                        print()

                    print("Summary JSON:")
                    print(json_path)
                    print()
                    print("Trial summary CSV:")
                    print(csv_summary_path)
                    print()
                    print(
                        "NOTE: This is a technical embodied-input "
                        "probe, not a user study."
                    )
                    print(
                        "========================================================"
                    )
                    print()
                else:
                    stage = "IDLE"

                    next_side = TRIAL_PROTOCOL[
                        completed_trials
                    ]

                    print(
                        "Return to START."
                    )
                    print(
                        f"Next trial: {next_side}"
                    )
                    print(
                        "Press SPACE when ready."
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
            (width, 105),
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
            "AdaptiveSkill - Hand Obstacle-Route Orientation Test",
            18,
            31,
            scale=0.66,
        )

        if stage == "RECORDING":
            expected_side = TRIAL_PROTOCOL[
                completed_trials
            ]
            status = (
                f"TRIAL {completed_trials + 1}/"
                f"{len(TRIAL_PROTOCOL)} | "
                f"{expected_side} | keep naturally upright"
            )
        elif stage == "CALIBRATING":
            status = (
                "CALIBRATING - hold neutral upright at START"
            )
            expected_side = (
                TRIAL_PROTOCOL[
                    completed_trials
                ]
                if completed_trials
                <
                len(TRIAL_PROTOCOL)
                else
                "UPPER"
            )
        elif stage == "COMPLETE":
            status = "COMPLETE - press Q to close"
            expected_side = "UPPER"
        else:
            if calibration is None:
                status = "Press C to calibrate"
            else:
                expected_side = TRIAL_PROTOCOL[
                    completed_trials
                ]
                status = (
                    f"Next: {expected_side} | "
                    "return to START and press SPACE"
                )

            expected_side = (
                TRIAL_PROTOCOL[
                    completed_trials
                ]
                if completed_trials
                <
                len(TRIAL_PROTOCOL)
                else
                "UPPER"
            )

        put_text(
            frame,
            status,
            18,
            67,
            scale=0.52,
            color=(240, 220, 120),
        )

        put_text(
            frame,
            "No live angle number is shown.",
            18,
            94,
            scale=0.44,
            color=(220, 220, 220),
        )

        draw_task_world(
            frame,
            x_norm,
            y_norm,
            expected_side,
        )

        cv2.imshow(
            "AdaptiveSkill - Hand Obstacle-Route Orientation Test",
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
            print("Session reset.")
            print("Press C to recalibrate.")
            print()

            stage = "IDLE"
            calibration_start_time = None
            calibration_samples = []
            calibration = None

            trial_start_time = None
            trial_rows = []
            completed_trials = 0

            trial_summaries = []
            all_rows = []
            previous_world_position = None

            continue

        if key in (ord("c"), ord("C")):
            if stage == "IDLE":
                calibration_start_time = (
                    time.perf_counter()
                )
                calibration_samples = []
                calibration = None
                stage = "CALIBRATING"

                print()
                print("Calibration started.")
                print(
                    "Hold neutral upright at START."
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

            if completed_trials >= len(TRIAL_PROTOCOL):
                continue

            expected_side = TRIAL_PROTOCOL[
                completed_trials
            ]

            trial_rows = []
            previous_world_position = None
            trial_start_time = time.perf_counter()
            stage = "RECORDING"

            print()
            print(
                f"Trial {completed_trials + 1} started | "
                f"{expected_side}"
            )
            print(
                (
                    "Move above the obstacle to TARGET."
                    if expected_side == "UPPER"
                    else
                    "Move below the obstacle to TARGET."
                )
            )
            print(
                "Keep the hand naturally upright; "
                "do not chase orientation numbers."
            )
            print()


# =========================================================
# CLEANUP
# =========================================================

cap.release()
cv2.destroyAllWindows()
