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
    / "hand_orientation_deliberate_tilt"
)

ROUTE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "hand_orientation_route"
)


# =========================================================
# PROTOCOL
#
# No target degree is shown to the user.
# We want natural subjective categories, not "chase 12°".
# =========================================================

NEUTRAL_CALIBRATION_SEC = 2.5
TRIAL_DURATION_SEC = 3.5
MIN_NEUTRAL_SAMPLES = 30
CAMERA_INDEX = 0

TRIAL_PROTOCOL = [
    {
        "label": "NEUTRAL",
        "instruction": "Hold naturally upright.",
    },
    {
        "label": "MILD_LEFT",
        "instruction": (
            "Tilt the TOP of your hand slightly to the LEFT "
            "on the screen, but still feel mostly upright."
        ),
    },
    {
        "label": "MILD_RIGHT",
        "instruction": (
            "Tilt the TOP of your hand slightly to the RIGHT "
            "on the screen, but still feel mostly upright."
        ),
    },
    {
        "label": "STRONG_LEFT",
        "instruction": (
            "Tilt the TOP of your hand clearly to the LEFT "
            "on the screen, like a cup is noticeably tilted."
        ),
    },
    {
        "label": "STRONG_RIGHT",
        "instruction": (
            "Tilt the TOP of your hand clearly to the RIGHT "
            "on the screen, like a cup is noticeably tilted."
        ),
    },
]


# =========================================================
# CHECKS / CONFIG
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

MILD_TOLERANCE_DEG = float(
    TASK_CONFIG["orientation"]["mild_tolerance_deg"]
)

STRONG_TOLERANCE_DEG = float(
    TASK_CONFIG["orientation"]["strong_tolerance_deg"]
)


# =========================================================
# MEDIAPIPE TASKS
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
# ANGLE HELPERS
#
# Matches the existing AdaptiveSkill hand proxy:
# wrist (0) -> middle MCP (9), image-plane angle.
# =========================================================

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


def robust_unwrapped_median_angle(angles_deg):
    values = np.asarray(
        angles_deg,
        dtype=float,
    )

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
# PREVIOUS ROUTE EVIDENCE
# =========================================================

def load_latest_route_summary():
    if not ROUTE_ROOT.exists():
        return None, None

    candidates = sorted(
        ROUTE_ROOT.glob("*/session_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        return None, None

    path = candidates[0]

    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        return payload, path

    except Exception:
        return None, None


LATEST_ROUTE_SUMMARY, LATEST_ROUTE_PATH = (
    load_latest_route_summary()
)


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

    signed_errors = [
        float(row["relative_angle_deg"])
        for row in valid_rows
    ]

    abs_errors = [
        abs(value)
        for value in signed_errors
    ]

    if signed_errors:
        median_signed = float(
            np.median(signed_errors)
        )
        mean_signed = float(
            np.mean(signed_errors)
        )
        mean_abs = float(
            np.mean(abs_errors)
        )
        p10_signed = percentile(
            signed_errors, 10
        )
        p90_signed = percentile(
            signed_errors, 90
        )
        p50_abs = percentile(
            abs_errors, 50
        )
        p90_abs = percentile(
            abs_errors, 90
        )
        p95_abs = percentile(
            abs_errors, 95
        )
        max_abs = float(
            np.max(abs_errors)
        )

        mild_exceed = float(
            np.mean(
                np.asarray(abs_errors)
                > MILD_TOLERANCE_DEG
            )
        )

        strong_exceed = float(
            np.mean(
                np.asarray(abs_errors)
                > STRONG_TOLERANCE_DEG
            )
        )
    else:
        median_signed = float("nan")
        mean_signed = float("nan")
        mean_abs = float("nan")
        p10_signed = float("nan")
        p90_signed = float("nan")
        p50_abs = float("nan")
        p90_abs = float("nan")
        p95_abs = float("nan")
        max_abs = float("nan")
        mild_exceed = float("nan")
        strong_exceed = float("nan")

    return {
        "total_frames": total_frames,
        "valid_frames": valid_frames,
        "tracking_ratio": round_or_none(
            tracking_ratio, 4
        ),
        "median_signed_error_deg": round_or_none(
            median_signed, 3
        ),
        "mean_signed_error_deg": round_or_none(
            mean_signed, 3
        ),
        "p10_signed_error_deg": round_or_none(
            p10_signed, 3
        ),
        "p90_signed_error_deg": round_or_none(
            p90_signed, 3
        ),
        "mean_abs_error_deg": round_or_none(
            mean_abs, 3
        ),
        "median_abs_error_deg": round_or_none(
            p50_abs, 3
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
            mild_exceed, 4
        ),
        "fraction_above_strong_tolerance": round_or_none(
            strong_exceed, 4
        ),
        "mild_tolerance_deg": MILD_TOLERANCE_DEG,
        "strong_tolerance_deg": STRONG_TOLERANCE_DEG,
    }


def get_summary_by_label(trial_summaries, label):
    for item in trial_summaries:
        if item["label"] == label:
            return item

    return None


def build_separability_summary(trial_summaries):
    neutral = get_summary_by_label(
        trial_summaries,
        "NEUTRAL",
    )

    mild_left = get_summary_by_label(
        trial_summaries,
        "MILD_LEFT",
    )

    mild_right = get_summary_by_label(
        trial_summaries,
        "MILD_RIGHT",
    )

    strong_left = get_summary_by_label(
        trial_summaries,
        "STRONG_LEFT",
    )

    strong_right = get_summary_by_label(
        trial_summaries,
        "STRONG_RIGHT",
    )

    mild_abs_medians = [
        value["median_abs_error_deg"]
        for value in (
            mild_left,
            mild_right,
        )
        if value is not None
        and value["median_abs_error_deg"] is not None
    ]

    strong_abs_medians = [
        value["median_abs_error_deg"]
        for value in (
            strong_left,
            strong_right,
        )
        if value is not None
        and value["median_abs_error_deg"] is not None
    ]

    mild_typical = (
        float(
            np.mean(mild_abs_medians)
        )
        if mild_abs_medians
        else float("nan")
    )

    strong_typical = (
        float(
            np.mean(strong_abs_medians)
        )
        if strong_abs_medians
        else float("nan")
    )

    route_p95 = None
    route_upper_p95 = None
    route_lower_p95 = None

    if LATEST_ROUTE_SUMMARY is not None:
        try:
            route_p95 = float(
                LATEST_ROUTE_SUMMARY[
                    "pooled_all"
                ][
                    "p95_abs_error_deg"
                ]
            )
        except Exception:
            route_p95 = None

        try:
            route_upper_p95 = float(
                LATEST_ROUTE_SUMMARY[
                    "pooled_upper"
                ][
                    "p95_abs_error_deg"
                ]
            )
        except Exception:
            route_upper_p95 = None

        try:
            route_lower_p95 = float(
                LATEST_ROUTE_SUMMARY[
                    "pooled_lower"
                ][
                    "p95_abs_error_deg"
                ]
            )
        except Exception:
            route_lower_p95 = None

    return {
        "neutral_median_abs_error_deg":
            (
                neutral[
                    "median_abs_error_deg"
                ]
                if neutral is not None
                else None
            ),

        "subjective_mild_typical_median_abs_error_deg":
            round_or_none(
                mild_typical, 3
            ),

        "subjective_strong_typical_median_abs_error_deg":
            round_or_none(
                strong_typical, 3
            ),

        "latest_natural_route_p95_abs_error_deg":
            round_or_none(
                route_p95, 3
            ),

        "latest_natural_upper_route_p95_abs_error_deg":
            round_or_none(
                route_upper_p95, 3
            ),

        "latest_natural_lower_route_p95_abs_error_deg":
            round_or_none(
                route_lower_p95, 3
            ),

        "mild_minus_natural_route_p95_deg":
            (
                round_or_none(
                    mild_typical - route_p95,
                    3,
                )
                if (
                    route_p95 is not None
                    and
                    not math.isnan(mild_typical)
                )
                else None
            ),

        "strong_minus_natural_route_p95_deg":
            (
                round_or_none(
                    strong_typical - route_p95,
                    3,
                )
                if (
                    route_p95 is not None
                    and
                    not math.isnan(strong_typical)
                )
                else None
            ),
    }


# =========================================================
# OUTPUT
# =========================================================

def save_trial_csv(trial_index, label, rows):
    path = (
        SESSION_DIR
        /
        f"trial_{trial_index:02d}_{label}.csv"
    )

    fieldnames = [
        "trial_index",
        "label",
        "timestamp_sec",
        "hand_detected",
        "raw_angle_deg",
        "relative_angle_deg",
        "abs_error_deg",
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
    neutral_angle_deg,
    trial_summaries,
    separability,
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

    payload = {
        "session_id": SESSION_ID,

        "protocol": {
            "neutral_calibration_sec":
                NEUTRAL_CALIBRATION_SEC,

            "trial_duration_sec":
                TRIAL_DURATION_SEC,

            "trial_labels":
                [
                    item["label"]
                    for item in TRIAL_PROTOCOL
                ],

            "orientation_definition":
                (
                    "2D wrist-to-middle-MCP angle "
                    "relative to calibrated neutral pose"
                ),

            "important_note":
                (
                    "No target degrees were shown during "
                    "the deliberate tilt trials."
                ),
        },

        "task_thresholds": {
            "mild_tolerance_deg":
                MILD_TOLERANCE_DEG,

            "strong_tolerance_deg":
                STRONG_TOLERANCE_DEG,
        },

        "neutral_angle_deg":
            round_or_none(
                neutral_angle_deg,
                3,
            ),

        "trials":
            trial_summaries,

        "separability":
            separability,

        "latest_route_summary_used":
            (
                str(LATEST_ROUTE_PATH)
                if LATEST_ROUTE_PATH is not None
                else None
            ),

        "claim_boundary":
            (
                "This is a single-operator technical "
                "calibration of subjective deliberate "
                "tilt categories using the current 2D "
                "hand-orientation proxy. It does not "
                "establish optimal thresholds, object "
                "orientation ground truth, or general "
                "human performance."
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


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW,
)

if not cap.isOpened():
    print(
        "ERROR: Camera cannot be opened."
    )
    raise SystemExit


# =========================================================
# STATE
# =========================================================

stage = "IDLE"

neutral_ready = False
neutral_angle_deg = None
neutral_samples = []
neutral_start_time = None

trial_rows = []
trial_start_time = None

completed_trials = 0
trial_summaries = []


# =========================================================
# INTRO
# =========================================================

print()
print("========================================================")
print("AdaptiveSkill - Deliberate Tilt Calibration")
print("========================================================")
print()
print(
    "Purpose: compare natural route-related orientation "
    "variation with deliberately tilted hand poses."
)
print()
print(
    "IMPORTANT: the program will NOT give you a target "
    "degree. Use your own natural judgement."
)
print()
print("Protocol:")
print("1. Press C and hold a natural upright pose.")
print(
    "2. Then complete 5 short trials:"
)
print("   NEUTRAL")
print("   MILD_LEFT")
print("   MILD_RIGHT")
print("   STRONG_LEFT")
print("   STRONG_RIGHT")
print()
print(
    "LEFT/RIGHT refers to the TOP of your hand on the "
    "mirrored camera screen."
)
print()
print(
    "Current prototype thresholds are only logged for "
    "comparison:"
)
print(
    f"mild={MILD_TOLERANCE_DEG} deg, "
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
            print(
                "ERROR: Cannot read camera frame."
            )
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

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        hand_detected = bool(
            result.hand_landmarks
        )

        raw_angle_deg = None
        relative_angle_deg = None

        now = time.perf_counter()

        if hand_detected:
            landmarks = result.hand_landmarks[0]

            raw_angle_deg = calculate_hand_angle(
                landmarks
            )

            if neutral_ready:
                relative_angle_deg = (
                    circular_angle_difference(
                        raw_angle_deg,
                        neutral_angle_deg,
                    )
                )

        # =================================================
        # CALIBRATION
        # =================================================

        if stage == "CALIBRATING":

            if hand_detected:
                neutral_samples.append(
                    raw_angle_deg
                )

            elapsed = (
                now
                -
                neutral_start_time
            )

            if elapsed >= NEUTRAL_CALIBRATION_SEC:

                if (
                    len(neutral_samples)
                    <
                    MIN_NEUTRAL_SAMPLES
                ):
                    print()
                    print(
                        "Neutral calibration FAILED: "
                        "not enough valid hand samples."
                    )
                    print(
                        "Press C and try again."
                    )
                    print()

                    neutral_samples = []
                    stage = "IDLE"

                else:
                    neutral_angle_deg = (
                        robust_unwrapped_median_angle(
                            neutral_samples
                        )
                    )

                    neutral_ready = True
                    stage = "IDLE"

                    print()
                    print(
                        "Neutral calibration complete."
                    )
                    print(
                        "Valid samples:",
                        len(neutral_samples)
                    )
                    print(
                        "Neutral raw angle:",
                        round(
                            neutral_angle_deg,
                            3,
                        ),
                        "deg"
                    )
                    print()
                    print(
                        "Next trial: NEUTRAL"
                    )
                    print(
                        "Press SPACE when ready."
                    )
                    print()

        # =================================================
        # RECORDING
        # =================================================

        if stage == "RECORDING":

            current_protocol = (
                TRIAL_PROTOCOL[
                    completed_trials
                ]
            )

            elapsed = (
                now
                -
                trial_start_time
            )

            row = {
                "trial_index":
                    completed_trials + 1,

                "label":
                    current_protocol["label"],

                "timestamp_sec":
                    round(
                        elapsed,
                        6,
                    ),

                "hand_detected":
                    1
                    if hand_detected
                    else 0,

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
                            abs(relative_angle_deg),
                            6,
                        )
                        if relative_angle_deg is not None
                        else ""
                    ),
            }

            trial_rows.append(row)

            if elapsed >= TRIAL_DURATION_SEC:

                trial_number = (
                    completed_trials + 1
                )

                label = (
                    current_protocol["label"]
                )

                csv_path = save_trial_csv(
                    trial_number,
                    label,
                    trial_rows,
                )

                summary = summarize_rows(
                    trial_rows
                )

                summary["trial_index"] = (
                    trial_number
                )

                summary["label"] = label

                summary["instruction"] = (
                    current_protocol[
                        "instruction"
                    ]
                )

                summary["csv_path"] = str(
                    csv_path
                )

                trial_summaries.append(
                    summary
                )

                completed_trials += 1

                print()
                print(
                    "--------------------------------------------------------"
                )
                print(
                    f"Trial {trial_number} complete | {label}"
                )
                print(
                    "--------------------------------------------------------"
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
                    "Median signed error:",
                    summary["median_signed_error_deg"],
                    "deg"
                )
                print(
                    "Mean abs error:",
                    summary["mean_abs_error_deg"],
                    "deg"
                )
                print(
                    "Median abs error:",
                    summary["median_abs_error_deg"],
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

                if (
                    completed_trials
                    >=
                    len(TRIAL_PROTOCOL)
                ):
                    stage = "COMPLETE"

                    separability = (
                        build_separability_summary(
                            trial_summaries
                        )
                    )

                    (
                        json_path,
                        csv_summary_path,
                    ) = save_session_summary(
                        neutral_angle_deg,
                        trial_summaries,
                        separability,
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
                        "SEPARABILITY SUMMARY"
                    )
                    print()

                    for key, value in (
                        (
                            "Neutral median abs",
                            separability[
                                "neutral_median_abs_error_deg"
                            ],
                        ),
                        (
                            "Subjective mild typical median abs",
                            separability[
                                "subjective_mild_typical_median_abs_error_deg"
                            ],
                        ),
                        (
                            "Subjective strong typical median abs",
                            separability[
                                "subjective_strong_typical_median_abs_error_deg"
                            ],
                        ),
                        (
                            "Latest natural route P95",
                            separability[
                                "latest_natural_route_p95_abs_error_deg"
                            ],
                        ),
                        (
                            "Latest natural UPPER route P95",
                            separability[
                                "latest_natural_upper_route_p95_abs_error_deg"
                            ],
                        ),
                        (
                            "Latest natural LOWER route P95",
                            separability[
                                "latest_natural_lower_route_p95_abs_error_deg"
                            ],
                        ),
                        (
                            "Mild - natural route P95",
                            separability[
                                "mild_minus_natural_route_p95_deg"
                            ],
                        ),
                        (
                            "Strong - natural route P95",
                            separability[
                                "strong_minus_natural_route_p95_deg"
                            ],
                        ),
                    ):
                        print(
                            f"{key}:",
                            (
                                f"{value} deg"
                                if value is not None
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
                        "NOTE:"
                    )
                    print(
                        "This is a single-operator technical "
                        "calibration. Do not treat the measured "
                        "category values as universal human "
                        "thresholds."
                    )
                    print(
                        "========================================================"
                    )
                    print()

                else:
                    stage = "IDLE"

                    next_protocol = (
                        TRIAL_PROTOCOL[
                            completed_trials
                        ]
                    )

                    print(
                        "Next trial:",
                        next_protocol["label"]
                    )
                    print(
                        next_protocol[
                            "instruction"
                        ]
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
            "AdaptiveSkill - Deliberate Tilt Calibration",
            18,
            32,
            scale=0.68,
        )

        if stage == "CALIBRATING":
            status = (
                "CALIBRATING: hold natural upright pose"
            )
            detail = (
                "No target degree is shown."
            )

        elif stage == "RECORDING":
            current_protocol = (
                TRIAL_PROTOCOL[
                    completed_trials
                ]
            )

            status = (
                f"TRIAL {completed_trials + 1}/"
                f"{len(TRIAL_PROTOCOL)} | "
                f"{current_protocol['label']}"
            )

            detail = (
                current_protocol[
                    "instruction"
                ]
            )

        elif stage == "COMPLETE":
            status = (
                "COMPLETE - press Q to close"
            )
            detail = (
                "Results saved."
            )

        else:
            if not neutral_ready:
                status = (
                    "Press C to calibrate neutral upright pose"
                )
                detail = (
                    "Keep one hand visible."
                )
            else:
                next_protocol = (
                    TRIAL_PROTOCOL[
                        completed_trials
                    ]
                )

                status = (
                    f"Next: {next_protocol['label']} | "
                    "press SPACE"
                )

                detail = (
                    next_protocol[
                        "instruction"
                    ]
                )

        put_text(
            frame,
            status,
            18,
            70,
            scale=0.53,
            color=(240, 220, 120),
        )

        put_text(
            frame,
            detail,
            18,
            106,
            scale=0.44,
            color=(220, 220, 220),
        )

        put_text(
            frame,
            "Do not chase a number; use your natural judgement.",
            18,
            138,
            scale=0.42,
            color=(190, 190, 190),
        )

        if hand_detected:
            landmarks = (
                result.hand_landmarks[0]
            )

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

        cv2.imshow(
            "AdaptiveSkill - Deliberate Tilt Calibration",
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

            neutral_ready = False
            neutral_angle_deg = None
            neutral_samples = []
            neutral_start_time = None

            trial_rows = []
            trial_start_time = None

            completed_trials = 0
            trial_summaries = []

            continue

        if key in (ord("c"), ord("C")):
            if stage == "IDLE":
                neutral_samples = []
                neutral_start_time = (
                    time.perf_counter()
                )
                neutral_ready = False
                neutral_angle_deg = None
                stage = "CALIBRATING"

                print()
                print(
                    "Neutral calibration started."
                )
                print(
                    "Hold a natural upright pose."
                )
                print()

            continue

        if key == 32:
            if not neutral_ready:
                print()
                print(
                    "Neutral calibration is not ready. "
                    "Press C first."
                )
                print()
                continue

            if stage != "IDLE":
                continue

            if (
                completed_trials
                >=
                len(TRIAL_PROTOCOL)
            ):
                continue

            current_protocol = (
                TRIAL_PROTOCOL[
                    completed_trials
                ]
            )

            trial_rows = []
            trial_start_time = (
                time.perf_counter()
            )
            stage = "RECORDING"

            print()
            print(
                f"Trial {completed_trials + 1} started | "
                f"{current_protocol['label']}"
            )
            print(
                current_protocol[
                    "instruction"
                ]
            )
            print(
                "Do not chase a degree value."
            )
            print()


# =========================================================
# CLEANUP
# =========================================================

cap.release()
cv2.destroyAllWindows()
