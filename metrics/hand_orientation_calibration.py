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
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# =========================================================
# PATHS
# =========================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "hand_landmarker.task"
)

TASK_CONFIG_PATH = (
    PROJECT_ROOT
    / "task"
    / "task_config.json"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "hand_orientation_calibration"
)


# =========================================================
# PROTOCOL
# =========================================================

NEUTRAL_CALIBRATION_SEC = 2.5

TRIAL_DURATION_SEC = 5.0

TARGET_TRIALS = 3

MIN_NEUTRAL_SAMPLES = 30

CAMERA_INDEX = 0


# =========================================================
# BASIC CHECKS
# =========================================================

if not MODEL_PATH.exists():

    print(
        "ERROR: hand_landmarker.task not found:"
    )

    print(
        MODEL_PATH
    )

    raise SystemExit


if not TASK_CONFIG_PATH.exists():

    print(
        "ERROR: task_config.json not found:"
    )

    print(
        TASK_CONFIG_PATH
    )

    raise SystemExit


with open(
    TASK_CONFIG_PATH,
    "r",
    encoding="utf-8",
) as file:

    TASK_CONFIG = json.load(
        file
    )


MILD_TOLERANCE_DEG = float(
    TASK_CONFIG[
        "orientation"
    ][
        "mild_tolerance_deg"
    ]
)

STRONG_TOLERANCE_DEG = float(
    TASK_CONFIG[
        "orientation"
    ][
        "strong_tolerance_deg"
    ]
)


# =========================================================
# MEDIAPIPE
# =========================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


OPTIONS = HandLandmarkerOptions(

    base_options=
        BaseOptions(
            model_asset_path=
                str(
                    MODEL_PATH
                )
        ),

    running_mode=
        VisionRunningMode.VIDEO,

    num_hands=
        1,

    min_hand_detection_confidence=
        0.5,

    min_hand_presence_confidence=
        0.5,

    min_tracking_confidence=
        0.5,
)


# =========================================================
# ANGLE HELPERS
#
# IMPORTANT:
# This deliberately matches the existing AdaptiveSkill
# hand-orientation definition:
#
# wrist (landmark 0) -> middle MCP (landmark 9)
#
# raw_angle = atan2(dy, dx)
#
# Relative orientation is then measured against one neutral
# calibration angle using circular angle difference.
# =========================================================

def circular_angle_difference(
    angle1,
    angle2,
):

    return (
        angle1
        -
        angle2
        +
        180.0
    ) % 360.0 - 180.0


def calculate_hand_angle(
    landmarks,
):

    wrist = landmarks[
        0
    ]

    middle_mcp = landmarks[
        9
    ]


    dx = (
        middle_mcp.x
        -
        wrist.x
    )


    dy = (
        middle_mcp.y
        -
        wrist.y
    )


    return math.degrees(
        math.atan2(
            dy,
            dx,
        )
    )


def robust_unwrapped_median_angle(
    angles_deg,
):

    values = np.asarray(
        angles_deg,
        dtype=float,
    )


    if len(values) == 0:

        raise ValueError(
            "No angles available."
        )


    unwrapped = np.rad2deg(
        np.unwrap(
            np.deg2rad(
                values
            )
        )
    )


    median_value = float(
        np.median(
            unwrapped
        )
    )


    return (
        median_value
        +
        180.0
    ) % 360.0 - 180.0


# =========================================================
# SUMMARY HELPERS
# =========================================================

def percentile(
    values,
    q,
):

    if not values:

        return float(
            "nan"
        )


    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=float,
            ),
            q,
        )
    )


def round_or_none(
    value,
    digits=4,
):

    if value is None:

        return None


    try:

        value = float(
            value
        )


    except Exception:

        return None


    if math.isnan(
        value
    ):

        return None


    return round(
        value,
        digits,
    )


def summarize_trial(
    rows,
    neutral_angle_deg,
    mild_tolerance_deg,
    strong_tolerance_deg,
):

    total_frames = len(
        rows
    )


    valid_rows = [

        row

        for row in rows

        if int(
            row[
                "hand_detected"
            ]
        )
        ==
        1

        and

        row[
            "raw_angle_deg"
        ]
        !=
        ""
    ]


    valid_frames = len(
        valid_rows
    )


    tracking_ratio = (

        valid_frames
        /
        total_frames

        if total_frames > 0

        else 0.0
    )


    signed_errors = [

        float(
            row[
                "relative_angle_deg"
            ]
        )

        for row in valid_rows
    ]


    abs_errors = [

        abs(
            value
        )

        for value in signed_errors
    ]


    raw_angles = [

        float(
            row[
                "raw_angle_deg"
            ]
        )

        for row in valid_rows
    ]


    frame_jitter = []


    for index in range(
        1,
        len(
            raw_angles
        ),
    ):

        jitter = abs(
            circular_angle_difference(
                raw_angles[
                    index
                ],
                raw_angles[
                    index - 1
                ],
            )
        )


        frame_jitter.append(
            jitter
        )


    if signed_errors:

        median_signed_error = float(
            np.median(
                signed_errors
            )
        )


        mean_signed_error = float(
            np.mean(
                signed_errors
            )
        )


        std_signed_error = float(
            np.std(
                signed_errors,
                ddof=0,
            )
        )


        mean_abs_error = float(
            np.mean(
                abs_errors
            )
        )


        p90_abs_error = percentile(
            abs_errors,
            90,
        )


        p95_abs_error = percentile(
            abs_errors,
            95,
        )


        max_abs_error = float(
            np.max(
                abs_errors
            )
        )


        mild_exceed_fraction = float(
            np.mean(
                np.asarray(
                    abs_errors
                )
                >
                mild_tolerance_deg
            )
        )


        strong_exceed_fraction = float(
            np.mean(
                np.asarray(
                    abs_errors
                )
                >
                strong_tolerance_deg
            )
        )


    else:

        median_signed_error = float(
            "nan"
        )

        mean_signed_error = float(
            "nan"
        )

        std_signed_error = float(
            "nan"
        )

        mean_abs_error = float(
            "nan"
        )

        p90_abs_error = float(
            "nan"
        )

        p95_abs_error = float(
            "nan"
        )

        max_abs_error = float(
            "nan"
        )

        mild_exceed_fraction = float(
            "nan"
        )

        strong_exceed_fraction = float(
            "nan"
        )


    if frame_jitter:

        mean_frame_jitter = float(
            np.mean(
                frame_jitter
            )
        )


        p95_frame_jitter = percentile(
            frame_jitter,
            95,
        )


    else:

        mean_frame_jitter = float(
            "nan"
        )

        p95_frame_jitter = float(
            "nan"
        )


    return {

        "neutral_angle_deg":
            round_or_none(
                neutral_angle_deg,
                3,
            ),

        "requested_duration_sec":
            TRIAL_DURATION_SEC,

        "total_frames":
            total_frames,

        "valid_frames":
            valid_frames,

        "tracking_ratio":
            round_or_none(
                tracking_ratio,
                4,
            ),

        "median_signed_error_deg":
            round_or_none(
                median_signed_error,
                3,
            ),

        "mean_signed_error_deg":
            round_or_none(
                mean_signed_error,
                3,
            ),

        "std_signed_error_deg":
            round_or_none(
                std_signed_error,
                3,
            ),

        "mean_abs_error_deg":
            round_or_none(
                mean_abs_error,
                3,
            ),

        "p90_abs_error_deg":
            round_or_none(
                p90_abs_error,
                3,
            ),

        "p95_abs_error_deg":
            round_or_none(
                p95_abs_error,
                3,
            ),

        "max_abs_error_deg":
            round_or_none(
                max_abs_error,
                3,
            ),

        "mean_frame_jitter_deg":
            round_or_none(
                mean_frame_jitter,
                3,
            ),

        "p95_frame_jitter_deg":
            round_or_none(
                p95_frame_jitter,
                3,
            ),

        "fraction_above_mild_tolerance":
            round_or_none(
                mild_exceed_fraction,
                4,
            ),

        "fraction_above_strong_tolerance":
            round_or_none(
                strong_exceed_fraction,
                4,
            ),

        "mild_tolerance_deg":
            mild_tolerance_deg,

        "strong_tolerance_deg":
            strong_tolerance_deg,
    }


def summarize_pooled(
    all_rows,
    neutral_angle_deg,
    mild_tolerance_deg,
    strong_tolerance_deg,
):

    return summarize_trial(

        rows=
            all_rows,

        neutral_angle_deg=
            neutral_angle_deg,

        mild_tolerance_deg=
            mild_tolerance_deg,

        strong_tolerance_deg=
            strong_tolerance_deg,
    )


# =========================================================
# FILE OUTPUT
# =========================================================

SESSION_ID = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)


SESSION_DIR = (
    OUTPUT_ROOT
    /
    SESSION_ID
)


SESSION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def save_trial_csv(
    trial_index,
    rows,
):

    path = (
        SESSION_DIR
        /
        f"trial_{trial_index:02d}.csv"
    )


    fieldnames = [
        "trial_index",
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

            fieldnames=
                fieldnames,
        )


        writer.writeheader()

        writer.writerows(
            rows
        )


    return path


def save_session_summary(
    neutral_angle_deg,
    trial_summaries,
    pooled_summary,
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

        "session_id":
            SESSION_ID,

        "protocol":
            {
                "neutral_calibration_sec":
                    NEUTRAL_CALIBRATION_SEC,

                "trial_duration_sec":
                    TRIAL_DURATION_SEC,

                "target_trials":
                    TARGET_TRIALS,

                "orientation_definition":
                    (
                        "2D wrist-to-middle-MCP angle "
                        "relative to calibrated neutral pose"
                    ),
            },

        "task_thresholds":
            {
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

        "pooled":
            pooled_summary,

        "claim_boundary":
            (
                "This is a technical calibration of 2D "
                "hand-orientation variability under the "
                "current MediaPipe sensing pipeline. "
                "It does not establish optimal task "
                "thresholds or infer cognitive state."
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

        fieldnames = list(
            trial_summaries[0].keys()
        )


        with open(
            csv_path,
            "w",
            encoding="utf-8",
            newline="",
        ) as file:

            writer = csv.DictWriter(

                file,

                fieldnames=
                    fieldnames,
            )


            writer.writeheader()

            writer.writerows(
                trial_summaries
            )


    return (
        json_path,
        csv_path,
    )


# =========================================================
# UI HELPERS
# =========================================================

def put_text(
    frame,
    text,
    x,
    y,
    scale=0.65,
    color=(255, 255, 255),
    thickness=2,
):

    cv2.putText(

        frame,

        text,

        (
            int(x),
            int(y),
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        float(
            scale
        ),

        color,

        int(
            thickness
        ),

        cv2.LINE_AA,
    )


def draw_status_panel(
    frame,
    stage,
    hand_detected,
    neutral_ready,
    completed_trials,
    trial_elapsed_sec,
):

    height, width, _ = (
        frame.shape
    )


    panel_height = 170


    overlay = (
        frame.copy()
    )


    cv2.rectangle(

        overlay,

        (
            0,
            0,
        ),

        (
            width,
            panel_height,
        ),

        (
            25,
            25,
            25,
        ),

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
        "AdaptiveSkill - Hand Orientation Calibration",
        20,
        34,
        scale=0.72,
    )


    put_text(
        frame,
        (
            "Hand: DETECTED"
            if hand_detected
            else
            "Hand: NOT DETECTED"
        ),
        20,
        68,
        scale=0.58,
        color=(
            (100, 230, 100)
            if hand_detected
            else
            (80, 80, 240)
        ),
    )


    if stage == "IDLE":

        message = (
            "Press C: calibrate neutral upright pose"
            if not neutral_ready
            else
            "Press SPACE: record 5-sec upright trial"
        )


    elif stage == "CALIBRATING":

        message = (
            "CALIBRATING: hold your hand naturally upright"
        )


    elif stage == "RECORDING":

        message = (
            f"RECORDING trial {completed_trials + 1}/{TARGET_TRIALS}: "
            "keep naturally upright - do not chase a number"
        )


    elif stage == "COMPLETE":

        message = (
            "COMPLETE - press Q to close"
        )


    else:

        message = stage


    put_text(
        frame,
        message,
        20,
        103,
        scale=0.56,
        color=(240, 220, 120),
    )


    if stage == "RECORDING":

        remaining = max(
            0.0,
            TRIAL_DURATION_SEC
            -
            trial_elapsed_sec,
        )


        secondary = (
            f"Time remaining: {remaining:.1f}s"
        )


    elif stage == "CALIBRATING":

        secondary = (
            "Keep one hand visible and stable."
        )


    else:

        secondary = (
            f"Completed trials: {completed_trials}/{TARGET_TRIALS} | "
            "R = reset session | Q = quit"
        )


    put_text(
        frame,
        secondary,
        20,
        138,
        scale=0.50,
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

all_trial_rows = []


# =========================================================
# CONSOLE INTRO
# =========================================================

print()

print(
    "========================================================"
)

print(
    "AdaptiveSkill - Hand Orientation Calibration"
)

print(
    "========================================================"
)

print()

print(
    "Purpose:"
)

print(
    "Measure natural 2D hand-orientation variability under "
    "the current MediaPipe sensing pipeline."
)

print()

print(
    "Protocol:"
)

print(
    "1. Put ONE hand in view in a comfortable upright pose."
)

print(
    "2. Press C and keep it naturally upright for "
    f"{NEUTRAL_CALIBRATION_SEC:.1f} sec."
)

print(
    "3. Press SPACE for each 5-sec trial."
)

print(
    f"4. Complete {TARGET_TRIALS} trials."
)

print()

print(
    "During recording, do NOT try to chase the 8/15-degree "
    "thresholds. Hold the pose naturally."
)

print()

print(
    "Current prototype thresholds:"
)

print(
    "Mild:",
    MILD_TOLERANCE_DEG,
    "deg"
)

print(
    "Strong:",
    STRONG_TOLERANCE_DEG,
    "deg"
)

print()

print(
    "Controls: C = neutral calibration | SPACE = trial | "
    "R = reset | Q = quit"
)

print(
    "========================================================"
)

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

        ret, frame = (
            cap.read()
        )


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

        relative_angle_deg = None


        if hand_detected:

            landmarks = (
                result.hand_landmarks[0]
            )


            raw_angle_deg = (
                calculate_hand_angle(
                    landmarks
                )
            )


            if neutral_ready:

                relative_angle_deg = (
                    circular_angle_difference(
                        raw_angle_deg,
                        neutral_angle_deg,
                    )
                )


        now = (
            time.perf_counter()
        )


        # =================================================
        # NEUTRAL CALIBRATION
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


            if (
                elapsed
                >=
                NEUTRAL_CALIBRATION_SEC
            ):

                if (
                    len(
                        neutral_samples
                    )
                    <
                    MIN_NEUTRAL_SAMPLES
                ):

                    print()

                    print(
                        "Neutral calibration FAILED:"
                    )

                    print(
                        "Only",
                        len(
                            neutral_samples
                        ),
                        "valid hand samples were detected."
                    )

                    print(
                        "Press C and try again with the hand "
                        "fully visible."
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
                        len(
                            neutral_samples
                        )
                    )

                    print(
                        "Neutral raw angle:",
                        f"{neutral_angle_deg:.3f}",
                        "deg"
                    )

                    print()

                    print(
                        "Press SPACE to start trial 1."
                    )

                    print()


        # =================================================
        # TRIAL RECORDING
        # =================================================

        if stage == "RECORDING":

            elapsed = (
                now
                -
                trial_start_time
            )


            row = {

                "trial_index":
                    completed_trials
                    +
                    1,

                "timestamp_sec":
                    round(
                        elapsed,
                        6,
                    ),

                "hand_detected":
                    1
                    if hand_detected
                    else
                    0,

                "raw_angle_deg":
                    (
                        round(
                            raw_angle_deg,
                            6,
                        )

                        if raw_angle_deg
                        is not None

                        else ""
                    ),

                "relative_angle_deg":
                    (
                        round(
                            relative_angle_deg,
                            6,
                        )

                        if relative_angle_deg
                        is not None

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

                        if relative_angle_deg
                        is not None

                        else ""
                    ),
            }


            trial_rows.append(
                row
            )


            if (
                elapsed
                >=
                TRIAL_DURATION_SEC
            ):

                trial_number = (
                    completed_trials
                    +
                    1
                )


                csv_path = (
                    save_trial_csv(
                        trial_number,
                        trial_rows,
                    )
                )


                summary = (
                    summarize_trial(

                        rows=
                            trial_rows,

                        neutral_angle_deg=
                            neutral_angle_deg,

                        mild_tolerance_deg=
                            MILD_TOLERANCE_DEG,

                        strong_tolerance_deg=
                            STRONG_TOLERANCE_DEG,
                    )
                )


                summary[
                    "trial_index"
                ] = trial_number


                summary[
                    "csv_path"
                ] = str(
                    csv_path
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
                    "Tracking ratio:",
                    (
                        f"{summary['tracking_ratio'] * 100:.1f}%"
                        if summary[
                            "tracking_ratio"
                        ]
                        is not None
                        else "N/A"
                    )
                )

                print(
                    "Mean abs error:",
                    summary[
                        "mean_abs_error_deg"
                    ],
                    "deg"
                )

                print(
                    "P90 abs error:",
                    summary[
                        "p90_abs_error_deg"
                    ],
                    "deg"
                )

                print(
                    "P95 abs error:",
                    summary[
                        "p95_abs_error_deg"
                    ],
                    "deg"
                )

                print(
                    "Max abs error:",
                    summary[
                        "max_abs_error_deg"
                    ],
                    "deg"
                )

                print(
                    "Frames > mild threshold:",
                    (
                        f"{summary['fraction_above_mild_tolerance'] * 100:.1f}%"
                        if summary[
                            "fraction_above_mild_tolerance"
                        ]
                        is not None
                        else "N/A"
                    )
                )

                print(
                    "Frames > strong threshold:",
                    (
                        f"{summary['fraction_above_strong_tolerance'] * 100:.1f}%"
                        if summary[
                            "fraction_above_strong_tolerance"
                        ]
                        is not None
                        else "N/A"
                    )
                )

                print(
                    "Mean frame jitter:",
                    summary[
                        "mean_frame_jitter_deg"
                    ],
                    "deg"
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
                    TARGET_TRIALS
                ):

                    stage = "COMPLETE"


                    pooled_summary = (
                        summarize_pooled(

                            all_rows=
                                all_trial_rows,

                            neutral_angle_deg=
                                neutral_angle_deg,

                            mild_tolerance_deg=
                                MILD_TOLERANCE_DEG,

                            strong_tolerance_deg=
                                STRONG_TOLERANCE_DEG,
                        )
                    )


                    (
                        json_path,
                        csv_summary_path,
                    ) = (
                        save_session_summary(

                            neutral_angle_deg=
                                neutral_angle_deg,

                            trial_summaries=
                                trial_summaries,

                            pooled_summary=
                                pooled_summary,
                        )
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
                        "POOLED RESULTS"
                    )

                    print(
                        "Tracking ratio:",
                        (
                            f"{pooled_summary['tracking_ratio'] * 100:.1f}%"
                            if pooled_summary[
                                "tracking_ratio"
                            ]
                            is not None
                            else "N/A"
                        )
                    )

                    print(
                        "Mean abs error:",
                        pooled_summary[
                            "mean_abs_error_deg"
                        ],
                        "deg"
                    )

                    print(
                        "P90 abs error:",
                        pooled_summary[
                            "p90_abs_error_deg"
                        ],
                        "deg"
                    )

                    print(
                        "P95 abs error:",
                        pooled_summary[
                            "p95_abs_error_deg"
                        ],
                        "deg"
                    )

                    print(
                        "Max abs error:",
                        pooled_summary[
                            "max_abs_error_deg"
                        ],
                        "deg"
                    )

                    print(
                        "Frames > mild threshold:",
                        (
                            f"{pooled_summary['fraction_above_mild_tolerance'] * 100:.1f}%"
                            if pooled_summary[
                                "fraction_above_mild_tolerance"
                            ]
                            is not None
                            else "N/A"
                        )
                    )

                    print(
                        "Frames > strong threshold:",
                        (
                            f"{pooled_summary['fraction_above_strong_tolerance'] * 100:.1f}%"
                            if pooled_summary[
                                "fraction_above_strong_tolerance"
                            ]
                            is not None
                            else "N/A"
                        )
                    )

                    print(
                        "Mean frame jitter:",
                        pooled_summary[
                            "mean_frame_jitter_deg"
                        ],
                        "deg"
                    )

                    print(
                        "P95 frame jitter:",
                        pooled_summary[
                            "p95_frame_jitter_deg"
                        ],
                        "deg"
                    )

                    print()

                    print(
                        "Summary JSON:"
                    )

                    print(
                        json_path
                    )

                    print()

                    print(
                        "Trial summary CSV:"
                    )

                    print(
                        csv_summary_path
                    )

                    print()

                    print(
                        "INTERPRETATION RULE:"
                    )

                    print(
                        "Do not change the 8/15-degree thresholds "
                        "just to make this calibration pass."
                    )

                    print(
                        "Use these results to decide whether the "
                        "current thresholds are compatible with "
                        "natural hand/sensor variability."
                    )

                    print()

                    print(
                        "NOTE:"
                    )

                    print(
                        "This is a technical calibration, not a "
                        "user study and not an estimate of optimal "
                        "orientation thresholds."
                    )

                    print(
                        "========================================================"
                    )

                    print()


                else:

                    stage = "IDLE"


                    print(
                        f"Press SPACE to start trial "
                        f"{completed_trials + 1}."
                    )

                    print()


        # =================================================
        # UI
        # =================================================

        trial_elapsed_sec = (

            (
                now
                -
                trial_start_time
            )

            if (
                stage == "RECORDING"
                and
                trial_start_time
                is not None
            )

            else 0.0
        )


        draw_status_panel(

            frame=
                frame,

            stage=
                stage,

            hand_detected=
                hand_detected,

            neutral_ready=
                neutral_ready,

            completed_trials=
                completed_trials,

            trial_elapsed_sec=
                trial_elapsed_sec,
        )


        # -------------------------------------------------
        # Draw a simple wrist -> middle-MCP orientation line
        # for visibility, but DO NOT show live numeric error
        # during recording because that could make the user
        # consciously chase the threshold.
        # -------------------------------------------------

        if hand_detected:

            landmarks = (
                result.hand_landmarks[0]
            )


            wrist = landmarks[
                0
            ]

            middle_mcp = landmarks[
                9
            ]


            height, width, _ = (
                frame.shape
            )


            wrist_px = (
                int(
                    wrist.x
                    *
                    width
                ),
                int(
                    wrist.y
                    *
                    height
                ),
            )


            middle_px = (
                int(
                    middle_mcp.x
                    *
                    width
                ),
                int(
                    middle_mcp.y
                    *
                    height
                ),
            )


            cv2.line(

                frame,

                wrist_px,

                middle_px,

                (
                    0,
                    220,
                    255,
                ),

                4,

                cv2.LINE_AA,
            )


            cv2.circle(

                frame,

                wrist_px,

                6,

                (
                    255,
                    255,
                    255,
                ),

                -1,
            )


            cv2.circle(

                frame,

                middle_px,

                6,

                (
                    0,
                    220,
                    255,
                ),

                -1,
            )


        cv2.imshow(
            "AdaptiveSkill - Hand Orientation Calibration",
            frame,
        )


        key = cv2.waitKey(
            1
        ) & 0xFF


        # =================================================
        # CONTROLS
        # =================================================

        if key in (
            ord(
                "q"
            ),
            ord(
                "Q"
            ),
        ):

            break


        if key in (
            ord(
                "r"
            ),
            ord(
                "R"
            ),
        ):

            print()

            print(
                "Session reset."
            )

            print(
                "Press C to recalibrate neutral orientation."
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

            all_trial_rows = []


            continue


        if key in (
            ord(
                "c"
            ),
            ord(
                "C"
            ),
        ):

            if (
                stage
                in
                (
                    "IDLE",
                    "COMPLETE",
                )
            ):

                if stage == "COMPLETE":

                    print()

                    print(
                        "Session already complete. Press R if "
                        "you want to start a new session."
                    )

                    print()

                    continue


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
                    "Hold your hand naturally upright."
                )

                print()


            continue


        if key == 32:

            if not neutral_ready:

                print()

                print(
                    "Neutral calibration is not ready."
                )

                print(
                    "Press C first."
                )

                print()

                continue


            if stage != "IDLE":

                continue


            if (
                completed_trials
                >=
                TARGET_TRIALS
            ):

                continue


            trial_rows = []

            trial_start_time = (
                time.perf_counter()
            )

            stage = "RECORDING"


            print()

            print(
                f"Trial {completed_trials + 1} started."
            )

            print(
                "Hold naturally upright for 5 seconds."
            )

            print(
                "Do not consciously chase the threshold."
            )

            print()


# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()
