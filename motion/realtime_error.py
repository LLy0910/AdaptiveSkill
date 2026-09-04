import cv2
import mediapipe as mp
import time
import math
import csv
from pathlib import Path
from collections import deque
from datetime import datetime

import numpy as np

from hesitation_detector import HesitationDetector
from online_motion_metrics import OnlineMotionMetrics
from assistance_policy import AssistancePolicy


# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "hand_landmarker.task"
)

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

SUMMARY_PATH = (
    TRIAL_DIR
    / "summary.csv"
)

TRIAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


if not MODEL_PATH.exists():
    print("ERROR: hand_landmarker.task not found.")
    raise SystemExit


if not REFERENCE_PATH.exists():
    print("ERROR: reference.csv not found.")
    raise SystemExit


# =========================================================
# 2. PARAMETERS
# =========================================================

CALIBRATION_FRAMES = 30

MIN_PALM_SIZE = 0.08
MAX_PALM_SIZE = 0.28

PALM_SIZE_TOLERANCE = 0.15

# unit = palm widths
START_POSITION_TOLERANCE = 0.35

SMOOTHING_WINDOW = 5
SPEED_SMOOTHING_WINDOW = 5

LEARNER_TRAIL_LENGTH = 600

MIN_VALID_TRIAL_FRAMES = 10

# Rotation-completion diagnostics.
# These gates only prevent unstable percentages very early in a trial.
# They are NOT assistance thresholds.
ROTATION_DIAGNOSTIC_MIN_PROGRESS = 0.25
ROTATION_DIAGNOSTIC_MIN_EXPECTED_DEG = 8.0


# =========================================================
# 3. TRIAL TYPES
# =========================================================

TRIAL_TYPES = {

    ord("1"): {
        "prefix": "A_correct",
        "label": "A - CORRECT"
    },

    ord("2"): {
        "prefix": "B_path_wrong",
        "label": "B - PATH WRONG"
    },

    ord("3"): {
        "prefix": "C_angle_wrong",
        "label": "C - ANGLE WRONG"
    }
}


# =========================================================
# 4. MEDIAPIPE SETTINGS
# =========================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),

    running_mode=VisionRunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


# =========================================================
# 5. BASIC FUNCTIONS
# =========================================================

def distance_xy(
    x1,
    y1,
    x2,
    y2
):
    dx = x1 - x2
    dy = y1 - y2

    return math.sqrt(
        dx * dx
        +
        dy * dy
    )


def landmark_distance(
    p1,
    p2
):
    return distance_xy(
        p1.x,
        p1.y,
        p2.x,
        p2.y
    )


def circular_angle_difference(
    angle1,
    angle2
):
    return (
        angle1
        -
        angle2
        +
        180
    ) % 360 - 180


# =========================================================
# 6. UI
# =========================================================

def get_ui_scale(frame):

    height, width, _ = frame.shape

    scale = min(
        width / 1280.0,
        height / 720.0
    )

    return max(
        0.65,
        min(scale, 1.6)
    )


def put_text(
    frame,
    text,
    position,
    scale,
    font_scale=0.60,
    thickness=2,
    color=(255, 255, 255)
):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * scale,
        color,
        max(
            1,
            int(thickness * scale)
        ),
        cv2.LINE_AA
    )


# =========================================================
# 7. LOAD REFERENCE
# =========================================================

def load_reference():

    reference = []

    with open(
        REFERENCE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        required_columns = [
            "timestamp",
            "x_raw",
            "y_raw",
            "x_norm",
            "y_norm",
            "angle_raw",
            "angle_rel",
            "palm_size"
        ]

        if reader.fieldnames is None:
            print("ERROR: reference.csv is empty.")
            raise SystemExit

        missing = [
            column
            for column in required_columns
            if column not in reader.fieldnames
        ]

        if missing:
            print(
                "ERROR: reference.csv "
                "is not v1.2 format."
            )

            print(
                "Missing:",
                missing
            )

            raise SystemExit

        for row in reader:

            reference.append({

                "timestamp":
                    float(row["timestamp"]),

                "x_raw":
                    float(row["x_raw"]),

                "y_raw":
                    float(row["y_raw"]),

                "x_norm":
                    float(row["x_norm"]),

                "y_norm":
                    float(row["y_norm"]),

                "angle_raw":
                    float(row["angle_raw"]),

                "angle_rel":
                    float(row["angle_rel"]),

                "palm_size":
                    float(row["palm_size"])
            })

    if len(reference) < 2:

        print(
            "ERROR: Reference is too short."
        )

        raise SystemExit

    print(
        f"Loaded reference: "
        f"{len(reference)} frames"
    )

    return reference


# =========================================================
# 8. CALIBRATION
# =========================================================

def calculate_calibration(samples):

    start_x = float(
        np.median([
            sample["x"]
            for sample in samples
        ])
    )

    start_y = float(
        np.median([
            sample["y"]
            for sample in samples
        ])
    )

    palm_size = float(
        np.median([
            sample["palm_size"]
            for sample in samples
        ])
    )

    angles = np.array(
        [
            sample["angle"]
            for sample in samples
        ],
        dtype=float
    )

    angles_unwrapped = (
        np.rad2deg(
            np.unwrap(
                np.deg2rad(
                    angles
                )
            )
        )
    )

    start_angle = float(
        np.median(
            angles_unwrapped
        )
    )

    return {
        "start_x":
            start_x,

        "start_y":
            start_y,

        "palm_size":
            palm_size,

        "start_angle":
            start_angle
    }


# =========================================================
# 9. SCALE CHECK
# =========================================================

def check_scale_against_calibration(
    current_palm_size,
    calibration
):

    baseline = (
        calibration["palm_size"]
    )

    lower_bound = (
        baseline
        *
        (
            1
            -
            PALM_SIZE_TOLERANCE
        )
    )

    upper_bound = (
        baseline
        *
        (
            1
            +
            PALM_SIZE_TOLERANCE
        )
    )

    if current_palm_size < lower_bound:
        return "TOO_FAR"

    if current_palm_size > upper_bound:
        return "TOO_CLOSE"

    return "READY"


# =========================================================
# 10. START POSITION ERROR
# =========================================================

def get_start_position_error(
    current_x,
    current_y,
    calibration
):

    scale = (
        calibration["palm_size"]
    )

    if scale < 1e-6:
        return None

    raw_distance = distance_xy(

        current_x,
        current_y,

        calibration["start_x"],
        calibration["start_y"]
    )

    return (
        raw_distance
        /
        scale
    )


# =========================================================
# 11. PATH ERROR
# =========================================================

def find_nearest_reference_point(
    x_norm,
    y_norm,
    reference
):

    reference_xy = np.array(

        [
            [
                point["x_norm"],
                point["y_norm"]
            ]
            for point in reference
        ],

        dtype=float
    )

    current_point = np.array(
        [
            x_norm,
            y_norm
        ],
        dtype=float
    )

    distances = np.linalg.norm(
        reference_xy
        -
        current_point,
        axis=1
    )

    nearest_index = int(
        np.argmin(distances)
    )

    path_error = float(
        distances[
            nearest_index
        ]
    )

    return (
        path_error,
        nearest_index
    )


# =========================================================
# 12. REFERENCE MAP BOUNDS
# =========================================================

def calculate_reference_bounds(
    reference
):

    xs = np.array(
        [
            point["x_norm"]
            for point in reference
        ]
    )

    ys = np.array(
        [
            point["y_norm"]
            for point in reference
        ]
    )

    min_x = float(xs.min())
    max_x = float(xs.max())

    min_y = float(ys.min())
    max_y = float(ys.max())

    range_x = max(
        max_x - min_x,
        0.1
    )

    range_y = max(
        max_y - min_y,
        0.1
    )

    padding_x = max(
        0.35,
        range_x * 0.20
    )

    padding_y = max(
        0.35,
        range_y * 0.20
    )

    return (
        min_x - padding_x,
        max_x + padding_x,
        min_y - padding_y,
        max_y + padding_y
    )


# =========================================================
# 13. START MARKER
# =========================================================

def draw_start_marker(
    frame,
    calibration
):

    if calibration is None:
        return

    height, width, _ = frame.shape

    ui = get_ui_scale(
        frame
    )

    px = int(
        calibration["start_x"]
        *
        width
    )

    py = int(
        calibration["start_y"]
        *
        height
    )

    radius = max(
        10,
        int(
            16
            *
            ui
        )
    )

    cv2.circle(
        frame,
        (px, py),
        radius,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.circle(
        frame,
        (px, py),
        max(
            4,
            int(
                5
                *
                ui
            )
        ),
        (255, 255, 255),
        -1,
        cv2.LINE_AA
    )

    put_text(
        frame,
        "START",
        (
            min(
                px + int(22 * ui),
                width - int(90 * ui)
            ),

            max(
                py - int(10 * ui),
                int(25 * ui)
            )
        ),
        ui,
        0.52,
        2
    )


# =========================================================
# 14. MOTION MAP
# =========================================================

def draw_motion_map(
    frame,
    reference,
    learner_trail,
    current_norm,
    bounds,
    mode
):

    height, width, _ = frame.shape

    ui = get_ui_scale(
        frame
    )

    panel_width = max(
        int(
            width
            *
            0.34
        ),
        int(
            220
            *
            ui
        )
    )

    panel_height = max(
        int(
            height
            *
            0.32
        ),
        int(
            150
            *
            ui
        )
    )

    margin = max(
        10,
        int(
            14
            *
            ui
        )
    )

    x0 = (
        width
        -
        panel_width
        -
        margin
    )

    y0 = (
        height
        -
        panel_height
        -
        margin
    )

    x1 = (
        x0
        +
        panel_width
    )

    y1 = (
        y0
        +
        panel_height
    )


    # -----------------------------------------------------
    # Background
    # -----------------------------------------------------

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x0, y0),
        (x1, y1),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.62,
        frame,
        0.38,
        0,
        frame
    )

    cv2.rectangle(
        frame,
        (x0, y0),
        (x1, y1),
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )


    # -----------------------------------------------------
    # Title
    # -----------------------------------------------------

    if mode == "READY":

        title = (
            "MOTION SPACE - START / END"
        )

    elif mode == "PRACTICING":

        title = (
            "LIVE MOTION - REFERENCE HIDDEN"
        )

    else:

        title = (
            "POST-TRIAL COMPARISON"
        )


    put_text(
        frame,
        title,
        (
            x0
            +
            int(
                10
                *
                ui
            ),

            y0
            +
            int(
                22
                *
                ui
            )
        ),
        ui,
        0.40,
        1
    )


    graph_left = (
        x0
        +
        int(
            18
            *
            ui
        )
    )

    graph_right = (
        x1
        -
        int(
            18
            *
            ui
        )
    )

    graph_top = (
        y0
        +
        int(
            36
            *
            ui
        )
    )

    graph_bottom = (
        y1
        -
        int(
            18
            *
            ui
        )
    )


    (
        min_x,
        max_x,
        min_y,
        max_y
    ) = bounds


    range_x = max(
        max_x - min_x,
        0.001
    )

    range_y = max(
        max_y - min_y,
        0.001
    )


    available_width = (
        graph_right
        -
        graph_left
    )

    available_height = (
        graph_bottom
        -
        graph_top
    )


    # X/Y same scale
    map_scale = min(

        available_width
        /
        range_x,

        available_height
        /
        range_y
    )


    drawing_width = (
        range_x
        *
        map_scale
    )

    drawing_height = (
        range_y
        *
        map_scale
    )


    offset_x = (

        graph_left

        +

        (
            available_width
            -
            drawing_width
        )
        /
        2
    )


    offset_y = (

        graph_top

        +

        (
            available_height
            -
            drawing_height
        )
        /
        2
    )


    def convert(
        x,
        y
    ):

        px = int(

            offset_x

            +

            (
                x
                -
                min_x
            )

            *

            map_scale
        )


        py = int(

            offset_y

            +

            (
                y
                -
                min_y
            )

            *

            map_scale
        )


        return (
            px,
            py
        )


    # -----------------------------------------------------
    # START / END
    # -----------------------------------------------------

    ref_start = convert(

        reference[0]["x_norm"],

        reference[0]["y_norm"]
    )


    ref_end = convert(

        reference[-1]["x_norm"],

        reference[-1]["y_norm"]
    )


    cv2.circle(
        frame,
        ref_start,
        max(
            3,
            int(
                5
                *
                ui
            )
        ),
        (255, 255, 255),
        -1,
        cv2.LINE_AA
    )


    cv2.circle(
        frame,
        ref_end,
        max(
            4,
            int(
                6
                *
                ui
            )
        ),
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # -----------------------------------------------------
    # Reference:
    # only visible after trial
    # -----------------------------------------------------

    if mode == "REVIEW":

        for i in range(
            1,
            len(reference)
        ):

            p1 = convert(

                reference[
                    i - 1
                ]["x_norm"],

                reference[
                    i - 1
                ]["y_norm"]
            )

            p2 = convert(

                reference[
                    i
                ]["x_norm"],

                reference[
                    i
                ]["y_norm"]
            )

            cv2.line(
                frame,
                p1,
                p2,
                (170, 170, 170),
                max(
                    1,
                    int(
                        2
                        *
                        ui
                    )
                ),
                cv2.LINE_AA
            )


    # -----------------------------------------------------
    # Learner trail
    # -----------------------------------------------------

    if len(
        learner_trail
    ) >= 2:

        for i in range(
            1,
            len(
                learner_trail
            )
        ):

            p1 = convert(

                learner_trail[
                    i - 1
                ][0],

                learner_trail[
                    i - 1
                ][1]
            )

            p2 = convert(

                learner_trail[
                    i
                ][0],

                learner_trail[
                    i
                ][1]
            )

            cv2.line(
                frame,
                p1,
                p2,
                (255, 255, 255),
                max(
                    1,
                    int(
                        2
                        *
                        ui
                    )
                ),
                cv2.LINE_AA
            )


    # -----------------------------------------------------
    # Current learner point
    # -----------------------------------------------------

    if (
        mode == "PRACTICING"
        and
        current_norm is not None
    ):

        (
            current_x,
            current_y
        ) = current_norm


        clipped_x = min(
            max(
                current_x,
                min_x
            ),
            max_x
        )

        clipped_y = min(
            max(
                current_y,
                min_y
            ),
            max_y
        )


        current_point = convert(
            clipped_x,
            clipped_y
        )


        cv2.circle(
            frame,
            current_point,
            max(
                5,
                int(
                    7
                    *
                    ui
                )
            ),
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


    # -----------------------------------------------------
    # Review legend
    # -----------------------------------------------------

    if mode == "REVIEW":

        put_text(
            frame,
            (
                "Gray=Reference | "
                "White=Your Motion"
            ),
            (
                x0
                +
                int(
                    10
                    *
                    ui
                ),

                y1
                -
                int(
                    8
                    *
                    ui
                )
            ),
            ui,
            0.28,
            1
        )


# =========================================================
# 15. UNIQUE TRIAL FILE NAME
# =========================================================

def get_next_trial_path(
    prefix
):

    timestamp = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
    )

    return (

        TRIAL_DIR

        /

        f"{prefix}_{timestamp}.csv"
    )


# =========================================================
# 16. QUICK SUMMARY
#     does NOT write summary.csv
# =========================================================

def calculate_trial_summary(
    records,
    trial_id,
    condition
):

    valid_records = [

        row

        for row in records

        if row["tracking"] == 1
    ]


    if len(valid_records) == 0:
        return None


    path_errors = np.array(
        [
            float(
                row["path_error"]
            )
            for row in valid_records
        ]
    )


    angle_errors = np.array(
        [
            float(
                row["angle_error"]
            )
            for row in valid_records
        ]
    )


    speeds = np.array(
        [
            float(
                row["speed"]
            )
            for row in valid_records
        ]
    )


    # Optional rotation diagnostics.
    # Older trial CSVs may not contain these fields, so keep
    # the summary backwards compatible.
    rotation_deficits = [
        float(row["rotation_deficit_deg"])
        for row in valid_records
        if row.get("rotation_deficit_deg") not in (None, "")
    ]

    rotation_completions = [
        float(row["rotation_completion_ratio"])
        for row in valid_records
        if row.get("rotation_completion_ratio") not in (None, "")
    ]


    # Optional assistance diagnostics.
    # Older trial CSVs do not contain these fields.
    assistance_levels = [
        int(row["assistance_level"])
        for row in valid_records
        if row.get("assistance_level") not in (None, "")
    ]


    duration = (

        float(
            records[-1]["timestamp"]
        )

        -

        float(
            records[0]["timestamp"]
        )
    )


    tracking_ratio = (

        len(valid_records)

        /

        len(records)
    )


    return {

        "trial_id":
            trial_id,

        "condition":
            condition,

        "duration_sec":
            round(
                duration,
                4
            ),

        "total_frames":
            len(records),

        "tracked_frames":
            len(valid_records),

        "tracking_ratio":
            round(
                tracking_ratio,
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

        "mean_angle_error":
            round(
                float(
                    np.mean(
                        angle_errors
                    )
                ),
                4
            ),

        "median_angle_error":
            round(
                float(
                    np.median(
                        angle_errors
                    )
                ),
                4
            ),

        "p90_angle_error":
            round(
                float(
                    np.percentile(
                        angle_errors,
                        90
                    )
                ),
                4
            ),

        "mean_speed":
            round(
                float(
                    np.mean(
                        speeds
                    )
                ),
                6
            ),

        "final_rotation_deficit_deg":
            (
                round(
                    float(rotation_deficits[-1]),
                    4
                )
                if rotation_deficits
                else ""
            ),

        "final_rotation_completion_pct":
            (
                round(
                    float(rotation_completions[-1]) * 100.0,
                    1
                )
                if rotation_completions
                else ""
            ),

        "max_assistance_level":
            (
                max(assistance_levels)
                if assistance_levels
                else ""
            )
    }


# =========================================================
# 17. SAVE ONE TRIAL
#
# IMPORTANT:
# S only saves this CSV.
# It does NOT update summary.csv.
# =========================================================

def save_trial(
    records,
    trial_type
):

    valid_records = [

        row

        for row in records

        if row["tracking"] == 1
    ]


    if (
        len(valid_records)
        <
        MIN_VALID_TRIAL_FRAMES
    ):

        print()
        print(
            "Trial too short."
        )

        print(
            "Trial NOT saved."
        )

        print()

        return (
            None,
            None
        )


    trial_path = (
        get_next_trial_path(
            trial_type["prefix"]
        )
    )


    trial_id = (
        trial_path.stem
    )


    fieldnames = [

        "trial_id",
        "condition",

        "timestamp",

        "tracking",

        "x_raw",
        "y_raw",

        "x_norm",
        "y_norm",

        "angle_raw",
        "angle_rel",

        "palm_size",

        "path_error",
        "angle_error",

        "speed",

        "nearest_reference_index",
        "reference_progress",

        "learner_relative_rotation",
        "expected_relative_rotation",
        "rotation_deficit_deg",
        "rotation_completion_ratio",

        "hesitation_state",
        "hesitation_event",
        "hesitation_duration",
        "hesitation_progress_delta",

        "assistance_level",
        "assistance_label",
        "assistance_reason",
        "assistance_cue",
        "assistance_dominant_error",
        "assistance_mild_duration",
        "assistance_strong_duration"
    ]


    with open(
        trial_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()


        for row in records:

            output_row = {
                "trial_id":
                    trial_id,

                "condition":
                    trial_type["label"]
            }

            output_row.update(
                row
            )

            writer.writerow(
                output_row
            )


    quick_summary = (
        calculate_trial_summary(
            records,
            trial_id,
            trial_type["label"]
        )
    )


    print()
    print(
        "======================================"
    )

    print(
        "TRIAL SAVED"
    )

    print(
        "======================================"
    )

    print(
        "Trial ID:"
    )

    print(
        trial_id
    )

    print(
        "Condition:",
        trial_type["label"]
    )


    if quick_summary is not None:

        print()

        print(
            "Mean Path Error:",
            quick_summary[
                "mean_path_error"
            ]
        )

        print(
            "Mean Angle Error:",
            quick_summary[
                "mean_angle_error"
            ],
            "deg"
        )

        print(
            "Tracking:",
            f"{quick_summary['tracking_ratio'] * 100:.1f}%"
        )


    print()

    print(
        "Saved:",
        trial_path
    )

    print()

    print(
        "Satisfied?"
    )

    print(
        "Keep it = do nothing"
    )

    print(
        "Technical failure = press D to delete it"
    )

    print()

    print(
        "Summary has NOT been generated yet."
    )

    print(
        "Press T only after all trials are ready."
    )

    print()


    return (
        trial_path,
        quick_summary
    )


# =========================================================
# 18. READ A SAVED TRIAL
# =========================================================

def read_trial_file(
    path
):

    records = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            try:

                tracking = int(
                    row["tracking"]
                )

            except Exception:

                continue


            record = {

                "tracking":
                    tracking,

                "timestamp":
                    float(
                        row["timestamp"]
                    )
            }


            if tracking == 1:

                try:

                    record[
                        "path_error"
                    ] = float(
                        row["path_error"]
                    )

                    record[
                        "angle_error"
                    ] = float(
                        row["angle_error"]
                    )

                    record[
                        "speed"
                    ] = float(
                        row["speed"]
                    )

                    rotation_deficit_value = row.get(
                        "rotation_deficit_deg",
                        ""
                    )

                    rotation_completion_value = row.get(
                        "rotation_completion_ratio",
                        ""
                    )

                    record["rotation_deficit_deg"] = (
                        float(rotation_deficit_value)
                        if rotation_deficit_value not in (None, "")
                        else ""
                    )

                    record["rotation_completion_ratio"] = (
                        float(rotation_completion_value)
                        if rotation_completion_value not in (None, "")
                        else ""
                    )

                except Exception:

                    continue


            else:

                record[
                    "path_error"
                ] = ""

                record[
                    "angle_error"
                ] = ""

                record[
                    "speed"
                ] = ""

                record[
                    "rotation_deficit_deg"
                ] = ""

                record[
                    "rotation_completion_ratio"
                ] = ""


            records.append(
                record
            )


    return records


# =========================================================
# 19. INFER CONDITION FROM FILE
# =========================================================

def infer_condition(
    path
):

    name = (
        path.name
    )


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
# 20. GENERATE FINAL SUMMARY
#
# T scans CURRENT CSV files only.
# Old summary is completely rebuilt.
# =========================================================

def generate_final_summary():

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
        key=lambda p: p.name
    )


    if len(trial_files) == 0:

        print()
        print(
            "No trial CSV files found."
        )
        print()

        return


    summaries = []


    for path in trial_files:

        records = (
            read_trial_file(
                path
            )
        )


        if len(records) == 0:
            continue


        condition = (
            infer_condition(
                path
            )
        )


        summary = (
            calculate_trial_summary(
                records,
                path.stem,
                condition
            )
        )


        if summary is not None:

            summaries.append(
                summary
            )


    if len(summaries) == 0:

        print()
        print(
            "No valid trials found."
        )
        print()

        return


    # -----------------------------------------------------
    # Completely rebuild summary.csv
    # -----------------------------------------------------

    fieldnames = [

        "trial_id",
        "condition",

        "duration_sec",

        "total_frames",
        "tracked_frames",
        "tracking_ratio",

        "mean_path_error",
        "median_path_error",
        "p90_path_error",

        "mean_angle_error",
        "median_angle_error",
        "p90_angle_error",

        "mean_speed",

        "final_rotation_deficit_deg",
        "final_rotation_completion_pct",

        "max_assistance_level"
    ]


    with open(
        SUMMARY_PATH,
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
            summaries
        )


    # -----------------------------------------------------
    # Counts
    # -----------------------------------------------------

    correct_count = sum(
        1
        for item in summaries
        if item["condition"]
        ==
        "A - CORRECT"
    )


    path_wrong_count = sum(
        1
        for item in summaries
        if item["condition"]
        ==
        "B - PATH WRONG"
    )


    angle_wrong_count = sum(
        1
        for item in summaries
        if item["condition"]
        ==
        "C - ANGLE WRONG"
    )


    print()
    print(
        "======================================"
    )

    print(
        "FINAL SUMMARY GENERATED"
    )

    print(
        "======================================"
    )

    print(
        "Correct:",
        correct_count
    )

    print(
        "Path Wrong:",
        path_wrong_count
    )

    print(
        "Angle Wrong:",
        angle_wrong_count
    )

    print(
        "Total:",
        len(summaries)
    )

    print()

    print(
        "Saved:",
        SUMMARY_PATH
    )

    print()


# =========================================================
# 21. CAMERA
# =========================================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)


if not cap.isOpened():

    print(
        "ERROR: Camera cannot be opened."
    )

    raise SystemExit


# =========================================================
# 22. STATE
# =========================================================

reference = (
    load_reference()
)


reference_bounds = (
    calculate_reference_bounds(
        reference
    )
)


# Online progress + relative-angle metrics
online_motion_metrics = OnlineMotionMetrics(
    reference
)

current_online_metric_result = None


# Hesitation detector
hesitation_detector = HesitationDetector()

current_hesitation_result = None


# Adaptive assistance policy
#
# The policy loads data-derived motion thresholds from
# data/assistance_config.json and maintains Level 0-3 state.
assistance_policy = AssistancePolicy()

current_assistance_result = None


# Calibration
calibrating = False

calibration_ready = False

calibration_samples = []

calibration = None


# Trial
practicing = False

trial_finished = False

selected_trial_type = None


trial_records = []

trial_start_time = None


# Last saved trial
last_saved_trial_path = None

last_review_summary = None


# Smoothing
x_history = deque(
    maxlen=
        SMOOTHING_WINDOW
)

y_history = deque(
    maxlen=
        SMOOTHING_WINDOW
)

speed_history = deque(
    maxlen=
        SPEED_SMOOTHING_WINDOW
)


# Learner trail
learner_trail = deque(
    maxlen=
        LEARNER_TRAIL_LENGTH
)


# Speed state
previous_norm_x = None

previous_norm_y = None

previous_time = None

current_speed = 0.0


# Error state
current_path_error = None

current_angle_error = None

current_nearest_index = None

current_reference_progress = None

current_learner_relative_rotation = None

current_expected_relative_rotation = None

current_rotation_deficit = None

current_rotation_completion_ratio = None


print()
print(
    "=========================================="
)

print(
    "AdaptiveSkill - Motion Evaluation"
)

print(
    "Adaptive Assistance Workflow"
)

print()

print(
    "1 = A Correct"
)

print(
    "2 = B Path Wrong"
)

print(
    "3 = C Angle Wrong"
)

print()

print(
    "C = Calibrate"
)

print(
    "R = Start Trial"
)

print(
    "S = Stop + Save Trial"
)

print(
    "D = Delete Last Saved Trial"
)

print(
    "T = Generate Final Summary"
)

print(
    "Q = Quit"
)

print()

print(
    "Hesitation detector:"
)

print(
    f"speed < {hesitation_detector.speed_threshold:.4f} | "
    f"hold >= {hesitation_detector.hold_duration:.3f}s | "
    f"progress delta <= {hesitation_detector.max_progress_delta:.3f}"
)

print(
    "=========================================="
)

print()


# =========================================================
# 23. MAIN LOOP
# =========================================================

with HandLandmarker.create_from_options(
    options
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
            1
        )


        height, width, _ = (
            frame.shape
        )


        ui = get_ui_scale(
            frame
        )


        # =================================================
        # MEDIAPIPE
        # =================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(

            image_format=
                mp.ImageFormat.SRGB,

            data=
                rgb_frame
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
                timestamp_ms
            )
        )


        # Current frame
        hand_detected = False


        current_x_raw = None
        current_y_raw = None


        current_x_norm = None
        current_y_norm = None


        current_angle = None
        current_angle_rel = None


        current_palm_size = None


        start_position_error = None

        current_reference_progress = None

        current_learner_relative_rotation = None

        current_expected_relative_rotation = None

        current_rotation_deficit = None

        current_rotation_completion_ratio = None


        # =================================================
        # HAND DETECTED
        # =================================================

        if result.hand_landmarks:


            hand_detected = True


            landmarks = (
                result.hand_landmarks[0]
            )


            wrist = landmarks[0]

            index_mcp = landmarks[5]

            index_tip = landmarks[8]

            middle_mcp = landmarks[9]

            pinky_mcp = landmarks[17]


            # Raw XY
            current_x_raw = (
                index_tip.x
            )

            current_y_raw = (
                index_tip.y
            )


            # Raw angle
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


            current_angle = (
                math.degrees(
                    math.atan2(
                        dy,
                        dx
                    )
                )
            )


            # Palm size
            current_palm_size = (
                landmark_distance(
                    index_mcp,
                    pinky_mcp
                )
            )


            # =================================================
            # CALIBRATION
            # =================================================

            if calibrating:


                calibration_samples.append({

                    "x":
                        current_x_raw,

                    "y":
                        current_y_raw,

                    "angle":
                        current_angle,

                    "palm_size":
                        current_palm_size
                })


                progress = int(

                    len(
                        calibration_samples
                    )

                    /

                    CALIBRATION_FRAMES

                    *

                    100
                )


                progress = min(
                    progress,
                    100
                )


                put_text(
                    frame,
                    f"CALIBRATING {progress}%",
                    (
                        int(
                            20
                            *
                            ui
                        ),

                        int(
                            150
                            *
                            ui
                        )
                    ),
                    ui,
                    0.62,
                    2
                )


                if (
                    len(
                        calibration_samples
                    )
                    >=
                    CALIBRATION_FRAMES
                ):


                    calibrating = False


                    calibration = (
                        calculate_calibration(
                            calibration_samples
                        )
                    )


                    calibration_ready = True


                    trial_finished = False


                    print()
                    print(
                        "CALIBRATION COMPLETE"
                    )

                    print(
                        "Return fingertip to START "
                        "before every trial."
                    )

                    print()


            # =================================================
            # CALIBRATED
            # =================================================

            if (
                calibration_ready
                and
                calibration is not None
            ):


                start_position_error = (
                    get_start_position_error(

                        current_x_raw,

                        current_y_raw,

                        calibration
                    )
                )


                # =================================================
                # PRACTICE
                # =================================================

                if practicing:


                    x_history.append(
                        current_x_raw
                    )


                    y_history.append(
                        current_y_raw
                    )


                    smooth_x = float(
                        np.mean(
                            x_history
                        )
                    )


                    smooth_y = float(
                        np.mean(
                            y_history
                        )
                    )


                    scale = (
                        calibration[
                            "palm_size"
                        ]
                    )


                    current_x_norm = (

                        smooth_x

                        -

                        calibration[
                            "start_x"
                        ]

                    ) / scale


                    current_y_norm = (

                        smooth_y

                        -

                        calibration[
                            "start_y"
                        ]

                    ) / scale


                    current_angle_rel = (
                        circular_angle_difference(

                            current_angle,

                            calibration[
                                "start_angle"
                            ]
                        )
                    )


                    # PATH ERROR
                    #
                    # Keep the validated global nearest-path distance
                    # as the path-quality metric.
                    (
                        current_path_error,
                        _
                    ) = (
                        find_nearest_reference_point(

                            current_x_norm,

                            current_y_norm,

                            reference
                        )
                    )


                    # ONLINE PROGRESS + RELATIVE ANGLE ERROR
                    #
                    # Progress is monotonic and angle error compares
                    # learner relative rotation against the expected
                    # reference rotation at the current progress.
                    current_online_metric_result = (
                        online_motion_metrics.update(

                            x_norm=
                                current_x_norm,

                            y_norm=
                                current_y_norm,

                            raw_angle=
                                current_angle,

                            tracking=True
                        )
                    )


                    current_nearest_index = (
                        current_online_metric_result[
                            "reference_index"
                        ]
                    )


                    current_reference_progress = (
                        current_online_metric_result[
                            "progress"
                        ]
                    )


                    current_angle_error = (
                        current_online_metric_result[
                            "relative_angle_error"
                        ]
                    )


                    current_learner_relative_rotation = (
                        current_online_metric_result[
                            "learner_relative_rotation"
                        ]
                    )


                    current_expected_relative_rotation = (
                        current_online_metric_result[
                            "expected_relative_rotation"
                        ]
                    )


                    # ROTATION DIAGNOSTICS
                    #
                    # Positive deficit means the learner has rotated
                    # less than expected in the reference direction.
                    # Completion ratio is only reported after the
                    # expected rotation is large enough to be stable.
                    reference_total_rotation = float(
                        online_motion_metrics.
                        reference_relative_rotation[-1]
                    )

                    rotation_direction = (
                        1.0
                        if reference_total_rotation >= 0.0
                        else -1.0
                    )

                    expected_aligned = (
                        current_expected_relative_rotation
                        * rotation_direction
                    )

                    learner_aligned = (
                        current_learner_relative_rotation
                        * rotation_direction
                    )

                    current_rotation_deficit = (
                        expected_aligned
                        - learner_aligned
                    )

                    if (
                        current_reference_progress
                        >= ROTATION_DIAGNOSTIC_MIN_PROGRESS
                        and
                        expected_aligned
                        >= ROTATION_DIAGNOSTIC_MIN_EXPECTED_DEG
                    ):

                        current_rotation_completion_ratio = (
                            learner_aligned
                            / expected_aligned
                        )

                    else:

                        current_rotation_completion_ratio = None


                    # SPEED
                    now = (
                        time.perf_counter()
                    )


                    if (
                        previous_norm_x
                        is not None
                        and
                        previous_norm_y
                        is not None
                        and
                        previous_time
                        is not None
                    ):


                        dt = (
                            now
                            -
                            previous_time
                        )


                        if dt > 1e-6:


                            movement = (
                                distance_xy(

                                    current_x_norm,

                                    current_y_norm,

                                    previous_norm_x,

                                    previous_norm_y
                                )
                            )


                            instant_speed = (
                                movement
                                /
                                dt
                            )


                            speed_history.append(
                                instant_speed
                            )


                            current_speed = (
                                float(
                                    np.mean(
                                        speed_history
                                    )
                                )
                            )


                    previous_norm_x = (
                        current_x_norm
                    )

                    previous_norm_y = (
                        current_y_norm
                    )

                    previous_time = (
                        now
                    )


                    # HESITATION
                    hesitation_timestamp = (

                        time.perf_counter()

                        -

                        trial_start_time
                    )


                    current_hesitation_result = (
                        hesitation_detector.update(

                            timestamp=
                                hesitation_timestamp,

                            speed=
                                current_speed,

                            progress=
                                current_reference_progress,

                            tracking=True
                        )
                    )


                    # ADAPTIVE ASSISTANCE
                    #
                    # Feed the current online motion evidence and
                    # hesitation state into the explainable Level 0-3
                    # policy. The policy itself handles persistence,
                    # recovery, and escalation.
                    current_assistance_result = (
                        assistance_policy.update(

                            timestamp=
                                hesitation_timestamp,

                            path_error=
                                current_path_error,

                            relative_angle_error=
                                current_angle_error,

                            progress=
                                current_reference_progress,

                            learner_rotation=
                                current_learner_relative_rotation,

                            expected_rotation=
                                current_expected_relative_rotation,

                            hesitation_result=
                                current_hesitation_result,

                            tracking=True
                        )
                    )


                    if current_hesitation_result[
                        "event"
                    ]:

                        print()

                        print(
                            "HESITATION DETECTED"
                        )

                        print(
                            f"Time: "
                            f"{hesitation_timestamp:.2f}s"
                        )

                        print(
                            f"Progress: "
                            f"{current_reference_progress:.3f}"
                        )

                        print(
                            f"Low-speed hold: "
                            f"{current_hesitation_result['candidate_duration']:.3f}s"
                        )

                        print()


                    learner_trail.append(
                        (
                            current_x_norm,
                            current_y_norm
                        )
                    )


            # Fingertip
            fingertip_x = int(
                current_x_raw
                *
                width
            )

            fingertip_y = int(
                current_y_raw
                *
                height
            )


            cv2.circle(
                frame,
                (
                    fingertip_x,
                    fingertip_y
                ),
                max(
                    6,
                    int(
                        9
                        *
                        ui
                    )
                ),
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )


        # =================================================
        # TRACKING LOST
        # =================================================

        else:


            if practicing:

                hesitation_timestamp = (

                    time.perf_counter()

                    -

                    trial_start_time
                )


                current_hesitation_result = (
                    hesitation_detector.update(

                        timestamp=
                            hesitation_timestamp,

                        speed=0.0,

                        progress=0.0,

                        tracking=False
                    )
                )


                current_assistance_result = (
                    assistance_policy.update(

                        timestamp=
                            hesitation_timestamp,

                        path_error=0.0,

                        relative_angle_error=0.0,

                        progress=0.0,

                        learner_rotation=0.0,

                        expected_rotation=0.0,

                        hesitation_result=
                            current_hesitation_result,

                        tracking=False
                    )
                )


            # TRACKING LOST is useful only during an active trial.
            # Never draw it on the post-trial review screen.
            if practicing:

                put_text(
                    frame,
                    "TRACKING LOST",
                    (
                        int(
                            20
                            *
                            ui
                        ),

                        int(
                            60
                            *
                            ui
                        )
                    ),
                    ui,
                    0.82,
                    3
                )


        # =================================================
        # RECORD CURRENT FRAME
        # =================================================

        if practicing:


            elapsed = (

                time.perf_counter()

                -

                trial_start_time
            )


            if (
                hand_detected

                and

                current_x_norm
                is not None

                and

                current_path_error
                is not None

                and

                current_angle_error
                is not None
            ):


                reference_progress = (
                    current_reference_progress
                )


                hesitation_state = (
                    current_hesitation_result["state"]
                    if current_hesitation_result is not None
                    else "NORMAL"
                )

                hesitation_event = (
                    int(
                        current_hesitation_result["event"]
                    )
                    if current_hesitation_result is not None
                    else 0
                )

                hesitation_duration = (
                    current_hesitation_result[
                        "candidate_duration"
                    ]
                    if current_hesitation_result is not None
                    else 0.0
                )

                hesitation_progress_delta = (
                    current_hesitation_result[
                        "progress_delta"
                    ]
                    if current_hesitation_result is not None
                    else 0.0
                )


                if current_assistance_result is None:

                    assistance_level = 0
                    assistance_label = "OBSERVE"
                    assistance_reason = "Within calibrated baseline"
                    assistance_cue = "No assistance"
                    assistance_dominant_error = "NONE"
                    assistance_mild_duration = 0.0
                    assistance_strong_duration = 0.0

                else:

                    assistance_level = int(
                        current_assistance_result["level"]
                    )

                    assistance_label = (
                        current_assistance_result["label"]
                    )

                    assistance_reason = (
                        current_assistance_result["reason"]
                    )

                    assistance_cue = (
                        current_assistance_result["cue"]
                    )

                    assistance_dominant_error = (
                        current_assistance_result["dominant_error"]
                    )

                    assistance_mild_duration = (
                        current_assistance_result["mild_duration_sec"]
                    )

                    assistance_strong_duration = (
                        current_assistance_result["strong_duration_sec"]
                    )


                trial_records.append({

                    "timestamp":
                        elapsed,

                    "tracking":
                        1,

                    "x_raw":
                        current_x_raw,

                    "y_raw":
                        current_y_raw,

                    "x_norm":
                        current_x_norm,

                    "y_norm":
                        current_y_norm,

                    "angle_raw":
                        current_angle,

                    "angle_rel":
                        current_angle_rel,

                    "palm_size":
                        current_palm_size,

                    "path_error":
                        current_path_error,

                    "angle_error":
                        current_angle_error,

                    "speed":
                        current_speed,

                    "nearest_reference_index":
                        current_nearest_index,

                    "reference_progress":
                        reference_progress,

                    "learner_relative_rotation":
                        current_learner_relative_rotation,

                    "expected_relative_rotation":
                        current_expected_relative_rotation,

                    "rotation_deficit_deg":
                        current_rotation_deficit,

                    "rotation_completion_ratio":
                        current_rotation_completion_ratio,

                    "hesitation_state":
                        hesitation_state,

                    "hesitation_event":
                        hesitation_event,

                    "hesitation_duration":
                        hesitation_duration,

                    "hesitation_progress_delta":
                        hesitation_progress_delta,

                    "assistance_level":
                        assistance_level,

                    "assistance_label":
                        assistance_label,

                    "assistance_reason":
                        assistance_reason,

                    "assistance_cue":
                        assistance_cue,

                    "assistance_dominant_error":
                        assistance_dominant_error,

                    "assistance_mild_duration":
                        assistance_mild_duration,

                    "assistance_strong_duration":
                        assistance_strong_duration
                })


            else:


                trial_records.append({

                    "timestamp":
                        elapsed,

                    "tracking":
                        0,

                    "x_raw": "",
                    "y_raw": "",

                    "x_norm": "",
                    "y_norm": "",

                    "angle_raw": "",
                    "angle_rel": "",

                    "palm_size": "",

                    "path_error": "",
                    "angle_error": "",

                    "speed": "",

                    "nearest_reference_index": "",

                    "reference_progress": "",

                    "learner_relative_rotation": "",

                    "expected_relative_rotation": "",

                    "rotation_deficit_deg": "",

                    "rotation_completion_ratio": "",

                    "hesitation_state":
                        (
                            current_hesitation_result["state"]
                            if current_hesitation_result is not None
                            else "NO_TRACKING"
                        ),

                    "hesitation_event": 0,

                    "hesitation_duration": 0.0,

                    "hesitation_progress_delta": 0.0,

                    "assistance_level":
                        (
                            int(current_assistance_result["level"])
                            if current_assistance_result is not None
                            else 0
                        ),

                    "assistance_label":
                        (
                            current_assistance_result["label"]
                            if current_assistance_result is not None
                            else "OBSERVE"
                        ),

                    "assistance_reason":
                        (
                            current_assistance_result["reason"]
                            if current_assistance_result is not None
                            else "Tracking unavailable"
                        ),

                    "assistance_cue":
                        (
                            current_assistance_result["cue"]
                            if current_assistance_result is not None
                            else "No assistance"
                        ),

                    "assistance_dominant_error":
                        (
                            current_assistance_result["dominant_error"]
                            if current_assistance_result is not None
                            else "NONE"
                        ),

                    "assistance_mild_duration":
                        (
                            current_assistance_result["mild_duration_sec"]
                            if current_assistance_result is not None
                            else 0.0
                        ),

                    "assistance_strong_duration":
                        (
                            current_assistance_result["strong_duration_sec"]
                            if current_assistance_result is not None
                            else 0.0
                        )
                })


        # =================================================
        # START MARKER
        # =================================================

        if (
            calibration_ready
            and
            calibration is not None
        ):

            draw_start_marker(
                frame,
                calibration
            )


        # =================================================
        # CONDITION TEXT
        # =================================================

        if selected_trial_type is None:

            condition_text = (
                "SELECT: "
                "1 Correct | "
                "2 Path Wrong | "
                "3 Angle Wrong"
            )

        else:

            condition_text = (
                "TRIAL TYPE: "
                +
                selected_trial_type[
                    "label"
                ]
            )


        put_text(
            frame,
            condition_text,
            (
                int(
                    18
                    *
                    ui
                ),

                int(
                    28
                    *
                    ui
                )
            ),
            ui,
            0.46,
            1
        )


        # =================================================
        # PRE-TRIAL STATUS
        # =================================================

        if (
            not practicing
            and
            hand_detected
            and
            current_palm_size is not None
        ):


            if (
                not calibration_ready
                and
                not calibrating
            ):


                if (
                    current_palm_size
                    <
                    MIN_PALM_SIZE
                ):

                    status_text = (
                        "MOVE CLOSER FOR CALIBRATION"
                    )


                elif (
                    current_palm_size
                    >
                    MAX_PALM_SIZE
                ):

                    status_text = (
                        "MOVE BACK FOR CALIBRATION"
                    )


                else:

                    status_text = (
                        "READY - Press C to Calibrate"
                    )


            elif calibrating:

                status_text = (
                    "HOLD STARTING POSE STILL"
                )


            else:


                scale_status = (
                    check_scale_against_calibration(

                        current_palm_size,

                        calibration
                    )
                )


                position_ready = (

                    start_position_error
                    is not None

                    and

                    start_position_error
                    <=
                    START_POSITION_TOLERANCE
                )


                if scale_status == "TOO_FAR":

                    status_text = (
                        "MOVE SLIGHTLY CLOSER"
                    )


                elif scale_status == "TOO_CLOSE":

                    status_text = (
                        "MOVE SLIGHTLY BACK"
                    )


                elif not position_ready:

                    status_text = (
                        "MOVE FINGERTIP TO START"
                    )


                elif selected_trial_type is None:

                    status_text = (
                        "SELECT 1 / 2 / 3 FIRST"
                    )


                else:

                    status_text = (
                        "START READY - Press R"
                    )


            put_text(
                frame,
                status_text,
                (
                    int(
                        18
                        *
                        ui
                    ),

                    height
                    -
                    int(
                        28
                        *
                        ui
                    )
                ),
                ui,
                0.56,
                2
            )


        # =================================================
        # LIVE ERROR UI
        # =================================================

        if practicing:


            top = int(
                54
                *
                ui
            )


            gap = int(
                30
                *
                ui
            )


            if current_path_error is not None:

                put_text(
                    frame,
                    (
                        "PATH ERROR: "
                        f"{current_path_error:.3f}"
                    ),
                    (
                        int(
                            18
                            *
                            ui
                        ),
                        top
                    ),
                    ui,
                    0.54,
                    2
                )


            if current_angle_error is not None:

                put_text(
                    frame,
                    (
                        "REL ANGLE ERROR: "
                        f"{current_angle_error:.1f} deg"
                    ),
                    (
                        int(
                            18
                            *
                            ui
                        ),
                        top
                        +
                        gap
                    ),
                    ui,
                    0.54,
                    2
                )


            put_text(
                frame,
                (
                    "SPEED: "
                    f"{current_speed:.2f}"
                ),
                (
                    int(
                        18
                        *
                        ui
                    ),
                    top
                    +
                    gap
                    *
                    2
                ),
                ui,
                0.54,
                2
            )


            if current_hesitation_result is None:

                hesitation_text = (
                    "HESITATION: NORMAL"
                )

                hesitation_color = (
                    255,
                    255,
                    255
                )


            else:

                hesitation_state = (
                    current_hesitation_result[
                        "state"
                    ]
                )


                if hesitation_state == "HESITATION":

                    hesitation_text = (
                        "HESITATION DETECTED"
                    )

                    hesitation_color = (
                        0,
                        0,
                        255
                    )


                elif hesitation_state == "LOW_SPEED":

                    hesitation_text = (
                        "LOW SPEED: "
                        f"{current_hesitation_result['candidate_duration']:.2f}"
                        "/"
                        f"{hesitation_detector.hold_duration:.2f}s"
                    )

                    hesitation_color = (
                        0,
                        255,
                        255
                    )


                elif hesitation_state == "NO_TRACKING":

                    hesitation_text = (
                        "HESITATION: NO TRACKING"
                    )

                    hesitation_color = (
                        255,
                        255,
                        255
                    )


                elif hesitation_state == "IGNORED":

                    hesitation_text = (
                        "HESITATION: BOUNDARY IGNORED"
                    )

                    hesitation_color = (
                        255,
                        255,
                        255
                    )


                else:

                    hesitation_text = (
                        "HESITATION: NORMAL"
                    )

                    hesitation_color = (
                        255,
                        255,
                        255
                    )


            put_text(
                frame,
                hesitation_text,
                (
                    int(
                        18
                        *
                        ui
                    ),
                    top
                    +
                    gap
                    *
                    3
                ),
                ui,
                0.54,
                2,
                hesitation_color
            )


            # ---------------------------------------------
            # ADAPTIVE ASSISTANCE UI
            # ---------------------------------------------

            if current_assistance_result is None:

                assistance_level = 0
                assistance_label = "OBSERVE"
                assistance_reason = "Within calibrated baseline"
                assistance_cue = "No assistance"

            else:

                assistance_level = int(
                    current_assistance_result["level"]
                )

                assistance_label = (
                    current_assistance_result["label"]
                )

                assistance_reason = (
                    current_assistance_result["reason"]
                )

                assistance_cue = (
                    current_assistance_result["cue"]
                )


            assistance_colors = {
                0: (255, 255, 255),
                1: (0, 255, 255),
                2: (0, 165, 255),
                3: (0, 0, 255)
            }

            assistance_color = assistance_colors.get(
                assistance_level,
                (255, 255, 255)
            )


            put_text(
                frame,
                (
                    "ASSISTANCE: L"
                    f"{assistance_level} "
                    f"{assistance_label}"
                ),
                (
                    int(18 * ui),
                    top + gap * 4
                ),
                ui,
                0.54,
                2,
                assistance_color
            )


            put_text(
                frame,
                (
                    "REASON: "
                    f"{assistance_reason}"
                ),
                (
                    int(18 * ui),
                    top + gap * 5
                ),
                ui,
                0.45,
                1,
                assistance_color
            )


            put_text(
                frame,
                (
                    "CUE: "
                    f"{assistance_cue}"
                ),
                (
                    int(18 * ui),
                    top + gap * 6
                ),
                ui,
                0.45,
                1,
                assistance_color
            )


        # =================================================
        # REVIEW SUMMARY
        # =================================================

        if (
            trial_finished
            and
            last_review_summary is not None
        ):


            text_x = int(
                18
                *
                ui
            )


            text_y = int(
                58
                *
                ui
            )


            text_gap = int(
                25
                *
                ui
            )


            put_text(
                frame,
                (
                    "MEAN PATH: "
                    f"{last_review_summary['mean_path_error']:.3f}"
                ),
                (
                    text_x,
                    text_y
                ),
                ui,
                0.46,
                1
            )


            put_text(
                frame,
                (
                    "MEAN REL ANGLE: "
                    f"{last_review_summary['mean_angle_error']:.1f} deg"
                ),
                (
                    text_x,
                    text_y
                    +
                    text_gap
                ),
                ui,
                0.46,
                1
            )


            completion_pct = last_review_summary.get(
                "final_rotation_completion_pct",
                ""
            )

            deficit_deg = last_review_summary.get(
                "final_rotation_deficit_deg",
                ""
            )

            if completion_pct != "":

                completion_text = (
                    "END ROT COMPLETE: "
                    f"{float(completion_pct):.0f}%"
                )

            else:

                completion_text = (
                    "END ROT COMPLETE: N/A"
                )

            put_text(
                frame,
                completion_text,
                (
                    text_x,
                    text_y
                    +
                    text_gap
                    *
                    2
                ),
                ui,
                0.46,
                1
            )


            if deficit_deg != "":

                deficit_text = (
                    "END ROT DEFICIT: "
                    f"{float(deficit_deg):.1f} deg"
                )

            else:

                deficit_text = (
                    "END ROT DEFICIT: N/A"
                )

            put_text(
                frame,
                deficit_text,
                (
                    text_x,
                    text_y
                    +
                    text_gap
                    *
                    3
                ),
                ui,
                0.46,
                1
            )


            put_text(
                frame,
                (
                    "TRACKING: "
                    f"{last_review_summary['tracking_ratio'] * 100:.1f}%"
                ),
                (
                    text_x,
                    text_y
                    +
                    text_gap
                    *
                    4
                ),
                ui,
                0.46,
                1
            )


            max_assistance_level = last_review_summary.get(
                "max_assistance_level",
                ""
            )

            if max_assistance_level != "":

                assistance_summary_text = (
                    "MAX ASSISTANCE: L"
                    f"{int(max_assistance_level)}"
                )

            else:

                assistance_summary_text = (
                    "MAX ASSISTANCE: N/A"
                )


            put_text(
                frame,
                assistance_summary_text,
                (
                    text_x,
                    text_y
                    +
                    text_gap
                    *
                    5
                ),
                ui,
                0.46,
                1
            )


            put_text(
                frame,
                "D = DELETE LAST TRIAL",
                (
                    text_x,
                    text_y
                    +
                    text_gap
                    *
                    6
                ),
                ui,
                0.42,
                1
            )


        # =================================================
        # MOTION MAP
        # =================================================

        if (
            calibration_ready
            and
            calibration is not None
        ):


            if practicing:

                map_mode = (
                    "PRACTICING"
                )


            elif trial_finished:

                map_mode = (
                    "REVIEW"
                )


            else:

                map_mode = (
                    "READY"
                )


            current_norm = None


            if (
                current_x_norm is not None
                and
                current_y_norm is not None
            ):

                current_norm = (
                    current_x_norm,
                    current_y_norm
                )


            draw_motion_map(

                frame,

                reference,

                learner_trail,

                current_norm,

                reference_bounds,

                map_mode
            )


        # =================================================
        # DISPLAY
        # =================================================

        cv2.imshow(
            (
                "AdaptiveSkill - "
                "Adaptive Assistance"
            ),
            frame
        )


        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        # =================================================
        # 1 / 2 / 3
        # =================================================

        if key in TRIAL_TYPES:


            if practicing:

                print(
                    "Cannot change Trial type "
                    "while recording."
                )


            else:

                selected_trial_type = (
                    TRIAL_TYPES[key]
                )

                print()
                print(
                    "Selected:",
                    selected_trial_type[
                        "label"
                    ]
                )
                print()


        # =================================================
        # C = CALIBRATE
        # =================================================

        elif key == ord("c"):


            if practicing:

                print(
                    "Stop Trial before recalibrating."
                )


            elif calibrating:

                print(
                    "Calibration already running."
                )


            elif not hand_detected:

                print(
                    "Cannot calibrate: "
                    "hand not detected."
                )


            elif (
                current_palm_size
                <
                MIN_PALM_SIZE
            ):

                print(
                    "Move closer."
                )


            elif (
                current_palm_size
                >
                MAX_PALM_SIZE
            ):

                print(
                    "Move back."
                )


            else:

                print()
                print(
                    "Calibration started."
                )

                print(
                    "Hold START position still."
                )


                calibrating = True

                calibration_ready = False


                calibration_samples = []

                calibration = None


                learner_trail.clear()

                hesitation_detector.reset()

                current_hesitation_result = None

                assistance_policy.reset()

                current_assistance_result = None


                trial_finished = False

                last_review_summary = None

                last_saved_trial_path = None


        # =================================================
        # R = START TRIAL
        # =================================================

        elif key == ord("r"):


            if practicing:

                print(
                    "Trial already running."
                )


            elif calibrating:

                print(
                    "Wait for Calibration."
                )


            elif not calibration_ready:

                print(
                    "Press C first."
                )


            elif selected_trial_type is None:

                print(
                    "Select 1, 2, or 3 first."
                )


            elif not hand_detected:

                print(
                    "Hand not detected."
                )


            else:


                scale_status = (
                    check_scale_against_calibration(

                        current_palm_size,

                        calibration
                    )
                )


                start_error = (
                    get_start_position_error(

                        current_x_raw,

                        current_y_raw,

                        calibration
                    )
                )


                if scale_status == "TOO_FAR":

                    print(
                        "Move slightly closer."
                    )


                elif scale_status == "TOO_CLOSE":

                    print(
                        "Move slightly back."
                    )


                elif (
                    start_error is None
                    or
                    start_error
                    >
                    START_POSITION_TOLERANCE
                ):

                    print(
                        "Move fingertip to START."
                    )


                else:

                    print()
                    print(
                        "======================================"
                    )

                    print(
                        "TRIAL STARTED"
                    )

                    print(
                        selected_trial_type[
                            "label"
                        ]
                    )

                    print(
                        "Reference hidden."
                    )

                    print(
                        "======================================"
                    )

                    print()


                    practicing = True

                    trial_finished = False


                    trial_records = []


                    trial_start_time = (
                        time.perf_counter()
                    )


                    learner_trail.clear()


                    x_history.clear()

                    y_history.clear()

                    speed_history.clear()

                    online_motion_metrics.reset()

                    current_online_metric_result = None

                    hesitation_detector.reset()

                    current_hesitation_result = None

                    assistance_policy.reset()

                    current_assistance_result = None


                    previous_norm_x = None

                    previous_norm_y = None

                    previous_time = None


                    current_speed = 0.0


                    current_path_error = None

                    current_angle_error = None

                    current_rotation_deficit = None

                    current_rotation_completion_ratio = None

                    current_nearest_index = None

                    current_reference_progress = None

                    current_learner_relative_rotation = None

                    current_expected_relative_rotation = None


                    last_review_summary = None

                    last_saved_trial_path = None


        # =================================================
        # S = STOP + SAVE ONLY
        # =================================================

        elif key == ord("s"):


            if not practicing:

                print(
                    "No Trial is running."
                )


            else:

                practicing = False

                trial_finished = True


                (
                    last_saved_trial_path,
                    last_review_summary
                ) = save_trial(

                    trial_records,

                    selected_trial_type
                )


                print(
                    "Check the POST-TRIAL COMPARISON."
                )

                print(
                    "Keep it if valid."
                )

                print(
                    "Press D only for a technical failure."
                )

                print()


        # =================================================
        # D = DELETE LAST SAVED TRIAL
        # =================================================

        elif key == ord("d"):


            if practicing:

                print(
                    "Cannot delete while Trial is running."
                )


            elif (
                last_saved_trial_path
                is None
            ):

                print(
                    "No newly saved Trial to delete."
                )


            elif not (
                last_saved_trial_path.exists()
            ):

                print(
                    "Last Trial file no longer exists."
                )

                last_saved_trial_path = None


            else:

                deleted_name = (
                    last_saved_trial_path.name
                )


                last_saved_trial_path.unlink()


                print()
                print(
                    "======================================"
                )

                print(
                    "TRIAL DELETED"
                )

                print(
                    deleted_name
                )

                print(
                    "======================================"
                )

                print()


                last_saved_trial_path = None

                last_review_summary = None


                trial_finished = False


                learner_trail.clear()

                assistance_policy.reset()

                current_assistance_result = None


        # =================================================
        # T = GENERATE CLEAN FINAL SUMMARY
        # =================================================

        elif key == ord("t"):


            if practicing:

                print(
                    "Stop Trial before generating summary."
                )


            else:

                generate_final_summary()


        # =================================================
        # Q = QUIT
        # =================================================

        elif key == ord("q"):


            if practicing:

                print(
                    "Trial is still running."
                )

                print(
                    "Press S first."
                )


            else:

                break


# =========================================================
# 24. CLOSE
# =========================================================

cap.release()

cv2.destroyAllWindows()

print()
print(
    "AdaptiveSkill - Motion Evaluation Closed."
)