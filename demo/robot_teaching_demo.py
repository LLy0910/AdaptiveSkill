import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# =========================================================
# PROJECT IMPORTS
# =========================================================

from task.task_constraints import (
    TaskConstraintEvaluator,
    STATE_VALID,
    STATE_RISK,
    STATE_VIOLATION,
    STATE_TRACKING_UNRELIABLE,
)

from interaction.intervention_comparator import (
    InterventionComparator,
    DECISION_STAY_QUIET,
    DECISION_INTERVENE,
    DECISION_PAUSE,
)

from metrics.assistance_metrics import (
    AssistanceMetrics,
    OUTCOME_IN_PROGRESS,
    OUTCOME_TASK_SUCCESS,
    OUTCOME_GOAL_WITH_VIOLATION,
    OUTCOME_VIOLATION_OCCURRED,
)

from guidance.correction_guidance import (
    CorrectionGuidanceGenerator,
    CorrectionGuidanceResult,
    GUIDANCE_NONE,
    GUIDANCE_DIRECTION,
    GUIDANCE_WAYPOINT,
    GUIDANCE_ORIENTATION,
    GUIDANCE_PAUSE,
)


# =========================================================
# EXTRA GUIDANCE TYPE
# =========================================================

GUIDANCE_RETURN_SAFE = "RETURN_TO_LAST_SAFE"


# =========================================================
# WINDOW
# =========================================================

WINDOW_NAME = (
    "AdaptiveSkill - Segment-Aware Task Assistance"
)

WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 860


# =========================================================
# TASK AREA
# =========================================================

TASK_LEFT = 60
TASK_RIGHT = 850

TASK_TOP = 135
TASK_BOTTOM = 700


# =========================================================
# RIGHT PANEL
# =========================================================

PANEL_LEFT = 880
PANEL_RIGHT = 1460


# =========================================================
# WORLD RANGE
# =========================================================

WORLD_X_MIN = -0.30
WORLD_X_MAX = 3.60

WORLD_Y_MIN = -1.50
WORLD_Y_MAX = 1.50


# =========================================================
# COLORS
# OpenCV = BGR
# =========================================================

COLOR_BACKGROUND = (
    247,
    247,
    247,
)

COLOR_TEXT = (
    45,
    45,
    45,
)

COLOR_SECONDARY = (
    115,
    115,
    115,
)

COLOR_VALID = (
    70,
    175,
    70,
)

COLOR_RISK = (
    0,
    170,
    255,
)

COLOR_VIOLATION = (
    65,
    65,
    225,
)

COLOR_START = (
    210,
    120,
    60,
)

COLOR_TARGET = (
    65,
    170,
    70,
)

COLOR_OBSTACLE = (
    70,
    70,
    70,
)

COLOR_SAFETY = (
    195,
    195,
    195,
)

COLOR_REFERENCE = (
    165,
    165,
    165,
)

COLOR_TRAIL = (
    120,
    120,
    120,
)

COLOR_CARD = (
    238,
    238,
    238,
)

COLOR_BORDER = (
    210,
    210,
    210,
)

COLOR_GUIDANCE = (
    0,
    135,
    235,
)

COLOR_GUIDANCE_STRONG = (
    60,
    60,
    230,
)

COLOR_GHOST = (
    75,
    185,
    90,
)

COLOR_SEGMENT_VIOLATION = (
    65,
    65,
    225,
)

COLOR_CONTROL_BAR = (
    238,
    238,
    238,
)


# =========================================================
# CONFIG
# =========================================================

CONFIG_PATH = (
    PROJECT_ROOT
    /
    "task"
    /
    "task_config.json"
)


with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as file:

    TASK_CONFIG = json.load(
        file
    )


TASK_EVALUATOR = (
    TaskConstraintEvaluator(
        CONFIG_PATH
    )
)


GUIDANCE_GENERATOR = (
    CorrectionGuidanceGenerator(
        CONFIG_PATH
    )
)


# =========================================================
# EXPERT PRIOR
# =========================================================

EXPERT_REFERENCE_POINTS = [

    (
        0.0,
        0.0,
    ),

    (
        0.70,
        -0.75,
    ),

    (
        1.10,
        -0.90,
    ),

    (
        2.20,
        -0.90,
    ),

    (
        2.60,
        -0.60,
    ),

    (
        3.20,
        0.0,
    ),
]


REFERENCE_THRESHOLD = 0.30


COMPARATOR = (
    InterventionComparator(

        expert_reference_points=
            EXPERT_REFERENCE_POINTS,

        reference_threshold=
            REFERENCE_THRESHOLD,
    )
)


# =========================================================
# MOUSE STATE
# =========================================================

mouse_x = 0
mouse_y = 0

mouse_inside_task = False


def mouse_callback(
    event,
    x,
    y,
    flags,
    param,
):

    global mouse_x
    global mouse_y
    global mouse_inside_task


    mouse_x = int(
        x
    )

    mouse_y = int(
        y
    )


    mouse_inside_task = (

        TASK_LEFT
        <=
        mouse_x
        <=
        TASK_RIGHT

        and

        TASK_TOP
        <=
        mouse_y
        <=
        TASK_BOTTOM
    )


# =========================================================
# COORDINATE CONVERSION
# =========================================================

def world_to_screen(
    x,
    y,
):

    x_ratio = (
        (
            float(x)
            -
            WORLD_X_MIN
        )
        /
        (
            WORLD_X_MAX
            -
            WORLD_X_MIN
        )
    )


    y_ratio = (
        (
            float(y)
            -
            WORLD_Y_MIN
        )
        /
        (
            WORLD_Y_MAX
            -
            WORLD_Y_MIN
        )
    )


    screen_x = int(
        TASK_LEFT
        +
        x_ratio
        *
        (
            TASK_RIGHT
            -
            TASK_LEFT
        )
    )


    screen_y = int(
        TASK_TOP
        +
        y_ratio
        *
        (
            TASK_BOTTOM
            -
            TASK_TOP
        )
    )


    return (
        screen_x,
        screen_y,
    )


def screen_to_world(
    x,
    y,
):

    x_ratio = (
        (
            float(x)
            -
            TASK_LEFT
        )
        /
        (
            TASK_RIGHT
            -
            TASK_LEFT
        )
    )


    y_ratio = (
        (
            float(y)
            -
            TASK_TOP
        )
        /
        (
            TASK_BOTTOM
            -
            TASK_TOP
        )
    )


    world_x = (
        WORLD_X_MIN
        +
        x_ratio
        *
        (
            WORLD_X_MAX
            -
            WORLD_X_MIN
        )
    )


    world_y = (
        WORLD_Y_MIN
        +
        y_ratio
        *
        (
            WORLD_Y_MAX
            -
            WORLD_Y_MIN
        )
    )


    return (
        world_x,
        world_y,
    )


# =========================================================
# TEXT
# =========================================================

def draw_text(
    frame,
    text,
    x,
    y,
    scale=0.50,
    color=COLOR_TEXT,
    thickness=1,
):

    cv2.putText(

        frame,

        str(text),

        (
            int(x),
            int(y),
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        float(scale),

        color,

        int(thickness),

        cv2.LINE_AA,
    )


# =========================================================
# CARD
# =========================================================

def draw_card(
    frame,
    x1,
    y1,
    x2,
    y2,
):

    cv2.rectangle(
        frame,
        (
            x1,
            y1,
        ),
        (
            x2,
            y2,
        ),
        COLOR_CARD,
        -1,
    )


    cv2.rectangle(
        frame,
        (
            x1,
            y1,
        ),
        (
            x2,
            y2,
        ),
        COLOR_BORDER,
        1,
    )


# =========================================================
# COLORS
# =========================================================

def get_task_color(
    state,
):

    if state == STATE_VALID:
        return COLOR_VALID

    if state == STATE_RISK:
        return COLOR_RISK

    if state == STATE_VIOLATION:
        return COLOR_VIOLATION

    return COLOR_SECONDARY


def get_decision_color(
    decision,
):

    if decision == DECISION_STAY_QUIET:
        return COLOR_VALID

    if decision == DECISION_INTERVENE:
        return COLOR_VIOLATION

    return COLOR_SECONDARY


# =========================================================
# TOP PERSISTENT CONTROLS
# =========================================================

def draw_top_controls(
    frame,
):

    draw_text(
        frame,
        "[S] Start/Restart",
        35,
        103,
        scale=0.37,
        color=COLOR_START,
        thickness=2,
    )

    draw_text(
        frame,
        "[MOUSE] Move cup",
        190,
        103,
        scale=0.37,
        color=COLOR_SECONDARY,
    )

    draw_text(
        frame,
        "[A/D] Rotate",
        355,
        103,
        scale=0.37,
        color=COLOR_SECONDARY,
    )

    draw_text(
        frame,
        "[R] Upright",
        480,
        103,
        scale=0.37,
        color=COLOR_SECONDARY,
    )

    draw_text(
        frame,
        "[C] Clear",
        590,
        103,
        scale=0.37,
        color=COLOR_SECONDARY,
    )

    draw_text(
        frame,
        "[Q] Quit",
        680,
        103,
        scale=0.37,
        color=COLOR_SECONDARY,
    )


# =========================================================
# EXPERT PRIOR
# =========================================================

def draw_expert_prior(
    frame,
):

    points = [

        world_to_screen(
            x,
            y
        )

        for x, y
        in EXPERT_REFERENCE_POINTS
    ]


    for index in range(
        len(points) - 1
    ):

        x1, y1 = points[
            index
        ]

        x2, y2 = points[
            index + 1
        ]


        length = math.hypot(
            x2 - x1,
            y2 - y1
        )


        if length < 1.0:
            continue


        dash = 12.0
        gap = 8.0

        step = (
            dash
            +
            gap
        )


        count = int(
            length
            /
            step
        ) + 1


        for number in range(
            count
        ):

            d1 = (
                number
                *
                step
            )


            d2 = min(
                d1 + dash,
                length
            )


            if d1 >= length:
                break


            t1 = (
                d1
                /
                length
            )


            t2 = (
                d2
                /
                length
            )


            sx = int(
                x1
                +
                (
                    x2 - x1
                )
                *
                t1
            )


            sy = int(
                y1
                +
                (
                    y2 - y1
                )
                *
                t1
            )


            ex = int(
                x1
                +
                (
                    x2 - x1
                )
                *
                t2
            )


            ey = int(
                y1
                +
                (
                    y2 - y1
                )
                *
                t2
            )


            cv2.line(

                frame,

                (
                    sx,
                    sy,
                ),

                (
                    ex,
                    ey,
                ),

                COLOR_REFERENCE,

                2,

                cv2.LINE_AA,
            )


    label = world_to_screen(
        1.15,
        -1.12
    )


    draw_text(
        frame,
        "EXPERT PRIOR",
        label[0],
        label[1],
        scale=0.40,
        color=COLOR_REFERENCE,
    )


    draw_text(
        frame,
        "(not the only valid path)",
        label[0],
        label[1] + 20,
        scale=0.32,
        color=COLOR_REFERENCE,
    )


# =========================================================
# WORLD
# =========================================================

def draw_task_world(
    frame,
):

    cv2.rectangle(
        frame,
        (
            TASK_LEFT,
            TASK_TOP,
        ),
        (
            TASK_RIGHT,
            TASK_BOTTOM,
        ),
        COLOR_BORDER,
        2,
    )


    draw_expert_prior(
        frame
    )


    # =====================================================
    # START
    # =====================================================

    start = (
        TASK_CONFIG["start"]
    )


    start_screen = world_to_screen(
        start["x"],
        start["y"],
    )


    cv2.circle(
        frame,
        start_screen,
        15,
        COLOR_START,
        3,
    )


    cv2.circle(
        frame,
        start_screen,
        4,
        COLOR_START,
        -1,
    )


    draw_text(
        frame,
        "START",
        start_screen[0] - 28,
        start_screen[1] - 23,
        scale=0.46,
        color=COLOR_START,
        thickness=2,
    )


    # =====================================================
    # TARGET
    # =====================================================

    target = (
        TASK_CONFIG["target"]
    )


    target_screen = world_to_screen(
        target["x"],
        target["y"],
    )


    radius_screen = world_to_screen(
        target["x"]
        +
        target["radius"],
        target["y"],
    )


    radius_px = abs(
        radius_screen[0]
        -
        target_screen[0]
    )


    cv2.circle(
        frame,
        target_screen,
        radius_px,
        COLOR_TARGET,
        3,
    )


    cv2.circle(
        frame,
        target_screen,
        5,
        COLOR_TARGET,
        -1,
    )


    draw_text(
        frame,
        "TARGET",
        target_screen[0] - 32,
        target_screen[1] - radius_px - 12,
        scale=0.46,
        color=COLOR_TARGET,
        thickness=2,
    )


    # =====================================================
    # OBSTACLES
    # =====================================================

    margin = float(
        TASK_CONFIG[
            "safety_margin"
        ]
    )


    for obstacle in TASK_CONFIG[
        "obstacles"
    ]:

        safety_a = world_to_screen(

            obstacle["x_min"]
            -
            margin,

            obstacle["y_min"]
            -
            margin,
        )


        safety_b = world_to_screen(

            obstacle["x_max"]
            +
            margin,

            obstacle["y_max"]
            +
            margin,
        )


        cv2.rectangle(
            frame,
            safety_a,
            safety_b,
            COLOR_SAFETY,
            2,
        )


        obstacle_a = world_to_screen(
            obstacle["x_min"],
            obstacle["y_min"],
        )


        obstacle_b = world_to_screen(
            obstacle["x_max"],
            obstacle["y_max"],
        )


        cv2.rectangle(
            frame,
            obstacle_a,
            obstacle_b,
            COLOR_OBSTACLE,
            -1,
        )


        draw_text(
            frame,
            "OBSTACLE",
            obstacle_a[0],
            obstacle_a[1] - 10,
            scale=0.40,
            color=COLOR_OBSTACLE,
            thickness=2,
        )


# =========================================================
# CUP
# =========================================================

def draw_virtual_cup(
    frame,
    x,
    y,
    orientation_deg,
    color,
):

    center = (
        int(x),
        int(y),
    )


    rectangle = (

        center,

        (
            28,
            42,
        ),

        float(
            orientation_deg
        ),
    )


    box = cv2.boxPoints(
        rectangle
    )


    box = np.int32(
        box
    )


    cv2.polylines(
        frame,
        [
            box
        ],
        True,
        color,
        3,
    )


    cv2.circle(
        frame,
        center,
        4,
        color,
        -1,
    )


    angle = math.radians(
        orientation_deg
        -
        90.0
    )


    end = (

        int(
            center[0]
            +
            52
            *
            math.cos(
                angle
            )
        ),

        int(
            center[1]
            +
            52
            *
            math.sin(
                angle
            )
        ),
    )


    cv2.line(
        frame,
        center,
        end,
        color,
        3,
        cv2.LINE_AA,
    )


# =========================================================
# TRAIL
# =========================================================

def draw_trail(
    frame,
    trail_points,
    violation_segments,
):

    if len(
        trail_points
    ) >= 2:

        points = np.array(
            trail_points,
            dtype=np.int32,
        )


        cv2.polylines(
            frame,
            [
                points
            ],
            False,
            COLOR_TRAIL,
            2,
            cv2.LINE_AA,
        )


    for (
        start,
        end
    ) in violation_segments:

        cv2.line(
            frame,
            start,
            end,
            COLOR_SEGMENT_VIOLATION,
            5,
            cv2.LINE_AA,
        )


        midpoint = (

            int(
                (
                    start[0]
                    +
                    end[0]
                )
                /
                2
            ),

            int(
                (
                    start[1]
                    +
                    end[1]
                )
                /
                2
            ),
        )


        draw_text(
            frame,
            "SEGMENT VIOLATION",
            midpoint[0] + 10,
            midpoint[1] - 10,
            scale=0.38,
            color=COLOR_VIOLATION,
            thickness=2,
        )


# =========================================================
# RETURN-TO-SAFE GUIDANCE
# =========================================================

def build_return_safe_guidance(
    current_x,
    current_y,
    last_safe_world,
):

    if last_safe_world is None:
        return None


    safe_x = float(
        last_safe_world[0]
    )

    safe_y = float(
        last_safe_world[1]
    )


    dx = (
        safe_x
        -
        float(current_x)
    )


    dy = (
        safe_y
        -
        float(current_y)
    )


    length = math.hypot(
        dx,
        dy
    )


    if length < 1e-9:

        direction_x = 0.0
        direction_y = 0.0


    else:

        direction_x = (
            dx
            /
            length
        )

        direction_y = (
            dy
            /
            length
        )


    return CorrectionGuidanceResult(

        guidance_type=
            GUIDANCE_RETURN_SAFE,

        should_intervene=
            True,

        target_x=
            safe_x,

        target_y=
            safe_y,

        direction_x=
            direction_x,

        direction_y=
            direction_y,

        rotation_direction=
            "NONE",

        rotation_correction_deg=
            0.0,

        dominant_constraint=
            "OBSTACLE",

        message=(
            "The movement crossed the obstacle. "
            "Return toward the last safe position."
        )
    )


# =========================================================
# GUIDANCE DRAWING
# =========================================================

def draw_guidance(
    frame,
    guidance,
    current_screen,
):

    if guidance is None:
        return


    if guidance.guidance_type in (
        GUIDANCE_NONE,
        GUIDANCE_PAUSE,
    ):
        return


    # =====================================================
    # DIRECTIONAL NUDGE
    # =====================================================

    if (
        guidance.guidance_type
        ==
        GUIDANCE_DIRECTION
    ):

        arrow_length = 85


        end = (

            int(
                current_screen[0]
                +
                guidance.direction_x
                *
                arrow_length
            ),

            int(
                current_screen[1]
                +
                guidance.direction_y
                *
                arrow_length
            ),
        )


        cv2.arrowedLine(

            frame,

            current_screen,

            end,

            COLOR_GUIDANCE,

            5,

            cv2.LINE_AA,

            tipLength=0.28,
        )


        draw_text(
            frame,
            "MOVE THIS WAY",
            end[0] + 10,
            end[1] - 8,
            scale=0.40,
            color=COLOR_GUIDANCE,
            thickness=2,
        )


        return


    # =====================================================
    # SAFE WAYPOINT
    # =====================================================

    if (
        guidance.guidance_type
        ==
        GUIDANCE_WAYPOINT
    ):

        if (
            guidance.target_x is None
            or
            guidance.target_y is None
        ):
            return


        target = world_to_screen(
            guidance.target_x,
            guidance.target_y,
        )


        cv2.arrowedLine(

            frame,

            current_screen,

            target,

            COLOR_GUIDANCE_STRONG,

            5,

            cv2.LINE_AA,

            tipLength=0.20,
        )


        cv2.circle(
            frame,
            target,
            17,
            COLOR_GUIDANCE_STRONG,
            3,
        )


        cv2.circle(
            frame,
            target,
            5,
            COLOR_GUIDANCE_STRONG,
            -1,
        )


        draw_text(
            frame,
            "SAFE WAYPOINT",
            target[0] + 15,
            target[1] - 15,
            scale=0.42,
            color=COLOR_GUIDANCE_STRONG,
            thickness=2,
        )


        return


    # =====================================================
    # SEGMENT CROSSING
    # =====================================================

    if (
        guidance.guidance_type
        ==
        GUIDANCE_RETURN_SAFE
    ):

        if (
            guidance.target_x is None
            or
            guidance.target_y is None
        ):
            return


        target = world_to_screen(
            guidance.target_x,
            guidance.target_y,
        )


        cv2.arrowedLine(

            frame,

            current_screen,

            target,

            COLOR_VIOLATION,

            5,

            cv2.LINE_AA,

            tipLength=0.18,
        )


        cv2.circle(
            frame,
            target,
            18,
            COLOR_VALID,
            3,
        )


        cv2.circle(
            frame,
            target,
            5,
            COLOR_VALID,
            -1,
        )


        draw_text(
            frame,
            "RETURN TO LAST SAFE",
            target[0] + 15,
            target[1] - 16,
            scale=0.42,
            color=COLOR_VIOLATION,
            thickness=2,
        )


        return


    # =====================================================
    # ORIENTATION
    # =====================================================

    if (
        guidance.guidance_type
        ==
        GUIDANCE_ORIENTATION
    ):

        ghost_rectangle = (

            current_screen,

            (
                34,
                48,
            ),

            0.0,
        )


        ghost_box = np.int32(
            cv2.boxPoints(
                ghost_rectangle
            )
        )


        cv2.polylines(

            frame,

            [
                ghost_box
            ],

            True,

            COLOR_GHOST,

            2,

            cv2.LINE_AA,
        )


        cv2.line(

            frame,

            current_screen,

            (
                current_screen[0],
                current_screen[1] - 65,
            ),

            COLOR_GHOST,

            3,

            cv2.LINE_AA,
        )


        if (
            guidance.rotation_direction
            ==
            "COUNTERCLOCKWISE"
        ):

            direction_text = (
                "ROTATE CCW"
            )


        elif (
            guidance.rotation_direction
            ==
            "CLOCKWISE"
        ):

            direction_text = (
                "ROTATE CW"
            )


        else:

            direction_text = (
                "ALIGN UPRIGHT"
            )


        draw_text(
            frame,
            (
                direction_text
                +
                " "
                +
                f"{guidance.rotation_correction_deg:.0f} deg"
            ),
            current_screen[0] + 35,
            current_screen[1] - 58,
            scale=0.46,
            color=COLOR_GUIDANCE,
            thickness=2,
        )


        draw_text(
            frame,
            "TARGET POSE",
            current_screen[0] + 35,
            current_screen[1] - 34,
            scale=0.34,
            color=COLOR_GHOST,
        )


# =========================================================
# CURRENT COMPARISON PANEL
# =========================================================

def draw_comparison_panel(
    frame,
    task_result,
    comparison,
    guidance,
):

    draw_text(
        frame,
        "CURRENT COMPARISON",
        PANEL_LEFT,
        125,
        scale=0.55,
        thickness=2,
    )


    # =====================================================
    # REFERENCE BASELINE
    # =====================================================

    draw_card(
        frame,
        PANEL_LEFT,
        150,
        PANEL_RIGHT,
        235,
    )


    draw_text(
        frame,
        "REFERENCE-SIMILARITY BASELINE",
        PANEL_LEFT + 15,
        175,
        scale=0.38,
        color=COLOR_SECONDARY,
    )


    draw_text(
        frame,
        comparison.reference_policy_decision,
        PANEL_LEFT + 15,
        207,
        scale=0.62,
        color=get_decision_color(
            comparison.reference_policy_decision
        ),
        thickness=2,
    )


    draw_text(
        frame,
        (
            "Reference distance: "
            +
            f"{comparison.reference_distance:.2f}"
        ),
        PANEL_LEFT + 300,
        203,
        scale=0.36,
    )


    # =====================================================
    # TASK AWARE
    # =====================================================

    draw_card(
        frame,
        PANEL_LEFT,
        250,
        PANEL_RIGHT,
        345,
    )


    draw_text(
        frame,
        "TASK-AWARE POLICY",
        PANEL_LEFT + 15,
        275,
        scale=0.38,
        color=COLOR_SECONDARY,
    )


    draw_text(
        frame,
        comparison.task_policy_decision,
        PANEL_LEFT + 15,
        309,
        scale=0.62,
        color=get_decision_color(
            comparison.task_policy_decision
        ),
        thickness=2,
    )


    draw_text(
        frame,
        (
            "Task state: "
            +
            task_result.state
        ),
        PANEL_LEFT + 300,
        303,
        scale=0.36,
    )


    draw_text(
        frame,
        (
            "Constraint: "
            +
            task_result.dominant_constraint
        ),
        PANEL_LEFT + 300,
        328,
        scale=0.34,
        color=COLOR_SECONDARY,
    )


    guidance_type = (

        guidance.guidance_type

        if guidance is not None

        else "NONE"
    )


    draw_text(
        frame,
        (
            "Guidance: "
            +
            guidance_type
        ),
        PANEL_LEFT + 15,
        335,
        scale=0.33,
        color=COLOR_SECONDARY,
    )


    # =====================================================
    # CURRENT RESULT
    # =====================================================

    draw_text(
        frame,
        "CURRENT RESULT",
        PANEL_LEFT,
        375,
        scale=0.37,
        color=COLOR_SECONDARY,
    )


    if comparison.unnecessary_intervention_avoided:

        result_color = (
            COLOR_VALID
        )


    elif comparison.task_relevant_intervention:

        result_color = (
            COLOR_RISK
        )


    else:

        result_color = (
            COLOR_SECONDARY
        )


    draw_text(
        frame,
        comparison.comparison_label,
        PANEL_LEFT,
        407,
        scale=0.49,
        color=result_color,
        thickness=2,
    )


# =========================================================
# QUICK START
# =========================================================

def draw_quick_start(
    frame,
):

    draw_text(
        frame,
        "QUICK START",
        PANEL_LEFT + 15,
        462,
        scale=0.52,
        thickness=2,
    )


    draw_text(
        frame,
        "READY",
        PANEL_LEFT + 430,
        462,
        scale=0.42,
        color=COLOR_SECONDARY,
        thickness=2,
    )


    draw_text(
        frame,
        "1",
        PANEL_LEFT + 18,
        507,
        scale=0.58,
        color=COLOR_START,
        thickness=2,
    )


    draw_text(
        frame,
        "Move the cursor to the blue START.",
        PANEL_LEFT + 50,
        507,
        scale=0.41,
    )


    draw_text(
        frame,
        "2",
        PANEL_LEFT + 18,
        550,
        scale=0.58,
        color=COLOR_START,
        thickness=2,
    )


    draw_text(
        frame,
        "Press S to start recording the trial.",
        PANEL_LEFT + 50,
        550,
        scale=0.41,
    )


    draw_text(
        frame,
        "3",
        PANEL_LEFT + 18,
        593,
        scale=0.58,
        color=COLOR_START,
        thickness=2,
    )


    draw_text(
        frame,
        "Move the mouse to carry the cup to TARGET.",
        PANEL_LEFT + 50,
        593,
        scale=0.41,
    )


    draw_text(
        frame,
        "4",
        PANEL_LEFT + 18,
        636,
        scale=0.58,
        color=COLOR_START,
        thickness=2,
    )


    draw_text(
        frame,
        "Avoid the obstacle and keep the cup upright.",
        PANEL_LEFT + 50,
        636,
        scale=0.41,
    )


    draw_text(
        frame,
        "Tip: only movement after pressing S is evaluated.",
        PANEL_LEFT + 18,
        692,
        scale=0.35,
        color=COLOR_SECONDARY,
    )


# =========================================================
# TRIAL SUMMARY
# =========================================================

def draw_trial_summary(
    frame,
    summary,
    trial_state,
):

    draw_card(
        frame,
        PANEL_LEFT,
        435,
        PANEL_RIGHT,
        745,
    )


    # =====================================================
    # READY
    # =====================================================

    if trial_state == "READY":

        draw_quick_start(
            frame
        )

        return


    # =====================================================
    # SUMMARY HEADER
    # =====================================================

    draw_text(
        frame,
        "TRIAL SUMMARY",
        PANEL_LEFT + 15,
        462,
        scale=0.52,
        thickness=2,
    )


    if trial_state == "ACTIVE":

        state_color = (
            COLOR_RISK
        )

    else:

        # FINISHED is neutral.
        state_color = (
            COLOR_SECONDARY
        )


    draw_text(
        frame,
        trial_state,
        PANEL_LEFT + 430,
        462,
        scale=0.42,
        color=state_color,
        thickness=2,
    )


    if summary is None:
        return


    # =====================================================
    # TASK OUTCOME
    # =====================================================

    if (
        summary.task_outcome
        ==
        OUTCOME_TASK_SUCCESS
    ):

        outcome_color = (
            COLOR_VALID
        )


    elif (
        summary.task_outcome
        in (
            OUTCOME_GOAL_WITH_VIOLATION,
            OUTCOME_VIOLATION_OCCURRED,
        )
    ):

        outcome_color = (
            COLOR_VIOLATION
        )


    else:

        outcome_color = (
            COLOR_SECONDARY
        )


    draw_text(
        frame,
        "TASK OUTCOME:",
        PANEL_LEFT + 15,
        495,
        scale=0.36,
        color=COLOR_SECONDARY,
    )


    draw_text(
        frame,
        summary.task_outcome,
        PANEL_LEFT + 135,
        495,
        scale=0.43,
        color=outcome_color,
        thickness=2,
    )


    draw_text(
        frame,
        (
            "Reference intervention episodes: "
            +
            str(
                summary.reference_intervention_episodes
            )
        ),
        PANEL_LEFT + 15,
        528,
        scale=0.37,
    )


    draw_text(
        frame,
        (
            "Task-aware intervention episodes: "
            +
            str(
                summary.task_intervention_episodes
            )
        ),
        PANEL_LEFT + 15,
        554,
        scale=0.37,
    )


    draw_text(
        frame,
        (
            "Unnecessary episodes avoided: "
            +
            str(
                summary.unnecessary_intervention_episodes_avoided
            )
        ),
        PANEL_LEFT + 15,
        584,
        scale=0.39,
        color=COLOR_VALID,
        thickness=2,
    )


    draw_text(
        frame,
        (
            "Task risks missed by baseline: "
            +
            str(
                summary.task_risk_missed_by_reference_episodes
            )
        ),
        PANEL_LEFT + 15,
        612,
        scale=0.37,
        color=COLOR_RISK,
    )


    draw_text(
        frame,
        (
            "Constraint violation episodes: "
            +
            str(
                summary.constraint_violation_episodes
            )
        ),
        PANEL_LEFT + 15,
        642,
        scale=0.37,
    )


    violation_text = (
        "YES"
        if summary.constraint_violation_occurred
        else "NO"
    )


    violation_color = (
        COLOR_VIOLATION
        if summary.constraint_violation_occurred
        else COLOR_VALID
    )


    draw_text(
        frame,
        (
            "Violation occurred: "
            +
            violation_text
        ),
        PANEL_LEFT + 15,
        670,
        scale=0.39,
        color=violation_color,
        thickness=2,
    )


    alternative_seen = (
        summary.alternative_valid_episodes
        >
        0
    )


    if (
        summary.goal_reached_seen
        and
        alternative_seen
        and
        not summary.constraint_violation_occurred
    ):

        alternative_text = (
            "Alternative-valid route preserved: YES"
        )


    else:

        alternative_text = (
            "Alternative-valid episode observed: "
            +
            (
                "YES"
                if alternative_seen
                else "NO"
            )
        )


    draw_text(
        frame,
        alternative_text,
        PANEL_LEFT + 15,
        698,
        scale=0.36,
        color=(
            COLOR_VALID
            if alternative_seen
            else COLOR_SECONDARY
        ),
    )


    draw_text(
        frame,
        (
            "Goal reached: "
            +
            (
                "YES"
                if summary.goal_reached_seen
                else "NO"
            )
        ),
        PANEL_LEFT + 15,
        725,
        scale=0.36,
    )


    if (
        summary.task_intervention_episodes
        ==
        0
    ):

        recovery_text = (
            "Recovery needed: NO"
        )

        recovery_color = (
            COLOR_SECONDARY
        )


    else:

        recovery_text = (
            "Returned to quiet: "
            +
            (
                "YES"
                if summary.returned_to_quiet
                else "NO"
            )
        )


        recovery_color = (
            COLOR_VALID
            if summary.returned_to_quiet
            else COLOR_RISK
        )


    draw_text(
        frame,
        recovery_text,
        PANEL_LEFT + 310,
        725,
        scale=0.36,
        color=recovery_color,
    )


# =========================================================
# BOTTOM CONTROL BAR
# =========================================================

def draw_control_bar(
    frame,
):

    # Put it high enough that it remains visible even if
    # Windows/OpenCV slightly crops the bottom.
    bar_top = 765
    bar_bottom = 850


    cv2.rectangle(
        frame,
        (
            0,
            bar_top,
        ),
        (
            WINDOW_WIDTH,
            bar_bottom,
        ),
        COLOR_CONTROL_BAR,
        -1,
    )


    cv2.line(
        frame,
        (
            0,
            bar_top,
        ),
        (
            WINDOW_WIDTH,
            bar_top,
        ),
        COLOR_BORDER,
        1,
    )


    draw_text(
        frame,
        "CONTROLS",
        35,
        792,
        scale=0.38,
        color=COLOR_SECONDARY,
        thickness=2,
    )


    draw_text(
        frame,
        "[S]",
        145,
        820,
        scale=0.42,
        color=COLOR_START,
        thickness=2,
    )

    draw_text(
        frame,
        "Start / Restart",
        180,
        820,
        scale=0.38,
    )


    draw_text(
        frame,
        "[MOUSE]",
        345,
        820,
        scale=0.42,
        color=COLOR_START,
        thickness=2,
    )

    draw_text(
        frame,
        "Move",
        415,
        820,
        scale=0.38,
    )


    draw_text(
        frame,
        "[A / D]",
        500,
        820,
        scale=0.42,
        color=COLOR_START,
        thickness=2,
    )

    draw_text(
        frame,
        "Rotate",
        565,
        820,
        scale=0.38,
    )


    draw_text(
        frame,
        "[R]",
        660,
        820,
        scale=0.42,
        color=COLOR_START,
        thickness=2,
    )

    draw_text(
        frame,
        "Upright",
        695,
        820,
        scale=0.38,
    )


    draw_text(
        frame,
        "[C]",
        805,
        820,
        scale=0.42,
        color=COLOR_START,
        thickness=2,
    )

    draw_text(
        frame,
        "Clear",
        840,
        820,
        scale=0.38,
    )


    draw_text(
        frame,
        "[Q]",
        930,
        820,
        scale=0.42,
        color=COLOR_START,
        thickness=2,
    )

    draw_text(
        frame,
        "Quit",
        965,
        820,
        scale=0.38,
    )


# =========================================================
# MAIN
# =========================================================

def main():

    cv2.namedWindow(
        WINDOW_NAME
    )


    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback
    )


    metrics = (
        AssistanceMetrics()
    )


    trail_points = []

    violation_segments = []


    trial_active = False

    trial_finished = False


    orientation_deg = 0.0


    previous_world_position = None

    previous_screen_position = None

    last_safe_world_position = None


    live_summary = None

    final_summary = None

    final_task_result = None

    final_comparison = None

    final_guidance = None


    while True:

        frame = np.full(
            (
                WINDOW_HEIGHT,
                WINDOW_WIDTH,
                3,
            ),
            COLOR_BACKGROUND,
            dtype=np.uint8,
        )


        # =================================================
        # HEADER
        # =================================================

        draw_text(
            frame,
            "AdaptiveSkill - Teach the Robot",
            35,
            42,
            scale=0.86,
            thickness=2,
        )


        if trial_active:

            subtitle = (
                "Guide the cup to TARGET. "
                "Valid strategies are preserved; "
                "task risks trigger assistance."
            )


        elif trial_finished:

            subtitle = (
                "Trial finished. Read TASK OUTCOME, "
                "or press S to start another trial."
            )


        else:

            subtitle = (
                "Quick start: move the cursor to START, "
                "then press S."
            )


        draw_text(
            frame,
            subtitle,
            35,
            72,
            scale=0.49,
            color=COLOR_SECONDARY,
        )


        # Persistent controls near top.
        draw_top_controls(
            frame
        )


        # =================================================
        # WORLD
        # =================================================

        draw_task_world(
            frame
        )


        draw_trail(
            frame,
            trail_points,
            violation_segments,
        )


        current_task_result = None

        current_point_result = None

        current_comparison = None

        current_guidance = None


        # =================================================
        # CURRENT POSITION
        # =================================================

        if mouse_inside_task:

            world_x, world_y = (
                screen_to_world(
                    mouse_x,
                    mouse_y,
                )
            )


            current_screen = (
                int(
                    mouse_x
                ),
                int(
                    mouse_y
                ),
            )


            # =============================================
            # POINT RESULT
            # =============================================

            current_point_result = (
                TASK_EVALUATOR.evaluate(

                    x=
                        world_x,

                    y=
                        world_y,

                    orientation_relative_deg=
                        orientation_deg,

                    tracking_missing_sec=
                        0.0,
                )
            )


            # =============================================
            # SEGMENT RESULT
            # =============================================

            if (
                trial_active

                and

                previous_world_position
                is not None
            ):

                current_task_result = (
                    TASK_EVALUATOR.evaluate_segment(

                        previous_x=
                            previous_world_position[0],

                        previous_y=
                            previous_world_position[1],

                        x=
                            world_x,

                        y=
                            world_y,

                        orientation_relative_deg=
                            orientation_deg,

                        tracking_missing_sec=
                            0.0,
                    )
                )


            else:

                current_task_result = (
                    current_point_result
                )


            # =============================================
            # SEGMENT CROSSING
            # =============================================

            segment_crossing_detected = (

                trial_active

                and

                previous_world_position
                is not None

                and

                current_task_result.obstacle_violation

                and

                not current_point_result.obstacle_violation
            )


            # =============================================
            # COMPARATOR
            # =============================================

            current_comparison = (
                COMPARATOR.compare(

                    x=
                        world_x,

                    y=
                        world_y,

                    task_result=
                        current_task_result,
                )
            )


            # =============================================
            # GUIDANCE
            # =============================================

            if (
                segment_crossing_detected

                and

                last_safe_world_position
                is not None
            ):

                current_guidance = (
                    build_return_safe_guidance(

                        current_x=
                            world_x,

                        current_y=
                            world_y,

                        last_safe_world=
                            last_safe_world_position,
                    )
                )


            else:

                current_guidance = (
                    GUIDANCE_GENERATOR.generate(

                        x=
                            world_x,

                        y=
                            world_y,

                        orientation_relative_deg=
                            orientation_deg,

                        task_result=
                            current_task_result,
                    )
                )


            # =============================================
            # ACTIVE TRIAL
            # =============================================

            if trial_active:

                now = (
                    time.perf_counter()
                )


                # -----------------------------------------
                # Trail
                # -----------------------------------------

                if (
                    not trail_points

                    or

                    math.hypot(

                        current_screen[0]
                        -
                        trail_points[-1][0],

                        current_screen[1]
                        -
                        trail_points[-1][1],

                    )
                    >=
                    4.0
                ):

                    trail_points.append(
                        current_screen
                    )


                # -----------------------------------------
                # Segment violation visualization
                # -----------------------------------------

                if (
                    segment_crossing_detected

                    and

                    previous_screen_position
                    is not None
                ):

                    violation_segments.append(

                        (
                            previous_screen_position,
                            current_screen,
                        )
                    )


                # -----------------------------------------
                # Metrics
                # -----------------------------------------

                metrics.update(

                    comparison=
                        current_comparison,

                    task_result=
                        current_task_result,

                    timestamp_sec=
                        now,
                )


                live_summary = (
                    metrics.get_summary()
                )


                # -----------------------------------------
                # Last valid position
                # -----------------------------------------

                if (
                    current_task_result.state
                    ==
                    STATE_VALID
                ):

                    last_safe_world_position = (

                        world_x,
                        world_y,
                    )


                # -----------------------------------------
                # Previous point
                # -----------------------------------------

                previous_world_position = (

                    world_x,
                    world_y,
                )


                previous_screen_position = (
                    current_screen
                )


                # =========================================
                # FINISH
                # =========================================

                if (
                    current_point_result.goal_reached

                    and

                    current_point_result.state
                    ==
                    STATE_VALID
                ):

                    final_summary = (
                        metrics.get_summary(
                            timestamp_sec=
                                now
                        )
                    )


                    final_task_result = (
                        current_task_result
                    )


                    final_comparison = (
                        current_comparison
                    )


                    final_guidance = (
                        current_guidance
                    )


                    trial_active = False

                    trial_finished = True


            # =============================================
            # CUP
            # =============================================

            draw_virtual_cup(

                frame,

                mouse_x,
                mouse_y,

                orientation_deg,

                get_task_color(
                    current_task_result.state
                ),
            )


            # =============================================
            # GUIDANCE
            # =============================================

            if trial_active:

                draw_guidance(

                    frame,

                    current_guidance,

                    current_screen,
                )


        # =================================================
        # COMPARISON PANEL
        # =================================================

        if (
            trial_finished

            and

            final_task_result is not None

            and

            final_comparison is not None
        ):

            draw_comparison_panel(

                frame,

                final_task_result,

                final_comparison,

                final_guidance,
            )


        elif (
            current_task_result is not None

            and

            current_comparison is not None
        ):

            draw_comparison_panel(

                frame,

                current_task_result,

                current_comparison,

                current_guidance,
            )


        # =================================================
        # SUMMARY
        # =================================================

        if trial_finished:

            trial_state = (
                "FINISHED"
            )

            summary_to_show = (
                final_summary
            )


        elif trial_active:

            trial_state = (
                "ACTIVE"
            )

            summary_to_show = (
                live_summary
            )


        else:

            trial_state = (
                "READY"
            )

            summary_to_show = None


        draw_trial_summary(

            frame,

            summary_to_show,

            trial_state,
        )


        # =================================================
        # BOTTOM CONTROLS
        # =================================================

        draw_control_bar(
            frame
        )


        # =================================================
        # SHOW
        # =================================================

        cv2.imshow(
            WINDOW_NAME,
            frame
        )


        key = (
            cv2.waitKey(
                16
            )
            &
            0xFF
        )


        # =================================================
        # QUIT
        # =================================================

        if key in (
            ord("q"),
            ord("Q"),
        ):

            break


        # =================================================
        # START / RESTART
        # =================================================

        if key in (
            ord("s"),
            ord("S"),
        ):

            metrics.reset()

            trail_points = []

            violation_segments = []


            live_summary = None

            final_summary = None

            final_task_result = None

            final_comparison = None

            final_guidance = None


            previous_world_position = None

            previous_screen_position = None

            last_safe_world_position = None


            orientation_deg = 0.0


            trial_active = True

            trial_finished = False


        # =================================================
        # ROTATE
        # =================================================

        if key in (
            ord("a"),
            ord("A"),
        ):

            orientation_deg -= 2.0


        if key in (
            ord("d"),
            ord("D"),
        ):

            orientation_deg += 2.0


        # =================================================
        # RESET POSE
        # =================================================

        if key in (
            ord("r"),
            ord("R"),
        ):

            orientation_deg = 0.0


        # =================================================
        # CLEAR
        # =================================================

        if key in (
            ord("c"),
            ord("C"),
        ):

            metrics.reset()

            trail_points = []

            violation_segments = []


            orientation_deg = 0.0


            previous_world_position = None

            previous_screen_position = None

            last_safe_world_position = None


            live_summary = None

            final_summary = None

            final_task_result = None

            final_comparison = None

            final_guidance = None


            trial_active = False

            trial_finished = False


    cv2.destroyAllWindows()


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":

    main()