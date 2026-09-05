import json
import math
import sys
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


# =========================================================
# WINDOW
# =========================================================

WINDOW_NAME = (
    "AdaptiveSkill - Intervention Comparator"
)

WINDOW_WIDTH = 1380
WINDOW_HEIGHT = 760


# =========================================================
# TASK AREA
# =========================================================

TASK_LEFT = 70
TASK_RIGHT = 870

TASK_TOP = 120
TASK_BOTTOM = 620


# =========================================================
# RIGHT PANEL
# =========================================================

PANEL_LEFT = 900
PANEL_RIGHT = 1350


# =========================================================
# WORLD COORDINATES
# =========================================================

WORLD_X_MIN = -0.30
WORLD_X_MAX = 3.60

WORLD_Y_MIN = -1.50
WORLD_Y_MAX = 1.50


# =========================================================
# COLORS
#
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

COLOR_PAUSE = (
    140,
    140,
    140,
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

COLOR_PANEL_BORDER = (
    210,
    210,
    210,
)

COLOR_CARD = (
    238,
    238,
    238,
)

COLOR_HIGHLIGHT = (
    70,
    160,
    70,
)

COLOR_WARNING = (
    0,
    150,
    220,
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
    encoding="utf-8",
) as file:
    TASK_CONFIG = json.load(
        file
    )


TASK_EVALUATOR = (
    TaskConstraintEvaluator(
        CONFIG_PATH
    )
)


# =========================================================
# EXPERT PRIOR
#
# IMPORTANT:
# This is an expert-reference PRIOR,
# not the only correct trajectory.
#
# It intentionally takes the upper route.
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


# =========================================================
# TRAJECTORY
# =========================================================

trail_points = []

MAX_TRAIL_POINTS = 700


# =========================================================
# WORLD <-> SCREEN
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
    screen_x,
    screen_y,
):
    x_ratio = (
        (
            float(screen_x)
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
            float(screen_y)
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
# MOUSE CALLBACK
# =========================================================

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

    mouse_x = int(x)
    mouse_y = int(y)

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
# TEXT
# =========================================================

def draw_text(
    frame,
    text,
    x,
    y,
    scale=0.52,
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
# STATE COLORS
# =========================================================

def get_task_state_color(
    state,
):
    if state == STATE_VALID:
        return COLOR_VALID

    if state == STATE_RISK:
        return COLOR_RISK

    if state == STATE_VIOLATION:
        return COLOR_VIOLATION

    return COLOR_PAUSE


def get_decision_color(
    decision,
):
    if decision == DECISION_STAY_QUIET:
        return COLOR_VALID

    if decision == DECISION_INTERVENE:
        return COLOR_VIOLATION

    return COLOR_PAUSE


# =========================================================
# DRAW EXPERT PRIOR
# =========================================================

def draw_expert_reference(
    frame,
):
    screen_points = [
        world_to_screen(
            x,
            y,
        )
        for x, y
        in EXPERT_REFERENCE_POINTS
    ]

    # Draw dashed-looking line using short segments.
    for index in range(
        len(screen_points) - 1
    ):
        x1, y1 = screen_points[
            index
        ]

        x2, y2 = screen_points[
            index + 1
        ]

        segment_length = math.hypot(
            x2 - x1,
            y2 - y1,
        )

        if segment_length < 1:
            continue

        dash_length = 12.0
        gap_length = 8.0

        step_length = (
            dash_length
            +
            gap_length
        )

        steps = int(
            segment_length
            /
            step_length
        ) + 1

        for step in range(
            steps
        ):
            start_distance = (
                step
                *
                step_length
            )

            end_distance = min(
                start_distance
                +
                dash_length,

                segment_length,
            )

            if (
                start_distance
                >=
                segment_length
            ):
                break

            t1 = (
                start_distance
                /
                segment_length
            )

            t2 = (
                end_distance
                /
                segment_length
            )

            sx = int(
                x1
                +
                (
                    x2
                    -
                    x1
                )
                *
                t1
            )

            sy = int(
                y1
                +
                (
                    y2
                    -
                    y1
                )
                *
                t1
            )

            ex = int(
                x1
                +
                (
                    x2
                    -
                    x1
                )
                *
                t2
            )

            ey = int(
                y1
                +
                (
                    y2
                    -
                    y1
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

    label_point = world_to_screen(
        1.20,
        -1.10,
    )

    draw_text(
        frame,
        "EXPERT PRIOR",
        label_point[0],
        label_point[1],
        scale=0.42,
        color=COLOR_REFERENCE,
        thickness=1,
    )

    draw_text(
        frame,
        "(not the only valid path)",
        label_point[0],
        label_point[1] + 21,
        scale=0.34,
        color=COLOR_REFERENCE,
        thickness=1,
    )


# =========================================================
# DRAW START
# =========================================================

def draw_start(
    frame,
):
    start = TASK_CONFIG[
        "start"
    ]

    center = world_to_screen(
        start["x"],
        start["y"],
    )

    cv2.circle(
        frame,
        center,
        15,
        COLOR_START,
        3,
    )

    cv2.circle(
        frame,
        center,
        4,
        COLOR_START,
        -1,
    )

    draw_text(
        frame,
        "START",
        center[0] - 28,
        center[1] - 24,
        scale=0.48,
        color=COLOR_START,
        thickness=2,
    )


# =========================================================
# DRAW TARGET
# =========================================================

def draw_target(
    frame,
):
    target = TASK_CONFIG[
        "target"
    ]

    center = world_to_screen(
        target["x"],
        target["y"],
    )

    edge = world_to_screen(
        target["x"]
        +
        target["radius"],
        target["y"],
    )

    radius_px = abs(
        edge[0]
        -
        center[0]
    )

    cv2.circle(
        frame,
        center,
        radius_px,
        COLOR_TARGET,
        3,
    )

    cv2.circle(
        frame,
        center,
        5,
        COLOR_TARGET,
        -1,
    )

    draw_text(
        frame,
        "TARGET",
        center[0] - 32,
        center[1] - radius_px - 12,
        scale=0.48,
        color=COLOR_TARGET,
        thickness=2,
    )


# =========================================================
# DRAW OBSTACLES
# =========================================================

def draw_obstacles(
    frame,
):
    safety_margin = float(
        TASK_CONFIG[
            "safety_margin"
        ]
    )

    for obstacle in TASK_CONFIG[
        "obstacles"
    ]:

        safe_top_left = world_to_screen(
            obstacle["x_min"]
            -
            safety_margin,

            obstacle["y_min"]
            -
            safety_margin,
        )

        safe_bottom_right = world_to_screen(
            obstacle["x_max"]
            +
            safety_margin,

            obstacle["y_max"]
            +
            safety_margin,
        )

        cv2.rectangle(
            frame,
            safe_top_left,
            safe_bottom_right,
            COLOR_SAFETY,
            2,
        )

        top_left = world_to_screen(
            obstacle["x_min"],
            obstacle["y_min"],
        )

        bottom_right = world_to_screen(
            obstacle["x_max"],
            obstacle["y_max"],
        )

        cv2.rectangle(
            frame,
            top_left,
            bottom_right,
            COLOR_OBSTACLE,
            -1,
        )

        draw_text(
            frame,
            "OBSTACLE",
            top_left[0],
            top_left[1] - 10,
            scale=0.42,
            color=COLOR_OBSTACLE,
            thickness=2,
        )


# =========================================================
# DRAW TASK WORLD
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
        COLOR_PANEL_BORDER,
        2,
    )

    draw_expert_reference(
        frame
    )

    draw_start(
        frame
    )

    draw_target(
        frame
    )

    draw_obstacles(
        frame
    )


# =========================================================
# DRAW VIRTUAL CUP
# =========================================================

def draw_virtual_cup(
    frame,
    center_x,
    center_y,
    orientation_deg,
    color,
):
    center = (
        int(center_x),
        int(center_y),
    )

    cup_width = 28
    cup_height = 42

    rectangle = (
        center,
        (
            cup_width,
            cup_height,
        ),
        float(orientation_deg),
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

    # 0 degrees = visually upright.
    display_angle_deg = (
        orientation_deg
        -
        90.0
    )

    angle_rad = math.radians(
        display_angle_deg
    )

    needle_length = 52

    end_x = int(
        center[0]
        +
        needle_length
        *
        math.cos(
            angle_rad
        )
    )

    end_y = int(
        center[1]
        +
        needle_length
        *
        math.sin(
            angle_rad
        )
    )

    cv2.line(
        frame,
        center,
        (
            end_x,
            end_y,
        ),
        color,
        3,
    )


# =========================================================
# TRAIL
# =========================================================

def draw_trail(
    frame,
):
    if len(
        trail_points
    ) < 2:
        return

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
        COLOR_PANEL_BORDER,
        1,
    )


# =========================================================
# COMPARATOR PANEL
# =========================================================

def draw_comparator_panel(
    frame,
    task_result,
    comparison,
    world_x,
    world_y,
    orientation_deg,
):
    # =====================================================
    # PANEL TITLE
    # =====================================================

    draw_text(
        frame,
        "INTERVENTION COMPARATOR",
        PANEL_LEFT,
        120,
        scale=0.58,
        thickness=2,
    )

    draw_text(
        frame,
        "Research comparison view",
        PANEL_LEFT,
        145,
        scale=0.38,
        color=COLOR_SECONDARY,
    )


    # =====================================================
    # REFERENCE BASELINE CARD
    # =====================================================

    draw_card(
        frame,
        PANEL_LEFT,
        170,
        PANEL_RIGHT,
        285,
    )

    draw_text(
        frame,
        "REFERENCE-SIMILARITY BASELINE",
        PANEL_LEFT + 15,
        198,
        scale=0.43,
        color=COLOR_SECONDARY,
        thickness=1,
    )

    reference_color = (
        get_decision_color(
            comparison.reference_policy_decision
        )
    )

    draw_text(
        frame,
        comparison.reference_policy_decision,
        PANEL_LEFT + 15,
        233,
        scale=0.68,
        color=reference_color,
        thickness=2,
    )

    draw_text(
        frame,
        (
            "Reference distance: "
            +
            f"{comparison.reference_distance:.2f}"
        ),
        PANEL_LEFT + 15,
        263,
        scale=0.42,
    )

    draw_text(
        frame,
        (
            "Threshold: "
            +
            f"{REFERENCE_THRESHOLD:.2f}"
        ),
        PANEL_LEFT + 240,
        263,
        scale=0.40,
        color=COLOR_SECONDARY,
    )


    # =====================================================
    # TASK-AWARE CARD
    # =====================================================

    draw_card(
        frame,
        PANEL_LEFT,
        305,
        PANEL_RIGHT,
        435,
    )

    draw_text(
        frame,
        "TASK-AWARE POLICY",
        PANEL_LEFT + 15,
        333,
        scale=0.43,
        color=COLOR_SECONDARY,
    )

    task_decision_color = (
        get_decision_color(
            comparison.task_policy_decision
        )
    )

    draw_text(
        frame,
        comparison.task_policy_decision,
        PANEL_LEFT + 15,
        368,
        scale=0.68,
        color=task_decision_color,
        thickness=2,
    )

    draw_text(
        frame,
        (
            "Task state: "
            +
            task_result.state
        ),
        PANEL_LEFT + 15,
        399,
        scale=0.42,
    )

    draw_text(
        frame,
        (
            "Constraint: "
            +
            task_result.dominant_constraint
        ),
        PANEL_LEFT + 15,
        424,
        scale=0.40,
        color=COLOR_SECONDARY,
    )


    # =====================================================
    # COMPARISON RESULT
    # =====================================================

    if (
        comparison.unnecessary_intervention_avoided
    ):

        result_color = (
            COLOR_HIGHLIGHT
        )

    elif (
        comparison.task_relevant_intervention
    ):

        result_color = (
            COLOR_WARNING
        )

    else:

        result_color = (
            COLOR_SECONDARY
        )


    draw_text(
        frame,
        "COMPARISON",
        PANEL_LEFT,
        475,
        scale=0.42,
        color=COLOR_SECONDARY,
    )


    # Wrap long labels.
    if (
        comparison.comparison_label
        ==
        "UNNECESSARY INTERVENTION AVOIDED"
    ):

        draw_text(
            frame,
            "UNNECESSARY",
            PANEL_LEFT,
            510,
            scale=0.65,
            color=result_color,
            thickness=2,
        )

        draw_text(
            frame,
            "INTERVENTION AVOIDED",
            PANEL_LEFT,
            540,
            scale=0.58,
            color=result_color,
            thickness=2,
        )

    elif (
        comparison.comparison_label
        ==
        "TASK-RELEVANT INTERVENTION"
    ):

        draw_text(
            frame,
            "TASK-RELEVANT",
            PANEL_LEFT,
            510,
            scale=0.62,
            color=result_color,
            thickness=2,
        )

        draw_text(
            frame,
            "INTERVENTION",
            PANEL_LEFT,
            540,
            scale=0.62,
            color=result_color,
            thickness=2,
        )

    else:

        draw_text(
            frame,
            comparison.comparison_label,
            PANEL_LEFT,
            520,
            scale=0.55,
            color=result_color,
            thickness=2,
        )


    # =====================================================
    # DEBUG
    # =====================================================

    draw_text(
        frame,
        (
            f"x={world_x:.2f}  "
            f"y={world_y:.2f}"
        ),
        PANEL_LEFT,
        585,
        scale=0.38,
        color=COLOR_SECONDARY,
    )

    draw_text(
        frame,
        (
            "Orientation error: "
            +
            f"{task_result.orientation_error_deg:.1f} deg"
        ),
        PANEL_LEFT,
        610,
        scale=0.38,
        color=COLOR_SECONDARY,
    )


# =========================================================
# MAIN
# =========================================================

def main():

    global trail_points


    cv2.namedWindow(
        WINDOW_NAME
    )

    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback
    )


    # =====================================================
    # SIMULATED CUP ORIENTATION
    # =====================================================

    orientation_deg = 0.0


    while True:

        # =================================================
        # BACKGROUND
        # =================================================

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

        draw_text(
            frame,
            (
                "Does deviation from an expert demonstration "
                "actually require intervention?"
            ),
            35,
            72,
            scale=0.53,
            color=COLOR_SECONDARY,
        )


        # =================================================
        # TASK WORLD
        # =================================================

        draw_task_world(
            frame
        )

        draw_trail(
            frame
        )


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


            # =============================================
            # TASK CONSTRAINTS
            # =============================================

            task_result = (
                TASK_EVALUATOR.evaluate(
                    x=world_x,
                    y=world_y,
                    orientation_relative_deg=
                        orientation_deg,
                    tracking_missing_sec=0.0,
                )
            )


            # =============================================
            # INTERVENTION COMPARISON
            # =============================================

            comparison = (
                COMPARATOR.compare(
                    x=world_x,
                    y=world_y,
                    task_result=
                        task_result,
                )
            )


            # =============================================
            # TRAIL
            # =============================================

            current_point = (
                int(mouse_x),
                int(mouse_y),
            )

            if (
                not trail_points
                or
                math.hypot(
                    current_point[0]
                    -
                    trail_points[-1][0],

                    current_point[1]
                    -
                    trail_points[-1][1],
                )
                >=
                4.0
            ):

                trail_points.append(
                    current_point
                )


            if (
                len(
                    trail_points
                )
                >
                MAX_TRAIL_POINTS
            ):

                trail_points = (
                    trail_points[
                        -MAX_TRAIL_POINTS:
                    ]
                )


            # =============================================
            # CURRENT CUP
            # =============================================

            state_color = (
                get_task_state_color(
                    task_result.state
                )
            )

            draw_virtual_cup(
                frame,
                mouse_x,
                mouse_y,
                orientation_deg,
                state_color,
            )


            # =============================================
            # RESEARCH PANEL
            # =============================================

            draw_comparator_panel(
                frame,
                task_result,
                comparison,
                world_x,
                world_y,
                orientation_deg,
            )


        else:

            draw_text(
                frame,
                "Move mouse into task area",
                PANEL_LEFT,
                210,
                scale=0.52,
            )


        # =================================================
        # CONTROLS
        # =================================================

        draw_text(
            frame,
            "Mouse = move virtual cup",
            50,
            690,
            scale=0.45,
            color=COLOR_SECONDARY,
        )

        draw_text(
            frame,
            "A / D = rotate",
            280,
            690,
            scale=0.45,
            color=COLOR_SECONDARY,
        )

        draw_text(
            frame,
            "R = reset pose",
            455,
            690,
            scale=0.45,
            color=COLOR_SECONDARY,
        )

        draw_text(
            frame,
            "C = clear trail",
            625,
            690,
            scale=0.45,
            color=COLOR_SECONDARY,
        )

        draw_text(
            frame,
            "Q = quit",
            790,
            690,
            scale=0.45,
            color=COLOR_SECONDARY,
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
        # KEYBOARD
        # =================================================

        if key in (
            ord("q"),
            ord("Q"),
        ):

            break


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


        if key in (
            ord("r"),
            ord("R"),
        ):

            orientation_deg = 0.0


        if key in (
            ord("c"),
            ord("C"),
        ):

            trail_points = []


    cv2.destroyAllWindows()


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":

    main()