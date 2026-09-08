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

from interaction.intervention_comparator import (
    InterventionComparator,
    DECISION_STAY_QUIET,
    DECISION_INTERVENE,
)

from interaction.intent_manager import (
    IntentManager,
    CHOICE_KEEP,
    CHOICE_GUIDANCE,
)

from vision.hand_input_adapter import HandInputAdapter

from metrics.assistance_metrics import (
    AssistanceMetrics,
    OUTCOME_TASK_SUCCESS,
    OUTCOME_GOAL_WITH_VIOLATION,
    OUTCOME_VIOLATION_OCCURRED,
)

from metrics.runtime_logger import RuntimeLogger
from metrics.runtime_scenarios import RuntimeScenarioCatalog

from guidance.correction_guidance import (
    CorrectionGuidanceGenerator,
    CorrectionGuidanceResult,
    GUIDANCE_NONE,
    GUIDANCE_DIRECTION,
    GUIDANCE_WAYPOINT,
    GUIDANCE_ORIENTATION,
    GUIDANCE_PAUSE,
)

from guidance.scaffolding_controller import (
    ScaffoldingController,
    LEVEL_0,
    LEVEL_1,
    LEVEL_2,
    LEVEL_3,
)

from guidance.scaffolded_guidance import (
    ScaffoldedGuidanceMapper,
    MODE_NONE,
    MODE_MINIMAL_DIRECTION,
    MODE_MINIMAL_ORIENTATION,
    MODE_EXPLICIT_DIRECTION,
    MODE_EXPLICIT_WAYPOINT,
    MODE_EXPLICIT_ORIENTATION,
    MODE_STRONG_WAYPOINT,
    MODE_STRONG_ORIENTATION,
    MODE_RECOVERY_FADE,
    MODE_PAUSED,
)


# =========================================================
# EXTRA GUIDANCE
# =========================================================

GUIDANCE_RETURN_SAFE = "RETURN_TO_LAST_SAFE"


# =========================================================
# WINDOW
# =========================================================

WINDOW_NAME = "AdaptiveSkill - Adaptive Task Assistance"

INPUT_MODE_MOUSE = "MOUSE"
INPUT_MODE_HAND = "HAND"

BUILD_TAG = "WAFA_HAND_V6"

# Central camera interaction region.
# This avoids forcing the user's hand to the extreme bottom/edges of
# the camera frame, where MediaPipe tracking becomes less reliable.
CAMERA_ROI_X_MIN = 0.10
CAMERA_ROI_X_MAX = 0.90
CAMERA_ROI_Y_MIN = 0.12
CAMERA_ROI_Y_MAX = 0.78

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
# PANEL
# =========================================================

PANEL_LEFT = 880
PANEL_RIGHT = 1460


# =========================================================
# WORLD
# =========================================================

WORLD_X_MIN = -0.30
WORLD_X_MAX = 3.60

WORLD_Y_MIN = -1.50
WORLD_Y_MAX = 1.50


# =========================================================
# COLORS
# OpenCV = BGR
# =========================================================

COLOR_BACKGROUND = (247, 247, 247)

COLOR_TEXT = (45, 45, 45)

COLOR_SECONDARY = (115, 115, 115)

COLOR_VALID = (70, 175, 70)

COLOR_RISK = (0, 170, 255)

COLOR_VIOLATION = (65, 65, 225)

COLOR_START = (210, 120, 60)

COLOR_TARGET = (65, 170, 70)

COLOR_OBSTACLE = (70, 70, 70)

COLOR_SAFETY = (195, 195, 195)

COLOR_REFERENCE = (165, 165, 165)

COLOR_TRAIL = (120, 120, 120)

COLOR_CARD = (238, 238, 238)

COLOR_BORDER = (210, 210, 210)

COLOR_GUIDANCE = (0, 135, 235)

COLOR_GUIDANCE_STRONG = (60, 60, 230)

COLOR_GHOST = (75, 185, 90)

COLOR_CONTROL_BAR = (238, 238, 238)

COLOR_INTENT = (185, 115, 35)


# =========================================================
# FAST UI ASSETS / GHOST VISUALS
# UI ONLY: no tracking, input, task, policy, or timing logic.
# =========================================================

UI_ASSET_DIR = PROJECT_ROOT / "assets" / "ui"
UI_CUP_PATH = UI_ASSET_DIR / "cup.png"
UI_OBSTACLE_PATH = UI_ASSET_DIR / "obstacle.png"
UI_TARGET_PATH = UI_ASSET_DIR / "target.png"

_UI_IMAGE_CACHE = {}
_UI_TRANSFORM_CACHE = {}
_UI_CONTENT_CROP_CACHE = {}
_HAND_INSET_CENTER_X = None
_HAND_INSET_CENTER_Y = None

HAND_INSET_WIDTH = 160
HAND_INSET_HEIGHT = 96
HAND_INSET_MARGIN = 18

# The source crop follows the fingertip, but is biased slightly
# downward so the palm/wrist remain visible more often.
HAND_INSET_SOURCE_WIDTH_RATIO = 0.34
HAND_INSET_VERTICAL_BIAS_RATIO = 0.085
HAND_INSET_SMOOTHING_ALPHA = 0.20

# Low-latency hand-control stabilization.
# This smooths the CUP control signal at the UI loop rate.
# It does not change MediaPipe inference or the hand tracker thread.
HAND_CONTROL_DEADZONE_PX = 2.0
HAND_CONTROL_NEAR_PX = 8.0
HAND_CONTROL_FAR_PX = 28.0
HAND_CONTROL_ALPHA_NEAR = 0.34
HAND_CONTROL_ALPHA_MID = 0.62
HAND_CONTROL_ALPHA_FAR = 0.90
HAND_CONTROL_REACQUIRE_RESET_SEC = 0.18

COLOR_GHOST_BLUE = (205, 170, 125)
COLOR_GHOST_HALO = (235, 220, 202)
COLOR_SOFT_TEXT = (108, 112, 120)
COLOR_PANEL_WHITE = (252, 252, 252)
COLOR_PANEL_LINE = (224, 224, 224)
COLOR_READY_PILL = (226, 246, 231)
COLOR_READY_TEXT = (42, 145, 69)


# =========================================================
# CONFIG
# =========================================================

CONFIG_PATH = (
    PROJECT_ROOT
    / "task"
    / "task_config.json"
)


with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as file:

    TASK_CONFIG = json.load(file)


TASK_EVALUATOR = TaskConstraintEvaluator(
    CONFIG_PATH
)


GUIDANCE_GENERATOR = (
    CorrectionGuidanceGenerator(
        CONFIG_PATH
    )
)


SCAFFOLDING_CONTROLLER = (
    ScaffoldingController()
)


SCAFFOLDED_GUIDANCE_MAPPER = (
    ScaffoldedGuidanceMapper()
)


# =========================================================
# SELECTED VALIDATION SCENARIO
# =========================================================

def draw_selected_scenario(
    frame,
    selected_scenario,
):

    if selected_scenario is None:
        return

    draw_text(
        frame,
        (
            "Scenario "
            + str(selected_scenario["key"])
            + "  |  "
            + selected_scenario["id"]
        ),
        PANEL_LEFT + 15,
        72,
        scale=0.34,
        color=COLOR_SECONDARY,
        thickness=2,
    )


# =========================================================
# EXPERT PRIOR
# =========================================================

EXPERT_REFERENCE_POINTS = [

    (0.0, 0.0),

    (0.70, -0.75),

    (1.10, -0.90),

    (2.20, -0.90),

    (2.60, -0.60),

    (3.20, 0.0),
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
# MOUSE
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
# COORDINATES
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

    _draw_rounded_card(
        frame,
        x1,
        y1,
        x2,
        y2,
        radius=12,
        fill=COLOR_PANEL_WHITE,
        border=COLOR_PANEL_LINE,
    )


# =========================================================
# FAST UI-ONLY HELPERS
# =========================================================

def _load_ui_rgba(
    path,
):

    key = str(path)

    if key in _UI_IMAGE_CACHE:
        return _UI_IMAGE_CACHE[key]

    image = cv2.imread(
        key,
        cv2.IMREAD_UNCHANGED,
    )

    if (
        image is None
        or
        image.ndim != 3
        or
        image.shape[2] != 4
    ):

        _UI_IMAGE_CACHE[key] = None
        return None

    _UI_IMAGE_CACHE[key] = image
    return image


def _get_cached_sprite(
    rgba,
    size,
    angle_deg=0.0,
):

    if rgba is None:
        return None

    width = max(
        1,
        int(size[0]),
    )

    height = max(
        1,
        int(size[1]),
    )

    # A/D changes orientation in discrete steps. Rounding the cache key
    # avoids rebuilding the same sprite every frame.
    angle_key = round(
        float(angle_deg),
        1,
    )

    key = (
        id(rgba),
        width,
        height,
        angle_key,
    )

    cached = _UI_TRANSFORM_CACHE.get(
        key
    )

    if cached is not None:
        return cached

    sprite = cv2.resize(
        rgba,
        (
            width,
            height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    if abs(
        angle_key
    ) > 0.01:

        matrix = cv2.getRotationMatrix2D(
            (
                width / 2.0,
                height / 2.0,
            ),
            angle_key,
            1.0,
        )

        sprite = cv2.warpAffine(
            sprite,
            matrix,
            (
                width,
                height,
            ),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(
                0,
                0,
                0,
                0,
            ),
        )

    # Pre-convert once. Per-frame work becomes only a small ROI blend.
    rgb = sprite[
        :,
        :,
        :3
    ].astype(
        np.float32
    )

    alpha = (
        sprite[
            :,
            :,
            3:4
        ].astype(
            np.float32
        )
        /
        255.0
    )

    cached = (
        rgb,
        alpha,
        width,
        height,
    )

    _UI_TRANSFORM_CACHE[
        key
    ] = cached

    return cached


def _alpha_blend_cached(
    frame,
    rgba,
    center,
    size,
    opacity=1.0,
    angle_deg=0.0,
):

    cached = _get_cached_sprite(
        rgba,
        size,
        angle_deg,
    )

    if cached is None:
        return False

    rgb, alpha, width, height = (
        cached
    )

    x1 = int(
        center[0]
        -
        width / 2
    )

    y1 = int(
        center[1]
        -
        height / 2
    )

    x2 = x1 + width
    y2 = y1 + height

    frame_h, frame_w = (
        frame.shape[:2]
    )

    cx1 = max(
        0,
        x1,
    )

    cy1 = max(
        0,
        y1,
    )

    cx2 = min(
        frame_w,
        x2,
    )

    cy2 = min(
        frame_h,
        y2,
    )

    if (
        cx1 >= cx2
        or
        cy1 >= cy2
    ):

        return False

    sx1 = cx1 - x1
    sy1 = cy1 - y1

    sx2 = (
        sx1
        +
        cx2
        -
        cx1
    )

    sy2 = (
        sy1
        +
        cy2
        -
        cy1
    )

    rgb_crop = rgb[
        sy1:sy2,
        sx1:sx2,
    ]

    alpha_crop = alpha[
        sy1:sy2,
        sx1:sx2,
    ]

    if opacity != 1.0:

        alpha_crop = (
            alpha_crop
            *
            float(opacity)
        )

    target = frame[
        cy1:cy2,
        cx1:cx2,
    ].astype(
        np.float32
    )

    frame[
        cy1:cy2,
        cx1:cx2,
    ] = (
        rgb_crop
        *
        alpha_crop
        +
        target
        *
        (
            1.0
            -
            alpha_crop
        )
    ).astype(
        np.uint8
    )

    return True


def _crop_rgba_to_content(
    rgba,
    alpha_threshold=24,
):

    """
    UI only.

    Remove transparent padding from a sprite once, then cache it.
    This makes the visible obstacle fill the SAME rectangle that the
    task evaluator uses as the obstacle geometry.
    """

    if rgba is None:
        return None

    key = (
        id(rgba),
        int(alpha_threshold),
    )

    cached = _UI_CONTENT_CROP_CACHE.get(
        key
    )

    if cached is not None:
        return cached

    alpha = rgba[
        :,
        :,
        3
    ]

    ys, xs = np.where(
        alpha
        >
        int(alpha_threshold)
    )

    if (
        len(xs) == 0
        or
        len(ys) == 0
    ):

        _UI_CONTENT_CROP_CACHE[
            key
        ] = rgba

        return rgba

    x1 = int(
        xs.min()
    )

    x2 = int(
        xs.max()
    ) + 1

    y1 = int(
        ys.min()
    )

    y2 = int(
        ys.max()
    ) + 1

    cropped = rgba[
        y1:y2,
        x1:x2,
    ].copy()

    _UI_CONTENT_CROP_CACHE[
        key
    ] = cropped

    return cropped


def _clip_segment_to_rect(
    start,
    end,
    rect_left,
    rect_top,
    rect_right,
    rect_bottom,
):

    """
    Liang-Barsky segment clipping in screen coordinates.

    Returns:
        ((x1, y1), (x2, y2))
        or None if the segment never enters the rectangle.

    UI ONLY: this does not change task-state evaluation.
    """

    x0 = float(
        start[0]
    )

    y0 = float(
        start[1]
    )

    x1 = float(
        end[0]
    )

    y1 = float(
        end[1]
    )

    dx = (
        x1
        -
        x0
    )

    dy = (
        y1
        -
        y0
    )

    p = (
        -dx,
        dx,
        -dy,
        dy,
    )

    q = (
        x0 - float(rect_left),
        float(rect_right) - x0,
        y0 - float(rect_top),
        float(rect_bottom) - y0,
    )

    u_enter = 0.0
    u_exit = 1.0

    for pi, qi in zip(
        p,
        q,
    ):

        if abs(
            pi
        ) < 1e-12:

            if qi < 0.0:
                return None

            continue

        ratio = (
            qi
            /
            pi
        )

        if pi < 0.0:

            if ratio > u_exit:
                return None

            u_enter = max(
                u_enter,
                ratio,
            )

        else:

            if ratio < u_enter:
                return None

            u_exit = min(
                u_exit,
                ratio,
            )

    clipped_start = (
        int(
            round(
                x0
                +
                u_enter
                *
                dx
            )
        ),
        int(
            round(
                y0
                +
                u_enter
                *
                dy
            )
        ),
    )

    clipped_end = (
        int(
            round(
                x0
                +
                u_exit
                *
                dx
            )
        ),
        int(
            round(
                y0
                +
                u_exit
                *
                dy
            )
        ),
    )

    return (
        clipped_start,
        clipped_end,
    )


def _precise_obstacle_intersections(
    start,
    end,
):

    """
    Return only the portions of a sampled motion segment that lie
    inside the actual obstacle rectangle(s).

    This is used ONLY for the red violation visualization.
    """

    intersections = []

    for obstacle in TASK_CONFIG[
        "obstacles"
    ]:

        a = world_to_screen(
            obstacle[
                "x_min"
            ],
            obstacle[
                "y_min"
            ],
        )

        b = world_to_screen(
            obstacle[
                "x_max"
            ],
            obstacle[
                "y_max"
            ],
        )

        left = min(
            a[0],
            b[0],
        )

        right = max(
            a[0],
            b[0],
        )

        top = min(
            a[1],
            b[1],
        )

        bottom = max(
            a[1],
            b[1],
        )

        clipped = _clip_segment_to_rect(
            start,
            end,
            left,
            top,
            right,
            bottom,
        )

        if clipped is not None:

            clipped_start, clipped_end = (
                clipped
            )

            if math.hypot(
                clipped_end[0]
                -
                clipped_start[0],
                clipped_end[1]
                -
                clipped_start[1],
            ) >= 1.0:

                intersections.append(
                    clipped
                )

    return intersections


def _draw_fast_ghost_line(
    frame,
    start,
    end,
    color,
    halo_color,
    width=2,
):

    # Two direct OpenCV lines: ghost appearance without full-frame alpha.
    cv2.line(
        frame,
        start,
        end,
        halo_color,
        width + 4,
        cv2.LINE_AA,
    )

    cv2.line(
        frame,
        start,
        end,
        color,
        width,
        cv2.LINE_AA,
    )


def _draw_rounded_card(
    frame,
    x1,
    y1,
    x2,
    y2,
    radius=12,
    fill=COLOR_PANEL_WHITE,
    border=COLOR_PANEL_LINE,
):

    r = max(
        2,
        int(radius),
    )

    cv2.rectangle(
        frame,
        (
            x1 + r,
            y1,
        ),
        (
            x2 - r,
            y2,
        ),
        fill,
        -1,
    )

    cv2.rectangle(
        frame,
        (
            x1,
            y1 + r,
        ),
        (
            x2,
            y2 - r,
        ),
        fill,
        -1,
    )

    for cx, cy in (
        (
            x1 + r,
            y1 + r,
        ),
        (
            x2 - r,
            y1 + r,
        ),
        (
            x1 + r,
            y2 - r,
        ),
        (
            x2 - r,
            y2 - r,
        ),
    ):

        cv2.circle(
            frame,
            (
                cx,
                cy,
            ),
            r,
            fill,
            -1,
            cv2.LINE_AA,
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
        border,
        1,
        cv2.LINE_AA,
    )


def _draw_ready_pill(
    frame,
    x,
    y,
):

    _draw_rounded_card(
        frame,
        x,
        y,
        x + 88,
        y + 30,
        radius=15,
        fill=COLOR_READY_PILL,
        border=COLOR_READY_PILL,
    )

    draw_text(
        frame,
        "READY",
        x + 15,
        y + 21,
        scale=0.38,
        color=COLOR_READY_TEXT,
        thickness=2,
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


def get_level_color(
    level,
):

    if level == LEVEL_0:
        return COLOR_VALID

    if level == LEVEL_1:
        return COLOR_RISK

    if level == LEVEL_2:
        return COLOR_GUIDANCE

    if level == LEVEL_3:
        return COLOR_VIOLATION

    return COLOR_SECONDARY


# =========================================================
# TOP CONTROLS
# =========================================================

def draw_top_controls(
    frame,
):

    controls = [
        ("[S] Start", 35),
        ("[H] Hand", 175),
        ("[M] Mouse", 285),
        ("[C] Clear", 405),
        ("[Q] Quit", 520),
        ("[1-6] Scenario", 625),
    ]

    for text, x in controls:

        draw_text(
            frame,
            text,
            x,
            103,
            scale=0.36,
            color=COLOR_SECONDARY,
            thickness=1,
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

        x1, y1 = points[index]

        x2, y2 = points[
            index + 1
        ]


        length = math.hypot(
            x2 - x1,
            y2 - y1
        )


        if length < 1:
            continue


        step = 22.0
        dash = 10.0


        count = int(
            length / step
        ) + 1


        for number in range(count):

            d1 = number * step

            d2 = min(
                d1 + dash,
                length
            )


            if d1 >= length:
                break


            t1 = d1 / length
            t2 = d2 / length


            start = (

                int(
                    x1
                    +
                    (
                        x2 - x1
                    )
                    *
                    t1
                ),

                int(
                    y1
                    +
                    (
                        y2 - y1
                    )
                    *
                    t1
                ),
            )


            end = (

                int(
                    x1
                    +
                    (
                        x2 - x1
                    )
                    *
                    t2
                ),

                int(
                    y1
                    +
                    (
                        y2 - y1
                    )
                    *
                    t2
                ),
            )


            # Very light ghost halo + thin core.
            cv2.line(
                frame,
                start,
                end,
                (230, 222, 214),
                3,
                cv2.LINE_AA,
            )

            cv2.line(
                frame,
                start,
                end,
                (186, 161, 136),
                1,
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
        scale=0.38,
        color=(135, 128, 121),
    )


    draw_text(
        frame,
        "(not the only valid path)",
        label[0],
        label[1] + 19,
        scale=0.30,
        color=(142, 146, 154),
    )


# =========================================================
# TASK WORLD
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


    # START

    start = TASK_CONFIG[
        "start"
    ]


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


    # TARGET

    target = TASK_CONFIG[
        "target"
    ]


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


    target_sprite = _load_ui_rgba(
        UI_TARGET_PATH
    )

    target_drawn = _alpha_blend_cached(
        frame,
        target_sprite,
        target_screen,
        (
            int(
                radius_px
                *
                2.55
            ),
            int(
                radius_px
                *
                1.85
            ),
        ),
        opacity=0.96,
    )

    if not target_drawn:

        cv2.circle(
            frame,
            target_screen,
            radius_px,
            COLOR_TARGET,
            3,
            cv2.LINE_AA,
        )

        cv2.circle(
            frame,
            target_screen,
            5,
            COLOR_TARGET,
            -1,
            cv2.LINE_AA,
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


    # OBSTACLE

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


        obstacle_center = (
            int(
                (
                    obstacle_a[0]
                    +
                    obstacle_b[0]
                )
                /
                2
            ),
            int(
                (
                    obstacle_a[1]
                    +
                    obstacle_b[1]
                )
                /
                2
            ),
        )

        obstacle_width = abs(
            obstacle_b[0]
            -
            obstacle_a[0]
        )

        obstacle_height = abs(
            obstacle_b[1]
            -
            obstacle_a[1]
        )

        obstacle_sprite = _load_ui_rgba(
            UI_OBSTACLE_PATH
        )

        obstacle_sprite = _crop_rgba_to_content(
            obstacle_sprite
        )

        obstacle_drawn = _alpha_blend_cached(
            frame,
            obstacle_sprite,
            obstacle_center,
            (
                max(
                    1,
                    obstacle_width,
                ),
                max(
                    1,
                    obstacle_height,
                ),
            ),
            opacity=0.98,
        )

        if not obstacle_drawn:

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
            scale=0.36,
            color=(110, 96, 84),
            thickness=1,
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

    cup_sprite = _load_ui_rgba(
        UI_CUP_PATH
    )

    drawn = _alpha_blend_cached(
        frame,
        cup_sprite,
        center,
        (
            60,
            66,
        ),
        opacity=1.0,
        angle_deg=float(
            orientation_deg
        ),
    )

    if not drawn:

        # Fallback only if the PNG is missing.
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

        box = np.int32(
            cv2.boxPoints(
                rectangle
            )
        )

        cv2.polylines(
            frame,
            [box],
            True,
            color,
            3,
        )


# =========================================================
# TRAIL
# =========================================================

def draw_trail(
    frame,
    trail_points,
    violation_segments,
):

    if len(trail_points) >= 2:

        points = np.array(
            trail_points,
            dtype=np.int32,
        )

        cv2.polylines(
            frame,
            [points],
            False,
            (128, 132, 138),
            1,
            cv2.LINE_AA,
        )


    latest_violation_midpoint = None


    for start, end in violation_segments:

        precise_segments = (
            _precise_obstacle_intersections(
                start,
                end,
            )
        )

        for (
            precise_start,
            precise_end,
        ) in precise_segments:

            # The red line is now ONLY the exact part of the sampled
            # movement segment that lies inside the task obstacle.
            cv2.line(
                frame,
                precise_start,
                precise_end,
                COLOR_VIOLATION,
                4,
                cv2.LINE_AA,
            )

            latest_violation_midpoint = (

                int(
                    (
                        precise_start[0]
                        +
                        precise_end[0]
                    )
                    /
                    2
                ),

                int(
                    (
                        precise_start[1]
                        +
                        precise_end[1]
                    )
                    /
                    2
                ),
            )


    if latest_violation_midpoint is not None:

        draw_text(
            frame,
            "TASK VIOLATION",
            latest_violation_midpoint[0] + 10,
            latest_violation_midpoint[1] - 12,
            scale=0.34,
            color=COLOR_VIOLATION,
            thickness=2,
        )


# =========================================================
# RETURN SAFE
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

        direction_x = dx / length
        direction_y = dy / length


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
            "Return toward the last safe position."
        )
    )


# =========================================================
# SCAFFOLDED USER GUIDANCE
# =========================================================

def draw_scaffolded_guidance(
    frame,
    base_guidance,
    presentation,
    current_screen,
):

    if presentation is None:
        return


    # =====================================================
    # NONE
    # =====================================================

    if presentation.display_mode == MODE_NONE:
        return


    # =====================================================
    # PAUSED
    # =====================================================

    if presentation.display_mode == MODE_PAUSED:

        draw_text(
            frame,
            "PAUSED",
            current_screen[0] + 30,
            current_screen[1] - 45,
            scale=0.45,
            color=COLOR_SECONDARY,
            thickness=2,
        )

        return


    # =====================================================
    # RECOVERY
    # =====================================================

    if (
        presentation.display_mode
        ==
        MODE_RECOVERY_FADE
    ):

        draw_text(
            frame,
            "RECOVERING",
            current_screen[0] + 30,
            current_screen[1] - 45,
            scale=0.43,
            color=COLOR_VALID,
            thickness=2,
        )

        return


    # =====================================================
    # STRONG BANNER
    # =====================================================

    if presentation.show_strong_banner:

        cv2.rectangle(
            frame,
            (
                TASK_LEFT + 18,
                TASK_TOP + 18,
            ),
            (
                TASK_RIGHT - 18,
                TASK_TOP + 62,
            ),
            COLOR_VIOLATION,
            -1,
        )


        draw_text(
            frame,
            "L3 STRONG ASSISTANCE - FOLLOW THE TASK-SAFE CORRECTION",
            TASK_LEFT + 35,
            TASK_TOP + 48,
            scale=0.50,
            color=(
                255,
                255,
                255,
            ),
            thickness=2,
        )


    # =====================================================
    # MINIMAL ORIENTATION
    # =====================================================

    if (
        presentation.display_mode
        ==
        MODE_MINIMAL_ORIENTATION
    ):

        if (
            base_guidance.rotation_direction
            ==
            "COUNTERCLOCKWISE"
        ):

            text = "ROTATE CCW"


        elif (
            base_guidance.rotation_direction
            ==
            "CLOCKWISE"
        ):

            text = "ROTATE CW"


        else:

            text = "ADJUST ORIENTATION"


        draw_text(
            frame,
            text,
            current_screen[0] + 35,
            current_screen[1] - 50,
            scale=0.46,
            color=COLOR_GUIDANCE,
            thickness=2,
        )


        return


    # =====================================================
    # ORIENTATION GHOST
    # =====================================================

    if presentation.show_orientation_ghost:

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
            [ghost_box],
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


    # =====================================================
    # ROTATION TEXT
    # =====================================================

    if presentation.show_rotation_direction:

        if (
            base_guidance.rotation_direction
            ==
            "COUNTERCLOCKWISE"
        ):

            rotation_text = (
                "ROTATE CCW"
            )


        elif (
            base_guidance.rotation_direction
            ==
            "CLOCKWISE"
        ):

            rotation_text = (
                "ROTATE CW"
            )


        else:

            rotation_text = (
                "ALIGN UPRIGHT"
            )


        if presentation.show_rotation_degrees:

            rotation_text += (

                " "

                +
                f"{base_guidance.rotation_correction_deg:.0f} deg"
            )


        draw_text(
            frame,
            rotation_text,
            current_screen[0] + 35,
            current_screen[1] - 58,
            scale=0.46,
            color=COLOR_GUIDANCE,
            thickness=2,
        )


    # =====================================================
    # DIRECTION / WAYPOINT
    # =====================================================

    if presentation.show_direction_arrow:

        # -------------------------------------------------
        # Actual target available
        # -------------------------------------------------

        if (
            presentation.show_waypoint

            and

            base_guidance.target_x
            is not None

            and

            base_guidance.target_y
            is not None
        ):

            target = world_to_screen(
                base_guidance.target_x,
                base_guidance.target_y,
            )


            cv2.arrowedLine(
                frame,
                current_screen,
                target,
                (175, 195, 245),
                8,
                cv2.LINE_AA,
                tipLength=0.20,
            )

            cv2.arrowedLine(
                frame,
                current_screen,
                target,
                COLOR_GUIDANCE_STRONG,
                3,
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


            label = (
                "SAFE WAYPOINT"
            )


            if (
                base_guidance.guidance_type
                ==
                GUIDANCE_RETURN_SAFE
            ):

                label = (
                    "LAST SAFE POSITION"
                )


            draw_text(
                frame,
                label,
                target[0] + 15,
                target[1] - 15,
                scale=0.42,
                color=COLOR_GUIDANCE_STRONG,
                thickness=2,
            )


        # -------------------------------------------------
        # L1 short arrow
        # -------------------------------------------------

        else:

            arrow_length = 70


            target = (

                int(
                    current_screen[0]
                    +
                    base_guidance.direction_x
                    *
                    arrow_length
                ),

                int(
                    current_screen[1]
                    +
                    base_guidance.direction_y
                    *
                    arrow_length
                ),
            )


            cv2.arrowedLine(
                frame,
                current_screen,
                target,
                COLOR_GUIDANCE,
                4,
                cv2.LINE_AA,
                tipLength=0.28,
            )


# =========================================================
# COMPARISON PANEL
# =========================================================

def draw_comparison_panel(
    frame,
    task_result,
    comparison,
    scaffolding_decision,
    presentation,
):

    draw_text(
        frame,
        "CURRENT COMPARISON",
        PANEL_LEFT,
        132,
        scale=0.55,
        thickness=2,
    )


    # REFERENCE

    draw_card(
        frame,
        PANEL_LEFT,
        158,
        PANEL_RIGHT,
        243,
    )


    draw_text(
        frame,
        "REFERENCE-SIMILARITY BASELINE",
        PANEL_LEFT + 15,
        183,
        scale=0.38,
        color=COLOR_SECONDARY,
    )


    draw_text(
        frame,
        comparison.reference_policy_decision,
        PANEL_LEFT + 15,
        215,
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
        211,
        scale=0.36,
    )


    # TASK AWARE

    draw_card(
        frame,
        PANEL_LEFT,
        258,
        PANEL_RIGHT,
        373,
    )


    draw_text(
        frame,
        "TASK-AWARE ADAPTIVE POLICY",
        PANEL_LEFT + 15,
        283,
        scale=0.38,
        color=COLOR_SECONDARY,
    )


    draw_text(
        frame,
        comparison.task_policy_decision,
        PANEL_LEFT + 15,
        317,
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
        306,
        scale=0.35,
    )


    draw_text(
        frame,
        (
            "Constraint: "
            +
            task_result.dominant_constraint
        ),
        PANEL_LEFT + 300,
        330,
        scale=0.33,
        color=COLOR_SECONDARY,
    )


    if scaffolding_decision is not None:

        level_color = get_level_color(
            scaffolding_decision.level
        )


        draw_text(
            frame,
            (
                "ASSISTANCE L"
                +
                str(
                    scaffolding_decision.level
                )
                +
                " - "
                +
                scaffolding_decision.label
            ),
            PANEL_LEFT + 15,
            351,
            scale=0.38,
            color=level_color,
            thickness=2,
        )


    if presentation is not None:

        draw_text(
            frame,
            (
                "Presentation: "
                +
                presentation.display_mode
            ),
            PANEL_LEFT + 300,
            353,
            scale=0.31,
            color=COLOR_SECONDARY,
        )


    # RESULT

    draw_text(
        frame,
        "CURRENT RESULT",
        PANEL_LEFT,
        405,
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
        433,
        scale=0.47,
        color=result_color,
        thickness=2,
    )


# =========================================================
# QUICK START
# =========================================================

def draw_quick_start(
    frame,
    selected_scenario,
    input_mode,
):

    draw_text(
        frame,
        "QUICK START",
        PANEL_LEFT + 15,
        468,
        scale=0.52,
        thickness=2,
    )


    _draw_ready_pill(
        frame,
        PANEL_LEFT + 430,
        446,
    )


    if selected_scenario is not None:

        draw_text(
            frame,
            (
                "Selected: ["
                + str(selected_scenario["key"])
                + "] "
                + selected_scenario["id"]
            ),
            PANEL_LEFT + 18,
            492,
            scale=0.34,
            color=COLOR_START,
            thickness=2,
        )


    if input_mode == INPUT_MODE_HAND:

        instructions = [

            "1  Move your hand until the CUP reaches blue START.",

            "2  Press S when the cup is on START.",

            "3  Move your hand to guide the cup around the obstacle.",

            "4  Guide the cup into TARGET to finish.",
        ]

    else:

        instructions = [

            "1  Move cursor to START.",

            "2  Press S to begin.",

            "3  Carry the cup to TARGET.",

            "4  Avoid obstacle; A/D/R control orientation.",
        ]


    y = 530


    for text in instructions:

        draw_text(
            frame,
            text,
            PANEL_LEFT + 18,
            y,
            scale=0.41,
        )

        y += 42


# =========================================================
# SUMMARY
# =========================================================

def draw_trial_summary(
    frame,
    summary,
    trial_state,
    selected_scenario,
    input_mode,
):

    draw_card(
        frame,
        PANEL_LEFT,
        442,
        PANEL_RIGHT,
        745,
    )


    if trial_state == "READY":

        draw_quick_start(
            frame,
            selected_scenario,
            input_mode,
        )

        return


    draw_text(
        frame,
        "TRIAL SUMMARY",
        PANEL_LEFT + 15,
        468,
        scale=0.52,
        thickness=2,
    )


    if trial_state == "ACTIVE":

        state_color = (
            COLOR_RISK
        )


    elif trial_state == "RECOVERING":

        state_color = (
            COLOR_GUIDANCE
        )


    else:

        state_color = (
            COLOR_SECONDARY
        )


    draw_text(
        frame,
        trial_state,
        PANEL_LEFT + 430,
        468,
        scale=0.42,
        color=state_color,
        thickness=2,
    )


    if trial_state == "RECOVERING":

        draw_text(
            frame,
            "Goal reached - assistance is fading to L0",
            PANEL_LEFT + 15,
            486,
            scale=0.31,
            color=COLOR_GUIDANCE,
            thickness=1,
        )


    if summary is None:
        return


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
        500,
        scale=0.35,
        color=COLOR_SECONDARY,
    )


    draw_text(
        frame,
        summary.task_outcome,
        PANEL_LEFT + 135,
        500,
        scale=0.42,
        color=outcome_color,
        thickness=2,
    )


    rows = [

        (
            "Reference intervention episodes",
            summary.reference_intervention_episodes,
        ),

        (
            "Task-aware intervention episodes",
            summary.task_intervention_episodes,
        ),

        (
            "Unnecessary episodes avoided",
            summary.unnecessary_intervention_episodes_avoided,
        ),

        (
            "Task risks missed by baseline",
            summary.task_risk_missed_by_reference_episodes,
        ),

        (
            "Constraint violation episodes",
            summary.constraint_violation_episodes,
        ),
    ]


    y = 530


    for label, value in rows:

        draw_text(
            frame,
            (
                label
                +
                ": "
                +
                str(value)
            ),
            PANEL_LEFT + 15,
            y,
            scale=0.36,
        )

        y += 27


    violation_text = (

        "YES"

        if summary.constraint_violation_occurred

        else "NO"
    )


    draw_text(
        frame,
        (
            "Violation occurred: "
            +
            violation_text
        ),
        PANEL_LEFT + 15,
        673,
        scale=0.38,
        color=(
            COLOR_VIOLATION

            if summary.constraint_violation_occurred

            else COLOR_VALID
        ),
        thickness=2,
    )


    alternative_seen = (

        summary.alternative_valid_episodes

        >
        0
    )


    draw_text(
        frame,
        (
            "Alternative-valid episode: "
            +
            (
                "YES"

                if alternative_seen

                else "NO"
            )
        ),
        PANEL_LEFT + 15,
        701,
        scale=0.35,
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
        728,
        scale=0.35,
    )


    if (
        summary.task_intervention_episodes
        ==
        0
    ):

        recovery_text = (
            "Recovery needed: NO"
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


    draw_text(
        frame,
        recovery_text,
        PANEL_LEFT + 310,
        728,
        scale=0.35,
        color=COLOR_SECONDARY,
    )


# =========================================================
# CONTROL BAR
# =========================================================

def draw_control_bar(
    frame,
):

    cv2.rectangle(
        frame,
        (0, 765),
        (
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
        ),
        COLOR_CONTROL_BAR,
        -1,
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

    line_1 = (
        "[H] HAND MODE     "
        "[M] MOUSE MODE     "
        "[S] START     "
        "[K] KEEP     "
        "[G] GUIDANCE"
    )

    draw_text(
        frame,
        line_1,
        145,
        805,
        scale=0.36,
        color=COLOR_TEXT,
    )

    line_2 = (
        "Mouse mode only: [A/D] rotate, [R] upright     "
        "Hand mode: fingertip controls cup | live hand inset"
    )

    draw_text(
        frame,
        line_2,
        145,
        842,
        scale=0.33,
        color=COLOR_SECONDARY,
    )


# =========================================================
# INPUT STATUS
# =========================================================

def draw_input_status(
    frame,
    input_mode,
    hand_state,
):

    x1 = TASK_LEFT + 14
    y1 = TASK_TOP + 14

    if input_mode == INPUT_MODE_MOUSE:

        label = "MOUSE · VALIDATION"
        color = COLOR_SECONDARY
        width = 158

    else:

        if (
            hand_state is not None
            and
            hand_state.hand_detected
        ):

            if (
                hand_state.index_x is not None
                and
                hand_state.index_y is not None
                and
                CAMERA_ROI_X_MIN
                <=
                float(hand_state.index_x)
                <=
                CAMERA_ROI_X_MAX
                and
                CAMERA_ROI_Y_MIN
                <=
                float(hand_state.index_y)
                <=
                CAMERA_ROI_Y_MAX
            ):

                label = "HAND · TRACKING"
                color = COLOR_VALID
                width = 148

            else:

                label = "HAND · MOVE INTO VIEW"
                color = COLOR_RISK
                width = 190

        else:

            label = "TRACKING LOST · CUP FROZEN"
            color = COLOR_RISK
            width = 220


    x2 = x1 + width
    y2 = y1 + 28


    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (250, 250, 250),
        -1,
    )


    cv2.circle(
        frame,
        (
            x1 + 13,
            y1 + 14,
        ),
        4,
        color,
        -1,
        cv2.LINE_AA,
    )


    draw_text(
        frame,
        label,
        x1 + 24,
        y1 + 19,
        scale=0.29,
        color=color,
        thickness=1,
    )


# =========================================================
# LOW-LATENCY HAND CONTROL STABILIZATION
# =========================================================

def stabilize_hand_screen_position(
    current_position,
    target_position,
):

    """
    Adaptive UI-rate smoothing for the hand-controlled cup.

    Small fingertip jitter is damped strongly.
    Large deliberate movements follow quickly.

    This operates AFTER the existing MediaPipe fingertip mapping and
    BEFORE task evaluation, so the rendered cup and evaluated task
    position stay aligned.
    """

    if target_position is None:
        return current_position

    target_x = float(
        target_position[0]
    )

    target_y = float(
        target_position[1]
    )

    if current_position is None:

        return (
            target_x,
            target_y,
        )

    current_x = float(
        current_position[0]
    )

    current_y = float(
        current_position[1]
    )

    dx = (
        target_x
        -
        current_x
    )

    dy = (
        target_y
        -
        current_y
    )

    distance = math.hypot(
        dx,
        dy
    )

    # Ignore tiny sub-pixel / few-pixel landmark noise.
    if distance <= HAND_CONTROL_DEADZONE_PX:

        return (
            current_x,
            current_y,
        )

    if distance <= HAND_CONTROL_NEAR_PX:

        alpha = HAND_CONTROL_ALPHA_NEAR

    elif distance <= HAND_CONTROL_FAR_PX:

        alpha = HAND_CONTROL_ALPHA_MID

    else:

        alpha = HAND_CONTROL_ALPHA_FAR

    return (
        current_x
        +
        alpha
        *
        dx,

        current_y
        +
        alpha
        *
        dy,
    )


# =========================================================
# EMBEDDED CAMERA / HAND OVERLAY
# =========================================================

def draw_hand_camera_background(
    frame,
    camera_frame,
    hand_state,
):

    """
    Presentation-only live hand inset.

    The inset is intentionally placed in the TOP-RIGHT HEADER,
    outside the task workspace, so it cannot cover the expert prior,
    target, obstacle, user trail, or guidance.

    MediaPipe still uses the same full camera frame in the background.
    """

    global _HAND_INSET_CENTER_X
    global _HAND_INSET_CENTER_Y


    # -----------------------------------------------------
    # Header placement: outside task area and above panels.
    # -----------------------------------------------------

    inset_x2 = (
        PANEL_RIGHT
        -
        HAND_INSET_MARGIN
    )

    inset_x1 = (
        inset_x2
        -
        HAND_INSET_WIDTH
    )

    inset_y1 = 12

    inset_y2 = (
        inset_y1
        +
        HAND_INSET_HEIGHT
    )


    status_color = COLOR_SECONDARY

    if (
        hand_state is not None
        and
        hand_state.hand_detected
    ):

        status_color = COLOR_VALID


    # -----------------------------------------------------
    # Neutral placeholder on tracking loss.
    # Never expose the full camera frame.
    # -----------------------------------------------------

    if (
        camera_frame is None
        or
        hand_state is None
        or
        not hand_state.hand_detected
        or
        hand_state.index_x is None
        or
        hand_state.index_y is None
    ):

        _HAND_INSET_CENTER_X = None
        _HAND_INSET_CENTER_Y = None

        cv2.rectangle(
            frame,
            (
                inset_x1,
                inset_y1,
            ),
            (
                inset_x2,
                inset_y2,
            ),
            (245, 245, 245),
            -1,
        )

        cv2.rectangle(
            frame,
            (
                inset_x1,
                inset_y1,
            ),
            (
                inset_x2,
                inset_y2,
            ),
            (214, 214, 214),
            1,
            cv2.LINE_AA,
        )

        cv2.circle(
            frame,
            (
                inset_x1 + 12,
                inset_y1 + 14,
            ),
            4,
            COLOR_SECONDARY,
            -1,
            cv2.LINE_AA,
        )

        draw_text(
            frame,
            "LIVE HAND",
            inset_x1 + 22,
            inset_y1 + 18,
            scale=0.27,
            color=COLOR_SECONDARY,
            thickness=1,
        )

        draw_text(
            frame,
            "HAND NOT VISIBLE",
            inset_x1 + 21,
            inset_y1 + 57,
            scale=0.30,
            color=COLOR_SECONDARY,
            thickness=1,
        )

        draw_text(
            frame,
            "camera stays private",
            inset_x1 + 23,
            inset_y1 + 76,
            scale=0.23,
            color=(148, 148, 148),
            thickness=1,
        )

        return


    source_h, source_w = (
        camera_frame.shape[:2]
    )


    fingertip_x = (
        float(
            hand_state.index_x
        )
        *
        source_w
    )

    fingertip_y = (
        float(
            hand_state.index_y
        )
        *
        source_h
    )


    desired_center_x = (
        fingertip_x
    )

    desired_center_y = (
        fingertip_y
        +
        HAND_INSET_VERTICAL_BIAS_RATIO
        *
        source_h
    )


    # Display-only crop smoothing.
    if (
        _HAND_INSET_CENTER_X is None
        or
        _HAND_INSET_CENTER_Y is None
    ):

        _HAND_INSET_CENTER_X = (
            desired_center_x
        )

        _HAND_INSET_CENTER_Y = (
            desired_center_y
        )

    else:

        alpha = float(
            HAND_INSET_SMOOTHING_ALPHA
        )

        _HAND_INSET_CENTER_X = (
            (
                1.0 - alpha
            )
            *
            _HAND_INSET_CENTER_X
            +
            alpha
            *
            desired_center_x
        )

        _HAND_INSET_CENTER_Y = (
            (
                1.0 - alpha
            )
            *
            _HAND_INSET_CENTER_Y
            +
            alpha
            *
            desired_center_y
        )


    crop_width = max(
        80,
        int(
            source_w
            *
            HAND_INSET_SOURCE_WIDTH_RATIO
        ),
    )

    crop_height = max(
        60,
        int(
            crop_width
            *
            HAND_INSET_HEIGHT
            /
            HAND_INSET_WIDTH
        ),
    )


    center_x = int(
        round(
            _HAND_INSET_CENTER_X
        )
    )

    center_y = int(
        round(
            _HAND_INSET_CENTER_Y
        )
    )


    crop_x1 = (
        center_x
        -
        crop_width // 2
    )

    crop_y1 = (
        center_y
        -
        crop_height // 2
    )


    crop_x1 = max(
        0,
        min(
            source_w
            -
            crop_width,
            crop_x1,
        ),
    )

    crop_y1 = max(
        0,
        min(
            source_h
            -
            crop_height,
            crop_y1,
        ),
    )


    crop_x2 = min(
        source_w,
        crop_x1
        +
        crop_width,
    )

    crop_y2 = min(
        source_h,
        crop_y1
        +
        crop_height,
    )


    cropped = camera_frame[
        crop_y1:crop_y2,
        crop_x1:crop_x2,
    ]

    if cropped.size == 0:
        return


    inset = cv2.resize(
        cropped,
        (
            HAND_INSET_WIDTH,
            HAND_INSET_HEIGHT,
        ),
        interpolation=cv2.INTER_LINEAR,
    )


    frame[
        inset_y1:inset_y2,
        inset_x1:inset_x2,
    ] = inset


    # -----------------------------------------------------
    # Small research-demo label INSIDE the inset.
    # -----------------------------------------------------

    cv2.rectangle(
        frame,
        (
            inset_x1,
            inset_y1,
        ),
        (
            inset_x2,
            inset_y2,
        ),
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )

    cv2.rectangle(
        frame,
        (
            inset_x1 + 6,
            inset_y1 + 6,
        ),
        (
            inset_x1 + 84,
            inset_y1 + 24,
        ),
        (250, 250, 250),
        -1,
    )

    cv2.circle(
        frame,
        (
            inset_x1 + 14,
            inset_y1 + 15,
        ),
        4,
        status_color,
        -1,
        cv2.LINE_AA,
    )

    draw_text(
        frame,
        "LIVE HAND",
        inset_x1 + 23,
        inset_y1 + 19,
        scale=0.25,
        color=status_color,
        thickness=1,
    )


def hand_index_to_task_screen(
    index_x,
    index_y,
):

    index_x = float(
        index_x
    )

    index_y = float(
        index_y
    )

    # Do NOT clamp an out-of-zone fingertip to the border.
    # That was the cause of the "cup stuck on the frame edge" bug.
    if (
        index_x < CAMERA_ROI_X_MIN
        or
        index_x > CAMERA_ROI_X_MAX
        or
        index_y < CAMERA_ROI_Y_MIN
        or
        index_y > CAMERA_ROI_Y_MAX
    ):

        return None

    nx = (
        (
            index_x
            -
            CAMERA_ROI_X_MIN
        )
        /
        (
            CAMERA_ROI_X_MAX
            -
            CAMERA_ROI_X_MIN
        )
    )

    ny = (
        (
            index_y
            -
            CAMERA_ROI_Y_MIN
        )
        /
        (
            CAMERA_ROI_Y_MAX
            -
            CAMERA_ROI_Y_MIN
        )
    )

    x = int(
        TASK_LEFT
        +
        nx
        *
        (
            TASK_RIGHT
            -
            TASK_LEFT
        )
    )

    y = int(
        TASK_TOP
        +
        ny
        *
        (
            TASK_BOTTOM
            -
            TASK_TOP
        )
    )

    return (
        x,
        y,
    )


def draw_camera_roi(
    frame,
):

    # Final presentation UI: camera-workspace guidance is hidden.
    # The underlying crop/mapping logic is unchanged.
    return


def draw_hand_pointer(
    frame,
    current_screen,
    detected,
):

    # UI only: the cup itself is the hand cursor.
    return


# =========================================================
# INTENT CLARIFICATION UI
# =========================================================

def draw_intent_prompt(
    frame,
):

    x1 = TASK_LEFT + 45
    y1 = TASK_TOP + 25

    x2 = TASK_RIGHT - 45
    y2 = TASK_TOP + 135

    _draw_rounded_card(
        frame,
        x1,
        y1,
        x2,
        y2,
        radius=14,
        fill=(252, 252, 252),
        border=COLOR_INTENT,
    )

    draw_text(
        frame,
        "DIFFERENT, BUT TASK-VALID",
        x1 + 18,
        y1 + 28,
        scale=0.46,
        color=COLOR_INTENT,
        thickness=2,
    )

    draw_text(
        frame,
        "This path differs from the expert demonstration.",
        x1 + 18,
        y1 + 56,
        scale=0.39,
        color=COLOR_TEXT,
    )

    draw_text(
        frame,
        "Was this intentional?",
        x1 + 18,
        y1 + 80,
        scale=0.39,
        color=COLOR_TEXT,
    )

    draw_text(
        frame,
        "[K] KEEP MY STRATEGY",
        x1 + 18,
        y1 + 104,
        scale=0.39,
        color=COLOR_VALID,
        thickness=2,
    )

    draw_text(
        frame,
        "[G] SHOW GUIDANCE",
        x1 + 330,
        y1 + 104,
        scale=0.39,
        color=COLOR_GUIDANCE,
        thickness=2,
    )


# =========================================================
# USER-REQUESTED EXPERT-PRIOR GUIDANCE
#
# This is not safety correction and does not alter the
# adaptive L0-L3 controller. It is shown only because the
# user explicitly asked to see guidance.
# =========================================================

def get_next_expert_prior_point(
    current_x,
):

    current_x = float(
        current_x
    )

    for point_x, point_y in EXPERT_REFERENCE_POINTS:

        if point_x > current_x + 0.12:

            return (
                float(point_x),
                float(point_y),
            )

    return (
        float(
            EXPERT_REFERENCE_POINTS[-1][0]
        ),
        float(
            EXPERT_REFERENCE_POINTS[-1][1]
        ),
    )


def draw_user_requested_guidance(
    frame,
    current_screen,
    current_x,
):

    target_world = (
        get_next_expert_prior_point(
            current_x
        )
    )

    target_screen = world_to_screen(
        target_world[0],
        target_world[1],
    )

    cv2.arrowedLine(
        frame,
        current_screen,
        target_screen,
        (205, 225, 245),
        7,
        cv2.LINE_AA,
        tipLength=0.22,
    )

    cv2.arrowedLine(
        frame,
        current_screen,
        target_screen,
        COLOR_GUIDANCE,
        3,
        cv2.LINE_AA,
        tipLength=0.22,
    )

    cv2.circle(
        frame,
        target_screen,
        13,
        COLOR_GUIDANCE,
        2,
    )

    draw_text(
        frame,
        "USER-REQUESTED EXPERT-PRIOR GUIDANCE",
        current_screen[0] + 25,
        current_screen[1] - 45,
        scale=0.35,
        color=COLOR_GUIDANCE,
        thickness=2,
    )


# =========================================================
# INTENT SIDECAR LOG
#
# RuntimeLogger remains untouched. Intent events are saved
# beside the existing runtime summary so the validated
# logger schema is not disturbed.
# =========================================================

def save_intent_sidecar(
    runtime_result,
    selected_scenario,
    trial_summary,
    prompt_count,
    keep_count,
    guidance_count,
    accepted_alternative,
    intent_events,
):

    if runtime_result is None:

        return None

    summary_path_value = (
        runtime_result.get(
            "summary_path"
        )
    )

    if not summary_path_value:

        return None

    summary_path = Path(
        summary_path_value
    )

    output_path = summary_path.with_name(
        summary_path.stem
        +
        "_intent.json"
    )

    goal_reached = bool(
        getattr(
            trial_summary,
            "goal_reached_seen",
            False,
        )
    )

    violation_occurred = bool(
        getattr(
            trial_summary,
            "constraint_violation_occurred",
            False,
        )
    )

    task_intervention_episodes = int(
        getattr(
            trial_summary,
            "task_intervention_episodes",
            0,
        )
        or
        0
    )

    eligible_for_learning = (
        bool(
            accepted_alternative
        )
        and
        goal_reached
        and
        not violation_occurred
        and
        task_intervention_episodes
        ==
        0
    )

    payload = {
        "scenario_id": (
            selected_scenario["id"]
            if selected_scenario is not None
            else None
        ),
        "intent_prompt_count": int(
            prompt_count
        ),
        "keep_count": int(
            keep_count
        ),
        "guidance_request_count": int(
            guidance_count
        ),
        "user_confirmed_alternative": bool(
            accepted_alternative
        ),
        "goal_reached": goal_reached,
        "constraint_violation_occurred": (
            violation_occurred
        ),
        "task_intervention_episodes": int(
            task_intervention_episodes
        ),
        "eligible_for_downstream_learning": (
            eligible_for_learning
        ),
        "events": list(
            intent_events
        ),
        "claim_boundary": (
            "The system does not infer user intention. "
            "KEEP is an explicit user confirmation of an "
            "alternative strategy; GUIDANCE is an explicit "
            "user request for expert-prior guidance."
        ),
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return output_path


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print(
        "=============================================="
    )
    print(
        "AdaptiveSkill build:",
        BUILD_TAG
    )
    print(
        "Hand input: legacy raw fingertip tracking + matched crop + freeze-on-loss"
    )
    print(
        "=============================================="
    )
    print()

    cv2.namedWindow(
        WINDOW_NAME
    )


    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback
    )


    metrics = AssistanceMetrics()

    scaffolding = (
        ScaffoldingController()
    )


    mapper = (
        ScaffoldedGuidanceMapper()
    )


    runtime_logger = RuntimeLogger()


    scenario_catalog = (
        RuntimeScenarioCatalog()
    )


    selected_scenario = (
        scenario_catalog.get_default()
    )


    input_mode = INPUT_MODE_MOUSE

    hand_input = HandInputAdapter(
        project_root=PROJECT_ROOT,
        camera_index=0,
    )

    hand_state = None

    # Smoothed hand-controlled cup position.
    # The raw tracker remains untouched; this interpolation runs at
    # the UI loop rate to reduce visible step/jitter.
    hand_control_screen_position = None
    hand_last_missing_sec = 999.0


    intent_manager = IntentManager(
        config_path=CONFIG_PATH,
        reference_deviation_threshold=
            REFERENCE_THRESHOLD,
    )


    trail_points = []

    violation_segments = []


    trial_active = False

    trial_finished = False

    # -----------------------------------------------------
    # Post-goal recovery phase
    #
    # Reaching TARGET and finishing the assistance lifecycle
    # are intentionally separate events. If the user reaches
    # the goal while assistance is still fading, the task pose
    # is frozen at the valid goal state and the controller is
    # allowed to return to L0 before the trial is finalized.
    # -----------------------------------------------------

    post_goal_recovery = False

    goal_recovery_world_position = None

    goal_recovery_screen_position = None

    goal_recovery_orientation_deg = 0.0


    orientation_deg = 0.0


    previous_world_position = None

    previous_screen_position = None

    last_safe_world_position = None


    live_summary = None

    final_summary = None

    final_task_result = None

    final_comparison = None

    final_scaffolding = None

    final_presentation = None


    # -----------------------------------------------------
    # User-confirmed intent state
    # -----------------------------------------------------

    current_intent_decision = None

    intent_prompt_was_visible = False

    intent_prompt_count = 0

    intent_keep_count = 0

    intent_guidance_count = 0

    accepted_alternative = False

    user_guidance_until = 0.0

    intent_events = []


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
            "AdaptiveSkill - Human-Guided Robot Teaching",
            35,
            42,
            scale=0.86,
            thickness=2,
        )


        if post_goal_recovery:

            subtitle = (
                "Goal reached. Completing assistance fade before finalizing the trial."
            )


        elif trial_active:

            if input_mode == INPUT_MODE_HAND:

                subtitle = (
                    "Hand-controlled demonstration | live hand inset | tracking loss freezes the cup."
                )

            else:

                subtitle = (
                    "Minimal assistance for task risk; "
                    "immediate strong assistance for hard violations."
                )


        elif trial_finished:

            subtitle = (
                "Trial finished. Press S to start again."
            )


        else:

            if input_mode == INPUT_MODE_HAND:

                subtitle = (
                    "Hand mode: move your hand until the cup reaches START, then press S."
                )

            else:

                subtitle = (
                    "Select a validation scenario with 1-6, "
                    "move cursor to START, then press S."
                )


        draw_text(
            frame,
            subtitle,
            35,
            72,
            scale=0.49,
            color=COLOR_SECONDARY,
        )


        draw_top_controls(
            frame
        )


        draw_selected_scenario(
            frame,
            selected_scenario,
        )


        if input_mode == INPUT_MODE_HAND:

            hand_state = hand_input.update()

        else:

            hand_state = None


        draw_input_status(
            frame,
            input_mode,
            hand_state,
        )


        draw_task_world(
            frame
        )


        if input_mode == INPUT_MODE_HAND:

            draw_camera_roi(
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

        base_guidance = None

        scaffolding_decision = None

        presentation = None

        current_intent_decision = None


        # =================================================
        # GENERIC INPUT POSITION
        # =================================================

        input_position_available = False

        input_world_x = None
        input_world_y = None

        input_screen_position = None

        input_tracking_missing_sec = 0.0


        if input_mode == INPUT_MODE_MOUSE:

            if mouse_inside_task:

                input_world_x, input_world_y = (
                    screen_to_world(
                        mouse_x,
                        mouse_y,
                    )
                )

                input_screen_position = (
                    int(mouse_x),
                    int(mouse_y),
                )

                input_position_available = True


        elif (
            input_mode == INPUT_MODE_HAND

            and

            hand_state is not None

            and

            hand_state.index_x is not None

            and

            hand_state.index_y is not None

            and

            (
                hand_state.hand_detected
                or
                trial_active
            )
        ):

            raw_hand_screen_position = (
                hand_index_to_task_screen(
                    hand_state.index_x,
                    hand_state.index_y,
                )
            )

            input_tracking_missing_sec = float(
                hand_state.tracking_missing_sec
            )

            if raw_hand_screen_position is not None:

                if hand_state.hand_detected:

                    # If tracking was absent for a meaningful period,
                    # reset once on reacquisition instead of slowly
                    # dragging the cup from a stale coordinate.
                    if (
                        hand_control_screen_position is None
                        or
                        hand_last_missing_sec
                        >=
                        HAND_CONTROL_REACQUIRE_RESET_SEC
                    ):

                        hand_control_screen_position = (
                            float(
                                raw_hand_screen_position[0]
                            ),
                            float(
                                raw_hand_screen_position[1]
                            ),
                        )

                    else:

                        hand_control_screen_position = (
                            stabilize_hand_screen_position(
                                hand_control_screen_position,
                                raw_hand_screen_position,
                            )
                        )

                    hand_last_missing_sec = 0.0

                else:

                    hand_last_missing_sec = (
                        input_tracking_missing_sec
                    )

                    # Freeze exactly where the stabilized cup was.
                    if hand_control_screen_position is None:

                        hand_control_screen_position = (
                            float(
                                raw_hand_screen_position[0]
                            ),
                            float(
                                raw_hand_screen_position[1]
                            ),
                        )


                input_screen_position = (
                    int(
                        round(
                            hand_control_screen_position[0]
                        )
                    ),
                    int(
                        round(
                            hand_control_screen_position[1]
                        )
                    ),
                )

                input_world_x, input_world_y = (
                    screen_to_world(
                        input_screen_position[0],
                        input_screen_position[1],
                    )
                )

                input_position_available = True


        # =================================================
        # CURRENT POSITION
        # =================================================

        if (
            input_position_available
            or
            post_goal_recovery
        ):

            if post_goal_recovery:

                world_x, world_y = (
                    goal_recovery_world_position
                )

                current_screen = (
                    goal_recovery_screen_position
                )

                evaluation_orientation_deg = (
                    goal_recovery_orientation_deg
                )

                tracking_missing_sec = 0.0


            else:

                world_x = float(
                    input_world_x
                )

                world_y = float(
                    input_world_y
                )

                current_screen = input_screen_position

                if input_mode == INPUT_MODE_HAND:

                    # Position-only embodied demo:
                    # do not treat 2D hand angle as true cup pose.
                    evaluation_orientation_deg = 0.0

                else:

                    evaluation_orientation_deg = (
                        orientation_deg
                    )

                tracking_missing_sec = float(
                    input_tracking_missing_sec
                )


            current_point_result = (
                TASK_EVALUATOR.evaluate(

                    x=
                        world_x,

                    y=
                        world_y,

                    orientation_relative_deg=
                        evaluation_orientation_deg,

                    tracking_missing_sec=
                        tracking_missing_sec,
                )
            )


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
                            evaluation_orientation_deg,

                        tracking_missing_sec=
                            tracking_missing_sec,
                    )
                )


            else:

                current_task_result = (
                    current_point_result
                )


            segment_crossing = (

                trial_active

                and

                previous_world_position
                is not None

                and

                current_task_result.obstacle_violation

                and

                not current_point_result.obstacle_violation
            )


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


            # =================================================
            # BASE TASK-AWARE CORRECTION
            # =================================================

            if (
                segment_crossing

                and

                last_safe_world_position
                is not None
            ):

                base_guidance = (
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

                base_guidance = (
                    GUIDANCE_GENERATOR.generate(

                        x=
                            world_x,

                        y=
                            world_y,

                        orientation_relative_deg=
                            evaluation_orientation_deg,

                        task_result=
                            current_task_result,
                    )
                )


            # =================================================
            # ADAPTIVE SCAFFOLDING
            # =================================================

            if trial_active:

                now = time.perf_counter()


                # =================================================
                # MINIMAL INTENT CLARIFICATION
                #
                # Only ask when the path remains task-valid,
                # tracking is reliable, and deviation from the
                # expert prior persists. A previously confirmed
                # alternative remains accepted for this trial;
                # task risk can still trigger normal assistance.
                # =================================================

                if (
                    not post_goal_recovery

                    and

                    not accepted_alternative
                ):

                    current_intent_decision = (
                        intent_manager.update(

                            task_state=
                                current_task_result.state,

                            reference_distance=
                                current_comparison.reference_distance,

                            tracking_reliable=
                                (
                                    tracking_missing_sec
                                    <=
                                    float(
                                        TASK_CONFIG[
                                            "tracking"
                                        ][
                                            "max_missing_sec"
                                        ]
                                    )
                                ),

                            now=
                                now,
                        )
                    )


                    if (
                        current_intent_decision.prompt_visible

                        and

                        not intent_prompt_was_visible
                    ):

                        intent_prompt_count += 1

                        intent_events.append(
                            {
                                "event":
                                    "PROMPT_SHOWN",

                                "timestamp_sec":
                                    round(
                                        now,
                                        6,
                                    ),

                                "reference_distance":
                                    round(
                                        float(
                                            current_comparison.reference_distance
                                        ),
                                        4,
                                    ),

                                "task_state":
                                    current_task_result.state,
                            }
                        )


                    intent_prompt_was_visible = (
                        current_intent_decision.prompt_visible
                    )


                else:

                    intent_prompt_was_visible = False


                scaffolding_decision = (
                    scaffolding.update(

                        current_task_result,

                        now,
                    )
                )


                presentation = (
                    mapper.map(

                        task_result=
                            current_task_result,

                        scaffolding_decision=
                            scaffolding_decision,

                        base_guidance=
                            base_guidance,
                    )
                )


                # Trail

                if (
                    (
                        input_mode != INPUT_MODE_HAND

                        or

                        (
                            hand_state is not None
                            and
                            hand_state.hand_detected
                        )
                    )

                    and

                    (
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
                        4
                    )
                ):

                    trail_points.append(
                        current_screen
                    )


                if (
                    previous_screen_position
                    is not None

                    and

                    current_task_result
                    is not None

                    and

                    current_task_result.obstacle_violation
                ):

                    # UI bookkeeping only.
                    #
                    # Store the sampled motion segment whenever the
                    # task evaluator reports obstacle violation.
                    # draw_trail() clips it to the exact obstacle
                    # intersection before anything is drawn red.
                    #
                    # This is mode-agnostic, so HAND and MOUSE use
                    # the identical precise red-line visualization.
                    violation_segments.append(
                        (
                            previous_screen_position,
                            current_screen,
                        )
                    )


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


                if runtime_logger.active:

                    runtime_logger.log_frame(

                        timestamp_sec=
                            now,

                        x=
                            world_x,

                        y=
                            world_y,

                        orientation_deg=
                            evaluation_orientation_deg,

                        task_result=
                            current_task_result,

                        comparison=
                            current_comparison,

                        base_guidance=
                            base_guidance,

                        scaffolding_decision=
                            scaffolding_decision,

                        presentation=
                            presentation,

                        trial_summary=
                            live_summary,

                        segment_crossing_detected=
                            segment_crossing,
                    )


                if (
                    current_task_result.state
                    ==
                    STATE_VALID
                ):

                    last_safe_world_position = (
                        world_x,
                        world_y,
                    )


                previous_world_position = (
                    world_x,
                    world_y,
                )


                previous_screen_position = (
                    current_screen
                )


                # =================================================
                # GOAL / COMPLETION LIFECYCLE
                #
                # Goal completion and assistance completion are not
                # treated as the same event. If the user reaches the
                # target while assistance is still above L0, freeze the
                # valid target pose and continue feeding that valid state
                # to the scaffolding controller until the assistance has
                # fully faded back to L0. Only then finalize the trial.
                # =================================================

                if (
                    current_point_result.goal_reached

                    and

                    current_point_result.state
                    ==
                    STATE_VALID
                ):

                    if (
                        not post_goal_recovery

                        and

                        scaffolding_decision is not None

                        and

                        scaffolding_decision.level
                        >
                        LEVEL_0
                    ):

                        post_goal_recovery = True

                        goal_recovery_world_position = (
                            world_x,
                            world_y,
                        )

                        goal_recovery_screen_position = (
                            current_screen
                        )

                        goal_recovery_orientation_deg = (
                            evaluation_orientation_deg
                        )

                        print()
                        print(
                            "Goal reached. Waiting for assistance "
                            "to return to L0 before finalizing."
                        )
                        print()


                    if (
                        scaffolding_decision is not None

                        and

                        scaffolding_decision.level
                        ==
                        LEVEL_0
                    ):

                        final_summary = (
                            metrics.get_summary(
                                timestamp_sec=now
                            )
                        )


                        runtime_result = None

                        if runtime_logger.active:

                            runtime_result = (
                                runtime_logger.finish_trial(

                                    timestamp_sec=
                                        now,

                                    final_trial_summary=
                                        final_summary,
                                )
                            )


                        if runtime_result is not None:

                            intent_path = (
                                save_intent_sidecar(

                                    runtime_result=
                                        runtime_result,

                                    selected_scenario=
                                        selected_scenario,

                                    trial_summary=
                                        final_summary,

                                    prompt_count=
                                        intent_prompt_count,

                                    keep_count=
                                        intent_keep_count,

                                    guidance_count=
                                        intent_guidance_count,

                                    accepted_alternative=
                                        accepted_alternative,

                                    intent_events=
                                        intent_events,
                                )
                            )

                            print()

                            print(
                                "=============================================="
                            )

                            print(
                                "Runtime trial saved"
                            )

                            print(
                                "CSV:",
                                runtime_result["csv_path"]
                            )

                            print(
                                "Summary:",
                                runtime_result["summary_path"]
                            )

                            if intent_path is not None:

                                print(
                                    "Intent:",
                                    intent_path
                                )

                            print(
                                "=============================================="
                            )

                            print()


                        final_task_result = (
                            current_task_result
                        )


                        final_comparison = (
                            current_comparison
                        )


                        final_scaffolding = (
                            scaffolding_decision
                        )


                        final_presentation = (
                            presentation
                        )


                        trial_active = False

                        trial_finished = True

                        post_goal_recovery = False

                        goal_recovery_world_position = None

                        goal_recovery_screen_position = None

                        goal_recovery_orientation_deg = 0.0


            # =================================================
            # CUP
            # =================================================

            draw_virtual_cup(

                frame,

                current_screen[0],
                current_screen[1],

                evaluation_orientation_deg,

                get_task_color(
                    current_task_result.state
                ),
            )


            if (
                input_mode == INPUT_MODE_HAND

                and

                hand_state is not None

                and

                hand_state.hand_detected
            ):

                draw_hand_pointer(
                    frame,
                    current_screen,
                    True,
                )


            # =================================================
            # USER GUIDANCE
            # =================================================

            if (
                trial_active

                and

                presentation is not None
            ):

                draw_scaffolded_guidance(

                    frame,

                    base_guidance,

                    presentation,

                    current_screen,
                )


            # -------------------------------------------------
            # Intent prompt is only shown for task-valid
            # persistent reference deviation.
            # -------------------------------------------------

            if (
                trial_active

                and

                current_intent_decision is not None

                and

                current_intent_decision.prompt_visible
            ):

                draw_intent_prompt(
                    frame
                )


            # -------------------------------------------------
            # Guidance requested explicitly by the user.
            # This does not masquerade as an automatic error
            # diagnosis and does not alter the L0-L3 state.
            # -------------------------------------------------

            if (
                trial_active

                and

                not post_goal_recovery

                and

                current_task_result.state
                ==
                STATE_VALID

                and

                time.perf_counter()
                <
                user_guidance_until
            ):

                draw_user_requested_guidance(

                    frame,

                    current_screen,

                    world_x,
                )


        # =================================================
        # PANEL
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

                final_scaffolding,

                final_presentation,
            )


        elif (
            trial_active

            and

            current_task_result is not None

            and

            current_comparison is not None
        ):

            draw_comparison_panel(

                frame,

                current_task_result,

                current_comparison,

                scaffolding_decision,

                presentation,
            )


        # =================================================
        # SUMMARY
        # =================================================

        if trial_finished:

            trial_state = "FINISHED"

            summary = final_summary


        elif post_goal_recovery:

            trial_state = "RECOVERING"

            summary = live_summary


        elif trial_active:

            trial_state = "ACTIVE"

            summary = live_summary


        else:

            trial_state = "READY"

            summary = None


        draw_trial_summary(
            frame,
            summary,
            trial_state,
            selected_scenario,
            input_mode,
        )


        if input_mode == INPUT_MODE_HAND:

            draw_hand_camera_background(
                frame,
                hand_input.latest_frame,
                hand_state,
            )


        draw_control_bar(
            frame
        )


        cv2.imshow(
            WINDOW_NAME,
            frame
        )


        key = (
            cv2.waitKey(16)
            &
            0xFF
        )


        # QUIT

        if key in (
            ord("q"),
            ord("Q"),
        ):

            if runtime_logger.active:

                runtime_result = (
                    runtime_logger.finish_trial(

                        timestamp_sec=
                            time.perf_counter(),

                        final_trial_summary=
                            live_summary,
                    )
                )


                if runtime_result is not None:

                    intent_path = (
                        save_intent_sidecar(

                            runtime_result=
                                runtime_result,

                            selected_scenario=
                                selected_scenario,

                            trial_summary=
                                live_summary,

                            prompt_count=
                                intent_prompt_count,

                            keep_count=
                                intent_keep_count,

                            guidance_count=
                                intent_guidance_count,

                            accepted_alternative=
                                accepted_alternative,

                            intent_events=
                                intent_events,
                        )
                    )

                    print()

                    print(
                        "Runtime trial saved before quit:"
                    )

                    print(
                        "CSV:",
                        runtime_result["csv_path"]
                    )

                    print(
                        "Summary:",
                        runtime_result["summary_path"]
                    )

                    print()

            break


        # START

        if key in (
            ord("s"),
            ord("S"),
        ):

            if (
                input_mode == INPUT_MODE_HAND

                and

                (
                    hand_state is None
                    or
                    not hand_state.hand_detected
                )
            ):

                print()
                print(
                    "HAND MODE: no hand is currently detected."
                )
                print(
                    "Show one hand to the camera, move until the "
                    "CUP reaches START, then press S."
                )
                print()

                continue


            if (
                input_mode == INPUT_MODE_HAND

                and

                input_screen_position is None
            ):

                print()
                print(
                    "HAND MODE: fingertip is outside the active camera crop."
                )
                print(
                    "Move your hand into the tracking region "
                    "until HAND INPUT shows TRACKING."
                )
                print()

                continue


            if (
                input_mode == INPUT_MODE_HAND

                and

                input_world_x is not None

                and

                input_world_y is not None
            ):

                start_x = float(
                    TASK_CONFIG["start"]["x"]
                )

                start_y = float(
                    TASK_CONFIG["start"]["y"]
                )

                start_distance = math.hypot(
                    input_world_x - start_x,
                    input_world_y - start_y,
                )

                if start_distance > 0.35:

                    print()
                    print(
                        "HAND MODE: move your INDEX FINGERTIP "
                        "onto the blue START circle first."
                    )
                    print(
                        f"Current distance from START: "
                        f"{start_distance:.2f}"
                    )
                    print()

                    continue


            if runtime_logger.active:

                runtime_result = (
                    runtime_logger.finish_trial(

                        timestamp_sec=
                            time.perf_counter(),

                        final_trial_summary=
                            live_summary,
                    )
                )


                if runtime_result is not None:

                    intent_path = (
                        save_intent_sidecar(

                            runtime_result=
                                runtime_result,

                            selected_scenario=
                                selected_scenario,

                            trial_summary=
                                live_summary,

                            prompt_count=
                                intent_prompt_count,

                            keep_count=
                                intent_keep_count,

                            guidance_count=
                                intent_guidance_count,

                            accepted_alternative=
                                accepted_alternative,

                            intent_events=
                                intent_events,
                        )
                    )

                    print()

                    print(
                        "Previous runtime trial saved:"
                    )

                    print(
                        "CSV:",
                        runtime_result["csv_path"]
                    )

                    print(
                        "Summary:",
                        runtime_result["summary_path"]
                    )

                    print()


            metrics.reset()

            scaffolding.reset()

            intent_manager.reset()

            current_intent_decision = None

            intent_prompt_was_visible = False

            intent_prompt_count = 0

            intent_keep_count = 0

            intent_guidance_count = 0

            accepted_alternative = False

            user_guidance_until = 0.0

            intent_events = []

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

            final_scaffolding = None

            final_presentation = None

            post_goal_recovery = False

            goal_recovery_world_position = None

            goal_recovery_screen_position = None

            goal_recovery_orientation_deg = 0.0


            runtime_logger.start_trial(

                trial_label=
                    selected_scenario["id"],

                timestamp_sec=
                    time.perf_counter(),
            )


            print()
            print(
                "Validation scenario:",
                selected_scenario["id"],
                "-",
                selected_scenario["title"],
            )
            print(
                "Target:",
                selected_scenario["validation_target"],
            )
            print()


            trial_active = True

            trial_finished = False


        # =================================================
        # INPUT MODE
        # =================================================

        if key in (
            ord("h"),
            ord("H"),
        ):

            if trial_active:

                print(
                    "Input mode is locked during an active trial."
                )

            else:

                if hand_input.start():

                    input_mode = INPUT_MODE_HAND

                    hand_state = hand_input.update()

                    hand_control_screen_position = None
                    hand_last_missing_sec = 999.0

                    trail_points = []
                    violation_segments = []

                    previous_world_position = None
                    previous_screen_position = None
                    last_safe_world_position = None

                    trial_finished = False

                    print()
                    print("Input mode: HAND")
                    print(
                        "Index fingertip now controls the virtual "
                        "cup directly on the SAME task image."
                    )
                    print(
                        "Move your hand until the CUP reaches START, "
                        "then press S."
                    )
                    print(
                        "2D hand orientation is intentionally "
                        "not used as object-orientation ground truth."
                    )
                    print()

                else:

                    print()
                    print("Could not start hand camera.")
                    print()


        if key in (
            ord("m"),
            ord("M"),
        ):

            if trial_active:

                print(
                    "Input mode is locked during an active trial."
                )

            else:

                input_mode = INPUT_MODE_MOUSE

                hand_control_screen_position = None
                hand_last_missing_sec = 999.0

                trail_points = []
                violation_segments = []

                previous_world_position = None
                previous_screen_position = None
                last_safe_world_position = None

                trial_finished = False

                print()
                print(
                    "Input mode: MOUSE "
                    "(controlled validation harness)"
                )
                print()


        # =================================================
        # USER INTENT CHOICE
        # =================================================

        if (
            trial_active

            and

            current_intent_decision is not None

            and

            current_intent_decision.prompt_visible

            and

            key in (
                ord("k"),
                ord("K"),
            )
        ):

            choice_time = time.perf_counter()

            intent_choice = (
                intent_manager.choose_keep(
                    now=
                        choice_time
                )
            )

            if (
                intent_choice.choice
                ==
                CHOICE_KEEP
            ):

                accepted_alternative = True

                intent_keep_count += 1

                intent_prompt_was_visible = False

                intent_events.append(
                    {
                        "event":
                            "KEEP",

                        "timestamp_sec":
                            round(
                                choice_time,
                                6,
                            ),

                        "meaning":
                            "user_confirmed_alternative",
                    }
                )

                print()
                print(
                    "Intent choice: KEEP MY STRATEGY"
                )
                print(
                    "Alternative remains task-constrained; "
                    "future risk can still trigger assistance."
                )
                print()


        if (
            trial_active

            and

            current_intent_decision is not None

            and

            current_intent_decision.prompt_visible

            and

            key in (
                ord("g"),
                ord("G"),
            )
        ):

            choice_time = time.perf_counter()

            intent_choice = (
                intent_manager.choose_guidance(
                    now=
                        choice_time
                )
            )

            if (
                intent_choice.choice
                ==
                CHOICE_GUIDANCE
            ):

                intent_guidance_count += 1

                intent_prompt_was_visible = False

                user_guidance_until = (
                    choice_time
                    +
                    1.8
                )

                intent_events.append(
                    {
                        "event":
                            "GUIDANCE",

                        "timestamp_sec":
                            round(
                                choice_time,
                                6,
                            ),

                        "meaning":
                            "user_requested_expert_prior_guidance",
                    }
                )

                print()
                print(
                    "Intent choice: SHOW GUIDANCE"
                )
                print(
                    "Showing temporary expert-prior guidance "
                    "because the user explicitly requested it."
                )
                print()


        # =================================================
        # SCENARIO SELECTION
        #
        # 1-6 may be changed only when no trial is active.
        # Changing scenario clears the previous on-screen result,
        # but does not delete any saved runtime data.
        # =================================================

        if key in (
            ord("1"),
            ord("2"),
            ord("3"),
            ord("4"),
            ord("5"),
            ord("6"),
        ):

            scenario_key = chr(key)

            if trial_active:

                print(
                    "Scenario selection is locked during an active trial."
                )

            else:

                scenario = (
                    scenario_catalog.get_by_key(
                        scenario_key
                    )
                )

                if scenario is not None:

                    selected_scenario = scenario

                    # Clear only the screen state so the newly selected
                    # scenario is unambiguous. Saved logs remain intact.
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
                    final_scaffolding = None
                    final_presentation = None

                    intent_manager.reset()
                    current_intent_decision = None
                    intent_prompt_was_visible = False
                    intent_prompt_count = 0
                    intent_keep_count = 0
                    intent_guidance_count = 0
                    accepted_alternative = False
                    user_guidance_until = 0.0
                    intent_events = []

                    post_goal_recovery = False
                    goal_recovery_world_position = None
                    goal_recovery_screen_position = None
                    goal_recovery_orientation_deg = 0.0
                    trial_finished = False

                    print()
                    print(
                        "Selected validation scenario:",
                        selected_scenario["id"],
                        "-",
                        selected_scenario["title"],
                    )
                    print(
                        "Instruction:",
                        selected_scenario["instruction"],
                    )
                    print()


        # ROTATE

        if (
            input_mode == INPUT_MODE_MOUSE

            and

            not post_goal_recovery

            and

            key in (
                ord("a"),
                ord("A"),
            )
        ):

            orientation_deg -= 2.0


        if (
            input_mode == INPUT_MODE_MOUSE

            and

            not post_goal_recovery

            and

            key in (
                ord("d"),
                ord("D"),
            )
        ):

            orientation_deg += 2.0


        # UPRIGHT

        if (
            input_mode == INPUT_MODE_MOUSE

            and

            not post_goal_recovery

            and

            key in (
                ord("r"),
                ord("R"),
            )
        ):

            orientation_deg = 0.0


        # CLEAR

        if key in (
            ord("c"),
            ord("C"),
        ):

            if runtime_logger.active:

                runtime_result = (
                    runtime_logger.finish_trial(

                        timestamp_sec=
                            time.perf_counter(),

                        final_trial_summary=
                            live_summary,
                    )
                )


                if runtime_result is not None:

                    intent_path = (
                        save_intent_sidecar(

                            runtime_result=
                                runtime_result,

                            selected_scenario=
                                selected_scenario,

                            trial_summary=
                                live_summary,

                            prompt_count=
                                intent_prompt_count,

                            keep_count=
                                intent_keep_count,

                            guidance_count=
                                intent_guidance_count,

                            accepted_alternative=
                                accepted_alternative,

                            intent_events=
                                intent_events,
                        )
                    )

                    print()

                    print(
                        "Runtime trial saved before clear:"
                    )

                    print(
                        "CSV:",
                        runtime_result["csv_path"]
                    )

                    print(
                        "Summary:",
                        runtime_result["summary_path"]
                    )

                    print()


            metrics.reset()

            scaffolding.reset()

            intent_manager.reset()

            current_intent_decision = None

            intent_prompt_was_visible = False

            intent_prompt_count = 0

            intent_keep_count = 0

            intent_guidance_count = 0

            accepted_alternative = False

            user_guidance_until = 0.0

            intent_events = []


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

            final_scaffolding = None

            final_presentation = None

            post_goal_recovery = False

            goal_recovery_world_position = None

            goal_recovery_screen_position = None

            goal_recovery_orientation_deg = 0.0


            trial_active = False

            trial_finished = False


    hand_input.close()

    cv2.destroyAllWindows()


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":

    main()