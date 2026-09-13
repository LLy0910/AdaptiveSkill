import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple


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
    TaskConstraintResult,
    TaskConstraintEvaluator,
    STATE_VALID,
    STATE_RISK,
    STATE_VIOLATION,
    STATE_TRACKING_UNRELIABLE,
)


# =========================================================
# GUIDANCE TYPES
# =========================================================

GUIDANCE_NONE = "NONE"

GUIDANCE_DIRECTION = "DIRECTIONAL_NUDGE"

GUIDANCE_WAYPOINT = "SAFE_WAYPOINT"

GUIDANCE_ORIENTATION = "ORIENTATION_CORRECTION"

GUIDANCE_PAUSE = "PAUSE"


# =========================================================
# GUIDANCE RESULT
# =========================================================

@dataclass
class CorrectionGuidanceResult:
    """
    Actionable guidance generated from TASK CONSTRAINTS.

    IMPORTANT:
    Guidance is NOT generated from expert-reference
    similarity.

    Core principle:

        correct task risk
        without correcting a valid user strategy.
    """

    guidance_type: str

    should_intervene: bool

    target_x: Optional[float]
    target_y: Optional[float]

    direction_x: float
    direction_y: float

    rotation_direction: str
    rotation_correction_deg: float

    dominant_constraint: str

    message: str


    def to_dict(self):
        return asdict(self)


# =========================================================
# BASIC HELPERS
# =========================================================

def normalize_vector(
    x,
    y,
) -> Tuple[float, float]:
    """
    Return a unit vector.

    Tiny vectors become (0, 0).
    """

    length = math.hypot(
        float(x),
        float(y),
    )

    if length < 1e-9:

        return (
            0.0,
            0.0,
        )

    return (
        float(x) / length,
        float(y) / length,
    )


def circular_difference_deg(
    current_deg,
    target_deg,
):
    """
    Signed shortest angular correction:

        target - current

    Range:
        [-180, 180)
    """

    return (
        (
            float(target_deg)
            -
            float(current_deg)
            +
            180.0
        )
        %
        360.0
    ) - 180.0


def clamp(
    value,
    minimum,
    maximum,
):

    return max(
        minimum,
        min(
            maximum,
            value,
        )
    )


# =========================================================
# RECTANGLE HELPERS
# =========================================================

def point_inside_rectangle(
    x,
    y,
    obstacle,
):

    return (
        float(obstacle["x_min"])
        <=
        float(x)
        <=
        float(obstacle["x_max"])

        and

        float(obstacle["y_min"])
        <=
        float(y)
        <=
        float(obstacle["y_max"])
    )


def nearest_point_on_rectangle(
    x,
    y,
    obstacle,
):
    """
    Closest point on / in an axis-aligned rectangle.
    """

    nearest_x = clamp(
        float(x),
        float(obstacle["x_min"]),
        float(obstacle["x_max"]),
    )

    nearest_y = clamp(
        float(y),
        float(obstacle["y_min"]),
        float(obstacle["y_max"]),
    )

    return (
        nearest_x,
        nearest_y,
    )


# =========================================================
# CORRECTION GUIDANCE GENERATOR
# =========================================================

class CorrectionGuidanceGenerator:
    """
    Generates minimum actionable task-aware guidance.

    Core rule:

        TASK CONSTRAINT
        ->
        CORRECTION

    NOT:

        REFERENCE DEVIATION
        ->
        CORRECTION

    Therefore:

        valid lower route
        ->
        no correction

        lower route close to obstacle
        ->
        correction remains on lower side

        upper route close to obstacle
        ->
        correction remains on upper side
    """


    def __init__(
        self,
        config_path=None
    ):

        if config_path is None:

            config_path = (
                PROJECT_ROOT
                /
                "task"
                /
                "task_config.json"
            )


        self.config_path = Path(
            config_path
        )


        with open(
            self.config_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.config = json.load(
                file
            )


        self.obstacles = (
            self.config["obstacles"]
        )


        self.safety_margin = float(
            self.config[
                "safety_margin"
            ]
        )


        self.waypoint_offset = float(
            self.config[
                "guidance"
            ][
                "waypoint_offset"
            ]
        )


        # Small engineering buffer beyond the configured
        # safety margin.
        #
        # This is a prototype heuristic, NOT a validated
        # optimal value.
        self.safe_waypoint_buffer = 0.05


        orientation_config = (
            self.config[
                "orientation"
            ]
        )


        self.target_orientation_deg = float(
            orientation_config[
                "target_relative_deg"
            ]
        )


    # =====================================================
    # FIND OBSTACLE
    # =====================================================

    def _find_obstacle_by_id(
        self,
        obstacle_id
    ):

        if obstacle_id is None:

            return None


        for obstacle in self.obstacles:

            if (
                obstacle.get(
                    "id"
                )
                ==
                obstacle_id
            ):

                return obstacle


        return None


    # =====================================================
    # SAFE TARGET FOR OUTSIDE POINT
    # =====================================================

    def _safe_target_from_outside(
        self,
        x,
        y,
        obstacle,
    ):
        """
        User is outside obstacle but within its safety margin.

        Find nearest obstacle point, then move outward until
        the target lies beyond:

            obstacle
            +
            safety margin
            +
            small prototype buffer

        This preserves the side / strategy chosen by user.
        """

        nearest_x, nearest_y = (
            nearest_point_on_rectangle(
                x,
                y,
                obstacle
            )
        )


        away_x = (
            float(x)
            -
            nearest_x
        )


        away_y = (
            float(y)
            -
            nearest_y
        )


        direction_x, direction_y = (
            normalize_vector(
                away_x,
                away_y
            )
        )


        # Defensive fallback.
        if (
            abs(direction_x) < 1e-9
            and
            abs(direction_y) < 1e-9
        ):

            direction_x = 0.0
            direction_y = -1.0


        safe_distance = (
            self.safety_margin
            +
            self.safe_waypoint_buffer
        )


        target_x = (
            nearest_x
            +
            direction_x
            *
            safe_distance
        )


        target_y = (
            nearest_y
            +
            direction_y
            *
            safe_distance
        )


        return (
            direction_x,
            direction_y,
            target_x,
            target_y,
        )


    # =====================================================
    # SAFE TARGET FOR INSIDE POINT
    # =====================================================

    def _safe_target_from_inside(
        self,
        x,
        y,
        obstacle,
    ):
        """
        User is already inside obstacle.

        Choose the nearest exit boundary and place
        waypoint OUTSIDE:

            obstacle
            +
            safety margin
            +
            prototype buffer

        This means SAFE_WAYPOINT is actually outside
        the configured obstacle risk region.
        """

        x = float(x)
        y = float(y)


        x_min = float(
            obstacle["x_min"]
        )

        x_max = float(
            obstacle["x_max"]
        )

        y_min = float(
            obstacle["y_min"]
        )

        y_max = float(
            obstacle["y_max"]
        )


        distances = {

            "LEFT":
                abs(
                    x
                    -
                    x_min
                ),

            "RIGHT":
                abs(
                    x_max
                    -
                    x
                ),

            "TOP":
                abs(
                    y
                    -
                    y_min
                ),

            "BOTTOM":
                abs(
                    y_max
                    -
                    y
                ),
        }


        nearest_side = min(
            distances,
            key=distances.get
        )


        safe_offset = (
            self.safety_margin
            +
            self.safe_waypoint_buffer
        )


        if nearest_side == "LEFT":

            direction_x = -1.0
            direction_y = 0.0

            target_x = (
                x_min
                -
                safe_offset
            )

            target_y = clamp(
                y,
                y_min,
                y_max,
            )


        elif nearest_side == "RIGHT":

            direction_x = 1.0
            direction_y = 0.0

            target_x = (
                x_max
                +
                safe_offset
            )

            target_y = clamp(
                y,
                y_min,
                y_max,
            )


        elif nearest_side == "TOP":

            direction_x = 0.0
            direction_y = -1.0

            target_x = clamp(
                x,
                x_min,
                x_max,
            )

            target_y = (
                y_min
                -
                safe_offset
            )


        else:

            direction_x = 0.0
            direction_y = 1.0

            target_x = clamp(
                x,
                x_min,
                x_max,
            )

            target_y = (
                y_max
                +
                safe_offset
            )


        return (
            direction_x,
            direction_y,
            target_x,
            target_y,
        )


    # =====================================================
    # OBSTACLE GUIDANCE
    # =====================================================

    def _generate_obstacle_guidance(
        self,
        x,
        y,
        task_result
    ):
        """
        Generate correction AWAY from obstacle.

        Crucially:
        this does NOT use expert reference.

        Therefore:

        lower-side strategy
        ->
        remains lower-side strategy

        upper-side strategy
        ->
        remains upper-side strategy
        """

        obstacle = self._find_obstacle_by_id(
            task_result.nearest_obstacle_id
        )


        if obstacle is None:

            return CorrectionGuidanceResult(

                guidance_type=
                    GUIDANCE_NONE,

                should_intervene=
                    False,

                target_x=
                    None,

                target_y=
                    None,

                direction_x=
                    0.0,

                direction_y=
                    0.0,

                rotation_direction=
                    "NONE",

                rotation_correction_deg=
                    0.0,

                dominant_constraint=
                    "NONE",

                message=
                    "No obstacle guidance available."
            )


        # =================================================
        # OUTSIDE OBSTACLE
        # =================================================

        if not point_inside_rectangle(
            x,
            y,
            obstacle
        ):

            (
                direction_x,
                direction_y,
                target_x,
                target_y,
            ) = self._safe_target_from_outside(

                x=
                    x,

                y=
                    y,

                obstacle=
                    obstacle,
            )


        # =================================================
        # INSIDE OBSTACLE
        # =================================================

        else:

            (
                direction_x,
                direction_y,
                target_x,
                target_y,
            ) = self._safe_target_from_inside(

                x=
                    x,

                y=
                    y,

                obstacle=
                    obstacle,
            )


        # =================================================
        # MINIMUM GUIDANCE
        # =================================================

        if (
            task_result.state
            ==
            STATE_RISK
        ):

            guidance_type = (
                GUIDANCE_DIRECTION
            )

            message = (
                "Move slightly away from the obstacle."
            )


        else:

            guidance_type = (
                GUIDANCE_WAYPOINT
            )

            message = (
                "Move toward the local safe waypoint."
            )


        return CorrectionGuidanceResult(

            guidance_type=
                guidance_type,

            should_intervene=
                True,

            target_x=
                target_x,

            target_y=
                target_y,

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

            message=
                message
        )


    # =====================================================
    # ORIENTATION GUIDANCE
    # =====================================================

    def _generate_orientation_guidance(
        self,
        current_orientation_deg
    ):
        """
        Generate shortest correction toward
        task-safe upright orientation.
        """

        correction_deg = (
            circular_difference_deg(

                current_orientation_deg,

                self.target_orientation_deg
            )
        )


        if correction_deg > 0.0:

            rotation_direction = (
                "CLOCKWISE"
            )


        elif correction_deg < 0.0:

            rotation_direction = (
                "COUNTERCLOCKWISE"
            )


        else:

            rotation_direction = (
                "NONE"
            )


        return CorrectionGuidanceResult(

            guidance_type=
                GUIDANCE_ORIENTATION,

            should_intervene=
                True,

            target_x=
                None,

            target_y=
                None,

            direction_x=
                0.0,

            direction_y=
                0.0,

            rotation_direction=
                rotation_direction,

            rotation_correction_deg=
                abs(
                    correction_deg
                ),

            dominant_constraint=
                "ORIENTATION",

            message=(
                "Rotate the virtual object "
                "toward the upright orientation."
            )
        )


    # =====================================================
    # MAIN GENERATOR
    # =====================================================

    def generate(
        self,
        x,
        y,
        orientation_relative_deg,
        task_result: TaskConstraintResult,
    ):
        """
        Generate guidance for current task state.
        """

        # =================================================
        # TRACKING
        # =================================================

        if (
            task_result.state
            ==
            STATE_TRACKING_UNRELIABLE
        ):

            return CorrectionGuidanceResult(

                guidance_type=
                    GUIDANCE_PAUSE,

                should_intervene=
                    False,

                target_x=
                    None,

                target_y=
                    None,

                direction_x=
                    0.0,

                direction_y=
                    0.0,

                rotation_direction=
                    "NONE",

                rotation_correction_deg=
                    0.0,

                dominant_constraint=
                    "TRACKING",

                message=(
                    "Tracking is unreliable. "
                    "Guidance is paused."
                )
            )


        # =================================================
        # VALID
        #
        # A valid alternative strategy must remain untouched.
        # =================================================

        if (
            task_result.state
            ==
            STATE_VALID
        ):

            return CorrectionGuidanceResult(

                guidance_type=
                    GUIDANCE_NONE,

                should_intervene=
                    False,

                target_x=
                    None,

                target_y=
                    None,

                direction_x=
                    0.0,

                direction_y=
                    0.0,

                rotation_direction=
                    "NONE",

                rotation_correction_deg=
                    0.0,

                dominant_constraint=
                    "NONE",

                message=(
                    "No task-relevant correction is needed."
                )
            )


        # =================================================
        # OBSTACLE
        # =================================================

        if (
            task_result.dominant_constraint
            ==
            "OBSTACLE"
        ):

            return self._generate_obstacle_guidance(

                x=
                    float(x),

                y=
                    float(y),

                task_result=
                    task_result
            )


        # =================================================
        # ORIENTATION
        # =================================================

        if (
            task_result.dominant_constraint
            ==
            "ORIENTATION"
        ):

            return self._generate_orientation_guidance(

                current_orientation_deg=
                    float(
                        orientation_relative_deg
                    )
            )


        # =================================================
        # FALLBACK
        # =================================================

        return CorrectionGuidanceResult(

            guidance_type=
                GUIDANCE_NONE,

            should_intervene=
                False,

            target_x=
                None,

            target_y=
                None,

            direction_x=
                0.0,

            direction_y=
                0.0,

            rotation_direction=
                "NONE",

            rotation_correction_deg=
                0.0,

            dominant_constraint=
                task_result.dominant_constraint,

            message=
                "No actionable correction generated."
        )


# =========================================================
# SELF TEST
# =========================================================

if __name__ == "__main__":

    evaluator = (
        TaskConstraintEvaluator()
    )


    generator = (
        CorrectionGuidanceGenerator()
    )


    print()

    print(
        "========================================================"
    )

    print(
        "AdaptiveSkill - Correction Guidance v2"
    )

    print(
        "========================================================"
    )


    tests = [

        # -------------------------------------------------
        # 1. Alternative lower route
        # -------------------------------------------------

        (
            "ALTERNATIVE VALID LOWER ROUTE",

            {
                "x": 1.65,
                "y": 0.90,

                "orientation_relative_deg":
                    0.0,

                "tracking_missing_sec":
                    0.0,
            },

            GUIDANCE_NONE,
        ),


        # -------------------------------------------------
        # 2. Valid upper route
        # -------------------------------------------------

        (
            "VALID UPPER ROUTE",

            {
                "x": 1.65,
                "y": -0.90,

                "orientation_relative_deg":
                    0.0,

                "tracking_missing_sec":
                    0.0,
            },

            GUIDANCE_NONE,
        ),


        # -------------------------------------------------
        # 3. Lower obstacle risk
        # -------------------------------------------------

        (
            "LOWER ROUTE - OBSTACLE RISK",

            {
                "x": 1.65,
                "y": 0.55,

                "orientation_relative_deg":
                    0.0,

                "tracking_missing_sec":
                    0.0,
            },

            GUIDANCE_DIRECTION,
        ),


        # -------------------------------------------------
        # 4. Upper obstacle risk
        # -------------------------------------------------

        (
            "UPPER ROUTE - OBSTACLE RISK",

            {
                "x": 1.65,
                "y": -0.55,

                "orientation_relative_deg":
                    0.0,

                "tracking_missing_sec":
                    0.0,
            },

            GUIDANCE_DIRECTION,
        ),


        # -------------------------------------------------
        # 5. Inside obstacle
        # -------------------------------------------------

        (
            "INSIDE OBSTACLE",

            {
                "x": 1.65,
                "y": 0.20,

                "orientation_relative_deg":
                    0.0,

                "tracking_missing_sec":
                    0.0,
            },

            GUIDANCE_WAYPOINT,
        ),


        # -------------------------------------------------
        # 6. Positive orientation
        # -------------------------------------------------

        (
            "ORIENTATION +12 DEG",

            {
                "x": 0.70,
                "y": -0.75,

                "orientation_relative_deg":
                    12.0,

                "tracking_missing_sec":
                    0.0,
            },

            GUIDANCE_ORIENTATION,
        ),


        # -------------------------------------------------
        # 7. Negative orientation
        # -------------------------------------------------

        (
            "ORIENTATION -12 DEG",

            {
                "x": 0.70,
                "y": -0.75,

                "orientation_relative_deg":
                    -12.0,

                "tracking_missing_sec":
                    0.0,
            },

            GUIDANCE_ORIENTATION,
        ),


        # -------------------------------------------------
        # 8. Tracking unreliable
        # -------------------------------------------------

        (
            "TRACKING UNRELIABLE",

            {
                "x": 1.00,
                "y": -0.80,

                "orientation_relative_deg":
                    0.0,

                "tracking_missing_sec":
                    0.50,
            },

            GUIDANCE_PAUSE,
        ),
    ]


    passed = 0


    for (
        name,
        kwargs,
        expected_guidance_type,
    ) in tests:

        task_result = (
            evaluator.evaluate(
                **kwargs
            )
        )


        guidance = (
            generator.generate(

                x=
                    kwargs["x"],

                y=
                    kwargs["y"],

                orientation_relative_deg=
                    kwargs[
                        "orientation_relative_deg"
                    ],

                task_result=
                    task_result,
            )
        )


        # =================================================
        # TYPE CHECK
        # =================================================

        type_ok = (

            guidance.guidance_type

            ==

            expected_guidance_type
        )


        # =================================================
        # CRITICAL LOGIC CHECK
        # =================================================

        logic_ok = True


        # -------------------------------------------------
        # Valid lower alternative must remain untouched
        # -------------------------------------------------

        if (
            name
            ==
            "ALTERNATIVE VALID LOWER ROUTE"
        ):

            logic_ok = (
                not guidance.should_intervene
            )


        # -------------------------------------------------
        # Lower-side risk must stay lower
        # -------------------------------------------------

        if (
            name
            ==
            "LOWER ROUTE - OBSTACLE RISK"
        ):

            logic_ok = (
                guidance.direction_y
                >
                0.0
            )


        # -------------------------------------------------
        # Upper-side risk must stay upper
        # -------------------------------------------------

        if (
            name
            ==
            "UPPER ROUTE - OBSTACLE RISK"
        ):

            logic_ok = (
                guidance.direction_y
                <
                0.0
            )


        # -------------------------------------------------
        # Orientation +12
        # -------------------------------------------------

        if (
            name
            ==
            "ORIENTATION +12 DEG"
        ):

            logic_ok = (

                guidance.rotation_direction
                ==
                "COUNTERCLOCKWISE"

                and

                abs(
                    guidance.rotation_correction_deg
                    -
                    12.0
                )
                <
                0.001
            )


        # -------------------------------------------------
        # Orientation -12
        # -------------------------------------------------

        if (
            name
            ==
            "ORIENTATION -12 DEG"
        ):

            logic_ok = (

                guidance.rotation_direction
                ==
                "CLOCKWISE"

                and

                abs(
                    guidance.rotation_correction_deg
                    -
                    12.0
                )
                <
                0.001
            )


        # =================================================
        # WAYPOINT SAFETY CHECK
        #
        # Any obstacle guidance target should itself
        # be outside obstacle RISK / VIOLATION.
        # =================================================

        waypoint_safe = True

        waypoint_state = None


        if (
            guidance.target_x
            is not None

            and

            guidance.target_y
            is not None
        ):

            waypoint_result = (
                evaluator.evaluate(

                    x=
                        guidance.target_x,

                    y=
                        guidance.target_y,

                    orientation_relative_deg=
                        0.0,

                    tracking_missing_sec=
                        0.0,
                )
            )


            waypoint_state = (
                waypoint_result.state
            )


            waypoint_safe = (

                not waypoint_result.obstacle_risk

                and

                not waypoint_result.obstacle_violation
            )


        # =================================================
        # FINAL CHECK
        # =================================================

        ok = (

            type_ok

            and

            logic_ok

            and

            waypoint_safe
        )


        if ok:

            passed += 1


        # =================================================
        # PRINT
        # =================================================

        print()

        print(
            name
        )

        print(
            "-" * 56
        )

        print(
            "Task state:",
            task_result.state
        )

        print(
            "Dominant constraint:",
            task_result.dominant_constraint
        )

        print(
            "Expected guidance:",
            expected_guidance_type
        )

        print(
            "Actual guidance:",
            guidance.guidance_type
        )

        print(
            "Should intervene:",
            guidance.should_intervene
        )

        print(
            "Direction:",
            (
                round(
                    guidance.direction_x,
                    3
                ),
                round(
                    guidance.direction_y,
                    3
                )
            )
        )


        if (
            guidance.target_x
            is not None
        ):

            print(
                "Safe waypoint:",
                (
                    round(
                        guidance.target_x,
                        3
                    ),
                    round(
                        guidance.target_y,
                        3
                    )
                )
            )

            print(
                "Waypoint task state:",
                waypoint_state
            )

            print(
                "Waypoint outside obstacle risk:",
                waypoint_safe
            )


        print(
            "Rotation:",
            guidance.rotation_direction,
            round(
                guidance.rotation_correction_deg,
                2
            ),
            "deg"
        )

        print(
            "Message:",
            guidance.message
        )

        print(
            "PASS:",
            ok
        )


    # =====================================================
    # FINAL
    # =====================================================

    print()

    print(
        "========================================================"
    )

    print(
        "RESULT:",
        f"{passed}/{len(tests)} tests passed"
    )


    if (
        passed
        ==
        len(tests)
    ):

        print(
            "CORRECTION GUIDANCE v2: PASSED"
        )

    else:

        print(
            "CORRECTION GUIDANCE v2: FAILED"
        )


    print(
        "========================================================"
    )

    print()