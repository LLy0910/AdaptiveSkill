import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional


# =========================================================
# TASK STATES
# =========================================================

STATE_VALID = "VALID"
STATE_RISK = "RISK"
STATE_VIOLATION = "VIOLATION"
STATE_TRACKING_UNRELIABLE = "TRACKING_UNRELIABLE"


# =========================================================
# RESULT
# =========================================================

@dataclass
class TaskConstraintResult:
    """
    Current task-constraint evaluation.

    IMPORTANT:
    This is NOT:
        - psychological-state detection
        - user competence estimation
        - complete demonstration-quality estimation
        - expert-reference correctness

    It only describes whether the current movement
    remains safe / valid with respect to the task.
    """

    state: str

    task_valid_so_far: bool
    tracking_reliable: bool

    goal_reached: bool
    distance_to_goal: float

    obstacle_violation: bool
    obstacle_risk: bool
    obstacle_clearance: float
    nearest_obstacle_id: Optional[str]

    orientation_violation: bool
    orientation_risk: bool
    orientation_error_deg: float

    dominant_constraint: str
    reason: str


    def to_dict(self):

        return asdict(self)


# =========================================================
# BASIC GEOMETRY
# =========================================================

def distance(
    x1,
    y1,
    x2,
    y2
):

    return math.hypot(
        float(x2) - float(x1),
        float(y2) - float(y1)
    )


def circular_difference_deg(
    angle_deg,
    target_deg
):

    """
    Signed shortest angular difference.

    Result range:
        [-180, 180)
    """

    return (
        (
            float(angle_deg)
            -
            float(target_deg)
            +
            180.0
        )
        %
        360.0
    ) - 180.0


def point_inside_rectangle(
    x,
    y,
    obstacle
):

    return (
        obstacle["x_min"]
        <=
        x
        <=
        obstacle["x_max"]

        and

        obstacle["y_min"]
        <=
        y
        <=
        obstacle["y_max"]
    )


def point_to_rectangle_distance(
    x,
    y,
    obstacle
):

    """
    Shortest Euclidean distance between
    a point and an axis-aligned rectangle.

    Inside rectangle:
        0.0
    """

    if point_inside_rectangle(
        x,
        y,
        obstacle
    ):

        return 0.0


    dx = max(
        obstacle["x_min"] - x,
        0.0,
        x - obstacle["x_max"]
    )


    dy = max(
        obstacle["y_min"] - y,
        0.0,
        y - obstacle["y_max"]
    )


    return math.hypot(
        dx,
        dy
    )


# =========================================================
# TASK CONSTRAINT EVALUATOR
# =========================================================

class TaskConstraintEvaluator:
    """
    Task-based evaluator for the Move-the-Cup demo.

    Current constraints:

        1. Goal region
        2. Obstacle collision
        3. Obstacle safety margin
        4. Virtual-object orientation
        5. Tracking reliability

    Reference-trajectory similarity is intentionally
    NOT used as the definition of correctness.
    """


    def __init__(
        self,
        config_path=None
    ):

        # -------------------------------------------------
        # Load configuration
        # -------------------------------------------------

        if config_path is None:

            config_path = (
                Path(__file__).resolve().parent
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
        ) as f:

            self.config = json.load(
                f
            )


        # -------------------------------------------------
        # Task configuration
        # -------------------------------------------------

        self.target = (
            self.config["target"]
        )


        self.obstacles = (
            self.config["obstacles"]
        )


        self.safety_margin = float(
            self.config["safety_margin"]
        )


        orientation_config = (
            self.config["orientation"]
        )


        self.target_orientation_deg = float(
            orientation_config[
                "target_relative_deg"
            ]
        )


        self.orientation_mild_deg = float(
            orientation_config[
                "mild_tolerance_deg"
            ]
        )


        self.orientation_strong_deg = float(
            orientation_config[
                "strong_tolerance_deg"
            ]
        )


        self.max_missing_sec = float(
            self.config[
                "tracking"
            ][
                "max_missing_sec"
            ]
        )


    # =====================================================
    # EVALUATE
    # =====================================================

    def evaluate(
        self,
        x,
        y,
        orientation_relative_deg,
        tracking_missing_sec=0.0
    ):

        x = float(x)
        y = float(y)

        orientation_relative_deg = float(
            orientation_relative_deg
        )

        tracking_missing_sec = float(
            tracking_missing_sec
        )


        # =================================================
        # 1. TRACKING RELIABILITY
        # =================================================

        tracking_reliable = (

            tracking_missing_sec

            <=

            self.max_missing_sec
        )


        if not tracking_reliable:

            return TaskConstraintResult(

                state=
                    STATE_TRACKING_UNRELIABLE,

                task_valid_so_far=
                    False,

                tracking_reliable=
                    False,

                goal_reached=
                    False,

                distance_to_goal=
                    float("nan"),

                obstacle_violation=
                    False,

                obstacle_risk=
                    False,

                obstacle_clearance=
                    float("nan"),

                nearest_obstacle_id=
                    None,

                orientation_violation=
                    False,

                orientation_risk=
                    False,

                orientation_error_deg=
                    float("nan"),

                dominant_constraint=
                    "TRACKING",

                reason=(
                    "Tracking is temporarily unreliable. "
                    "Task judgement is paused."
                )
            )


        # =================================================
        # 2. GOAL
        # =================================================

        target_x = float(
            self.target["x"]
        )


        target_y = float(
            self.target["y"]
        )


        target_radius = float(
            self.target["radius"]
        )


        distance_to_goal = distance(

            x,
            y,

            target_x,
            target_y
        )


        goal_reached = (

            distance_to_goal

            <=

            target_radius
        )


        # =================================================
        # 3. OBSTACLE
        # =================================================

        obstacle_violation = False

        obstacle_risk = False

        nearest_clearance = float(
            "inf"
        )

        nearest_obstacle_id = None


        for obstacle in self.obstacles:

            inside = point_inside_rectangle(

                x,
                y,
                obstacle
            )


            clearance = (
                point_to_rectangle_distance(

                    x,
                    y,
                    obstacle
                )
            )


            # ---------------------------------------------
            # Track nearest obstacle
            # ---------------------------------------------

            if (
                clearance
                <
                nearest_clearance
            ):

                nearest_clearance = (
                    clearance
                )

                nearest_obstacle_id = (
                    obstacle.get(
                        "id",
                        "obstacle"
                    )
                )


            # ---------------------------------------------
            # Collision
            # ---------------------------------------------

            if inside:

                obstacle_violation = True


            # ---------------------------------------------
            # Safety-margin risk
            # ---------------------------------------------

            elif (
                clearance
                <=
                self.safety_margin
            ):

                obstacle_risk = True


        # No obstacle configuration.
        if not self.obstacles:

            nearest_clearance = float(
                "inf"
            )

            nearest_obstacle_id = None


        # =================================================
        # 4. ORIENTATION
        # =================================================

        orientation_difference = (
            circular_difference_deg(

                orientation_relative_deg,

                self.target_orientation_deg
            )
        )


        orientation_error_deg = abs(
            orientation_difference
        )


        # ---------------------------------------------
        # Strong violation
        # ---------------------------------------------

        orientation_violation = (

            orientation_error_deg

            >

            self.orientation_strong_deg
        )


        # ---------------------------------------------
        # Mild risk
        # ---------------------------------------------

        orientation_risk = (

            not orientation_violation

            and

            orientation_error_deg
            >
            self.orientation_mild_deg
        )


        # =================================================
        # 5. COMBINE CONSTRAINTS
        # =================================================

        if (
            obstacle_violation
            or
            orientation_violation
        ):

            state = (
                STATE_VIOLATION
            )


        elif (
            obstacle_risk
            or
            orientation_risk
        ):

            state = (
                STATE_RISK
            )


        else:

            state = (
                STATE_VALID
            )


        # =================================================
        # TASK VALID SO FAR
        # =================================================

        task_valid_so_far = (

            not obstacle_violation

            and

            not orientation_violation
        )


        # =================================================
        # DOMINANT CONSTRAINT
        # =================================================

        dominant_constraint = "NONE"


        if obstacle_violation:

            dominant_constraint = (
                "OBSTACLE"
            )


        elif orientation_violation:

            dominant_constraint = (
                "ORIENTATION"
            )


        elif obstacle_risk:

            dominant_constraint = (
                "OBSTACLE"
            )


        elif orientation_risk:

            dominant_constraint = (
                "ORIENTATION"
            )


        elif goal_reached:

            dominant_constraint = (
                "GOAL"
            )


        # =================================================
        # EXPLANATION
        # =================================================

        if obstacle_violation:

            reason = (
                "The current position enters "
                "the obstacle region."
            )


        elif orientation_violation:

            reason = (
                "Virtual object orientation exceeds "
                "the strong safety tolerance."
            )


        elif obstacle_risk:

            reason = (
                "The current position is within "
                "the obstacle safety margin."
            )


        elif orientation_risk:

            reason = (
                "Virtual object orientation is approaching "
                "the safety boundary."
            )


        elif goal_reached:

            reason = (
                "The target region has been reached "
                "without a current constraint violation."
            )


        else:

            reason = (
                "The current movement is task-valid so far."
            )


        # =================================================
        # RETURN
        # =================================================

        return TaskConstraintResult(

            state=
                state,

            task_valid_so_far=
                task_valid_so_far,

            tracking_reliable=
                tracking_reliable,

            goal_reached=
                goal_reached,

            distance_to_goal=
                distance_to_goal,

            obstacle_violation=
                obstacle_violation,

            obstacle_risk=
                obstacle_risk,

            obstacle_clearance=
                nearest_clearance,

            nearest_obstacle_id=
                nearest_obstacle_id,

            orientation_violation=
                orientation_violation,

            orientation_risk=
                orientation_risk,

            orientation_error_deg=
                orientation_error_deg,

            dominant_constraint=
                dominant_constraint,

            reason=
                reason
        )


# =========================================================
# SELF TEST
# =========================================================

if __name__ == "__main__":

    evaluator = (
        TaskConstraintEvaluator()
    )


    print()

    print(
        "=================================================="
    )

    print(
        "AdaptiveSkill - Task Constraint Evaluator"
    )

    print(
        "=================================================="
    )


    tests = [

        # =================================================
        # A
        # =================================================

        (
            "A. VALID - START AREA",

            {
                "x": 0.3,
                "y": -0.8,

                "orientation_relative_deg":
                    2.0,

                "tracking_missing_sec":
                    0.0
            },

            STATE_VALID
        ),


        # =================================================
        # B
        #
        # Upper alternative route.
        # =================================================

        (
            "B. VALID - UPPER ROUTE",

            {
                "x": 1.65,
                "y": -0.9,

                "orientation_relative_deg":
                    3.0,

                "tracking_missing_sec":
                    0.0
            },

            STATE_VALID
        ),


        # =================================================
        # C
        #
        # Lower alternative route.
        # =================================================

        (
            "C. VALID - LOWER ROUTE",

            {
                "x": 1.65,
                "y": 0.9,

                "orientation_relative_deg":
                    -4.0,

                "tracking_missing_sec":
                    0.0
            },

            STATE_VALID
        ),


        # =================================================
        # D
        #
        # Obstacle top boundary = -0.45
        # Current y = -0.55
        #
        # Clearance = 0.10
        # Safety margin = 0.20
        # =================================================

        (
            "D. RISK - NEAR OBSTACLE",

            {
                "x": 1.65,
                "y": -0.55,

                "orientation_relative_deg":
                    0.0,

                "tracking_missing_sec":
                    0.0
            },

            STATE_RISK
        ),


        # =================================================
        # E
        # =================================================

        (
            "E. VIOLATION - INSIDE OBSTACLE",

            {
                "x": 1.65,
                "y": 0.0,

                "orientation_relative_deg":
                    0.0,

                "tracking_missing_sec":
                    0.0
            },

            STATE_VIOLATION
        ),


        # =================================================
        # F
        #
        # Mild threshold = 8 deg
        # Strong threshold = 15 deg
        # =================================================

        (
            "F. RISK - ORIENTATION",

            {
                "x": 0.8,
                "y": -0.8,

                "orientation_relative_deg":
                    11.0,

                "tracking_missing_sec":
                    0.0
            },

            STATE_RISK
        ),


        # =================================================
        # G
        # =================================================

        (
            "G. VIOLATION - ORIENTATION",

            {
                "x": 0.8,
                "y": -0.8,

                "orientation_relative_deg":
                    22.0,

                "tracking_missing_sec":
                    0.0
            },

            STATE_VIOLATION
        ),


        # =================================================
        # H
        # =================================================

        (
            "H. VALID - GOAL REACHED",

            {
                "x": 3.15,
                "y": 0.05,

                "orientation_relative_deg":
                    2.0,

                "tracking_missing_sec":
                    0.0
            },

            STATE_VALID
        ),


        # =================================================
        # I
        #
        # max_missing_sec = 0.30
        # =================================================

        (
            "I. TRACKING UNRELIABLE",

            {
                "x": 1.0,
                "y": -0.8,

                "orientation_relative_deg":
                    0.0,

                "tracking_missing_sec":
                    0.5
            },

            STATE_TRACKING_UNRELIABLE
        )
    ]


    passed = 0


    for (
        name,
        kwargs,
        expected
    ) in tests:

        result = evaluator.evaluate(
            **kwargs
        )


        ok = (
            result.state
            ==
            expected
        )


        if ok:

            passed += 1


        print()

        print(
            name
        )

        print(
            "-" * 50
        )

        print(
            "Expected:",
            expected
        )

        print(
            "Actual:",
            result.state
        )

        print(
            "PASS:",
            ok
        )

        print(
            "Task valid so far:",
            result.task_valid_so_far
        )

        print(
            "Goal reached:",
            result.goal_reached
        )

        print(
            "Obstacle risk:",
            result.obstacle_risk
        )

        print(
            "Obstacle violation:",
            result.obstacle_violation
        )


        if math.isfinite(
            result.obstacle_clearance
        ):

            clearance_text = round(
                result.obstacle_clearance,
                3
            )

        else:

            clearance_text = (
                result.obstacle_clearance
            )


        print(
            "Obstacle clearance:",
            clearance_text
        )


        if math.isfinite(
            result.orientation_error_deg
        ):

            orientation_text = round(
                result.orientation_error_deg,
                2
            )

        else:

            orientation_text = (
                result.orientation_error_deg
            )


        print(
            "Orientation error:",
            orientation_text
        )

        print(
            "Dominant constraint:",
            result.dominant_constraint
        )

        print(
            "Reason:",
            result.reason
        )


    print()

    print(
        "=================================================="
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
            "TASK CONSTRAINT EVALUATOR: PASSED"
        )

    else:

        print(
            "TASK CONSTRAINT EVALUATOR: FAILED"
        )


    print(
        "=================================================="
    )

    print()