import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


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
    remains safe / valid with respect to task constraints.
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
        float(x)
        <=
        obstacle["x_max"]

        and

        obstacle["y_min"]
        <=
        float(y)
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
        obstacle["x_min"] - float(x),
        0.0,
        float(x) - obstacle["x_max"]
    )


    dy = max(
        obstacle["y_min"] - float(y),
        0.0,
        float(y) - obstacle["y_max"]
    )


    return math.hypot(
        dx,
        dy
    )


# =========================================================
# SEGMENT GEOMETRY
# =========================================================

def cross_product(
    ax,
    ay,
    bx,
    by,
    cx,
    cy
):
    """
    2D cross product for orientation tests.
    """

    return (
        (
            float(bx) - float(ax)
        )
        *
        (
            float(cy) - float(ay)
        )
        -
        (
            float(by) - float(ay)
        )
        *
        (
            float(cx) - float(ax)
        )
    )


def point_on_segment(
    px,
    py,
    ax,
    ay,
    bx,
    by,
    epsilon=1e-9
):
    """
    Check whether point P lies on line segment AB.
    """

    cross = cross_product(
        ax,
        ay,
        bx,
        by,
        px,
        py
    )

    if abs(cross) > epsilon:
        return False


    return (
        min(
            float(ax),
            float(bx)
        ) - epsilon
        <=
        float(px)
        <=
        max(
            float(ax),
            float(bx)
        ) + epsilon

        and

        min(
            float(ay),
            float(by)
        ) - epsilon
        <=
        float(py)
        <=
        max(
            float(ay),
            float(by)
        ) + epsilon
    )


def segments_intersect(
    ax,
    ay,
    bx,
    by,
    cx,
    cy,
    dx,
    dy,
    epsilon=1e-9
):
    """
    Inclusive segment intersection.

    Touching a boundary counts as intersection.
    """

    o1 = cross_product(
        ax,
        ay,
        bx,
        by,
        cx,
        cy
    )

    o2 = cross_product(
        ax,
        ay,
        bx,
        by,
        dx,
        dy
    )

    o3 = cross_product(
        cx,
        cy,
        dx,
        dy,
        ax,
        ay
    )

    o4 = cross_product(
        cx,
        cy,
        dx,
        dy,
        bx,
        by
    )


    # Proper intersection.
    if (
        (
            o1 > epsilon
            and
            o2 < -epsilon
        )
        or
        (
            o1 < -epsilon
            and
            o2 > epsilon
        )
    ) and (
        (
            o3 > epsilon
            and
            o4 < -epsilon
        )
        or
        (
            o3 < -epsilon
            and
            o4 > epsilon
        )
    ):
        return True


    # Collinear / boundary cases.
    if (
        abs(o1) <= epsilon
        and
        point_on_segment(
            cx,
            cy,
            ax,
            ay,
            bx,
            by
        )
    ):
        return True


    if (
        abs(o2) <= epsilon
        and
        point_on_segment(
            dx,
            dy,
            ax,
            ay,
            bx,
            by
        )
    ):
        return True


    if (
        abs(o3) <= epsilon
        and
        point_on_segment(
            ax,
            ay,
            cx,
            cy,
            dx,
            dy
        )
    ):
        return True


    if (
        abs(o4) <= epsilon
        and
        point_on_segment(
            bx,
            by,
            cx,
            cy,
            dx,
            dy
        )
    ):
        return True


    return False


def point_to_segment_distance(
    px,
    py,
    ax,
    ay,
    bx,
    by
):
    """
    Shortest Euclidean distance
    from point P to segment AB.
    """

    ab_x = float(bx) - float(ax)
    ab_y = float(by) - float(ay)

    ap_x = float(px) - float(ax)
    ap_y = float(py) - float(ay)


    ab_len_sq = (
        ab_x * ab_x
        +
        ab_y * ab_y
    )


    if ab_len_sq < 1e-12:

        return math.hypot(
            float(px) - float(ax),
            float(py) - float(ay)
        )


    t = (
        ap_x * ab_x
        +
        ap_y * ab_y
    ) / ab_len_sq


    t = max(
        0.0,
        min(
            1.0,
            t
        )
    )


    closest_x = (
        float(ax)
        +
        t * ab_x
    )


    closest_y = (
        float(ay)
        +
        t * ab_y
    )


    return math.hypot(
        float(px) - closest_x,
        float(py) - closest_y
    )


def segment_to_segment_distance(
    ax,
    ay,
    bx,
    by,
    cx,
    cy,
    dx,
    dy
):
    """
    Shortest distance between two 2D line segments.
    """

    if segments_intersect(
        ax,
        ay,
        bx,
        by,
        cx,
        cy,
        dx,
        dy
    ):
        return 0.0


    return min(

        point_to_segment_distance(
            ax,
            ay,
            cx,
            cy,
            dx,
            dy
        ),

        point_to_segment_distance(
            bx,
            by,
            cx,
            cy,
            dx,
            dy
        ),

        point_to_segment_distance(
            cx,
            cy,
            ax,
            ay,
            bx,
            by
        ),

        point_to_segment_distance(
            dx,
            dy,
            ax,
            ay,
            bx,
            by
        )
    )


def segment_intersects_rectangle(
    x1,
    y1,
    x2,
    y2,
    obstacle
):
    """
    Returns True when any part of segment
    intersects or touches the obstacle rectangle.
    """

    # Endpoint already inside.
    if point_inside_rectangle(
        x1,
        y1,
        obstacle
    ):
        return True


    if point_inside_rectangle(
        x2,
        y2,
        obstacle
    ):
        return True


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


    edges = [

        # Top
        (
            x_min,
            y_min,
            x_max,
            y_min
        ),

        # Right
        (
            x_max,
            y_min,
            x_max,
            y_max
        ),

        # Bottom
        (
            x_max,
            y_max,
            x_min,
            y_max
        ),

        # Left
        (
            x_min,
            y_max,
            x_min,
            y_min
        ),
    ]


    for (
        ax,
        ay,
        bx,
        by
    ) in edges:

        if segments_intersect(
            x1,
            y1,
            x2,
            y2,
            ax,
            ay,
            bx,
            by
        ):
            return True


    return False


def segment_to_rectangle_distance(
    x1,
    y1,
    x2,
    y2,
    obstacle
):
    """
    Shortest distance between a movement segment
    and an axis-aligned obstacle rectangle.

    If segment crosses / touches obstacle:
        0.0
    """

    if segment_intersects_rectangle(
        x1,
        y1,
        x2,
        y2,
        obstacle
    ):
        return 0.0


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


    edges = [

        (
            x_min,
            y_min,
            x_max,
            y_min
        ),

        (
            x_max,
            y_min,
            x_max,
            y_max
        ),

        (
            x_max,
            y_max,
            x_min,
            y_max
        ),

        (
            x_min,
            y_max,
            x_min,
            y_min
        ),
    ]


    best_distance = float(
        "inf"
    )


    for (
        ax,
        ay,
        bx,
        by
    ) in edges:

        current_distance = (
            segment_to_segment_distance(
                x1,
                y1,
                x2,
                y2,
                ax,
                ay,
                bx,
                by
            )
        )


        best_distance = min(
            best_distance,
            current_distance
        )


    return best_distance


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

    It supports:

        evaluate(...)
            point / current-frame evaluation

        evaluate_segment(...)
            movement-segment evaluation

    Reference-trajectory similarity is intentionally
    NOT used as the definition of correctness.
    """


    def __init__(
        self,
        config_path=None
    ):

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
        ) as file:

            self.config = json.load(
                file
            )


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
    # POINT EVALUATION
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
        # TRACKING
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
        # GOAL
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
        # OBSTACLE
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


            if inside:

                obstacle_violation = True


            elif (
                clearance
                <=
                self.safety_margin
            ):

                obstacle_risk = True


        if not self.obstacles:

            nearest_clearance = float(
                "inf"
            )

            nearest_obstacle_id = None


        # =================================================
        # ORIENTATION
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


        orientation_violation = (

            orientation_error_deg

            >

            self.orientation_strong_deg
        )


        orientation_risk = (

            not orientation_violation

            and

            orientation_error_deg

            >

            self.orientation_mild_deg
        )


        # =================================================
        # COMBINE
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


        task_valid_so_far = (

            not obstacle_violation

            and

            not orientation_violation
        )


        # =================================================
        # DOMINANT CONSTRAINT
        # =================================================

        dominant_constraint = (
            "NONE"
        )


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
        # REASON
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


    # =====================================================
    # SEGMENT EVALUATION
    # =====================================================

    def evaluate_segment(
        self,
        previous_x,
        previous_y,
        x,
        y,
        orientation_relative_deg,
        tracking_missing_sec=0.0
    ):
        """
        Evaluate the ENTIRE movement segment:

            previous position
            ->
            current position

        This prevents fast motion from skipping
        obstacle detection between sampled frames.

        Spatial task validity therefore depends on
        the swept movement segment, not only
        the current endpoint.

        Orientation is currently evaluated at the
        current endpoint only.
        """

        # -------------------------------------------------
        # First evaluate current endpoint normally.
        # -------------------------------------------------

        current_result = self.evaluate(

            x=
                x,

            y=
                y,

            orientation_relative_deg=
                orientation_relative_deg,

            tracking_missing_sec=
                tracking_missing_sec
        )


        # -------------------------------------------------
        # Tracking unreliable:
        # do not make geometric judgement.
        # -------------------------------------------------

        if (
            current_result.state
            ==
            STATE_TRACKING_UNRELIABLE
        ):

            return current_result


        previous_x = float(
            previous_x
        )

        previous_y = float(
            previous_y
        )

        x = float(
            x
        )

        y = float(
            y
        )


        # =================================================
        # SEGMENT -> OBSTACLE
        # =================================================

        segment_obstacle_violation = False
        segment_obstacle_risk = False


        nearest_segment_clearance = float(
            "inf"
        )


        nearest_segment_obstacle_id = None


        for obstacle in self.obstacles:

            intersects = (
                segment_intersects_rectangle(

                    previous_x,
                    previous_y,

                    x,
                    y,

                    obstacle
                )
            )


            clearance = (
                segment_to_rectangle_distance(

                    previous_x,
                    previous_y,

                    x,
                    y,

                    obstacle
                )
            )


            if (
                clearance
                <
                nearest_segment_clearance
            ):

                nearest_segment_clearance = (
                    clearance
                )

                nearest_segment_obstacle_id = (
                    obstacle.get(
                        "id",
                        "obstacle"
                    )
                )


            if intersects:

                segment_obstacle_violation = (
                    True
                )


            elif (
                clearance
                <=
                self.safety_margin
            ):

                segment_obstacle_risk = (
                    True
                )


        # =================================================
        # COMBINE ENDPOINT + SEGMENT
        # =================================================

        obstacle_violation = (

            current_result.obstacle_violation

            or

            segment_obstacle_violation
        )


        obstacle_risk = (

            not obstacle_violation

            and

            (
                current_result.obstacle_risk

                or

                segment_obstacle_risk
            )
        )


        # Use the most conservative spatial clearance.
        obstacle_clearance = min(

            current_result.obstacle_clearance,

            nearest_segment_clearance
        )


        if (
            nearest_segment_clearance
            <=
            current_result.obstacle_clearance
        ):

            nearest_obstacle_id = (
                nearest_segment_obstacle_id
            )


        else:

            nearest_obstacle_id = (
                current_result.nearest_obstacle_id
            )


        # =================================================
        # ORIENTATION
        # =================================================

        orientation_violation = (
            current_result.orientation_violation
        )


        orientation_risk = (
            current_result.orientation_risk
        )


        # =================================================
        # FINAL STATE
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


        task_valid_so_far = (

            not obstacle_violation

            and

            not orientation_violation
        )


        # =================================================
        # DOMINANT CONSTRAINT
        # =================================================

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


        elif current_result.goal_reached:

            dominant_constraint = (
                "GOAL"
            )


        else:

            dominant_constraint = (
                "NONE"
            )


        # =================================================
        # REASON
        # =================================================

        if segment_obstacle_violation:

            reason = (
                "The movement segment crosses "
                "the obstacle region."
            )


        elif current_result.obstacle_violation:

            reason = (
                current_result.reason
            )


        elif segment_obstacle_risk:

            reason = (
                "The movement segment enters "
                "the obstacle safety margin."
            )


        else:

            reason = (
                current_result.reason
            )


        return TaskConstraintResult(

            state=
                state,

            task_valid_so_far=
                task_valid_so_far,

            tracking_reliable=
                current_result.tracking_reliable,

            goal_reached=
                current_result.goal_reached,

            distance_to_goal=
                current_result.distance_to_goal,

            obstacle_violation=
                obstacle_violation,

            obstacle_risk=
                obstacle_risk,

            obstacle_clearance=
                obstacle_clearance,

            nearest_obstacle_id=
                nearest_obstacle_id,

            orientation_violation=
                orientation_violation,

            orientation_risk=
                orientation_risk,

            orientation_error_deg=
                current_result.orientation_error_deg,

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
        "AdaptiveSkill - Task Constraint Evaluator v2"
    )

    print(
        "=================================================="
    )


    # =====================================================
    # POINT TESTS
    # =====================================================

    point_tests = [

        (
            "POINT A - VALID UPPER ROUTE",

            {
                "x":
                    1.65,

                "y":
                    -0.90,

                "orientation_relative_deg":
                    0.0,

                "tracking_missing_sec":
                    0.0,
            },

            STATE_VALID,
        ),


        (
            "POINT B - VALID LOWER ROUTE",

            {
                "x":
                    1.65,

                "y":
                    0.90,

                "orientation_relative_deg":
                    0.0,

                "tracking_missing_sec":
                    0.0,
            },

            STATE_VALID,
        ),


        (
            "POINT C - OBSTACLE RISK",

            {
                "x":
                    1.65,

                "y":
                    0.55,

                "orientation_relative_deg":
                    0.0,

                "tracking_missing_sec":
                    0.0,
            },

            STATE_RISK,
        ),


        (
            "POINT D - OBSTACLE VIOLATION",

            {
                "x":
                    1.65,

                "y":
                    0.0,

                "orientation_relative_deg":
                    0.0,

                "tracking_missing_sec":
                    0.0,
            },

            STATE_VIOLATION,
        ),


        (
            "POINT E - ORIENTATION RISK",

            {
                "x":
                    0.70,

                "y":
                    -0.75,

                "orientation_relative_deg":
                    12.0,

                "tracking_missing_sec":
                    0.0,
            },

            STATE_RISK,
        ),


        (
            "POINT F - ORIENTATION VIOLATION",

            {
                "x":
                    0.70,

                "y":
                    -0.75,

                "orientation_relative_deg":
                    20.0,

                "tracking_missing_sec":
                    0.0,
            },

            STATE_VIOLATION,
        ),
    ]


    # =====================================================
    # SEGMENT TESTS
    # =====================================================

    segment_tests = [

        # -------------------------------------------------
        # Both endpoints valid,
        # segment passes through obstacle.
        # -------------------------------------------------

        (
            "SEGMENT A - FAST CROSS THROUGH OBSTACLE",

            {
                "previous_x":
                    0.90,

                "previous_y":
                    -0.90,

                "x":
                    2.40,

                "y":
                    0.90,

                "orientation_relative_deg":
                    0.0,

                "tracking_missing_sec":
                    0.0,
            },

            STATE_VIOLATION,
        ),


        # -------------------------------------------------
        # Both endpoints valid,
        # segment passes close enough to enter
        # safety margin but does not touch obstacle.
        # -------------------------------------------------

        (
            "SEGMENT B - PASSES THROUGH SAFETY MARGIN",

            {
                "previous_x":
                    0.90,

                "previous_y":
                    0.60,

                "x":
                    2.40,

                "y":
                    0.60,

                "orientation_relative_deg":
                    0.0,

                "tracking_missing_sec":
                    0.0,
            },

            STATE_RISK,
        ),


        # -------------------------------------------------
        # Safe lower route.
        # -------------------------------------------------

        (
            "SEGMENT C - SAFE LOWER ROUTE",

            {
                "previous_x":
                    0.90,

                "previous_y":
                    0.90,

                "x":
                    2.40,

                "y":
                    0.90,

                "orientation_relative_deg":
                    0.0,

                "tracking_missing_sec":
                    0.0,
            },

            STATE_VALID,
        ),


        # -------------------------------------------------
        # Safe upper route.
        # -------------------------------------------------

        (
            "SEGMENT D - SAFE UPPER ROUTE",

            {
                "previous_x":
                    0.90,

                "previous_y":
                    -0.90,

                "x":
                    2.40,

                "y":
                    -0.90,

                "orientation_relative_deg":
                    0.0,

                "tracking_missing_sec":
                    0.0,
            },

            STATE_VALID,
        ),


        # -------------------------------------------------
        # Endpoint orientation can still trigger
        # task risk independently of path.
        # -------------------------------------------------

        (
            "SEGMENT E - SAFE PATH BUT ORIENTATION RISK",

            {
                "previous_x":
                    0.20,

                "previous_y":
                    -1.00,

                "x":
                    0.70,

                "y":
                    -0.75,

                "orientation_relative_deg":
                    12.0,

                "tracking_missing_sec":
                    0.0,
            },

            STATE_RISK,
        ),
    ]


    passed = 0

    total = (
        len(point_tests)
        +
        len(segment_tests)
    )


    # =====================================================
    # RUN POINT TESTS
    # =====================================================

    for (
        name,
        kwargs,
        expected,
    ) in point_tests:

        result = (
            evaluator.evaluate(
                **kwargs
            )
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
            "-" * 54
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
            "Obstacle clearance:",
            round(
                result.obstacle_clearance,
                3
            )
        )

        print(
            "Reason:",
            result.reason
        )

        print(
            "PASS:",
            ok
        )


    # =====================================================
    # RUN SEGMENT TESTS
    # =====================================================

    for (
        name,
        kwargs,
        expected,
    ) in segment_tests:

        result = (
            evaluator.evaluate_segment(
                **kwargs
            )
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
            "-" * 54
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
            "Obstacle violation:",
            result.obstacle_violation
        )

        print(
            "Obstacle risk:",
            result.obstacle_risk
        )

        print(
            "Segment clearance:",
            round(
                result.obstacle_clearance,
                3
            )
        )

        print(
            "Reason:",
            result.reason
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
        "=================================================="
    )

    print(
        "RESULT:",
        f"{passed}/{total} tests passed"
    )


    if passed == total:

        print(
            "SEGMENT-AWARE TASK CONSTRAINTS: PASSED"
        )

    else:

        print(
            "SEGMENT-AWARE TASK CONSTRAINTS: FAILED"
        )


    print(
        "=================================================="
    )

    print()