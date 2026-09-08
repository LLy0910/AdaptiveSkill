import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

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
# PROJECT IMPORTS
# =========================================================

from task.task_constraints import (
    TaskConstraintEvaluator,
    STATE_VALID,
    STATE_RISK,
    STATE_VIOLATION,
)


# =========================================================
# PATHS
# =========================================================

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "task"
    / "task_config.json"
)


# =========================================================
# DATA TYPES
# =========================================================

@dataclass
class TrajectoryDemonstration:
    demo_id: str
    points: np.ndarray
    source_path: str = ""
    strategy: str = ""


@dataclass
class LearnedPrototype:
    strategy: str
    points: np.ndarray
    demonstration_ids: list


@dataclass
class ReplayEvaluation:
    goal_reached: bool
    risk_seen: bool
    violation_seen: bool
    segment_violation_seen: bool
    min_obstacle_clearance: float
    max_orientation_error_deg: float
    task_valid: bool
    strict_safe: bool


# =========================================================
# BASIC HELPERS
# =========================================================

def safe_float(
    value,
    default=float("nan"),
):

    try:
        return float(value)

    except Exception:
        return default


def safe_int(
    value,
    default=0,
):

    try:
        return int(
            float(value)
        )

    except Exception:
        return default


# =========================================================
# TRAJECTORY RESAMPLING
#
# We align demonstrations by 2D arc length rather than
# raw time. This is intentionally simple and transparent.
# =========================================================

def resample_trajectory(
    points,
    num_points=120,
):

    points = np.asarray(
        points,
        dtype=float,
    )


    if (
        points.ndim != 2

        or

        points.shape[1] != 3
    ):

        raise ValueError(
            "Trajectory points must have shape (N, 3): "
            "[x, y, orientation_deg]."
        )


    if len(points) < 2:

        raise ValueError(
            "A trajectory needs at least two points."
        )


    xy = points[:, :2]


    step_distances = np.linalg.norm(
        np.diff(
            xy,
            axis=0,
        ),
        axis=1,
    )


    cumulative = np.concatenate(
        (
            [0.0],
            np.cumsum(
                step_distances
            ),
        )
    )


    total_length = float(
        cumulative[-1]
    )


    if total_length <= 1e-9:

        raise ValueError(
            "Trajectory has near-zero path length."
        )


    # -----------------------------------------------------
    # Remove repeated arc-length coordinates so np.interp
    # receives a strictly increasing independent variable.
    # -----------------------------------------------------

    unique_cumulative, unique_indices = np.unique(
        cumulative,
        return_index=True,
    )


    unique_points = points[
        unique_indices
    ]


    if len(unique_cumulative) < 2:

        raise ValueError(
            "Trajectory does not contain enough unique "
            "positions for resampling."
        )


    target_s = np.linspace(
        0.0,
        total_length,
        int(num_points),
    )


    x = np.interp(
        target_s,
        unique_cumulative,
        unique_points[:, 0],
    )


    y = np.interp(
        target_s,
        unique_cumulative,
        unique_points[:, 1],
    )


    # -----------------------------------------------------
    # Interpolate orientation after unwrapping so values
    # around +180/-180 do not average incorrectly.
    # -----------------------------------------------------

    orientation_rad = np.unwrap(
        np.deg2rad(
            unique_points[:, 2]
        )
    )


    orientation_interp_rad = np.interp(
        target_s,
        unique_cumulative,
        orientation_rad,
    )


    orientation_deg = np.rad2deg(
        orientation_interp_rad
    )


    return np.column_stack(
        (
            x,
            y,
            orientation_deg,
        )
    )


# =========================================================
# RUNTIME CSV LOADER
#
# This is for completed demonstration logs produced by the
# current RuntimeLogger.
#
# By default, demonstrations that contain a violation or do
# not reach the goal are rejected from learner training.
# =========================================================

def load_runtime_demonstration(
    csv_path,
    require_goal=True,
    reject_violation=True,
):

    csv_path = Path(
        csv_path
    )


    if not csv_path.exists():

        raise FileNotFoundError(
            f"Runtime CSV not found: {csv_path}"
        )


    with open(
        csv_path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        rows = list(
            csv.DictReader(
                file
            )
        )


    if len(rows) < 2:

        raise ValueError(
            f"Runtime CSV is too short: {csv_path}"
        )


    points = []


    goal_reached = False

    violation_occurred = False


    for row in rows:

        x = safe_float(
            row.get(
                "x",
                float("nan"),
            )
        )


        y = safe_float(
            row.get(
                "y",
                float("nan"),
            )
        )


        orientation_deg = safe_float(
            row.get(
                "orientation_deg",
                0.0,
            ),
            0.0,
        )


        if (
            math.isnan(x)

            or

            math.isnan(y)
        ):

            continue


        points.append(
            (
                x,
                y,
                orientation_deg,
            )
        )


        goal_reached = (
            goal_reached

            or

            safe_int(
                row.get(
                    "goal_reached",
                    0,
                )
            )
            == 1

            or

            safe_int(
                row.get(
                    "trial_goal_reached_seen",
                    0,
                )
            )
            == 1
        )


        violation_occurred = (
            violation_occurred

            or

            str(
                row.get(
                    "task_state",
                    "",
                )
            )
            ==
            STATE_VIOLATION

            or

            safe_int(
                row.get(
                    "trial_violation_occurred",
                    0,
                )
            )
            == 1
        )


    if len(points) < 2:

        raise ValueError(
            f"No usable trajectory points in: {csv_path}"
        )


    if (
        require_goal

        and

        not goal_reached
    ):

        raise ValueError(
            "Demonstration rejected: goal was not reached."
        )


    if (
        reject_violation

        and

        violation_occurred
    ):

        raise ValueError(
            "Demonstration rejected: a task violation "
            "occurred."
        )


    demo_id = rows[0].get(
        "trial_id",
        csv_path.stem,
    )


    return TrajectoryDemonstration(

        demo_id=str(
            demo_id
        ),

        points=np.asarray(
            points,
            dtype=float,
        ),

        source_path=str(
            csv_path
        ),
    )


# =========================================================
# STRATEGY CLASSIFICATION
#
# The current Move-the-Cup task has a dominant obstacle.
# We use the median Y while passing the obstacle's X-range
# as a transparent strategy signature.
#
# This is NOT claimed to be a general intent recogniser.
# It is a small engineering mechanism for preserving
# distinct upper/lower task-valid demonstration modes.
# =========================================================

def classify_route_strategy(
    points,
    task_config,
):

    points = np.asarray(
        points,
        dtype=float,
    )


    obstacles = task_config.get(
        "obstacles",
        [],
    )


    if not obstacles:

        return "UNSPECIFIED"


    obstacle = obstacles[0]


    x_min = float(
        obstacle[
            "x_min"
        ]
    )


    x_max = float(
        obstacle[
            "x_max"
        ]
    )


    y_min = float(
        obstacle[
            "y_min"
        ]
    )


    y_max = float(
        obstacle[
            "y_max"
        ]
    )


    obstacle_center_y = (
        y_min
        +
        y_max
    ) / 2.0


    corridor_mask = (

        (points[:, 0] >= x_min)

        &

        (points[:, 0] <= x_max)
    )


    if np.any(
        corridor_mask
    ):

        strategy_y = float(
            np.median(
                points[
                    corridor_mask,
                    1,
                ]
            )
        )


    else:

        strategy_y = float(
            np.median(
                points[:, 1]
            )
        )


    neutral_band = 0.05


    if (
        strategy_y
        <
        obstacle_center_y
        -
        neutral_band
    ):

        return "UPPER"


    if (
        strategy_y
        >
        obstacle_center_y
        +
        neutral_band
    ):

        return "LOWER"


    return "CENTER"


# =========================================================
# STRATEGY-AWARE PROTOTYPE LEARNER
#
# This is a lightweight learner:
#
# 1. resample demonstrations by arc length
# 2. keep distinct route strategies separate
# 3. average demonstrations within each strategy
#
# Important claim boundary:
# This learns simple trajectory prototypes.
# It is NOT a general robot control policy and does not
# establish physical-robot generalisation.
# =========================================================

class StrategyAwareTrajectoryLearner:

    def __init__(
        self,
        config_path=DEFAULT_CONFIG_PATH,
        num_points=120,
    ):

        self.config_path = Path(
            config_path
        )


        with open(
            self.config_path,
            "r",
            encoding="utf-8",
        ) as file:

            self.task_config = json.load(
                file
            )


        self.num_points = int(
            num_points
        )


        self.prototypes = {}


    # =====================================================
    # FIT
    # =====================================================

    def fit(
        self,
        demonstrations,
    ):

        if not demonstrations:

            raise ValueError(
                "At least one demonstration is required."
            )


        groups = {}


        for demonstration in demonstrations:

            resampled = resample_trajectory(

                demonstration.points,

                num_points=
                    self.num_points,
            )


            strategy = (
                demonstration.strategy.strip().upper()

                if demonstration.strategy

                else classify_route_strategy(
                    resampled,
                    self.task_config,
                )
            )


            groups.setdefault(
                strategy,
                [],
            ).append(
                (
                    demonstration,
                    resampled,
                )
            )


        self.prototypes = {}


        for strategy, items in groups.items():

            stacked = np.stack(

                [
                    points

                    for (
                        _demonstration,
                        points
                    )
                    in items
                ],

                axis=0,
            )


            # -------------------------------------------------
            # X/Y arithmetic mean.
            # -------------------------------------------------

            mean_x = np.mean(
                stacked[:, :, 0],
                axis=0,
            )


            mean_y = np.mean(
                stacked[:, :, 1],
                axis=0,
            )


            # -------------------------------------------------
            # Circular mean for orientation.
            # -------------------------------------------------

            orientation_rad = np.deg2rad(
                stacked[:, :, 2]
            )


            mean_sin = np.mean(
                np.sin(
                    orientation_rad
                ),
                axis=0,
            )


            mean_cos = np.mean(
                np.cos(
                    orientation_rad
                ),
                axis=0,
            )


            mean_orientation = np.rad2deg(
                np.arctan2(
                    mean_sin,
                    mean_cos,
                )
            )


            prototype_points = np.column_stack(
                (
                    mean_x,
                    mean_y,
                    mean_orientation,
                )
            )


            self.prototypes[
                strategy
            ] = LearnedPrototype(

                strategy=
                    strategy,

                points=
                    prototype_points,

                demonstration_ids=
                    [
                        demonstration.demo_id

                        for (
                            demonstration,
                            _points,
                        )
                        in items
                    ],
            )


        return self.prototypes


    # =====================================================
    # AVAILABLE STRATEGIES
    # =====================================================

    def available_strategies(
        self,
    ):

        return sorted(
            self.prototypes.keys()
        )


    # =====================================================
    # PREDICT / REPLAY TRAJECTORY
    # =====================================================

    def predict(
        self,
        strategy=None,
    ):

        if not self.prototypes:

            raise RuntimeError(
                "Learner has not been fitted yet."
            )


        if strategy is None:

            if len(
                self.prototypes
            ) == 1:

                return next(
                    iter(
                        self.prototypes.values()
                    )
                )


            raise ValueError(
                "Multiple valid strategy prototypes exist. "
                "Specify a strategy instead of collapsing "
                "them into one trajectory."
            )


        strategy = str(
            strategy
        ).strip().upper()


        if (
            strategy
            not in
            self.prototypes
        ):

            raise KeyError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {self.available_strategies()}"
            )


        return self.prototypes[
            strategy
        ]


# =========================================================
# NAIVE SINGLE-MEAN BASELINE
#
# This intentionally ignores distinct strategy modes.
# It is useful as a transparent downstream baseline.
# =========================================================

def learn_naive_single_mean(
    demonstrations,
    num_points=120,
):

    if not demonstrations:

        raise ValueError(
            "At least one demonstration is required."
        )


    stacked = np.stack(

        [
            resample_trajectory(
                demonstration.points,
                num_points=
                    num_points,
            )

            for demonstration
            in demonstrations
        ],

        axis=0,
    )


    mean_x = np.mean(
        stacked[:, :, 0],
        axis=0,
    )


    mean_y = np.mean(
        stacked[:, :, 1],
        axis=0,
    )


    orientation_rad = np.deg2rad(
        stacked[:, :, 2]
    )


    mean_orientation = np.rad2deg(
        np.arctan2(

            np.mean(
                np.sin(
                    orientation_rad
                ),
                axis=0,
            ),

            np.mean(
                np.cos(
                    orientation_rad
                ),
                axis=0,
            ),
        )
    )


    return LearnedPrototype(

        strategy=
            "NAIVE_SINGLE_MEAN",

        points=
            np.column_stack(
                (
                    mean_x,
                    mean_y,
                    mean_orientation,
                )
            ),

        demonstration_ids=
            [
                demonstration.demo_id

                for demonstration
                in demonstrations
            ],
    )


# =========================================================
# LEARNED TRAJECTORY EVALUATION
#
# Reuse the SAME TaskConstraintEvaluator as the live demo.
# We do not create a second definition of correctness.
# =========================================================

def evaluate_learned_trajectory(
    points,
    config_path=DEFAULT_CONFIG_PATH,
):

    points = np.asarray(
        points,
        dtype=float,
    )


    if len(points) < 2:

        raise ValueError(
            "Learned trajectory needs at least two points."
        )


    evaluator = TaskConstraintEvaluator(
        config_path
    )


    risk_seen = False

    violation_seen = False

    segment_violation_seen = False


    min_clearance = float(
        "inf"
    )


    max_orientation_error = 0.0


    final_point_result = None


    for index, point in enumerate(
        points
    ):

        x = float(
            point[0]
        )


        y = float(
            point[1]
        )


        orientation_deg = float(
            point[2]
        )


        point_result = evaluator.evaluate(

            x=
                x,

            y=
                y,

            orientation_relative_deg=
                orientation_deg,

            tracking_missing_sec=
                0.0,
        )


        final_point_result = (
            point_result
        )


        state = str(
            point_result.state
        )


        risk_seen = (
            risk_seen

            or

            state == STATE_RISK
        )


        violation_seen = (
            violation_seen

            or

            state == STATE_VIOLATION
        )


        clearance = safe_float(
            getattr(
                point_result,
                "obstacle_clearance",
                float("nan"),
            )
        )


        if not math.isnan(
            clearance
        ):

            min_clearance = min(
                min_clearance,
                clearance,
            )


        orientation_error = safe_float(
            getattr(
                point_result,
                "orientation_error_deg",
                0.0,
            ),
            0.0,
        )


        max_orientation_error = max(

            max_orientation_error,

            abs(
                orientation_error
            ),
        )


        # -------------------------------------------------
        # Segment-aware check catches fast crossings that
        # endpoint-only evaluation could miss.
        # -------------------------------------------------

        if index > 0:

            previous = points[
                index - 1
            ]


            segment_result = (
                evaluator.evaluate_segment(

                    previous_x=
                        float(
                            previous[0]
                        ),

                    previous_y=
                        float(
                            previous[1]
                        ),

                    x=
                        x,

                    y=
                        y,

                    orientation_relative_deg=
                        orientation_deg,

                    tracking_missing_sec=
                        0.0,
                )
            )


            segment_state = str(
                segment_result.state
            )


            risk_seen = (
                risk_seen

                or

                segment_state
                ==
                STATE_RISK
            )


            if (
                segment_state
                ==
                STATE_VIOLATION
            ):

                violation_seen = True

                segment_violation_seen = True


            segment_clearance = safe_float(
                getattr(
                    segment_result,
                    "obstacle_clearance",
                    float("nan"),
                )
            )


            if not math.isnan(
                segment_clearance
            ):

                min_clearance = min(
                    min_clearance,
                    segment_clearance,
                )


    if math.isinf(
        min_clearance
    ):

        min_clearance = float(
            "nan"
        )


    goal_reached = bool(
        getattr(
            final_point_result,
            "goal_reached",
            False,
        )
    )


    # -----------------------------------------------------
    # task_valid:
    # goal reached and no hard violation.
    #
    # strict_safe:
    # additionally never entered RISK.
    # -----------------------------------------------------

    task_valid = (

        goal_reached

        and

        not violation_seen
    )


    strict_safe = (

        task_valid

        and

        not risk_seen
    )


    return ReplayEvaluation(

        goal_reached=
            goal_reached,

        risk_seen=
            risk_seen,

        violation_seen=
            violation_seen,

        segment_violation_seen=
            segment_violation_seen,

        min_obstacle_clearance=
            min_clearance,

        max_orientation_error_deg=
            max_orientation_error,

        task_valid=
            task_valid,

        strict_safe=
            strict_safe,
    )
