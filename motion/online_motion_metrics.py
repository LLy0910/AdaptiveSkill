import csv
import math
from pathlib import Path

import numpy as np


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_REFERENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "reference.csv"
)


# =========================================================
# BASIC FUNCTIONS
# =========================================================

def circular_angle_difference(
    angle1,
    angle2
):
    """
    Smallest signed angular difference.

    Example:
    179 -> -179
    should be approximately +2 degrees,
    not -358 degrees.
    """

    return (
        angle1
        -
        angle2
        +
        180.0
    ) % 360.0 - 180.0


# =========================================================
# LOAD REFERENCE
# =========================================================

def load_reference(
    path=DEFAULT_REFERENCE_PATH
):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Reference not found: {path}"
        )


    reference = []


    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(
            file
        )


        required = [
            "x_norm",
            "y_norm",
            "angle_raw"
        ]


        if reader.fieldnames is None:

            raise ValueError(
                "reference.csv is empty."
            )


        missing = [
            column
            for column in required
            if column not in reader.fieldnames
        ]


        if missing:

            raise ValueError(
                "reference.csv missing: "
                +
                ", ".join(missing)
            )


        for row in reader:

            reference.append({

                "x_norm":
                    float(
                        row["x_norm"]
                    ),

                "y_norm":
                    float(
                        row["y_norm"]
                    ),

                "angle_raw":
                    float(
                        row["angle_raw"]
                    )
            })


    if len(reference) < 2:

        raise ValueError(
            "Reference is too short."
        )


    return reference


# =========================================================
# ONLINE MOTION METRICS
# =========================================================

class OnlineMotionMetrics:
    """
    Online motion-state estimator.

    Responsibilities:

    1. Estimate reference progress.
    2. Force progress to move monotonically forward.
    3. Track learner relative 2D hand rotation.
    4. Compare learner relative rotation against the
       expected reference rotation at the current progress.

    Important:
    The angle is still a 2D hand-orientation proxy,
    not true 3D wrist rotation.
    """

    def __init__(
        self,
        reference,
        max_advance_fraction=0.08
    ):

        if len(reference) < 2:

            raise ValueError(
                "Reference must contain at least 2 points."
            )


        self.reference = reference

        self.reference_xy = np.array(
            [
                [
                    point["x_norm"],
                    point["y_norm"]
                ]
                for point in reference
            ],
            dtype=float
        )


        self.reference_raw_angles = np.array(
            [
                point["angle_raw"]
                for point in reference
            ],
            dtype=float
        )


        # -------------------------------------------------
        # Convert reference absolute angle into
        # relative rotation from its own first frame.
        # -------------------------------------------------

        reference_unwrapped = np.rad2deg(
            np.unwrap(
                np.deg2rad(
                    self.reference_raw_angles
                )
            )
        )


        self.reference_relative_rotation = (
            reference_unwrapped
            -
            reference_unwrapped[0]
        )


        self.reference_progress = np.linspace(
            0.0,
            1.0,
            len(reference)
        )


        # -------------------------------------------------
        # Maximum number of reference frames that online
        # progress may advance in one update.
        #
        # This prevents a noisy nearest-point match from
        # suddenly jumping from e.g. 25% to 80%.
        # -------------------------------------------------

        self.max_advance = max(
            2,
            int(
                round(
                    len(reference)
                    *
                    max_advance_fraction
                )
            )
        )


        self.reset()


    # =====================================================
    # RESET
    # =====================================================

    def reset(self):

        # Progress state
        self.current_reference_index = 0

        self.current_progress = 0.0


        # Learner angle state
        self.start_raw_angle = None

        self.previous_raw_angle = None

        self.accumulated_rotation = 0.0


        # Current output
        self.current_expected_rotation = 0.0

        self.current_angle_error = 0.0

        self.current_path_distance = None


    # =====================================================
    # MONOTONIC PROGRESS
    # =====================================================

    def _estimate_progress(
        self,
        x_norm,
        y_norm
    ):
        """
        Search only from the current progress forward.

        The returned reference index is never allowed
        to move backward.
        """

        n = len(
            self.reference_xy
        )


        start_index = (
            self.current_reference_index
        )


        end_index = min(
            n,
            start_index
            +
            self.max_advance
            +
            1
        )


        candidate_points = (
            self.reference_xy[
                start_index:end_index
            ]
        )


        current_point = np.array(
            [
                x_norm,
                y_norm
            ],
            dtype=float
        )


        distances = np.linalg.norm(
            candidate_points
            -
            current_point,
            axis=1
        )


        local_index = int(
            np.argmin(
                distances
            )
        )


        candidate_index = (
            start_index
            +
            local_index
        )


        # -------------------------------------------------
        # Explicit monotonic constraint.
        # -------------------------------------------------

        candidate_index = max(
            self.current_reference_index,
            candidate_index
        )


        candidate_index = min(
            candidate_index,
            n - 1
        )


        self.current_reference_index = (
            candidate_index
        )


        self.current_progress = (

            candidate_index

            /

            max(
                n - 1,
                1
            )
        )


        self.current_path_distance = float(
            np.linalg.norm(
                self.reference_xy[
                    candidate_index
                ]
                -
                current_point
            )
        )


    # =====================================================
    # RELATIVE LEARNER ROTATION
    # =====================================================

    def _update_rotation(
        self,
        raw_angle
    ):
        """
        Track how much the learner has rotated relative
        to the first frame of the current trial.

        Uses incremental circular differences so
        +179 / -179 does not create a huge jump.
        """

        raw_angle = float(
            raw_angle
        )


        # First valid frame
        if self.start_raw_angle is None:

            self.start_raw_angle = (
                raw_angle
            )

            self.previous_raw_angle = (
                raw_angle
            )

            self.accumulated_rotation = 0.0

            return


        angle_step = (
            circular_angle_difference(
                raw_angle,
                self.previous_raw_angle
            )
        )


        self.accumulated_rotation += (
            angle_step
        )


        self.previous_raw_angle = (
            raw_angle
        )


    # =====================================================
    # EXPECTED REFERENCE ROTATION
    # =====================================================

    def _calculate_expected_rotation(
        self
    ):

        self.current_expected_rotation = float(
            np.interp(
                self.current_progress,
                self.reference_progress,
                self.reference_relative_rotation
            )
        )


    # =====================================================
    # UPDATE
    # =====================================================

    def update(
        self,
        x_norm,
        y_norm,
        raw_angle,
        tracking=True
    ):
        """
        Call once per tracked frame.

        Returns:

        {
            "tracking": True,
            "reference_index": 20,
            "progress": 0.2532,
            "path_distance": 0.08,
            "learner_relative_rotation": 9.5,
            "expected_relative_rotation": 10.8,
            "relative_angle_error": 1.3
        }
        """

        # -------------------------------------------------
        # Tracking lost:
        #
        # Do NOT reset progress or learner rotation.
        # The trial may resume after temporary tracking loss.
        # -------------------------------------------------

        if not tracking:

            return {

                "tracking":
                    False,

                "reference_index":
                    self.current_reference_index,

                "progress":
                    self.current_progress,

                "path_distance":
                    self.current_path_distance,

                "learner_relative_rotation":
                    self.accumulated_rotation,

                "expected_relative_rotation":
                    self.current_expected_rotation,

                "relative_angle_error":
                    self.current_angle_error
            }


        x_norm = float(
            x_norm
        )

        y_norm = float(
            y_norm
        )

        raw_angle = float(
            raw_angle
        )


        # =================================================
        # 1. ONLINE PROGRESS
        # =================================================

        self._estimate_progress(
            x_norm,
            y_norm
        )


        # =================================================
        # 2. LEARNER RELATIVE ROTATION
        # =================================================

        self._update_rotation(
            raw_angle
        )


        # =================================================
        # 3. EXPECTED ROTATION AT CURRENT PROGRESS
        # =================================================

        self._calculate_expected_rotation()


        # =================================================
        # 4. RELATIVE ANGLE ERROR
        # =================================================

        self.current_angle_error = abs(

            self.accumulated_rotation

            -

            self.current_expected_rotation
        )


        return {

            "tracking":
                True,

            "reference_index":
                int(
                    self.current_reference_index
                ),

            "progress":
                float(
                    self.current_progress
                ),

            "path_distance":
                float(
                    self.current_path_distance
                ),

            "learner_relative_rotation":
                float(
                    self.accumulated_rotation
                ),

            "expected_relative_rotation":
                float(
                    self.current_expected_rotation
                ),

            "relative_angle_error":
                float(
                    self.current_angle_error
                )
        }


# =========================================================
# SIMPLE SELF TEST
# =========================================================

if __name__ == "__main__":

    reference = load_reference()


    metrics = OnlineMotionMetrics(
        reference
    )


    print()
    print(
        "=========================================="
    )

    print(
        "AdaptiveSkill - Online Motion Metrics"
    )

    print(
        "=========================================="
    )

    print()

    print(
        "Reference frames:",
        len(reference)
    )

    print(
        "Maximum forward search:",
        metrics.max_advance,
        "reference frames"
    )

    print(
        "Reference total relative rotation:",
        round(
            float(
                metrics.reference_relative_rotation[-1]
            ),
            2
        ),
        "deg"
    )

    print()


    # =====================================================
    # SELF TEST USING REFERENCE ITSELF
    #
    # Feed the reference back into the online estimator.
    #
    # Expected:
    # - progress should never move backward
    # - angle error should remain close to zero
    # =====================================================

    progress_values = []

    angle_errors = []


    for point in reference:

        result = metrics.update(

            x_norm=
                point["x_norm"],

            y_norm=
                point["y_norm"],

            raw_angle=
                point["angle_raw"],

            tracking=True
        )


        progress_values.append(
            result["progress"]
        )

        angle_errors.append(
            result[
                "relative_angle_error"
            ]
        )


    monotonic = all(

        progress_values[i]
        >=
        progress_values[i - 1]

        for i in range(
            1,
            len(progress_values)
        )
    )


    mean_angle_error = float(
        np.mean(
            angle_errors
        )
    )


    max_angle_error = float(
        np.max(
            angle_errors
        )
    )


    print(
        "SELF TEST"
    )

    print(
        "------------------------------------------"
    )

    print(
        "Monotonic progress:",
        monotonic
    )

    print(
        "Final progress:",
        round(
            progress_values[-1],
            4
        )
    )

    print(
        "Mean relative angle error:",
        round(
            mean_angle_error,
            4
        ),
        "deg"
    )

    print(
        "Max relative angle error:",
        round(
            max_angle_error,
            4
        ),
        "deg"
    )

    print()


    if (
        monotonic
        and
        progress_values[-1] >= 0.95
        and
        mean_angle_error < 1.0
    ):

        print(
            "Online metric self-test PASSED."
        )

    else:

        print(
            "Online metric self-test needs inspection."
        )

    print()