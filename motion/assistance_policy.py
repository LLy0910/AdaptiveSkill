import csv
import json
from collections import deque
from pathlib import Path

import numpy as np

from online_motion_metrics import (
    load_reference,
    OnlineMotionMetrics
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRIAL_DIR = PROJECT_ROOT / "data" / "trials"

CONFIG_PATH = PROJECT_ROOT / "data" / "assistance_config.json"

HESITATION_CONFIG_PATH = (
    PROJECT_ROOT / "data" / "hesitation_config.json"
)


# =========================================================
# POLICY TIMING
#
# Path / angle thresholds:
# derived from Correct baseline data.
#
# Timing values:
# engineering / prototype policy choices.
# They should NOT be described as experimentally optimal.
# =========================================================

LEVEL_1_HOLD_SEC = 0.35
LEVEL_2_HOLD_SEC = 0.65

# Conservative final escalation.
LEVEL_3_HOLD_SEC = 3.00

# Required time before stepping DOWN one assistance level.
RECOVERY_SEC = 0.60

HESITATION_WINDOW_SEC = 8.0
HESITATION_EVENTS_FOR_LEVEL_3 = 2


# =========================================================
# LOAD ONE CORRECT TRIAL
# =========================================================

def load_trial(path):

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        required = [
            "tracking",
            "x_norm",
            "y_norm",
            "angle_raw"
        ]

        if reader.fieldnames is None:
            return []

        missing = [
            column
            for column in required
            if column not in reader.fieldnames
        ]

        if missing:

            print(
                f"Skipping {path.name}: "
                f"missing {missing}"
            )

            return []

        for row in reader:

            try:
                tracking = int(
                    row["tracking"]
                )

            except Exception:
                continue

            if tracking != 1:
                continue

            try:

                rows.append({

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

            except Exception:
                continue

    return rows


# =========================================================
# LOAD PROGRESS EXCLUSION
# =========================================================

def load_progress_boundaries():

    default_start = 0.08
    default_end = 0.95

    if not HESITATION_CONFIG_PATH.exists():

        return (
            default_start,
            default_end
        )

    try:

        with open(
            HESITATION_CONFIG_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            config = json.load(file)

        start = float(
            config.get(
                "ignore_start_progress",
                default_start
            )
        )

        end = float(
            config.get(
                "ignore_end_progress",
                default_end
            )
        )

        return (
            start,
            end
        )

    except Exception:

        return (
            default_start,
            default_end
        )


# =========================================================
# BUILD BASELINE FROM CORRECT TRIALS
# =========================================================

def build_assistance_config():

    reference = load_reference()

    correct_files = sorted(
        TRIAL_DIR.glob(
            "A_correct_*.csv"
        )
    )

    if len(correct_files) == 0:

        raise RuntimeError(
            "No A_correct trials found."
        )

    (
        ignore_start,
        ignore_end
    ) = load_progress_boundaries()

    all_path_errors = []
    all_angle_errors = []

    valid_trial_count = 0

    for path in correct_files:

        rows = load_trial(
            path
        )

        if len(rows) < 2:
            continue

        metrics = OnlineMotionMetrics(
            reference
        )

        trial_has_valid_frame = False

        for row in rows:

            result = metrics.update(

                x_norm=
                    row["x_norm"],

                y_norm=
                    row["y_norm"],

                raw_angle=
                    row["angle_raw"],

                tracking=True
            )

            progress = float(
                result["progress"]
            )

            if progress < ignore_start:
                continue

            if progress > ignore_end:
                continue

            path_error = float(
                result[
                    "path_distance"
                ]
            )

            angle_error = float(
                result[
                    "relative_angle_error"
                ]
            )

            all_path_errors.append(
                path_error
            )

            all_angle_errors.append(
                angle_error
            )

            trial_has_valid_frame = True

        if trial_has_valid_frame:
            valid_trial_count += 1

    if (
        len(all_path_errors) < 20
        or
        len(all_angle_errors) < 20
    ):

        raise RuntimeError(
            "Not enough valid baseline frames."
        )

    path_array = np.asarray(
        all_path_errors,
        dtype=float
    )

    angle_array = np.asarray(
        all_angle_errors,
        dtype=float
    )

    path_mild = float(
        np.percentile(
            path_array,
            90
        )
    )

    path_strong = float(
        np.percentile(
            path_array,
            99
        )
    )

    angle_mild = float(
        np.percentile(
            angle_array,
            90
        )
    )

    angle_strong = float(
        np.percentile(
            angle_array,
            99
        )
    )

    path_strong = max(
        path_strong,
        path_mild + 0.01
    )

    angle_strong = max(
        angle_strong,
        angle_mild + 1.0
    )

    config = {

        "baseline": {

            "correct_trials":
                valid_trial_count,

            "valid_frames":
                len(
                    all_path_errors
                ),

            "path_median":
                round(
                    float(
                        np.median(
                            path_array
                        )
                    ),
                    6
                ),

            "path_p90":
                round(
                    path_mild,
                    6
                ),

            "path_p99":
                round(
                    path_strong,
                    6
                ),

            "angle_median_deg":
                round(
                    float(
                        np.median(
                            angle_array
                        )
                    ),
                    4
                ),

            "angle_p90_deg":
                round(
                    angle_mild,
                    4
                ),

            "angle_p99_deg":
                round(
                    angle_strong,
                    4
                )
        },


        "thresholds": {

            "path_mild":
                round(
                    path_mild,
                    6
                ),

            "path_strong":
                round(
                    path_strong,
                    6
                ),

            "angle_mild_deg":
                round(
                    angle_mild,
                    4
                ),

            "angle_strong_deg":
                round(
                    angle_strong,
                    4
                )
        },


        "progress": {

            "ignore_start":
                ignore_start,

            "ignore_end":
                ignore_end
        },


        "timing": {

            "level_1_hold_sec":
                LEVEL_1_HOLD_SEC,

            "level_2_hold_sec":
                LEVEL_2_HOLD_SEC,

            "level_3_hold_sec":
                LEVEL_3_HOLD_SEC,

            "recovery_sec":
                RECOVERY_SEC,

            "hesitation_window_sec":
                HESITATION_WINDOW_SEC,

            "hesitation_events_for_level_3":
                HESITATION_EVENTS_FOR_LEVEL_3
        }
    }

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=4
        )

    return config


# =========================================================
# LOAD / CREATE CONFIG
# =========================================================

def load_assistance_config():

    if not CONFIG_PATH.exists():

        print(
            "assistance_config.json not found."
        )

        print(
            "Building baseline from Correct trials..."
        )

        return build_assistance_config()

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


# =========================================================
# ASSISTANCE POLICY
# =========================================================

class AssistancePolicy:
    """
    Explainable stateful assistance policy.

    Levels:
    0 = Observe
    1 = Gentle Cue
    2 = Explicit Guidance
    3 = Human Assistance
    """

    def __init__(
        self,
        config=None
    ):

        if config is None:
            config = load_assistance_config()

        self.config = config

        thresholds = (
            config["thresholds"]
        )

        self.path_mild = float(
            thresholds[
                "path_mild"
            ]
        )

        self.path_strong = float(
            thresholds[
                "path_strong"
            ]
        )

        self.angle_mild = float(
            thresholds[
                "angle_mild_deg"
            ]
        )

        self.angle_strong = float(
            thresholds[
                "angle_strong_deg"
            ]
        )

        progress_config = (
            config["progress"]
        )

        self.ignore_start = float(
            progress_config[
                "ignore_start"
            ]
        )

        self.ignore_end = float(
            progress_config[
                "ignore_end"
            ]
        )

        timing = (
            config["timing"]
        )

        self.level_1_hold = float(
            timing[
                "level_1_hold_sec"
            ]
        )

        self.level_2_hold = float(
            timing[
                "level_2_hold_sec"
            ]
        )

        self.level_3_hold = float(
            timing[
                "level_3_hold_sec"
            ]
        )

        self.recovery_sec = float(
            timing[
                "recovery_sec"
            ]
        )

        self.hesitation_window = float(
            timing[
                "hesitation_window_sec"
            ]
        )

        self.hesitation_events_for_level_3 = int(
            timing[
                "hesitation_events_for_level_3"
            ]
        )

        self.reset()


    def reset(self):

        self.current_level = 0

        self.mild_start_time = None
        self.strong_start_time = None

        self.recovery_start_time = None

        self.hesitation_event_times = deque()

        self.last_reason = (
            "Within calibrated baseline"
        )


    def _update_hesitation_history(
        self,
        timestamp,
        hesitation_event
    ):

        if hesitation_event:

            self.hesitation_event_times.append(
                timestamp
            )

        while self.hesitation_event_times:

            oldest = (
                self.hesitation_event_times[0]
            )

            if (
                timestamp
                -
                oldest
                <=
                self.hesitation_window
            ):

                break

            self.hesitation_event_times.popleft()


    def _status(
        self,
        value,
        mild_threshold,
        strong_threshold
    ):

        if value >= strong_threshold:
            return "STRONG"

        if value >= mild_threshold:
            return "MILD"

        return "NORMAL"


    def update(
        self,
        timestamp,
        path_error,
        relative_angle_error,
        progress,
        learner_rotation,
        expected_rotation,
        hesitation_result=None,
        tracking=True
    ):

        timestamp = float(
            timestamp
        )

        if not tracking:

            return self._make_result(

                path_status=
                    "NO_TRACKING",

                angle_status=
                    "NO_TRACKING",

                rotation_deficit=
                    0.0,

                hesitation_count=
                    len(
                        self.hesitation_event_times
                    )
            )

        path_error = float(
            path_error
        )

        relative_angle_error = float(
            relative_angle_error
        )

        progress = float(
            progress
        )

        learner_rotation = float(
            learner_rotation
        )

        expected_rotation = float(
            expected_rotation
        )

        # Ignore natural start/end transients.
        if (
            progress < self.ignore_start
            or
            progress > self.ignore_end
        ):

            self.mild_start_time = None
            self.strong_start_time = None

            return self._make_result(

                path_status=
                    "IGNORED",

                angle_status=
                    "IGNORED",

                rotation_deficit=(
                    expected_rotation
                    -
                    learner_rotation
                ),

                hesitation_count=
                    len(
                        self.hesitation_event_times
                    )
            )

        hesitation_state = "NORMAL"
        hesitation_event = False
        hesitating = False

        if hesitation_result is not None:

            hesitation_state = str(
                hesitation_result.get(
                    "state",
                    "NORMAL"
                )
            )

            hesitation_event = bool(
                hesitation_result.get(
                    "event",
                    False
                )
            )

            hesitating = bool(
                hesitation_result.get(
                    "hesitating",
                    False
                )
            )

        self._update_hesitation_history(
            timestamp,
            hesitation_event
        )

        hesitation_count = len(
            self.hesitation_event_times
        )

        path_status = self._status(

            path_error,
            self.path_mild,
            self.path_strong
        )

        angle_status = self._status(

            relative_angle_error,
            self.angle_mild,
            self.angle_strong
        )

        rotation_deficit = (
            expected_rotation
            -
            learner_rotation
        )

        path_ratio = (
            path_error
            /
            max(
                self.path_mild,
                1e-6
            )
        )

        angle_ratio = (
            relative_angle_error
            /
            max(
                self.angle_mild,
                1e-6
            )
        )

        if path_ratio > angle_ratio:
            dominant_error = "PATH"
        else:
            dominant_error = "ROTATION"

        mild_issue = (

            path_status
            in
            (
                "MILD",
                "STRONG"
            )

            or

            angle_status
            in
            (
                "MILD",
                "STRONG"
            )
        )

        strong_issue = (

            path_status
            ==
            "STRONG"

            or

            angle_status
            ==
            "STRONG"

            or

            hesitating
        )

        if mild_issue:

            if self.mild_start_time is None:
                self.mild_start_time = timestamp

        else:
            self.mild_start_time = None

        if strong_issue:

            if self.strong_start_time is None:
                self.strong_start_time = timestamp

        else:
            self.strong_start_time = None

        if self.mild_start_time is None:
            mild_duration = 0.0
        else:
            mild_duration = max(
                0.0,
                timestamp
                -
                self.mild_start_time
            )

        if self.strong_start_time is None:
            strong_duration = 0.0
        else:
            strong_duration = max(
                0.0,
                timestamp
                -
                self.strong_start_time
            )

        requested_level = 0

        # Repeated hesitation only TRIGGERS L3 at the
        # moment a new hesitation event arrives.
        #
        # It does not lock the system at L3 for the
        # entire 8-second history window.
        repeated_hesitation_trigger = (

            hesitation_event

            and

            hesitation_count
            >=
            self.hesitation_events_for_level_3
        )

        if (

            repeated_hesitation_trigger

            or

            strong_duration
            >=
            self.level_3_hold
        ):

            requested_level = 3

        elif (

            hesitation_event

            or

            strong_duration
            >=
            self.level_2_hold
        ):

            requested_level = 2

        elif (

            mild_duration
            >=
            self.level_1_hold
        ):

            requested_level = 1

        # =================================================
        # ESCALATION
        # =================================================

        if (
            requested_level
            >
            self.current_level
        ):

            self.current_level = (
                requested_level
            )

            self.recovery_start_time = None

        # =================================================
        # RECOVERY / DE-ESCALATION
        #
        # FIX:
        # As soon as requested_level becomes lower than
        # current_level, start the recovery timer.
        #
        # After RECOVERY_SEC, step down ONE level.
        #
        # This allows:
        #
        # L2 -> L1 when the user improves to mild error
        # L1 -> L0 when the user returns to normal
        # L3 -> L2 -> L1 -> L0 gradually
        # =================================================

        elif (
            requested_level
            <
            self.current_level
        ):

            if self.recovery_start_time is None:

                self.recovery_start_time = (
                    timestamp
                )

            recovery_duration = (
                timestamp
                -
                self.recovery_start_time
            )

            if (
                recovery_duration
                >=
                self.recovery_sec
            ):

                self.current_level = max(

                    requested_level,

                    self.current_level
                    -
                    1
                )

                self.recovery_start_time = (
                    timestamp
                )

        else:

            # requested_level == current_level
            self.recovery_start_time = None

        reason, cue = self._build_feedback(

            level=
                self.current_level,

            dominant_error=
                dominant_error,

            rotation_deficit=
                rotation_deficit,

            hesitation_state=
                hesitation_state,

            hesitation_count=
                hesitation_count
        )

        self.last_reason = reason

        return self._make_result(

            path_status=
                path_status,

            angle_status=
                angle_status,

            rotation_deficit=
                rotation_deficit,

            hesitation_count=
                hesitation_count,

            mild_duration=
                mild_duration,

            strong_duration=
                strong_duration,

            dominant_error=
                dominant_error,

            reason=
                reason,

            cue=
                cue
        )


    def _build_feedback(
        self,
        level,
        dominant_error,
        rotation_deficit,
        hesitation_state,
        hesitation_count
    ):

        if level == 0:

            return (
                "Within calibrated baseline",
                "No assistance"
            )

        if level == 3:

            if (
                hesitation_count
                >=
                self.hesitation_events_for_level_3
            ):

                return (
                    "Repeated behavioral hesitation detected",
                    "Suggest human assistance"
                )

            return (
                "Severe error persisted despite assistance",
                "Suggest human assistance"
            )

        if level == 2:

            if (
                hesitation_state
                ==
                "HESITATION"
            ):

                return (
                    "Behavioral hesitation detected",
                    "Show explicit next-step guidance"
                )

            if dominant_error == "PATH":

                return (
                    "Persistent path deviation",
                    "Show reference path guidance"
                )

            if rotation_deficit > 0:

                return (
                    "Persistent rotation under-completion",
                    "Show stronger rotation guidance"
                )

            return (
                "Persistent rotation over-completion",
                "Show rotation correction guidance"
            )

        if dominant_error == "PATH":

            return (
                "Mild path deviation",
                "Gentle path cue"
            )

        if rotation_deficit > 0:

            return (
                "Mild rotation under-completion",
                "Gentle rotation cue"
            )

        return (
            "Mild rotation over-completion",
            "Gentle rotation cue"
        )


    def _make_result(
        self,
        path_status,
        angle_status,
        rotation_deficit,
        hesitation_count,
        mild_duration=0.0,
        strong_duration=0.0,
        dominant_error="NONE",
        reason=None,
        cue=None
    ):

        labels = {

            0:
                "OBSERVE",

            1:
                "GENTLE CUE",

            2:
                "EXPLICIT GUIDANCE",

            3:
                "HUMAN ASSISTANCE"
        }

        if reason is None:
            reason = self.last_reason

        if cue is None:
            cue = "No assistance"

        return {

            "level":
                self.current_level,

            "label":
                labels[
                    self.current_level
                ],

            "reason":
                reason,

            "cue":
                cue,

            "path_status":
                path_status,

            "angle_status":
                angle_status,

            "dominant_error":
                dominant_error,

            "rotation_deficit_deg":
                round(
                    float(
                        rotation_deficit
                    ),
                    3
                ),

            "hesitation_events":
                int(
                    hesitation_count
                ),

            "mild_duration_sec":
                round(
                    float(
                        mild_duration
                    ),
                    3
                ),

            "strong_duration_sec":
                round(
                    float(
                        strong_duration
                    ),
                    3
                )
        }


# =========================================================
# SELF TEST
# =========================================================

if __name__ == "__main__":

    config = build_assistance_config()

    policy = AssistancePolicy(
        config
    )

    print()
    print(
        "=============================================="
    )
    print(
        "AdaptiveSkill - Assistance Policy"
    )
    print(
        "=============================================="
    )
    print()

    print(
        "Correct baseline trials:",
        config[
            "baseline"
        ][
            "correct_trials"
        ]
    )

    print(
        "Valid baseline frames:",
        config[
            "baseline"
        ][
            "valid_frames"
        ]
    )

    print()

    print(
        "PATH"
    )

    print(
        "  Mild:",
        config[
            "thresholds"
        ][
            "path_mild"
        ]
    )

    print(
        "  Strong:",
        config[
            "thresholds"
        ][
            "path_strong"
        ]
    )

    print()

    print(
        "RELATIVE ANGLE"
    )

    print(
        "  Mild:",
        config[
            "thresholds"
        ][
            "angle_mild_deg"
        ],
        "deg"
    )

    print(
        "  Strong:",
        config[
            "thresholds"
        ][
            "angle_strong_deg"
        ],
        "deg"
    )

    print()

    print(
        "Policy timing:"
    )

    print(
        "  Level 1 hold:",
        policy.level_1_hold,
        "sec"
    )

    print(
        "  Level 2 hold:",
        policy.level_2_hold,
        "sec"
    )

    print(
        "  Level 3 hold:",
        policy.level_3_hold,
        "sec"
    )

    print(
        "  Recovery:",
        policy.recovery_sec,
        "sec / level"
    )

    print()

    print(
        "=============================================="
    )
    print(
        "POLICY LOGIC SELF TEST"
    )
    print(
        "=============================================="
    )

    # -----------------------------------------------------
    # 1. NORMAL
    # -----------------------------------------------------

    policy.reset()

    result = None
    t = 0.0

    for _ in range(30):

        result = policy.update(

            timestamp=t,

            path_error=
                policy.path_mild
                *
                0.5,

            relative_angle_error=
                policy.angle_mild
                *
                0.5,

            progress=0.50,

            learner_rotation=18.0,

            expected_rotation=18.5,

            hesitation_result={
                "state": "NORMAL",
                "event": False,
                "hesitating": False
            },

            tracking=True
        )

        t += 1.0 / 30.0

    print()
    print(
        "NORMAL"
    )
    print(
        "  Expected: Level 0"
    )
    print(
        "  Actual:",
        result["level"],
        result["label"]
    )

    # -----------------------------------------------------
    # 2. MILD PATH
    # -----------------------------------------------------

    policy.reset()

    result = None
    t = 0.0

    for _ in range(20):

        result = policy.update(

            timestamp=t,

            path_error=
                (
                    policy.path_mild
                    +
                    policy.path_strong
                )
                /
                2.0,

            relative_angle_error=
                policy.angle_mild
                *
                0.5,

            progress=0.50,

            learner_rotation=18.0,

            expected_rotation=18.0,

            hesitation_result={
                "state": "NORMAL",
                "event": False,
                "hesitating": False
            },

            tracking=True
        )

        t += 1.0 / 30.0

    print()
    print(
        "MILD PATH"
    )
    print(
        "  Expected: Level 1"
    )
    print(
        "  Actual:",
        result["level"],
        result["label"]
    )

    # -----------------------------------------------------
    # 3. STRONG ROTATION
    # -----------------------------------------------------

    policy.reset()

    result = None
    t = 0.0

    for _ in range(30):

        result = policy.update(

            timestamp=t,

            path_error=
                policy.path_mild
                *
                0.5,

            relative_angle_error=
                policy.angle_strong
                *
                1.5,

            progress=0.60,

            learner_rotation=8.0,

            expected_rotation=24.0,

            hesitation_result={
                "state": "NORMAL",
                "event": False,
                "hesitating": False
            },

            tracking=True
        )

        t += 1.0 / 30.0

    print()
    print(
        "STRONG ROTATION"
    )
    print(
        "  Expected: Level 2"
    )
    print(
        "  Actual:",
        result["level"],
        result["label"]
    )
    print(
        "  Cue:",
        result["cue"]
    )

    # -----------------------------------------------------
    # 4. REPEATED HESITATION
    # -----------------------------------------------------

    policy.reset()
    t = 0.0

    result = policy.update(

        timestamp=t,

        path_error=
            policy.path_mild
            *
            0.5,

        relative_angle_error=
            policy.angle_mild
            *
            0.5,

        progress=0.45,

        learner_rotation=15.0,

        expected_rotation=15.0,

        hesitation_result={
            "state": "HESITATION",
            "event": True,
            "hesitating": True
        },

        tracking=True
    )

    t += 1.0

    result = policy.update(

        timestamp=t,

        path_error=
            policy.path_mild
            *
            0.5,

        relative_angle_error=
            policy.angle_mild
            *
            0.5,

        progress=0.55,

        learner_rotation=20.0,

        expected_rotation=20.0,

        hesitation_result={
            "state": "NORMAL",
            "event": False,
            "hesitating": False
        },

        tracking=True
    )

    t += 1.0

    result = policy.update(

        timestamp=t,

        path_error=
            policy.path_mild
            *
            0.5,

        relative_angle_error=
            policy.angle_mild
            *
            0.5,

        progress=0.65,

        learner_rotation=23.0,

        expected_rotation=23.0,

        hesitation_result={
            "state": "HESITATION",
            "event": True,
            "hesitating": True
        },

        tracking=True
    )

    print()
    print(
        "REPEATED HESITATION"
    )
    print(
        "  Expected: Level 3"
    )
    print(
        "  Actual:",
        result["level"],
        result["label"]
    )
    print(
        "  Cue:",
        result["cue"]
    )

    # -----------------------------------------------------
    # 5. RECOVERY L2 -> L1 -> L0
    # -----------------------------------------------------

    policy.reset()
    t = 0.0

    # Force L2.
    for _ in range(30):

        result = policy.update(

            timestamp=t,

            path_error=
                policy.path_strong
                *
                1.5,

            relative_angle_error=
                policy.angle_mild
                *
                0.5,

            progress=0.50,

            learner_rotation=18.0,

            expected_rotation=18.0,

            hesitation_result={
                "state": "NORMAL",
                "event": False,
                "hesitating": False
            },

            tracking=True
        )

        t += 1.0 / 30.0

    reached_l2 = result["level"]

    # Improve to mild error long enough to step down to L1.
    mild_path_value = (
        policy.path_mild
        +
        policy.path_strong
    ) / 2.0

    for _ in range(40):

        result = policy.update(

            timestamp=t,

            path_error=
                mild_path_value,

            relative_angle_error=
                policy.angle_mild
                *
                0.5,

            progress=0.55,

            learner_rotation=20.0,

            expected_rotation=20.0,

            hesitation_result={
                "state": "NORMAL",
                "event": False,
                "hesitating": False
            },

            tracking=True
        )

        t += 1.0 / 30.0

    recovered_l1 = result["level"]

    # Then fully normal long enough to step down to L0.
    for _ in range(40):

        result = policy.update(

            timestamp=t,

            path_error=
                policy.path_mild
                *
                0.5,

            relative_angle_error=
                policy.angle_mild
                *
                0.5,

            progress=0.60,

            learner_rotation=22.0,

            expected_rotation=22.0,

            hesitation_result={
                "state": "NORMAL",
                "event": False,
                "hesitating": False
            },

            tracking=True
        )

        t += 1.0 / 30.0

    recovered_l0 = result["level"]

    print()
    print(
        "RECOVERY L2 -> L1 -> L0"
    )
    print(
        "  Expected: 2 -> 1 -> 0"
    )
    print(
        "  Actual:",
        reached_l2,
        "->",
        recovered_l1,
        "->",
        recovered_l0
    )

    print()
    print(
        "=============================================="
    )
    print(
        "Saved configuration:"
    )
    print(
        CONFIG_PATH
    )
    print(
        "=============================================="
    )
    print()
