import json
from collections import deque
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "hesitation_config.json"
)


# =========================================================
# HESITATION DETECTOR
# =========================================================

class HesitationDetector:
    """
    Detect hesitation from:

    1. Low movement speed
    2. Very little reference-progress change
    3. The condition lasts long enough

    Important:

    Slow movement alone is NOT hesitation.
    """

    def __init__(
        self,
        config_path=DEFAULT_CONFIG_PATH
    ):

        self.config_path = Path(
            config_path
        )

        self._load_config()

        self.reset()


    # =====================================================
    # LOAD CONFIG
    # =====================================================

    def _load_config(self):

        if not self.config_path.exists():

            raise FileNotFoundError(
                f"Hesitation config not found: "
                f"{self.config_path}"
            )


        with open(
            self.config_path,
            "r",
            encoding="utf-8"
        ) as file:

            config = json.load(
                file
            )


        required = [
            "speed_threshold",
            "hold_duration_sec",
            "max_progress_delta",
            "ignore_start_progress",
            "ignore_end_progress",
            "cooldown_sec"
        ]


        missing = [
            key
            for key in required
            if key not in config
        ]


        if missing:

            raise ValueError(
                "Missing hesitation config values: "
                + ", ".join(missing)
            )


        self.speed_threshold = float(
            config[
                "speed_threshold"
            ]
        )

        self.hold_duration = float(
            config[
                "hold_duration_sec"
            ]
        )

        self.max_progress_delta = float(
            config[
                "max_progress_delta"
            ]
        )

        self.ignore_start_progress = float(
            config[
                "ignore_start_progress"
            ]
        )

        self.ignore_end_progress = float(
            config[
                "ignore_end_progress"
            ]
        )

        self.cooldown_sec = float(
            config[
                "cooldown_sec"
            ]
        )


    # =====================================================
    # RESET
    # =====================================================

    def reset(self):

        # Low-speed samples:
        #
        # (timestamp, progress)
        #
        self.low_speed_samples = deque()

        self.is_hesitating = False

        self.last_event_time = None

        self.last_state = "NORMAL"


    # =====================================================
    # CLEAR LOW-SPEED CANDIDATE
    # =====================================================

    def _clear_candidate(self):

        self.low_speed_samples.clear()

        self.is_hesitating = False


    # =====================================================
    # UPDATE
    # =====================================================

    def update(
        self,
        timestamp,
        speed,
        progress,
        tracking=True
    ):
        """
        Call once per frame.

        Returns a dictionary such as:

        {
            "state": "NORMAL",
            "hesitating": False,
            "event": False,
            "low_speed": False,
            "candidate_duration": 0.0,
            "progress_delta": 0.0
        }

        Possible states:

        NORMAL
        LOW_SPEED
        HESITATION
        IGNORED
        NO_TRACKING
        """

        timestamp = float(
            timestamp
        )

        speed = float(
            speed
        )

        progress = float(
            progress
        )


        # =================================================
        # 1. NO TRACKING
        # =================================================

        if not tracking:

            self._clear_candidate()

            self.last_state = (
                "NO_TRACKING"
            )

            return self._result(
                state="NO_TRACKING"
            )


        # =================================================
        # 2. IGNORE START / END
        # =================================================

        usable_progress = (

            self.ignore_start_progress
            <=
            progress
            <=
            self.ignore_end_progress

        )


        if not usable_progress:

            self._clear_candidate()

            self.last_state = (
                "IGNORED"
            )

            return self._result(
                state="IGNORED"
            )


        # =================================================
        # 3. MOVING NORMALLY
        # =================================================

        low_speed = (
            speed
            <
            self.speed_threshold
        )


        if not low_speed:

            self._clear_candidate()

            self.last_state = (
                "NORMAL"
            )

            return self._result(
                state="NORMAL",
                low_speed=False
            )


        # =================================================
        # 4. LOW SPEED
        # =================================================

        self.low_speed_samples.append(
            (
                timestamp,
                progress
            )
        )


        # Keep enough history to examine the current
        # hesitation window.
        #
        # We keep a little more than hold_duration so that
        # timing remains robust across frame-rate changes.

        history_limit = (
            self.hold_duration
            +
            0.25
        )


        while (
            len(
                self.low_speed_samples
            )
            >
            1
        ):

            oldest_time = (
                self.low_speed_samples[0][0]
            )


            if (
                timestamp
                -
                oldest_time
                <=
                history_limit
            ):

                break


            self.low_speed_samples.popleft()


        # =================================================
        # 5. CURRENT LOW-SPEED RUN
        # =================================================

        candidate_start_time = (
            self.low_speed_samples[0][0]
        )


        candidate_duration = max(
            0.0,
            timestamp
            -
            candidate_start_time
        )


        progresses = [
            item[1]
            for item in self.low_speed_samples
        ]


        progress_delta = (

            max(progresses)
            -
            min(progresses)

        )


        # =================================================
        # 6. SLOW BUT STILL PROGRESSING
        #
        # If movement is slow but progress keeps changing,
        # this should NOT immediately become hesitation.
        # =================================================

        if (
            progress_delta
            >
            self.max_progress_delta
        ):

            # Start a fresh low-speed candidate
            # from the current position.
            #
            # This prevents one long slow movement from
            # accumulating forever.

            self.low_speed_samples.clear()

            self.low_speed_samples.append(
                (
                    timestamp,
                    progress
                )
            )

            self.is_hesitating = False

            self.last_state = (
                "LOW_SPEED"
            )

            return self._result(
                state="LOW_SPEED",
                low_speed=True,
                candidate_duration=0.0,
                progress_delta=0.0
            )


        # =================================================
        # 7. POSSIBLE HESITATION
        # =================================================

        if (
            candidate_duration
            <
            self.hold_duration
        ):

            self.is_hesitating = False

            self.last_state = (
                "LOW_SPEED"
            )

            return self._result(
                state="LOW_SPEED",
                low_speed=True,
                candidate_duration=(
                    candidate_duration
                ),
                progress_delta=(
                    progress_delta
                )
            )


        # =================================================
        # 8. HESITATION CONFIRMED
        # =================================================

        self.is_hesitating = True

        event = False


        # -------------------------------------------------
        # Event is emitted only once per hesitation episode
        # and obeys cooldown.
        # -------------------------------------------------

        if self.last_state != "HESITATION":

            cooldown_ok = (

                self.last_event_time
                is None

                or

                (
                    timestamp
                    -
                    self.last_event_time
                )
                >=
                self.cooldown_sec

            )


            if cooldown_ok:

                event = True

                self.last_event_time = (
                    timestamp
                )


        self.last_state = (
            "HESITATION"
        )


        return self._result(
            state="HESITATION",
            hesitating=True,
            event=event,
            low_speed=True,
            candidate_duration=(
                candidate_duration
            ),
            progress_delta=(
                progress_delta
            )
        )


    # =====================================================
    # RESULT FORMAT
    # =====================================================

    def _result(
        self,
        state,
        hesitating=False,
        event=False,
        low_speed=False,
        candidate_duration=0.0,
        progress_delta=0.0
    ):

        return {

            "state":
                state,

            "hesitating":
                bool(
                    hesitating
                ),

            "event":
                bool(
                    event
                ),

            "low_speed":
                bool(
                    low_speed
                ),

            "candidate_duration":
                round(
                    float(
                        candidate_duration
                    ),
                    4
                ),

            "progress_delta":
                round(
                    float(
                        progress_delta
                    ),
                    6
                ),

            "speed_threshold":
                self.speed_threshold,

            "hold_duration":
                self.hold_duration,

            "max_progress_delta":
                self.max_progress_delta
        }


# =========================================================
# SIMPLE SELF TEST
# =========================================================

if __name__ == "__main__":

    detector = (
        HesitationDetector()
    )


    print()
    print(
        "=========================================="
    )

    print(
        "AdaptiveSkill - Hesitation Detector"
    )

    print(
        "=========================================="
    )

    print()

    print(
        "Speed threshold:",
        detector.speed_threshold
    )

    print(
        "Hold duration:",
        detector.hold_duration,
        "sec"
    )

    print(
        "Max progress delta:",
        detector.max_progress_delta
    )

    print(
        "Ignore start:",
        detector.ignore_start_progress
    )

    print(
        "Ignore end:",
        detector.ignore_end_progress
    )

    print()

    print(
        "Detector loaded successfully."
    )

    print()