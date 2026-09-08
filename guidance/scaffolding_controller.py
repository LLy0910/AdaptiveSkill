import sys
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path


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


# =========================================================
# SCAFFOLDING LEVELS
# =========================================================

LEVEL_0 = 0
LEVEL_1 = 1
LEVEL_2 = 2
LEVEL_3 = 3


LEVEL_LABELS = {

    LEVEL_0:
        "OBSERVE",

    LEVEL_1:
        "MINIMAL CUE",

    LEVEL_2:
        "EXPLICIT CORRECTION",

    LEVEL_3:
        "STRONG ASSISTANCE",
}


# =========================================================
# RESULT
# =========================================================

@dataclass
class ScaffoldingDecision:
    """
    Adaptive assistance decision.

    IMPORTANT:

    This controller uses TASK STATE only.

    It does NOT use:
        - expert-reference distance
        - trajectory similarity
        - DTW score
        - "how expert-like" the user looks

    Therefore a valid alternative strategy remains L0.
    """

    level: int
    label: str

    requested_level: int

    task_state: str

    paused: bool

    risk_duration_sec: float
    violation_duration_sec: float

    recent_violation_episodes: int

    reason: str


    def to_dict(self):
        return asdict(self)


# =========================================================
# CONTROLLER
# =========================================================

class ScaffoldingController:
    """
    Task-aware adaptive scaffolding.

    Core behavior:

        VALID
            -> L0

        new RISK
            -> L1

        persistent RISK
            -> L2

        VIOLATION
            -> L3 immediately

        continued / repeated VIOLATION
            -> remain at L3

        recovery
            -> fade one level at a time

    Prototype thresholds are engineering defaults.
    They are NOT experimentally validated optimal values.
    """


    def __init__(
        self,

        persistent_risk_sec=0.80,

        persistent_violation_sec=2.00,

        repeated_violation_count=2,

        repeated_violation_window_sec=6.00,

        fade_hold_sec=0.60,
    ):

        self.persistent_risk_sec = float(
            persistent_risk_sec
        )

        self.persistent_violation_sec = float(
            persistent_violation_sec
        )

        self.repeated_violation_count = int(
            repeated_violation_count
        )

        self.repeated_violation_window_sec = float(
            repeated_violation_window_sec
        )

        self.fade_hold_sec = float(
            fade_hold_sec
        )


        self.reset()


    # =====================================================
    # RESET
    # =====================================================

    def reset(self):

        self.current_level = (
            LEVEL_0
        )


        self.previous_task_state = (
            None
        )


        self.risk_start_timestamp = (
            None
        )


        self.violation_start_timestamp = (
            None
        )


        self.fade_start_timestamp = (
            None
        )


        self.violation_episode_times = (
            deque()
        )


    # =====================================================
    # TIME HELPERS
    # =====================================================

    def _prune_violation_history(
        self,
        timestamp_sec,
    ):

        minimum_time = (

            float(timestamp_sec)

            -

            self.repeated_violation_window_sec
        )


        while (
            self.violation_episode_times

            and

            self.violation_episode_times[0]
            <
            minimum_time
        ):

            self.violation_episode_times.popleft()


    # =====================================================
    # STATE DURATIONS
    # =====================================================

    def _update_state_timers(
        self,
        task_state,
        timestamp_sec,
    ):

        timestamp_sec = float(
            timestamp_sec
        )


        # -------------------------------------------------
        # RISK
        # -------------------------------------------------

        if task_state == STATE_RISK:

            if (
                self.previous_task_state
                !=
                STATE_RISK
            ):

                self.risk_start_timestamp = (
                    timestamp_sec
                )


            self.violation_start_timestamp = (
                None
            )


        # -------------------------------------------------
        # VIOLATION
        # -------------------------------------------------

        elif task_state == STATE_VIOLATION:

            if (
                self.previous_task_state
                !=
                STATE_VIOLATION
            ):

                self.violation_start_timestamp = (
                    timestamp_sec
                )


                self.violation_episode_times.append(
                    timestamp_sec
                )


            self.risk_start_timestamp = (
                None
            )


        # -------------------------------------------------
        # VALID
        # -------------------------------------------------

        else:

            self.risk_start_timestamp = (
                None
            )

            self.violation_start_timestamp = (
                None
            )


        self._prune_violation_history(
            timestamp_sec
        )


    # =====================================================
    # DURATIONS
    # =====================================================

    def _get_risk_duration(
        self,
        timestamp_sec,
    ):

        if self.risk_start_timestamp is None:

            return 0.0


        return max(
            0.0,

            float(timestamp_sec)

            -

            self.risk_start_timestamp
        )


    def _get_violation_duration(
        self,
        timestamp_sec,
    ):

        if self.violation_start_timestamp is None:

            return 0.0


        return max(
            0.0,

            float(timestamp_sec)

            -

            self.violation_start_timestamp
        )


    # =====================================================
    # REQUESTED LEVEL
    # =====================================================

    def _compute_requested_level(
        self,
        task_state,
        risk_duration,
        violation_duration,
    ):

        # -------------------------------------------------
        # VALID
        # -------------------------------------------------

        if task_state == STATE_VALID:

            return (
                LEVEL_0,
                "Task constraints are currently satisfied."
            )


        # -------------------------------------------------
        # RISK
        # -------------------------------------------------

        if task_state == STATE_RISK:

            if (
                risk_duration
                >=
                self.persistent_risk_sec
            ):

                return (
                    LEVEL_2,
                    "Task risk persisted beyond the minimal-cue period."
                )


            return (
                LEVEL_1,
                "A task risk appeared; start with minimal assistance."
            )


        # -------------------------------------------------
        # VIOLATION
        # -------------------------------------------------

        if task_state == STATE_VIOLATION:

            # Hard task-constraint violations receive immediate
            # strong assistance.
            #
            # IMPORTANT:
            # This controller is input-mode agnostic. The same rule
            # therefore applies to BOTH hand and mouse input.
            #
            # Duration / repetition are still logged as engineering
            # metrics, but they no longer delay escalation to L3.

            return (
                LEVEL_3,
                "A hard task-constraint violation requires immediate strong assistance."
            )


        # -------------------------------------------------
        # FALLBACK
        # -------------------------------------------------

        return (
            LEVEL_0,
            "No task-aware assistance requested."
        )


    # =====================================================
    # LEVEL TRANSITION
    # =====================================================

    def _update_level(
        self,
        requested_level,
        timestamp_sec,
    ):

        requested_level = int(
            requested_level
        )

        timestamp_sec = float(
            timestamp_sec
        )


        # =================================================
        # ESCALATION
        #
        # Safety escalation is immediate.
        # =================================================

        if (
            requested_level
            >
            self.current_level
        ):

            self.current_level = (
                requested_level
            )


            self.fade_start_timestamp = (
                None
            )


            return


        # =================================================
        # SAME LEVEL
        # =================================================

        if (
            requested_level
            ==
            self.current_level
        ):

            self.fade_start_timestamp = (
                None
            )

            return


        # =================================================
        # DE-ESCALATION
        #
        # Fade only one level at a time.
        # =================================================

        if self.fade_start_timestamp is None:

            self.fade_start_timestamp = (
                timestamp_sec
            )

            return


        fade_duration = (

            timestamp_sec

            -

            self.fade_start_timestamp
        )


        if (
            fade_duration
            >=
            self.fade_hold_sec
        ):

            self.current_level = max(

                requested_level,

                self.current_level - 1
            )


            # One hold period per downward step.
            if (
                self.current_level
                >
                requested_level
            ):

                self.fade_start_timestamp = (
                    timestamp_sec
                )

            else:

                self.fade_start_timestamp = (
                    None
                )


    # =====================================================
    # MAIN UPDATE
    # =====================================================

    def update(
        self,
        task_result,
        timestamp_sec,
    ):

        timestamp_sec = float(
            timestamp_sec
        )


        task_state = (
            task_result.state
        )


        # =================================================
        # TRACKING UNRELIABLE
        #
        # Do not interpret tracking loss as user error.
        # Freeze current assistance level.
        # =================================================

        if (
            task_state
            ==
            STATE_TRACKING_UNRELIABLE
        ):

            return ScaffoldingDecision(

                level=
                    self.current_level,

                label=
                    "PAUSED",

                requested_level=
                    self.current_level,

                task_state=
                    task_state,

                paused=
                    True,

                risk_duration_sec=
                    0.0,

                violation_duration_sec=
                    0.0,

                recent_violation_episodes=
                    len(
                        self.violation_episode_times
                    ),

                reason=(
                    "Tracking is unreliable; "
                    "assistance escalation is paused."
                )
            )


        # =================================================
        # UPDATE STATE HISTORY
        # =================================================

        self._update_state_timers(

            task_state=
                task_state,

            timestamp_sec=
                timestamp_sec,
        )


        risk_duration = (
            self._get_risk_duration(
                timestamp_sec
            )
        )


        violation_duration = (
            self._get_violation_duration(
                timestamp_sec
            )
        )


        (
            requested_level,
            reason,
        ) = self._compute_requested_level(

            task_state=
                task_state,

            risk_duration=
                risk_duration,

            violation_duration=
                violation_duration,
        )


        # =================================================
        # APPLY ESCALATION / FADING
        # =================================================

        self._update_level(

            requested_level=
                requested_level,

            timestamp_sec=
                timestamp_sec,
        )


        self.previous_task_state = (
            task_state
        )


        return ScaffoldingDecision(

            level=
                self.current_level,

            label=
                LEVEL_LABELS[
                    self.current_level
                ],

            requested_level=
                requested_level,

            task_state=
                task_state,

            paused=
                False,

            risk_duration_sec=
                risk_duration,

            violation_duration_sec=
                violation_duration,

            recent_violation_episodes=
                len(
                    self.violation_episode_times
                ),

            reason=
                reason,
        )


# =========================================================
# SELF TEST
# =========================================================

if __name__ == "__main__":

    evaluator = (
        TaskConstraintEvaluator()
    )


    # =====================================================
    # TASK STATES
    # =====================================================

    valid_result = (
        evaluator.evaluate(

            x=
                1.65,

            y=
                0.90,

            orientation_relative_deg=
                0.0,

            tracking_missing_sec=
                0.0,
        )
    )


    risk_result = (
        evaluator.evaluate(

            x=
                1.65,

            y=
                0.55,

            orientation_relative_deg=
                0.0,

            tracking_missing_sec=
                0.0,
        )
    )


    violation_result = (
        evaluator.evaluate(

            x=
                1.65,

            y=
                0.0,

            orientation_relative_deg=
                0.0,

            tracking_missing_sec=
                0.0,
        )
    )


    tracking_result = (
        evaluator.evaluate(

            x=
                1.0,

            y=
                -0.80,

            orientation_relative_deg=
                0.0,

            tracking_missing_sec=
                0.50,
        )
    )


    print()

    print(
        "========================================================"
    )

    print(
        "AdaptiveSkill - Adaptive Scaffolding Controller"
    )

    print(
        "========================================================"
    )


    total_tests = 0
    passed_tests = 0


    # =====================================================
    # TEST 1
    # VALID ALTERNATIVE -> L0
    # =====================================================

    controller = (
        ScaffoldingController()
    )


    decision = (
        controller.update(
            valid_result,
            0.0,
        )
    )


    ok = (
        decision.level
        ==
        LEVEL_0
    )


    total_tests += 1
    passed_tests += int(ok)


    print()

    print(
        "1. VALID ALTERNATIVE STRATEGY"
    )

    print(
        "-" * 56
    )

    print(
        "Task state:",
        decision.task_state
    )

    print(
        "Expected level: 0"
    )

    print(
        "Actual level:",
        decision.level,
        decision.label
    )

    print(
        "PASS:",
        ok
    )


    # =====================================================
    # TEST 2
    # NEW RISK -> L1
    # =====================================================

    controller = (
        ScaffoldingController()
    )


    decision = (
        controller.update(
            risk_result,
            0.0,
        )
    )


    ok = (
        decision.level
        ==
        LEVEL_1
    )


    total_tests += 1
    passed_tests += int(ok)


    print()

    print(
        "2. NEW TASK RISK"
    )

    print(
        "-" * 56
    )

    print(
        "Expected level: 1"
    )

    print(
        "Actual level:",
        decision.level,
        decision.label
    )

    print(
        "Risk duration:",
        round(
            decision.risk_duration_sec,
            2
        )
    )

    print(
        "PASS:",
        ok
    )


    # =====================================================
    # TEST 3
    # PERSISTENT RISK -> L2
    # =====================================================

    controller = (
        ScaffoldingController()
    )


    controller.update(
        risk_result,
        0.0,
    )


    decision = (
        controller.update(
            risk_result,
            0.90,
        )
    )


    ok = (
        decision.level
        ==
        LEVEL_2
    )


    total_tests += 1
    passed_tests += int(ok)


    print()

    print(
        "3. PERSISTENT TASK RISK"
    )

    print(
        "-" * 56
    )

    print(
        "Expected level: 2"
    )

    print(
        "Actual level:",
        decision.level,
        decision.label
    )

    print(
        "Risk duration:",
        round(
            decision.risk_duration_sec,
            2
        )
    )

    print(
        "PASS:",
        ok
    )


    # =====================================================
    # TEST 4
    # RECOVERY -> L2 -> L1 -> L0
    # =====================================================

    controller = (
        ScaffoldingController()
    )


    controller.update(
        risk_result,
        0.0,
    )


    level_2 = (
        controller.update(
            risk_result,
            0.90,
        )
    )


    # First valid sample starts fade timer.
    controller.update(
        valid_result,
        1.00,
    )


    level_1 = (
        controller.update(
            valid_result,
            1.70,
        )
    )


    level_0 = (
        controller.update(
            valid_result,
            2.40,
        )
    )


    ok = (

        level_2.level == LEVEL_2

        and

        level_1.level == LEVEL_1

        and

        level_0.level == LEVEL_0
    )


    total_tests += 1
    passed_tests += int(ok)


    print()

    print(
        "4. STEPWISE FADING"
    )

    print(
        "-" * 56
    )

    print(
        "Expected: 2 -> 1 -> 0"
    )

    print(
        "Actual:",
        level_2.level,
        "->",
        level_1.level,
        "->",
        level_0.level
    )

    print(
        "PASS:",
        ok
    )


    # =====================================================
    # TEST 5
    # VIOLATION -> L3 IMMEDIATELY
    # =====================================================

    controller = (
        ScaffoldingController()
    )


    decision = (
        controller.update(
            violation_result,
            0.0,
        )
    )


    ok = (
        decision.level
        ==
        LEVEL_3
    )


    total_tests += 1
    passed_tests += int(ok)


    print()

    print(
        "5. TASK VIOLATION"
    )

    print(
        "-" * 56
    )

    print(
        "Expected level: 3"
    )

    print(
        "Actual level:",
        decision.level,
        decision.label
    )

    print(
        "PASS:",
        ok
    )


    # =====================================================
    # TEST 6
    # CONTINUED VIOLATION -> REMAIN L3
    # =====================================================

    controller = (
        ScaffoldingController()
    )


    controller.update(
        violation_result,
        0.0,
    )


    decision = (
        controller.update(
            violation_result,
            2.10,
        )
    )


    ok = (
        decision.level
        ==
        LEVEL_3
    )


    total_tests += 1
    passed_tests += int(ok)


    print()

    print(
        "6. CONTINUED VIOLATION"
    )

    print(
        "-" * 56
    )

    print(
        "Expected level: 3"
    )

    print(
        "Actual level:",
        decision.level,
        decision.label
    )

    print(
        "Violation duration:",
        round(
            decision.violation_duration_sec,
            2
        )
    )

    print(
        "PASS:",
        ok
    )


    # =====================================================
    # TEST 7
    # REPEATED VIOLATION -> REMAIN L3
    # =====================================================

    controller = (
        ScaffoldingController()
    )


    first = (
        controller.update(
            violation_result,
            0.0,
        )
    )


    controller.update(
        valid_result,
        0.20,
    )


    second = (
        controller.update(
            violation_result,
            0.40,
        )
    )


    ok = (

        first.level
        ==
        LEVEL_3

        and

        second.level
        ==
        LEVEL_3

        and

        second.recent_violation_episodes
        >=
        2
    )


    total_tests += 1
    passed_tests += int(ok)


    print()

    print(
        "7. REPEATED VIOLATION"
    )

    print(
        "-" * 56
    )

    print(
        "Expected: first L3, second L3"
    )

    print(
        "Actual:",
        first.level,
        "->",
        second.level
    )

    print(
        "Recent violation episodes:",
        second.recent_violation_episodes
    )

    print(
        "PASS:",
        ok
    )


    # =====================================================
    # TEST 8
    # TRACKING LOSS MUST NOT ESCALATE
    # =====================================================

    controller = (
        ScaffoldingController()
    )


    decision = (
        controller.update(
            tracking_result,
            0.0,
        )
    )


    ok = (

        decision.paused

        and

        decision.level
        ==
        LEVEL_0

        and

        decision.label
        ==
        "PAUSED"
    )


    total_tests += 1
    passed_tests += int(ok)


    print()

    print(
        "8. TRACKING UNRELIABLE"
    )

    print(
        "-" * 56
    )

    print(
        "Expected: PAUSED, no escalation"
    )

    print(
        "Actual:",
        decision.level,
        decision.label
    )

    print(
        "Paused:",
        decision.paused
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
        f"{passed_tests}/{total_tests} tests passed"
    )


    if (
        passed_tests
        ==
        total_tests
    ):

        print(
            "ADAPTIVE SCAFFOLDING CONTROLLER: PASSED"
        )

    else:

        print(
            "ADAPTIVE SCAFFOLDING CONTROLLER: FAILED"
        )


    print(
        "========================================================"
    )

    print()