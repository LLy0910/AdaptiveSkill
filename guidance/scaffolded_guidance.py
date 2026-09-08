import sys
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
# IMPORTS
# =========================================================

from task.task_constraints import (
    TaskConstraintEvaluator,
    STATE_VALID,
    STATE_TRACKING_UNRELIABLE,
)

from guidance.correction_guidance import (
    CorrectionGuidanceGenerator,
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


# =========================================================
# EXTRA GUIDANCE TYPE
#
# Used by the current demo when a sampled movement
# segment jumps through an obstacle.
# =========================================================

GUIDANCE_RETURN_SAFE = "RETURN_TO_LAST_SAFE"


# =========================================================
# PRESENTATION MODES
# =========================================================

MODE_NONE = "NONE"

MODE_MINIMAL_DIRECTION = (
    "MINIMAL_DIRECTION"
)

MODE_MINIMAL_ORIENTATION = (
    "MINIMAL_ORIENTATION"
)

MODE_EXPLICIT_DIRECTION = (
    "EXPLICIT_DIRECTION"
)

MODE_EXPLICIT_WAYPOINT = (
    "EXPLICIT_WAYPOINT"
)

MODE_EXPLICIT_ORIENTATION = (
    "EXPLICIT_ORIENTATION"
)

MODE_STRONG_WAYPOINT = (
    "STRONG_WAYPOINT"
)

MODE_STRONG_ORIENTATION = (
    "STRONG_ORIENTATION"
)

MODE_RECOVERY_FADE = (
    "RECOVERY_FADE"
)

MODE_PAUSED = (
    "PAUSED"
)


# =========================================================
# RESULT
# =========================================================

@dataclass
class ScaffoldedGuidancePresentation:
    """
    Describes HOW the current guidance should be shown.

    This object does not decide whether the task
    is correct.

    Task constraints:
        decide whether assistance is warranted.

    Scaffolding controller:
        decides HOW MUCH assistance to provide.

    This mapper:
        decides WHAT the user should actually see.
    """

    level: int

    level_label: str

    display_mode: str

    show_direction_arrow: bool

    show_waypoint: bool

    show_orientation_ghost: bool

    show_rotation_direction: bool

    show_rotation_degrees: bool

    show_strong_banner: bool

    show_recovery_status: bool

    paused: bool

    message: str


    def to_dict(self):
        return asdict(self)


# =========================================================
# MAPPER
# =========================================================

class ScaffoldedGuidanceMapper:
    """
    Maps:

        task state
        +
        scaffolding level
        +
        task-aware correction

    into a user-facing presentation.

    IMPORTANT:

    Reference similarity is NOT used here.
    """


    # =====================================================
    # EMPTY
    # =====================================================

    def _none(
        self,
        decision,
    ):

        return ScaffoldedGuidancePresentation(

            level=
                decision.level,

            level_label=
                decision.label,

            display_mode=
                MODE_NONE,

            show_direction_arrow=
                False,

            show_waypoint=
                False,

            show_orientation_ghost=
                False,

            show_rotation_direction=
                False,

            show_rotation_degrees=
                False,

            show_strong_banner=
                False,

            show_recovery_status=
                False,

            paused=
                False,

            message=
                "No assistance is needed."
        )


    # =====================================================
    # PAUSED
    # =====================================================

    def _paused(
        self,
        decision,
    ):

        return ScaffoldedGuidancePresentation(

            level=
                decision.level,

            level_label=
                "PAUSED",

            display_mode=
                MODE_PAUSED,

            show_direction_arrow=
                False,

            show_waypoint=
                False,

            show_orientation_ghost=
                False,

            show_rotation_direction=
                False,

            show_rotation_degrees=
                False,

            show_strong_banner=
                False,

            show_recovery_status=
                False,

            paused=
                True,

            message=(
                "Tracking is unreliable. "
                "Assistance is paused."
            )
        )


    # =====================================================
    # RECOVERY
    # =====================================================

    def _recovery(
        self,
        decision,
    ):
        """
        The user has already returned to VALID.

        The controller may still be fading from:
            L3 -> L2 -> L1 -> L0

        Do NOT keep showing stale directional commands.
        """

        return ScaffoldedGuidancePresentation(

            level=
                decision.level,

            level_label=
                decision.label,

            display_mode=
                MODE_RECOVERY_FADE,

            show_direction_arrow=
                False,

            show_waypoint=
                False,

            show_orientation_ghost=
                False,

            show_rotation_direction=
                False,

            show_rotation_degrees=
                False,

            show_strong_banner=
                False,

            show_recovery_status=
                True,

            paused=
                False,

            message=(
                "Task state recovered. "
                "Assistance is fading."
            )
        )


    # =====================================================
    # LEVEL 1
    # =====================================================

    def _level_1(
        self,
        decision,
        base_guidance,
    ):
        """
        Minimal cue.

        Do not reveal more information than necessary.
        """

        # -------------------------------------------------
        # Obstacle:
        # short direction only.
        # -------------------------------------------------

        if (
            base_guidance.guidance_type
            ==
            GUIDANCE_DIRECTION
        ):

            return ScaffoldedGuidancePresentation(

                level=
                    decision.level,

                level_label=
                    decision.label,

                display_mode=
                    MODE_MINIMAL_DIRECTION,

                show_direction_arrow=
                    True,

                show_waypoint=
                    False,

                show_orientation_ghost=
                    False,

                show_rotation_direction=
                    False,

                show_rotation_degrees=
                    False,

                show_strong_banner=
                    False,

                show_recovery_status=
                    False,

                paused=
                    False,

                message=(
                    "Move slightly away from the task risk."
                )
            )


        # -------------------------------------------------
        # Orientation:
        # tell direction, but do NOT reveal exact degrees.
        # -------------------------------------------------

        if (
            base_guidance.guidance_type
            ==
            GUIDANCE_ORIENTATION
        ):

            return ScaffoldedGuidancePresentation(

                level=
                    decision.level,

                level_label=
                    decision.label,

                display_mode=
                    MODE_MINIMAL_ORIENTATION,

                show_direction_arrow=
                    False,

                show_waypoint=
                    False,

                show_orientation_ghost=
                    False,

                show_rotation_direction=
                    True,

                show_rotation_degrees=
                    False,

                show_strong_banner=
                    False,

                show_recovery_status=
                    False,

                paused=
                    False,

                message=(
                    "Adjust the object orientation."
                )
            )


        # Defensive fallback.
        return ScaffoldedGuidancePresentation(

            level=
                decision.level,

            level_label=
                decision.label,

            display_mode=
                MODE_EXPLICIT_DIRECTION,

            show_direction_arrow=
                True,

            show_waypoint=
                False,

            show_orientation_ghost=
                False,

            show_rotation_direction=
                False,

            show_rotation_degrees=
                False,

            show_strong_banner=
                False,

            show_recovery_status=
                False,

            paused=
                False,

            message=
                base_guidance.message
        )


    # =====================================================
    # LEVEL 2
    # =====================================================

    def _level_2(
        self,
        decision,
        base_guidance,
    ):
        """
        Explicit actionable correction.
        """

        # -------------------------------------------------
        # Obstacle / segment crossing
        #
        # At L2 expose the actual safe target.
        # -------------------------------------------------

        if (
            base_guidance.guidance_type
            in (
                GUIDANCE_DIRECTION,
                GUIDANCE_WAYPOINT,
                GUIDANCE_RETURN_SAFE,
            )
        ):

            if (
                base_guidance.target_x
                is not None

                and

                base_guidance.target_y
                is not None
            ):

                return ScaffoldedGuidancePresentation(

                    level=
                        decision.level,

                    level_label=
                        decision.label,

                    display_mode=
                        MODE_EXPLICIT_WAYPOINT,

                    show_direction_arrow=
                        True,

                    show_waypoint=
                        True,

                    show_orientation_ghost=
                        False,

                    show_rotation_direction=
                        False,

                    show_rotation_degrees=
                        False,

                    show_strong_banner=
                        False,

                    show_recovery_status=
                        False,

                    paused=
                        False,

                    message=(
                        "Move toward the task-safe waypoint."
                    )
                )


            return ScaffoldedGuidancePresentation(

                level=
                    decision.level,

                level_label=
                    decision.label,

                display_mode=
                    MODE_EXPLICIT_DIRECTION,

                show_direction_arrow=
                    True,

                show_waypoint=
                    False,

                show_orientation_ghost=
                    False,

                show_rotation_direction=
                    False,

                show_rotation_degrees=
                    False,

                show_strong_banner=
                    False,

                show_recovery_status=
                    False,

                paused=
                    False,

                message=
                    base_guidance.message
            )


        # -------------------------------------------------
        # Orientation
        #
        # L2 reveals:
        #   - target ghost
        #   - direction
        #   - correction degrees
        # -------------------------------------------------

        if (
            base_guidance.guidance_type
            ==
            GUIDANCE_ORIENTATION
        ):

            return ScaffoldedGuidancePresentation(

                level=
                    decision.level,

                level_label=
                    decision.label,

                display_mode=
                    MODE_EXPLICIT_ORIENTATION,

                show_direction_arrow=
                    False,

                show_waypoint=
                    False,

                show_orientation_ghost=
                    True,

                show_rotation_direction=
                    True,

                show_rotation_degrees=
                    True,

                show_strong_banner=
                    False,

                show_recovery_status=
                    False,

                paused=
                    False,

                message=(
                    "Follow the explicit orientation correction."
                )
            )


        return self._none(
            decision
        )


    # =====================================================
    # LEVEL 3
    # =====================================================

    def _level_3(
        self,
        decision,
        base_guidance,
    ):
        """
        Strong assistance.

        L3 does not invent a different task goal.
        It makes the existing task-aware correction
        much more explicit.
        """

        # -------------------------------------------------
        # Orientation
        # -------------------------------------------------

        if (
            base_guidance.guidance_type
            ==
            GUIDANCE_ORIENTATION
        ):

            return ScaffoldedGuidancePresentation(

                level=
                    decision.level,

                level_label=
                    decision.label,

                display_mode=
                    MODE_STRONG_ORIENTATION,

                show_direction_arrow=
                    False,

                show_waypoint=
                    False,

                show_orientation_ghost=
                    True,

                show_rotation_direction=
                    True,

                show_rotation_degrees=
                    True,

                show_strong_banner=
                    True,

                show_recovery_status=
                    False,

                paused=
                    False,

                message=(
                    "Strong assistance: "
                    "follow the target orientation."
                )
            )


        # -------------------------------------------------
        # Spatial correction
        # -------------------------------------------------

        return ScaffoldedGuidancePresentation(

            level=
                decision.level,

            level_label=
                decision.label,

            display_mode=
                MODE_STRONG_WAYPOINT,

            show_direction_arrow=
                True,

            show_waypoint=
                (
                    base_guidance.target_x
                    is not None

                    and

                    base_guidance.target_y
                    is not None
                ),

            show_orientation_ghost=
                False,

            show_rotation_direction=
                False,

            show_rotation_degrees=
                False,

            show_strong_banner=
                True,

            show_recovery_status=
                False,

            paused=
                False,

            message=(
                "Strong assistance: "
                "follow the task-safe correction."
            )
        )


    # =====================================================
    # MAIN
    # =====================================================

    def map(
        self,
        task_result,
        scaffolding_decision,
        base_guidance,
    ):
        """
        Create the final user-facing presentation.
        """

        # =================================================
        # TRACKING
        # =================================================

        if (
            scaffolding_decision.paused

            or

            task_result.state
            ==
            STATE_TRACKING_UNRELIABLE
        ):

            return self._paused(
                scaffolding_decision
            )


        # =================================================
        # VALID
        #
        # Critical rule:
        #
        # When the task is already valid,
        # do NOT continue giving old movement commands.
        #
        # If assistance is still fading,
        # show recovery status only.
        # =================================================

        if (
            task_result.state
            ==
            STATE_VALID
        ):

            if (
                scaffolding_decision.level
                ==
                LEVEL_0
            ):

                return self._none(
                    scaffolding_decision
                )


            return self._recovery(
                scaffolding_decision
            )


        # =================================================
        # L0
        # =================================================

        if (
            scaffolding_decision.level
            ==
            LEVEL_0
        ):

            return self._none(
                scaffolding_decision
            )


        # =================================================
        # L1
        # =================================================

        if (
            scaffolding_decision.level
            ==
            LEVEL_1
        ):

            return self._level_1(

                scaffolding_decision,

                base_guidance,
            )


        # =================================================
        # L2
        # =================================================

        if (
            scaffolding_decision.level
            ==
            LEVEL_2
        ):

            return self._level_2(

                scaffolding_decision,

                base_guidance,
            )


        # =================================================
        # L3
        # =================================================

        return self._level_3(

            scaffolding_decision,

            base_guidance,
        )


# =========================================================
# SELF TEST
# =========================================================

if __name__ == "__main__":

    evaluator = (
        TaskConstraintEvaluator()
    )


    correction_generator = (
        CorrectionGuidanceGenerator()
    )


    mapper = (
        ScaffoldedGuidanceMapper()
    )


    print()

    print(
        "========================================================"
    )

    print(
        "AdaptiveSkill - Scaffolded Guidance Mapping"
    )

    print(
        "========================================================"
    )


    passed = 0
    total = 0


    # =====================================================
    # HELPER
    # =====================================================

    def run_case(
        name,
        controller,
        task_result,
        timestamp,
        expected_level,
        expected_mode,
    ):

        global passed
        global total


        base_guidance = (
            correction_generator.generate(

                x=
                    current_x,

                y=
                    current_y,

                orientation_relative_deg=
                    current_orientation,

                task_result=
                    task_result,
            )
        )


        decision = (
            controller.update(

                task_result,

                timestamp,
            )
        )


        presentation = (
            mapper.map(

                task_result=
                    task_result,

                scaffolding_decision=
                    decision,

                base_guidance=
                    base_guidance,
            )
        )


        ok = (

            decision.level
            ==
            expected_level

            and

            presentation.display_mode
            ==
            expected_mode
        )


        total += 1

        if ok:
            passed += 1


        print()

        print(
            name
        )

        print(
            "-" * 58
        )

        print(
            "Task state:",
            task_result.state
        )

        print(
            "Scaffolding level:",
            decision.level,
            decision.label
        )

        print(
            "Base guidance:",
            base_guidance.guidance_type
        )

        print(
            "Expected mode:",
            expected_mode
        )

        print(
            "Actual mode:",
            presentation.display_mode
        )

        print(
            "Arrow:",
            presentation.show_direction_arrow
        )

        print(
            "Waypoint:",
            presentation.show_waypoint
        )

        print(
            "Ghost:",
            presentation.show_orientation_ghost
        )

        print(
            "Degrees:",
            presentation.show_rotation_degrees
        )

        print(
            "Strong banner:",
            presentation.show_strong_banner
        )

        print(
            "Recovery:",
            presentation.show_recovery_status
        )

        print(
            "PASS:",
            ok
        )


    # =====================================================
    # 1. VALID ALTERNATIVE -> NONE
    # =====================================================

    current_x = 1.65
    current_y = 0.90
    current_orientation = 0.0


    valid_result = (
        evaluator.evaluate(

            x=
                current_x,

            y=
                current_y,

            orientation_relative_deg=
                current_orientation,

            tracking_missing_sec=
                0.0,
        )
    )


    controller = (
        ScaffoldingController()
    )


    run_case(

        "1. VALID ALTERNATIVE -> NO ASSISTANCE",

        controller,

        valid_result,

        0.0,

        LEVEL_0,

        MODE_NONE,
    )


    # =====================================================
    # 2. NEW OBSTACLE RISK -> L1 MINIMAL ARROW
    # =====================================================

    current_x = 1.65
    current_y = 0.55
    current_orientation = 0.0


    risk_result = (
        evaluator.evaluate(

            x=
                current_x,

            y=
                current_y,

            orientation_relative_deg=
                current_orientation,

            tracking_missing_sec=
                0.0,
        )
    )


    controller = (
        ScaffoldingController()
    )


    run_case(

        "2. NEW OBSTACLE RISK -> MINIMAL DIRECTION",

        controller,

        risk_result,

        0.0,

        LEVEL_1,

        MODE_MINIMAL_DIRECTION,
    )


    # =====================================================
    # 3. PERSISTENT OBSTACLE RISK -> L2 WAYPOINT
    # =====================================================

    controller = (
        ScaffoldingController()
    )


    controller.update(
        risk_result,
        0.0,
    )


    run_case(

        "3. PERSISTENT OBSTACLE RISK -> EXPLICIT WAYPOINT",

        controller,

        risk_result,

        0.90,

        LEVEL_2,

        MODE_EXPLICIT_WAYPOINT,
    )


    # =====================================================
    # 4. NEW ORIENTATION RISK -> L1 DIRECTION ONLY
    # =====================================================

    current_x = 0.70
    current_y = -0.75
    current_orientation = 12.0


    orientation_result = (
        evaluator.evaluate(

            x=
                current_x,

            y=
                current_y,

            orientation_relative_deg=
                current_orientation,

            tracking_missing_sec=
                0.0,
        )
    )


    controller = (
        ScaffoldingController()
    )


    run_case(

        "4. NEW ORIENTATION RISK -> MINIMAL ORIENTATION",

        controller,

        orientation_result,

        0.0,

        LEVEL_1,

        MODE_MINIMAL_ORIENTATION,
    )


    # =====================================================
    # 5. PERSISTENT ORIENTATION RISK -> L2 EXPLICIT
    # =====================================================

    controller = (
        ScaffoldingController()
    )


    controller.update(
        orientation_result,
        0.0,
    )


    run_case(

        "5. PERSISTENT ORIENTATION RISK -> EXPLICIT ORIENTATION",

        controller,

        orientation_result,

        0.90,

        LEVEL_2,

        MODE_EXPLICIT_ORIENTATION,
    )


    # =====================================================
    # 6. VIOLATION -> L3 STRONG IMMEDIATELY
    # =====================================================

    current_x = 1.65
    current_y = 0.0
    current_orientation = 0.0


    violation_result = (
        evaluator.evaluate(

            x=
                current_x,

            y=
                current_y,

            orientation_relative_deg=
                current_orientation,

            tracking_missing_sec=
                0.0,
        )
    )


    controller = (
        ScaffoldingController()
    )


    run_case(

        "6. VIOLATION -> STRONG ASSISTANCE IMMEDIATELY",

        controller,

        violation_result,

        0.0,

        LEVEL_3,

        MODE_STRONG_WAYPOINT,
    )


    # =====================================================
    # 7. CONTINUED VIOLATION -> REMAIN L3 STRONG
    # =====================================================

    controller = (
        ScaffoldingController()
    )


    controller.update(
        violation_result,
        0.0,
    )


    run_case(

        "7. CONTINUED VIOLATION -> REMAIN STRONG",

        controller,

        violation_result,

        2.10,

        LEVEL_3,

        MODE_STRONG_WAYPOINT,
    )


    # =====================================================
    # 8. RECOVERY FROM L2 -> RECOVERY FADE
    # =====================================================

    current_x = 1.65
    current_y = 0.90
    current_orientation = 0.0


    controller = (
        ScaffoldingController()
    )


    # Force L2 first.
    controller.update(
        risk_result,
        0.0,
    )

    controller.update(
        risk_result,
        0.90,
    )


    run_case(

        "8. RECOVERY -> NO STALE CORRECTION",

        controller,

        valid_result,

        1.00,

        LEVEL_2,

        MODE_RECOVERY_FADE,
    )


    # =====================================================
    # 9. TRACKING LOSS -> PAUSED
    # =====================================================

    current_x = 1.0
    current_y = -0.80
    current_orientation = 0.0


    tracking_result = (
        evaluator.evaluate(

            x=
                current_x,

            y=
                current_y,

            orientation_relative_deg=
                current_orientation,

            tracking_missing_sec=
                0.50,
        )
    )


    controller = (
        ScaffoldingController()
    )


    run_case(

        "9. TRACKING LOSS -> PAUSED",

        controller,

        tracking_result,

        0.0,

        LEVEL_0,

        MODE_PAUSED,
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
        f"{passed}/{total} tests passed"
    )


    if passed == total:

        print(
            "SCAFFOLDED GUIDANCE MAPPING: PASSED"
        )

    else:

        print(
            "SCAFFOLDED GUIDANCE MAPPING: FAILED"
        )


    print(
        "========================================================"
    )

    print()