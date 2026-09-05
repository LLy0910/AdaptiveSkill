from dataclasses import dataclass
from typing import Optional


# =========================================================
# FEEDBACK SETTINGS
# =========================================================

VOICE_COOLDOWN_SEC = 3.0


# =========================================================
# FEEDBACK RESULT
# =========================================================

@dataclass
class FeedbackResult:

    level: int

    label: str

    visual_title: str

    visual_message: str

    voice_message: Optional[str]

    should_speak: bool

    feedback_type: str

    reason: str


# =========================================================
# FEEDBACK CONTROLLER
# =========================================================

class FeedbackController:
    """
    Converts AssistancePolicy output into user-facing
    visual and spoken feedback.

    Design:

    Level 0:
        Observe only.
        No spoken feedback.

    Level 1:
        Gentle visual cue.
        No spoken feedback.

    Level 2:
        Explicit visual guidance.
        Spoken guidance may be triggered.

    Level 3:
        Human-assistance recommendation.
        Spoken guidance may be triggered.

    Speech is only requested when:
        1. the assistance level meaningfully changes, OR
        2. the feedback message changes,

    AND the voice cooldown has expired.

    This module decides WHAT should be communicated.
    It does not directly play audio.
    """


    def __init__(
        self,
        voice_cooldown_sec=VOICE_COOLDOWN_SEC
    ):

        self.voice_cooldown_sec = float(
            voice_cooldown_sec
        )

        self.reset()


    # =====================================================
    # RESET
    # =====================================================

    def reset(self):

        self.last_level = 0

        self.last_feedback_type = "NONE"

        self.last_visual_message = ""

        self.last_voice_message = None

        self.last_voice_time = None


    # =====================================================
    # MAIN UPDATE
    # =====================================================

    def update(
        self,
        timestamp,
        assistance_result
    ):

        timestamp = float(
            timestamp
        )


        # =================================================
        # READ ASSISTANCE RESULT
        # =================================================

        level = int(
            assistance_result.get(
                "level",
                0
            )
        )

        label = str(
            assistance_result.get(
                "label",
                "OBSERVE"
            )
        )

        reason = str(
            assistance_result.get(
                "reason",
                ""
            )
        )

        cue = str(
            assistance_result.get(
                "cue",
                ""
            )
        )

        dominant_error = str(
            assistance_result.get(
                "dominant_error",
                "NONE"
            )
        )

        rotation_deficit = float(
            assistance_result.get(
                "rotation_deficit_deg",
                0.0
            )
        )

        hesitation_events = int(
            assistance_result.get(
                "hesitation_events",
                0
            )
        )


        # =================================================
        # BUILD FEEDBACK CONTENT
        # =================================================

        feedback = self._build_feedback(

            level=
                level,

            label=
                label,

            reason=
                reason,

            cue=
                cue,

            dominant_error=
                dominant_error,

            rotation_deficit=
                rotation_deficit,

            hesitation_events=
                hesitation_events
        )


        # =================================================
        # SPEECH DECISION
        # =================================================

        should_speak = self._should_speak(

            timestamp=
                timestamp,

            level=
                level,

            voice_message=
                feedback["voice_message"],

            feedback_type=
                feedback["feedback_type"]
        )


        # =================================================
        # UPDATE HISTORY
        # =================================================

        if should_speak:

            self.last_voice_time = (
                timestamp
            )

            self.last_voice_message = (
                feedback[
                    "voice_message"
                ]
            )


        self.last_level = (
            level
        )

        self.last_feedback_type = (
            feedback[
                "feedback_type"
            ]
        )

        self.last_visual_message = (
            feedback[
                "visual_message"
            ]
        )


        # =================================================
        # RESULT
        # =================================================

        return FeedbackResult(

            level=
                level,

            label=
                label,

            visual_title=
                feedback[
                    "visual_title"
                ],

            visual_message=
                feedback[
                    "visual_message"
                ],

            voice_message=
                feedback[
                    "voice_message"
                ],

            should_speak=
                should_speak,

            feedback_type=
                feedback[
                    "feedback_type"
                ],

            reason=
                reason
        )


    # =====================================================
    # FEEDBACK CONTENT
    # =====================================================

    def _build_feedback(
        self,
        level,
        label,
        reason,
        cue,
        dominant_error,
        rotation_deficit,
        hesitation_events
    ):

        # =================================================
        # LEVEL 0
        # =================================================

        if level <= 0:

            return {

                "visual_title":
                    "OBSERVE",

                "visual_message":
                    "Continue the movement.",

                "voice_message":
                    None,

                "feedback_type":
                    "NONE"
            }


        # =================================================
        # LEVEL 1
        #
        # Visual only.
        # =================================================

        if level == 1:

            if dominant_error == "PATH":

                return {

                    "visual_title":
                        "GENTLE CUE",

                    "visual_message":
                        "Adjust your path slightly.",

                    "voice_message":
                        None,

                    "feedback_type":
                        "PATH"
                }


            if dominant_error == "ROTATION":

                if rotation_deficit > 0:

                    message = (
                        "Rotate your hand "
                        "a little more."
                    )

                elif rotation_deficit < 0:

                    message = (
                        "Reduce the rotation "
                        "slightly."
                    )

                else:

                    message = (
                        "Adjust your hand "
                        "rotation slightly."
                    )


                return {

                    "visual_title":
                        "GENTLE CUE",

                    "visual_message":
                        message,

                    "voice_message":
                        None,

                    "feedback_type":
                        "ROTATION"
                }


            return {

                "visual_title":
                    "GENTLE CUE",

                "visual_message":
                    "Make a small adjustment.",

                "voice_message":
                    None,

                "feedback_type":
                    "GENERAL"
            }


        # =================================================
        # LEVEL 2
        #
        # Explicit visual + possible speech.
        # =================================================

        if level == 2:

            # ---------------------------------------------
            # HESITATION
            # ---------------------------------------------

            if (
                "hesitation"
                in
                reason.lower()
            ):

                message = (
                    "Continue when you are ready."
                )

                return {

                    "visual_title":
                        "EXPLICIT GUIDANCE",

                    "visual_message":
                        message,

                    "voice_message":
                        message,

                    "feedback_type":
                        "HESITATION"
                }


            # ---------------------------------------------
            # PATH
            # ---------------------------------------------

            if dominant_error == "PATH":

                message = (
                    "Move back toward "
                    "the reference path."
                )

                return {

                    "visual_title":
                        "EXPLICIT GUIDANCE",

                    "visual_message":
                        message,

                    "voice_message":
                        message,

                    "feedback_type":
                        "PATH"
                }


            # ---------------------------------------------
            # ROTATION
            # ---------------------------------------------

            if dominant_error == "ROTATION":

                if rotation_deficit > 0:

                    message = (
                        "Rotate your hand more."
                    )

                elif rotation_deficit < 0:

                    message = (
                        "Reduce your hand rotation."
                    )

                else:

                    message = (
                        "Adjust your hand rotation."
                    )


                return {

                    "visual_title":
                        "EXPLICIT GUIDANCE",

                    "visual_message":
                        message,

                    "voice_message":
                        message,

                    "feedback_type":
                        "ROTATION"
                }


            # ---------------------------------------------
            # FALLBACK
            # ---------------------------------------------

            message = (
                cue
                if cue
                else
                "Adjust the movement."
            )

            return {

                "visual_title":
                    "EXPLICIT GUIDANCE",

                "visual_message":
                    message,

                "voice_message":
                    message,

                "feedback_type":
                    "GENERAL"
            }


        # =================================================
        # LEVEL 3
        #
        # Human assistance recommendation.
        # =================================================

        if level >= 3:

            if (
                "repeated behavioral hesitation"
                in
                reason.lower()
                or
                hesitation_events >= 2
            ):

                visual_message = (
                    "Repeated hesitation detected. "
                    "Human assistance is recommended."
                )

                voice_message = (
                    "You may need human assistance."
                )

                feedback_type = (
                    "HUMAN_HESITATION"
                )


            else:

                visual_message = (
                    "The difficulty is continuing. "
                    "Human assistance is recommended."
                )

                voice_message = (
                    "You may need human assistance."
                )

                feedback_type = (
                    "HUMAN_ASSISTANCE"
                )


            return {

                "visual_title":
                    "HUMAN ASSISTANCE",

                "visual_message":
                    visual_message,

                "voice_message":
                    voice_message,

                "feedback_type":
                    feedback_type
            }


        # =================================================
        # SAFETY FALLBACK
        # =================================================

        return {

            "visual_title":
                label,

            "visual_message":
                cue,

            "voice_message":
                None,

            "feedback_type":
                "GENERAL"
        }


    # =====================================================
    # SPEECH GATING
    # =====================================================

    def _should_speak(
        self,
        timestamp,
        level,
        voice_message,
        feedback_type
    ):

        # -------------------------------------------------
        # No voice content.
        # -------------------------------------------------

        if not voice_message:
            return False


        # -------------------------------------------------
        # Only Level 2 / Level 3 may speak.
        # -------------------------------------------------

        if level < 2:
            return False


        # -------------------------------------------------
        # First spoken message.
        # -------------------------------------------------

        if self.last_voice_time is None:
            return True


        elapsed = (
            timestamp
            -
            self.last_voice_time
        )


        # -------------------------------------------------
        # Voice cooldown.
        # -------------------------------------------------

        if (
            elapsed
            <
            self.voice_cooldown_sec
        ):
            return False


        # -------------------------------------------------
        # Speak when entering a higher assistance level.
        # -------------------------------------------------

        if (
            level
            >
            self.last_level
        ):
            return True


        # -------------------------------------------------
        # Speak when the required feedback changes.
        #
        # Example:
        #
        # L2 PATH
        # ->
        # L2 ROTATION
        # -------------------------------------------------

        if (
            feedback_type
            !=
            self.last_feedback_type
        ):
            return True


        # -------------------------------------------------
        # Speak when the actual sentence changes.
        # -------------------------------------------------

        if (
            voice_message
            !=
            self.last_voice_message
        ):
            return True


        # -------------------------------------------------
        # Same level + same message:
        # do not repeatedly speak every frame.
        # -------------------------------------------------

        return False


# =========================================================
# SIMPLE SELF TEST
# =========================================================

if __name__ == "__main__":

    controller = FeedbackController(
        voice_cooldown_sec=3.0
    )


    print()
    print(
        "=============================================="
    )

    print(
        "AdaptiveSkill - Feedback Controller"
    )

    print(
        "=============================================="
    )


    # =====================================================
    # TEST HELPER
    # =====================================================

    def run_test(
        name,
        timestamp,
        assistance_result
    ):

        result = controller.update(
            timestamp=
                timestamp,

            assistance_result=
                assistance_result
        )

        print()
        print(name)
        print("-" * 46)

        print(
            "LEVEL:",
            result.level,
            result.label
        )

        print(
            "TYPE:",
            result.feedback_type
        )

        print(
            "VISUAL TITLE:",
            result.visual_title
        )

        print(
            "VISUAL:",
            result.visual_message
        )

        print(
            "VOICE:",
            result.voice_message
        )

        print(
            "SHOULD SPEAK:",
            result.should_speak
        )


    # =====================================================
    # 1. LEVEL 0
    # =====================================================

    controller.reset()

    run_test(

        name=
            "LEVEL 0 - NORMAL",

        timestamp=
            0.0,

        assistance_result={

            "level":
                0,

            "label":
                "OBSERVE",

            "reason":
                "Within calibrated baseline",

            "cue":
                "No assistance",

            "dominant_error":
                "NONE",

            "rotation_deficit_deg":
                0.0,

            "hesitation_events":
                0
        }
    )


    # =====================================================
    # 2. LEVEL 1 PATH
    # =====================================================

    run_test(

        name=
            "LEVEL 1 - PATH",

        timestamp=
            1.0,

        assistance_result={

            "level":
                1,

            "label":
                "GENTLE CUE",

            "reason":
                "Mild path deviation",

            "cue":
                "Gentle path cue",

            "dominant_error":
                "PATH",

            "rotation_deficit_deg":
                0.5,

            "hesitation_events":
                0
        }
    )


    # =====================================================
    # 3. LEVEL 2 PATH
    # =====================================================

    run_test(

        name=
            "LEVEL 2 - PATH",

        timestamp=
            2.0,

        assistance_result={

            "level":
                2,

            "label":
                "EXPLICIT GUIDANCE",

            "reason":
                "Persistent path deviation",

            "cue":
                "Show reference path guidance",

            "dominant_error":
                "PATH",

            "rotation_deficit_deg":
                1.0,

            "hesitation_events":
                0
        }
    )


    # =====================================================
    # 4. SAME L2 PATH IMMEDIATELY
    #
    # Should not repeat speech.
    # =====================================================

    run_test(

        name=
            "LEVEL 2 - PATH REPEAT",

        timestamp=
            2.5,

        assistance_result={

            "level":
                2,

            "label":
                "EXPLICIT GUIDANCE",

            "reason":
                "Persistent path deviation",

            "cue":
                "Show reference path guidance",

            "dominant_error":
                "PATH",

            "rotation_deficit_deg":
                1.0,

            "hesitation_events":
                0
        }
    )


    # =====================================================
    # 5. LEVEL 2 ROTATION
    #
    # Timestamp is after cooldown.
    # Feedback changed from PATH to ROTATION.
    # Should speak.
    # =====================================================

    run_test(

        name=
            "LEVEL 2 - ROTATION UNDER",

        timestamp=
            5.5,

        assistance_result={

            "level":
                2,

            "label":
                "EXPLICIT GUIDANCE",

            "reason":
                "Persistent rotation under-completion",

            "cue":
                "Show stronger rotation guidance",

            "dominant_error":
                "ROTATION",

            "rotation_deficit_deg":
                12.0,

            "hesitation_events":
                0
        }
    )


    # =====================================================
    # 6. LEVEL 2 HESITATION
    # =====================================================

    run_test(

        name=
            "LEVEL 2 - HESITATION",

        timestamp=
            9.0,

        assistance_result={

            "level":
                2,

            "label":
                "EXPLICIT GUIDANCE",

            "reason":
                "Behavioral hesitation detected",

            "cue":
                "Show explicit next-step guidance",

            "dominant_error":
                "PATH",

            "rotation_deficit_deg":
                0.0,

            "hesitation_events":
                1
        }
    )


    # =====================================================
    # 7. LEVEL 3 REPEATED HESITATION
    # =====================================================

    run_test(

        name=
            "LEVEL 3 - REPEATED HESITATION",

        timestamp=
            12.5,

        assistance_result={

            "level":
                3,

            "label":
                "HUMAN ASSISTANCE",

            "reason":
                "Repeated behavioral hesitation detected",

            "cue":
                "Suggest human assistance",

            "dominant_error":
                "PATH",

            "rotation_deficit_deg":
                0.0,

            "hesitation_events":
                2
        }
    )


    # =====================================================
    # 8. SAME L3 IMMEDIATELY
    #
    # Should not repeat speech.
    # =====================================================

    run_test(

        name=
            "LEVEL 3 - REPEAT",

        timestamp=
            13.0,

        assistance_result={

            "level":
                3,

            "label":
                "HUMAN ASSISTANCE",

            "reason":
                "Repeated behavioral hesitation detected",

            "cue":
                "Suggest human assistance",

            "dominant_error":
                "PATH",

            "rotation_deficit_deg":
                0.0,

            "hesitation_events":
                2
        }
    )


    # =====================================================
    # 9. RECOVERY TO L1
    #
    # Visual only, no speech.
    # =====================================================

    run_test(

        name=
            "RECOVERY - LEVEL 1",

        timestamp=
            14.0,

        assistance_result={

            "level":
                1,

            "label":
                "GENTLE CUE",

            "reason":
                "Mild path deviation",

            "cue":
                "Gentle path cue",

            "dominant_error":
                "PATH",

            "rotation_deficit_deg":
                0.5,

            "hesitation_events":
                2
        }
    )


    print()

    print(
        "=============================================="
    )

    print(
        "Expected behavior:"
    )

    print(
        "L0 -> visual only, no speech"
    )

    print(
        "L1 -> gentle visual only"
    )

    print(
        "L2 -> explicit visual + gated speech"
    )

    print(
        "L3 -> human-help visual + gated speech"
    )

    print(
        "Repeated identical frames -> no repeated speech"
    )

    print(
        "Recovery -> no unnecessary speech"
    )

    print(
        "=============================================="
    )

    print()