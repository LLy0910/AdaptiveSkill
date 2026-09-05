import time

from feedback_controller import (
    FeedbackController
)

from voice_feedback import (
    VoiceFeedback
)


# =========================================================
# HELPERS
# =========================================================

def show_feedback(
    name,
    feedback
):

    print()
    print(
        "=" * 56
    )

    print(
        name
    )

    print(
        "=" * 56
    )

    print(
        "LEVEL:",
        feedback.level,
        feedback.label
    )

    print(
        "TYPE:",
        feedback.feedback_type
    )

    print(
        "VISUAL TITLE:",
        feedback.visual_title
    )

    print(
        "VISUAL MESSAGE:",
        feedback.visual_message
    )

    print(
        "VOICE MESSAGE:",
        feedback.voice_message
    )

    print(
        "SHOULD SPEAK:",
        feedback.should_speak
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print()
    print(
        "========================================================"
    )

    print(
        "AdaptiveSkill - Multimodal Feedback Integration Test"
    )

    print(
        "========================================================"
    )


    # =====================================================
    # CONTROLLER
    # =====================================================

    controller = FeedbackController(
        voice_cooldown_sec=3.0
    )


    # =====================================================
    # VOICE
    # =====================================================

    voice = VoiceFeedback(

        enabled=True,

        rate=0,

        volume=100
    )


    # Wait briefly for Windows voice worker.
    for _ in range(30):

        if (
            voice.ready
            or
            voice.error is not None
        ):

            break

        time.sleep(
            0.1
        )


    print()
    print(
        "Voice ready:",
        voice.ready
    )

    print(
        "Voice error:",
        voice.error
    )


    if not voice.ready:

        print(
            "Voice system unavailable."
        )

        voice.shutdown()

        raise SystemExit(
            1
        )


    # =====================================================
    # SIMULATED ASSISTANCE SEQUENCE
    # =====================================================

    tests = [

        # -------------------------------------------------
        # L0
        # -------------------------------------------------

        (
            "1. NORMAL - L0",
            0.0,
            {
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
        ),


        # -------------------------------------------------
        # L1 path
        # -------------------------------------------------

        (
            "2. MILD PATH - L1",
            1.0,
            {
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
                    1.0,

                "hesitation_events":
                    0
            }
        ),


        # -------------------------------------------------
        # L2 path
        # -------------------------------------------------

        (
            "3. STRONG PATH - L2",
            2.0,
            {
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
                    2.0,

                "hesitation_events":
                    0
            }
        ),


        # -------------------------------------------------
        # Same L2 immediately.
        #
        # Must NOT speak again.
        # -------------------------------------------------

        (
            "4. SAME PATH - L2 REPEAT",
            2.5,
            {
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
                    2.0,

                "hesitation_events":
                    0
            }
        ),


        # -------------------------------------------------
        # Recover to L1.
        #
        # Visual only.
        # -------------------------------------------------

        (
            "5. RECOVERY - L1",
            3.5,
            {
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
                    1.0,

                "hesitation_events":
                    0
            }
        ),


        # -------------------------------------------------
        # L2 rotation after cooldown.
        #
        # Should speak.
        # -------------------------------------------------

        (
            "6. ROTATION ERROR - L2",
            5.5,
            {
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
                    14.0,

                "hesitation_events":
                    0
            }
        ),


        # -------------------------------------------------
        # L2 hesitation after cooldown.
        #
        # Should speak.
        # -------------------------------------------------

        (
            "7. HESITATION - L2",
            9.0,
            {
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
        ),


        # -------------------------------------------------
        # L3 repeated hesitation.
        #
        # Should speak.
        # -------------------------------------------------

        (
            "8. REPEATED HESITATION - L3",
            12.5,
            {
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
        ),


        # -------------------------------------------------
        # Same L3 immediately.
        #
        # Must NOT repeat.
        # -------------------------------------------------

        (
            "9. SAME L3 REPEAT",
            13.0,
            {
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
        ),


        # -------------------------------------------------
        # Recovery.
        # -------------------------------------------------

        (
            "10. RECOVERY - L1",
            14.5,
            {
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
                    1.0,

                "hesitation_events":
                    2
            }
        )
    ]


    # =====================================================
    # RUN
    # =====================================================

    spoken_count = 0


    for (
        name,
        timestamp,
        assistance_result
    ) in tests:

        feedback = controller.update(

            timestamp=
                timestamp,

            assistance_result=
                assistance_result
        )


        show_feedback(
            name,
            feedback
        )


        # =============================================
        # THIS IS THE IMPORTANT INTEGRATION
        # =============================================

        if (
            feedback.should_speak
            and
            feedback.voice_message
        ):

            queued = voice.speak(
                feedback.voice_message
            )


            if queued:

                spoken_count += 1


                print(
                    "VOICE QUEUED: YES"
                )

            else:

                print(
                    "VOICE QUEUED: NO"
                )

        else:

            print(
                "VOICE QUEUED: NO"
            )


    # =====================================================
    # WAIT FOR BACKGROUND SPEECH
    # =====================================================

    print()
    print(
        "========================================================"
    )

    print(
        "All simulated frames processed."
    )

    print(
        "Main loop remained responsive."
    )

    print(
        "Spoken messages queued:",
        spoken_count
    )

    print(
        "Waiting for voice queue..."
    )


    voice.wait_until_idle()


    print()
    print(
        "Voice queue finished."
    )

    print(
        "Pending messages:",
        voice.pending_count
    )

    print(
        "Voice error:",
        voice.error
    )


    voice.shutdown()


    print()
    print(
        "========================================================"
    )

    print(
        "EXPECTED RESULT"
    )

    print(
        "========================================================"
    )

    print()

    print(
        "L0:"
    )

    print(
        "  visual only"
    )

    print(
        "  no voice"
    )

    print()

    print(
        "L1:"
    )

    print(
        "  gentle visual only"
    )

    print(
        "  no voice"
    )

    print()

    print(
        "L2 PATH:"
    )

    print(
        "  speak once"
    )

    print(
        "  immediate repeat stays silent"
    )

    print()

    print(
        "L2 ROTATION:"
    )

    print(
        "  speak rotation guidance"
    )

    print()

    print(
        "L2 HESITATION:"
    )

    print(
        "  speak hesitation guidance"
    )

    print()

    print(
        "L3:"
    )

    print(
        "  speak human-assistance guidance"
    )

    print(
        "  immediate repeat stays silent"
    )

    print()

    print(
        "RECOVERY:"
    )

    print(
        "  no unnecessary speech"
    )

    print()

    print(
        "Expected spoken messages queued: 4"
    )

    print(
        "========================================================"
    )

    print()