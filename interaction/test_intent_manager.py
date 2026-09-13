import sys
from pathlib import Path


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from interaction.intent_manager import (
    IntentManager,
    STATE_IDLE,
    STATE_HOLDING,
    STATE_WAITING,
    STATE_COOLDOWN,
    CHOICE_KEEP,
    CHOICE_GUIDANCE,
)


CONFIG_PATH = (
    PROJECT_ROOT
    / "task"
    / "task_config.json"
)


def main():

    manager = IntentManager(
        config_path=CONFIG_PATH,
        reference_deviation_threshold=0.30,
    )

    print()
    print(
        "========================================================"
    )
    print(
        "AdaptiveSkill - Minimal Intent Manager Test"
    )
    print(
        "========================================================"
    )

    # -----------------------------------------------------
    # 1. Reference-like + valid -> no prompt
    # -----------------------------------------------------

    d1 = manager.update(
        task_state="VALID",
        reference_distance=0.10,
        tracking_reliable=True,
        now=0.0,
    )

    test_1 = (
        d1.state == STATE_IDLE
        and
        not d1.prompt_visible
    )

    # -----------------------------------------------------
    # 2. Large but transient deviation -> holding only
    # -----------------------------------------------------

    d2a = manager.update(
        task_state="VALID",
        reference_distance=0.70,
        tracking_reliable=True,
        now=1.0,
    )

    d2b = manager.update(
        task_state="VALID",
        reference_distance=0.70,
        tracking_reliable=True,
        now=1.4,
    )

    test_2 = (
        d2a.state == STATE_HOLDING
        and
        d2b.state == STATE_HOLDING
        and
        not d2b.prompt_visible
    )

    # Reset transient condition by returning near reference.
    manager.update(
        task_state="VALID",
        reference_distance=0.10,
        tracking_reliable=True,
        now=1.5,
    )

    # -----------------------------------------------------
    # 3. Persistent valid difference -> prompt
    # -----------------------------------------------------

    manager.update(
        task_state="VALID",
        reference_distance=0.75,
        tracking_reliable=True,
        now=2.0,
    )

    d3 = manager.update(
        task_state="VALID",
        reference_distance=0.75,
        tracking_reliable=True,
        now=2.81,
    )

    test_3 = (
        d3.state == STATE_WAITING
        and
        d3.prompt_visible
        and
        d3.reason == "task_valid_but_different"
    )

    # -----------------------------------------------------
    # 4. KEEP -> user-confirmed alternative + cooldown
    # -----------------------------------------------------

    d4 = manager.choose_keep(
        now=2.82,
    )

    test_4 = (
        d4.state == STATE_COOLDOWN
        and
        d4.choice == CHOICE_KEEP
        and
        d4.user_confirmed_alternative
        and
        not d4.guidance_requested
    )

    # -----------------------------------------------------
    # 5. Same valid difference during cooldown -> no prompt
    # -----------------------------------------------------

    d5 = manager.update(
        task_state="VALID",
        reference_distance=0.80,
        tracking_reliable=True,
        now=3.5,
    )

    test_5 = (
        d5.state == STATE_COOLDOWN
        and
        not d5.prompt_visible
    )

    # -----------------------------------------------------
    # 6. Risk state -> never intent prompt
    # -----------------------------------------------------

    manager.reset()

    manager.update(
        task_state="VALID",
        reference_distance=0.80,
        tracking_reliable=True,
        now=10.0,
    )

    d6 = manager.update(
        task_state="RISK",
        reference_distance=0.80,
        tracking_reliable=True,
        now=10.9,
    )

    test_6 = (
        d6.state == STATE_IDLE
        and
        not d6.prompt_visible
        and
        d6.reason == "task_not_valid"
    )

    # -----------------------------------------------------
    # 7. Tracking unreliable -> never intent prompt
    # -----------------------------------------------------

    manager.reset()

    manager.update(
        task_state="VALID",
        reference_distance=0.80,
        tracking_reliable=True,
        now=20.0,
    )

    d7 = manager.update(
        task_state="VALID",
        reference_distance=0.80,
        tracking_reliable=False,
        now=20.9,
    )

    test_7 = (
        d7.state == STATE_IDLE
        and
        not d7.prompt_visible
        and
        d7.reason == "tracking_unreliable"
    )

    # -----------------------------------------------------
    # 8. GUIDANCE -> explicit user request event
    # -----------------------------------------------------

    manager.reset()

    manager.update(
        task_state="VALID",
        reference_distance=0.80,
        tracking_reliable=True,
        now=30.0,
    )

    manager.update(
        task_state="VALID",
        reference_distance=0.80,
        tracking_reliable=True,
        now=30.81,
    )

    d8 = manager.choose_guidance(
        now=30.82,
    )

    test_8 = (
        d8.state == STATE_COOLDOWN
        and
        d8.choice == CHOICE_GUIDANCE
        and
        d8.guidance_requested
        and
        not d8.user_confirmed_alternative
    )

    # -----------------------------------------------------
    # FINAL
    # -----------------------------------------------------

    final_pass = all(
        [
            test_1,
            test_2,
            test_3,
            test_4,
            test_5,
            test_6,
            test_7,
            test_8,
        ]
    )

    print()
    print(
        "1. Reference-like valid -> no prompt:",
        test_1
    )
    print(
        "2. Transient valid deviation -> hold only:",
        test_2
    )
    print(
        "3. Persistent valid difference -> ask user:",
        test_3
    )
    print(
        "4. KEEP -> user-confirmed alternative:",
        test_4
    )
    print(
        "5. KEEP cooldown -> no repeated nagging:",
        test_5
    )
    print(
        "6. Risk -> task assistance, not intent prompt:",
        test_6
    )
    print(
        "7. Tracking unreliable -> no intent inference:",
        test_7
    )
    print(
        "8. GUIDANCE -> explicit user request:",
        test_8
    )

    print()
    print(
        "========================================================"
    )
    print(
        "PASS:",
        final_pass
    )

    if final_pass:
        print(
            "MINIMAL INTENT MANAGER: PASSED"
        )
    else:
        print(
            "MINIMAL INTENT MANAGER: FAILED"
        )

    print(
        "========================================================"
    )
    print()
    print(
        "NOTE:"
    )
    print(
        "This module does not infer user intention. "
        "It only asks for user confirmation when a "
        "persistently different trajectory remains "
        "task-valid and tracking is reliable."
    )
    print()


if __name__ == "__main__":
    main()
