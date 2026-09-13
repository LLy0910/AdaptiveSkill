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


from interaction.policy_baselines import (
    ReferenceOnlyPolicy,
    TaskAwareBinaryPolicy,
    DECISION_STAY_QUIET,
    DECISION_INTERVENE,
    DECISION_PAUSE,
)


def main():

    print()

    print(
        "========================================================"
    )

    print(
        "AdaptiveSkill - Policy Baseline Test"
    )

    print(
        "========================================================"
    )


    reference = (
        ReferenceOnlyPolicy(
            reference_threshold=
                0.30
        )
    )


    binary = (
        TaskAwareBinaryPolicy()
    )


    # -----------------------------------------------------
    # 1. Reference-like + task-valid
    # -----------------------------------------------------

    r1 = reference.decide(
        reference_distance=
            0.10
    )

    b1 = binary.decide(
        "VALID"
    )


    test_1 = (

        r1.decision
        ==
        DECISION_STAY_QUIET

        and

        b1.decision
        ==
        DECISION_STAY_QUIET
    )


    # -----------------------------------------------------
    # 2. Alternative-valid:
    # reference baseline intervenes, task-aware binary does
    # not.
    # -----------------------------------------------------

    r2 = reference.decide(
        reference_distance=
            0.75
    )

    b2 = binary.decide(
        "VALID"
    )


    test_2 = (

        r2.decision
        ==
        DECISION_INTERVENE

        and

        b2.decision
        ==
        DECISION_STAY_QUIET
    )


    # -----------------------------------------------------
    # 3. Task risk:
    # binary baseline always gives full correction.
    # -----------------------------------------------------

    b3 = binary.decide(
        "RISK"
    )


    test_3 = (

        b3.decision
        ==
        DECISION_INTERVENE

        and

        b3.nominal_level
        ==
        2

        and

        b3.presentation_mode
        ==
        "FULL_CORRECTION"
    )


    # -----------------------------------------------------
    # 4. Violation:
    # binary baseline still uses same full correction;
    # it has no progressive L2 -> L3 distinction.
    # -----------------------------------------------------

    b4 = binary.decide(
        "VIOLATION"
    )


    test_4 = (

        b4.decision
        ==
        DECISION_INTERVENE

        and

        b4.nominal_level
        ==
        2
    )


    # -----------------------------------------------------
    # 5. Recovery:
    # once task state is VALID, binary goes immediately
    # quiet. It has no recovery-fade lifecycle.
    # -----------------------------------------------------

    b5 = binary.decide(
        "VALID"
    )


    test_5 = (

        b5.decision
        ==
        DECISION_STAY_QUIET
    )


    # -----------------------------------------------------
    # 6. Tracking unreliable:
    # neither policy should interpret this as user error.
    # -----------------------------------------------------

    r6 = reference.decide(
        reference_distance=
            1.00,

        tracking_reliable=
            False,
    )

    b6 = binary.decide(
        "TRACKING_UNRELIABLE"
    )


    test_6 = (

        r6.decision
        ==
        DECISION_PAUSE

        and

        b6.decision
        ==
        DECISION_PAUSE
    )


    final_pass = all(
        [
            test_1,
            test_2,
            test_3,
            test_4,
            test_5,
            test_6,
        ]
    )


    print()

    print(
        "1. Reference-like valid -> both quiet:",
        test_1
    )

    print(
        "2. Alternative-valid -> reference intervenes, "
        "task-aware binary stays quiet:",
        test_2
    )

    print(
        "3. Risk -> binary full correction:",
        test_3
    )

    print(
        "4. Violation -> binary same full correction:",
        test_4
    )

    print(
        "5. Recovery VALID -> binary immediately quiet:",
        test_5
    )

    print(
        "6. Tracking unreliable -> both pause:",
        test_6
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
            "POLICY BASELINES: PASSED"
        )

    else:

        print(
            "POLICY BASELINES: FAILED"
        )


    print(
        "========================================================"
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "These are transparent engineering baselines. "
        "They are not claimed to represent the strongest "
        "possible methods in the literature."
    )

    print()


if __name__ == "__main__":

    main()
