from dataclasses import dataclass


# =========================================================
# CONSTANTS
# =========================================================

DECISION_STAY_QUIET = "STAY_QUIET"
DECISION_INTERVENE = "INTERVENE"
DECISION_PAUSE = "PAUSE"

MODE_NONE = "NONE"
MODE_FULL_CORRECTION = "FULL_CORRECTION"

STATE_VALID = "VALID"
STATE_RISK = "RISK"
STATE_VIOLATION = "VIOLATION"
STATE_TRACKING_UNRELIABLE = "TRACKING_UNRELIABLE"


# =========================================================
# RESULT
# =========================================================

@dataclass
class BaselineDecision:
    policy_name: str
    decision: str
    nominal_level: int
    presentation_mode: str
    reason: str


# =========================================================
# A. REFERENCE-ONLY BASELINE
#
# This is the original/simple comparison:
# deviation from the expert prior can trigger intervention.
#
# It is intentionally transparent, not claimed to be a
# state-of-the-art LfD baseline.
# =========================================================

class ReferenceOnlyPolicy:

    def __init__(
        self,
        reference_threshold=0.30,
    ):

        self.reference_threshold = float(
            reference_threshold
        )


    def decide(
        self,
        reference_distance,
        tracking_reliable=True,
    ):

        if not tracking_reliable:

            return BaselineDecision(

                policy_name=
                    "REFERENCE_ONLY",

                decision=
                    DECISION_PAUSE,

                nominal_level=
                    0,

                presentation_mode=
                    MODE_NONE,

                reason=
                    "tracking_unreliable",
            )


        if (
            float(
                reference_distance
            )
            >
            self.reference_threshold
        ):

            return BaselineDecision(

                policy_name=
                    "REFERENCE_ONLY",

                decision=
                    DECISION_INTERVENE,

                nominal_level=
                    2,

                presentation_mode=
                    MODE_FULL_CORRECTION,

                reason=
                    "reference_deviation",
            )


        return BaselineDecision(

            policy_name=
                "REFERENCE_ONLY",

            decision=
                DECISION_STAY_QUIET,

            nominal_level=
                0,

            presentation_mode=
                MODE_NONE,

            reason=
                "reference_similarity_acceptable",
        )


# =========================================================
# B. TASK-AWARE BINARY BASELINE
#
# Stronger comparator:
# it knows task validity, but has no progressive
# scaffolding. Any task risk/violation receives the same
# full correction.
#
# This separates two questions:
#
# A vs B:
#   Is task validity a better intervention trigger than
#   reference deviation?
#
# B vs AdaptiveSkill:
#   Once task risk is known, is progressive information
#   allocation different from always giving full guidance?
# =========================================================

class TaskAwareBinaryPolicy:

    def decide(
        self,
        task_state,
    ):

        task_state = str(
            task_state
        )


        if (
            task_state
            ==
            STATE_TRACKING_UNRELIABLE
        ):

            return BaselineDecision(

                policy_name=
                    "TASK_AWARE_BINARY",

                decision=
                    DECISION_PAUSE,

                nominal_level=
                    0,

                presentation_mode=
                    MODE_NONE,

                reason=
                    "tracking_unreliable",
            )


        if (
            task_state
            in
            (
                STATE_RISK,
                STATE_VIOLATION,
            )
        ):

            return BaselineDecision(

                policy_name=
                    "TASK_AWARE_BINARY",

                decision=
                    DECISION_INTERVENE,

                nominal_level=
                    2,

                presentation_mode=
                    MODE_FULL_CORRECTION,

                reason=
                    (
                        "task_risk"

                        if task_state
                        ==
                        STATE_RISK

                        else
                        "task_violation"
                    ),
            )


        return BaselineDecision(

            policy_name=
                "TASK_AWARE_BINARY",

            decision=
                DECISION_STAY_QUIET,

            nominal_level=
                0,

            presentation_mode=
                MODE_NONE,

            reason=
                "task_valid",
        )
