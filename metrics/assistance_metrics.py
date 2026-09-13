import sys
import time
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
)

from interaction.intervention_comparator import (
    InterventionComparator,
    DECISION_STAY_QUIET,
    DECISION_INTERVENE,
    DECISION_PAUSE,
)


# =========================================================
# TASK OUTCOME LABELS
# =========================================================

OUTCOME_IN_PROGRESS = "IN_PROGRESS"

OUTCOME_TASK_SUCCESS = "TASK_SUCCESS"

OUTCOME_GOAL_WITH_VIOLATION = (
    "GOAL_REACHED_WITH_VIOLATION"
)

OUTCOME_VIOLATION_OCCURRED = (
    "VIOLATION_OCCURRED"
)


# =========================================================
# SUMMARY
# =========================================================

@dataclass
class TrialInterventionSummary:
    """
    Episode-level summary for one demonstration trial.

    IMPORTANT:

    Counts are contiguous EPISODES,
    not video / UI frames.

    The summary also keeps sticky task-history state:

        constraint violation occurred

    Once a violation has happened,
    returning to a valid state does NOT erase it.
    """

    trial_duration_sec: float

    # -----------------------------------------------------
    # Intervention episodes
    # -----------------------------------------------------

    reference_intervention_episodes: int
    task_intervention_episodes: int

    unnecessary_intervention_episodes_avoided: int
    task_risk_missed_by_reference_episodes: int
    task_relevant_intervention_episodes: int

    # -----------------------------------------------------
    # Alternative strategies
    # -----------------------------------------------------

    alternative_valid_episodes: int
    alternative_valid_route_preserved: bool

    # -----------------------------------------------------
    # Task constraint history
    # -----------------------------------------------------

    constraint_risk_episodes: int
    constraint_violation_episodes: int

    constraint_risk_occurred: bool
    constraint_violation_occurred: bool

    # -----------------------------------------------------
    # Time
    # -----------------------------------------------------

    time_reference_intervention_sec: float
    time_task_intervention_sec: float

    time_unnecessary_intervention_sec: float
    time_task_risk_missed_by_reference_sec: float

    time_in_risk_sec: float
    time_in_violation_sec: float

    # -----------------------------------------------------
    # Trial observations
    # -----------------------------------------------------

    max_reference_distance: float

    goal_reached_seen: bool

    returned_to_quiet: bool

    current_reference_decision: str
    current_task_decision: str

    task_outcome: str


    def to_dict(self):
        return asdict(self)


# =========================================================
# ASSISTANCE METRICS
# =========================================================

class AssistanceMetrics:
    """
    Episode-level trial metrics.

    Main research purpose:

    Distinguish:

        reference deviation

    from:

        task-relevant intervention need

    while also preserving whether a real task
    constraint violation happened anywhere
    during the demonstration.

    This prototype does NOT claim:

        - improved learning
        - lower workload
        - improved user experience
        - validated assistance superiority

    Those would require a user study.
    """


    def __init__(self):
        self.reset()


    # =====================================================
    # RESET
    # =====================================================

    def reset(self):

        self.started = False

        self.start_timestamp = None
        self.last_timestamp = None


        # =================================================
        # INTERVENTION EPISODES
        # =================================================

        self.reference_intervention_episodes = 0

        self.task_intervention_episodes = 0

        self.unnecessary_intervention_episodes_avoided = 0

        self.task_risk_missed_by_reference_episodes = 0

        self.task_relevant_intervention_episodes = 0


        # =================================================
        # ALTERNATIVE VALID STRATEGY
        # =================================================

        self.alternative_valid_episodes = 0


        # =================================================
        # TASK CONSTRAINT HISTORY
        # =================================================

        self.constraint_risk_episodes = 0

        self.constraint_violation_episodes = 0


        self.constraint_risk_occurred = False

        self.constraint_violation_occurred = False


        # =================================================
        # TIME
        # =================================================

        self.time_reference_intervention_sec = 0.0

        self.time_task_intervention_sec = 0.0

        self.time_unnecessary_intervention_sec = 0.0

        self.time_task_risk_missed_by_reference_sec = 0.0

        self.time_in_risk_sec = 0.0

        self.time_in_violation_sec = 0.0


        # =================================================
        # OBSERVATIONS
        # =================================================

        self.max_reference_distance = 0.0

        self.goal_reached_seen = False

        self.ever_task_intervened = False


        # =================================================
        # PREVIOUS EPISODE FLAGS
        # =================================================

        self.previous_reference_intervention = False

        self.previous_task_intervention = False

        self.previous_unnecessary_intervention = False

        self.previous_task_risk_missed = False

        self.previous_task_relevant_intervention = False

        self.previous_alternative_valid = False

        self.previous_constraint_risk = False

        self.previous_constraint_violation = False


        # =================================================
        # CURRENT ACTIVE FLAGS
        # =================================================

        self.current_reference_intervention = False

        self.current_task_intervention = False

        self.current_unnecessary_intervention = False

        self.current_task_risk_missed = False

        self.current_task_relevant_intervention = False

        self.current_alternative_valid = False

        self.current_constraint_risk = False

        self.current_constraint_violation = False


        # =================================================
        # CURRENT DECISIONS
        # =================================================

        self.current_reference_decision = (
            DECISION_PAUSE
        )

        self.current_task_decision = (
            DECISION_PAUSE
        )


    # =====================================================
    # TIME HELPER
    # =====================================================

    def _get_timestamp(
        self,
        timestamp_sec=None
    ):

        if timestamp_sec is None:

            return time.perf_counter()

        return float(
            timestamp_sec
        )


    # =====================================================
    # ACCUMULATE DURATION
    # =====================================================

    def _accumulate_previous_duration(
        self,
        timestamp_sec
    ):

        if self.last_timestamp is None:

            return


        dt = (
            float(timestamp_sec)
            -
            float(self.last_timestamp)
        )


        if dt <= 0.0:

            return


        # Defensive guard against accidental huge jumps.
        dt = min(
            dt,
            5.0
        )


        if self.current_reference_intervention:

            self.time_reference_intervention_sec += dt


        if self.current_task_intervention:

            self.time_task_intervention_sec += dt


        if self.current_unnecessary_intervention:

            self.time_unnecessary_intervention_sec += dt


        if self.current_task_risk_missed:

            self.time_task_risk_missed_by_reference_sec += dt


        if self.current_constraint_risk:

            self.time_in_risk_sec += dt


        if self.current_constraint_violation:

            self.time_in_violation_sec += dt


    # =====================================================
    # UPDATE
    # =====================================================

    def update(
        self,
        comparison,
        task_result,
        timestamp_sec=None
    ):

        timestamp_sec = self._get_timestamp(
            timestamp_sec
        )


        # =================================================
        # START CLOCK
        # =================================================

        if not self.started:

            self.started = True

            self.start_timestamp = (
                timestamp_sec
            )

            self.last_timestamp = (
                timestamp_sec
            )


        else:

            self._accumulate_previous_duration(
                timestamp_sec
            )


        # =================================================
        # CURRENT DECISIONS
        # =================================================

        reference_decision = (
            comparison.reference_policy_decision
        )

        task_decision = (
            comparison.task_policy_decision
        )


        self.current_reference_decision = (
            reference_decision
        )

        self.current_task_decision = (
            task_decision
        )


        # =================================================
        # INTERVENTION STATES
        # =================================================

        reference_intervention = (

            reference_decision
            ==
            DECISION_INTERVENE
        )


        task_intervention = (

            task_decision
            ==
            DECISION_INTERVENE
        )


        unnecessary_intervention = (

            reference_decision
            ==
            DECISION_INTERVENE

            and

            task_decision
            ==
            DECISION_STAY_QUIET
        )


        task_risk_missed = (

            reference_decision
            ==
            DECISION_STAY_QUIET

            and

            task_decision
            ==
            DECISION_INTERVENE

            and

            task_result.state
            in (
                STATE_RISK,
                STATE_VIOLATION,
            )
        )


        task_relevant_intervention = (

            task_decision
            ==
            DECISION_INTERVENE

            and

            task_result.state
            in (
                STATE_RISK,
                STATE_VIOLATION,
            )
        )


        # =================================================
        # ALTERNATIVE VALID
        # =================================================

        alternative_valid = (

            unnecessary_intervention

            and

            task_result.state
            ==
            STATE_VALID

            and

            task_result.task_valid_so_far
        )


        # =================================================
        # TASK CONSTRAINT STATES
        # =================================================

        constraint_risk = (

            task_result.state
            ==
            STATE_RISK
        )


        constraint_violation = (

            task_result.state
            ==
            STATE_VIOLATION
        )


        # =================================================
        # EPISODE EDGE DETECTION
        # =================================================

        if (
            reference_intervention

            and

            not self.previous_reference_intervention
        ):

            self.reference_intervention_episodes += 1


        if (
            task_intervention

            and

            not self.previous_task_intervention
        ):

            self.task_intervention_episodes += 1


        if (
            unnecessary_intervention

            and

            not self.previous_unnecessary_intervention
        ):

            self.unnecessary_intervention_episodes_avoided += 1


        if (
            task_risk_missed

            and

            not self.previous_task_risk_missed
        ):

            self.task_risk_missed_by_reference_episodes += 1


        if (
            task_relevant_intervention

            and

            not self.previous_task_relevant_intervention
        ):

            self.task_relevant_intervention_episodes += 1


        if (
            alternative_valid

            and

            not self.previous_alternative_valid
        ):

            self.alternative_valid_episodes += 1


        if (
            constraint_risk

            and

            not self.previous_constraint_risk
        ):

            self.constraint_risk_episodes += 1


        if (
            constraint_violation

            and

            not self.previous_constraint_violation
        ):

            self.constraint_violation_episodes += 1


        # =================================================
        # STICKY HISTORY
        # =================================================

        if constraint_risk:

            self.constraint_risk_occurred = True


        if constraint_violation:

            self.constraint_violation_occurred = True


        if task_result.goal_reached:

            self.goal_reached_seen = True


        if task_intervention:

            self.ever_task_intervened = True


        # =================================================
        # MAX REFERENCE DISTANCE
        # =================================================

        try:

            reference_distance = float(
                comparison.reference_distance
            )


            if reference_distance >= 0.0:

                self.max_reference_distance = max(

                    self.max_reference_distance,

                    reference_distance
                )

        except Exception:

            pass


        # =================================================
        # SAVE CURRENT STATES
        # =================================================

        self.current_reference_intervention = (
            reference_intervention
        )

        self.current_task_intervention = (
            task_intervention
        )

        self.current_unnecessary_intervention = (
            unnecessary_intervention
        )

        self.current_task_risk_missed = (
            task_risk_missed
        )

        self.current_task_relevant_intervention = (
            task_relevant_intervention
        )

        self.current_alternative_valid = (
            alternative_valid
        )

        self.current_constraint_risk = (
            constraint_risk
        )

        self.current_constraint_violation = (
            constraint_violation
        )


        # =================================================
        # PREVIOUS FLAGS
        # =================================================

        self.previous_reference_intervention = (
            reference_intervention
        )

        self.previous_task_intervention = (
            task_intervention
        )

        self.previous_unnecessary_intervention = (
            unnecessary_intervention
        )

        self.previous_task_risk_missed = (
            task_risk_missed
        )

        self.previous_task_relevant_intervention = (
            task_relevant_intervention
        )

        self.previous_alternative_valid = (
            alternative_valid
        )

        self.previous_constraint_risk = (
            constraint_risk
        )

        self.previous_constraint_violation = (
            constraint_violation
        )


        self.last_timestamp = (
            timestamp_sec
        )


    # =====================================================
    # ADVANCE CLOCK
    # =====================================================

    def advance_time(
        self,
        timestamp_sec=None
    ):

        if not self.started:

            return


        timestamp_sec = self._get_timestamp(
            timestamp_sec
        )


        self._accumulate_previous_duration(
            timestamp_sec
        )


        self.last_timestamp = (
            timestamp_sec
        )


    # =====================================================
    # TASK OUTCOME
    # =====================================================

    def _get_task_outcome(self):

        # -------------------------------------------------
        # Goal reached, no violation:
        # genuine task success.
        # -------------------------------------------------

        if (
            self.goal_reached_seen

            and

            not self.constraint_violation_occurred
        ):

            return (
                OUTCOME_TASK_SUCCESS
            )


        # -------------------------------------------------
        # Goal reached, but violation happened earlier.
        # -------------------------------------------------

        if (
            self.goal_reached_seen

            and

            self.constraint_violation_occurred
        ):

            return (
                OUTCOME_GOAL_WITH_VIOLATION
            )


        # -------------------------------------------------
        # Violation happened but goal not reached.
        # -------------------------------------------------

        if self.constraint_violation_occurred:

            return (
                OUTCOME_VIOLATION_OCCURRED
            )


        return (
            OUTCOME_IN_PROGRESS
        )


    # =====================================================
    # SUMMARY
    # =====================================================

    def get_summary(
        self,
        timestamp_sec=None
    ):

        if timestamp_sec is not None:

            self.advance_time(
                timestamp_sec
            )


        if (
            not self.started

            or

            self.start_timestamp is None

            or

            self.last_timestamp is None
        ):

            duration = 0.0


        else:

            duration = max(
                0.0,
                self.last_timestamp
                -
                self.start_timestamp
            )


        returned_to_quiet = (

            self.ever_task_intervened

            and

            self.current_task_decision
            ==
            DECISION_STAY_QUIET
        )


        task_outcome = (
            self._get_task_outcome()
        )


        return TrialInterventionSummary(

            trial_duration_sec=
                duration,

            reference_intervention_episodes=
                self.reference_intervention_episodes,

            task_intervention_episodes=
                self.task_intervention_episodes,

            unnecessary_intervention_episodes_avoided=
                self.unnecessary_intervention_episodes_avoided,

            task_risk_missed_by_reference_episodes=
                self.task_risk_missed_by_reference_episodes,

            task_relevant_intervention_episodes=
                self.task_relevant_intervention_episodes,

            alternative_valid_episodes=
                self.alternative_valid_episodes,

            alternative_valid_route_preserved=
                (
                    self.alternative_valid_episodes
                    >
                    0
                ),

            constraint_risk_episodes=
                self.constraint_risk_episodes,

            constraint_violation_episodes=
                self.constraint_violation_episodes,

            constraint_risk_occurred=
                self.constraint_risk_occurred,

            constraint_violation_occurred=
                self.constraint_violation_occurred,

            time_reference_intervention_sec=
                self.time_reference_intervention_sec,

            time_task_intervention_sec=
                self.time_task_intervention_sec,

            time_unnecessary_intervention_sec=
                self.time_unnecessary_intervention_sec,

            time_task_risk_missed_by_reference_sec=
                self.time_task_risk_missed_by_reference_sec,

            time_in_risk_sec=
                self.time_in_risk_sec,

            time_in_violation_sec=
                self.time_in_violation_sec,

            max_reference_distance=
                self.max_reference_distance,

            goal_reached_seen=
                self.goal_reached_seen,

            returned_to_quiet=
                returned_to_quiet,

            current_reference_decision=
                self.current_reference_decision,

            current_task_decision=
                self.current_task_decision,

            task_outcome=
                task_outcome,
        )


# =========================================================
# SELF TEST
# =========================================================

if __name__ == "__main__":

    evaluator = (
        TaskConstraintEvaluator()
    )


    expert_reference = [

        (
            0.0,
            0.0,
        ),

        (
            0.70,
            -0.75,
        ),

        (
            1.10,
            -0.90,
        ),

        (
            2.20,
            -0.90,
        ),

        (
            2.60,
            -0.60,
        ),

        (
            3.20,
            0.0,
        ),
    ]


    comparator = (
        InterventionComparator(

            expert_reference_points=
                expert_reference,

            reference_threshold=
                0.30,
        )
    )


    # =====================================================
    # TEST 1
    # SAFE ALTERNATIVE ROUTE
    # =====================================================

    safe_metrics = (
        AssistanceMetrics()
    )


    safe_sequence = [

        (
            0.0,
            0.0,
            0.0,
        ),

        (
            0.1,
            1.65,
            0.90,
        ),

        (
            0.2,
            2.40,
            0.90,
        ),

        (
            0.3,
            3.20,
            0.0,
        ),
    ]


    for (
        timestamp,
        x,
        y,
    ) in safe_sequence:

        task_result = (
            evaluator.evaluate(

                x=
                    x,

                y=
                    y,

                orientation_relative_deg=
                    0.0,

                tracking_missing_sec=
                    0.0,
            )
        )


        comparison = (
            comparator.compare(

                x=
                    x,

                y=
                    y,

                task_result=
                    task_result,
            )
        )


        safe_metrics.update(

            comparison=
                comparison,

            task_result=
                task_result,

            timestamp_sec=
                timestamp,
        )


    safe_summary = (
        safe_metrics.get_summary(
            timestamp_sec=
                0.4
        )
    )


    # =====================================================
    # TEST 2
    # VIOLATION THEN RECOVERY + GOAL
    # =====================================================

    violation_metrics = (
        AssistanceMetrics()
    )


    violation_sequence = [

        # Valid start.
        (
            0.0,
            0.0,
            0.0,
        ),

        # Actual obstacle violation.
        (
            0.1,
            1.65,
            0.0,
        ),

        # Recover to safe lower route.
        (
            0.2,
            2.40,
            0.90,
        ),

        # Reach target.
        (
            0.3,
            3.20,
            0.0,
        ),
    ]


    for (
        timestamp,
        x,
        y,
    ) in violation_sequence:

        task_result = (
            evaluator.evaluate(

                x=
                    x,

                y=
                    y,

                orientation_relative_deg=
                    0.0,

                tracking_missing_sec=
                    0.0,
            )
        )


        comparison = (
            comparator.compare(

                x=
                    x,

                y=
                    y,

                task_result=
                    task_result,
            )
        )


        violation_metrics.update(

            comparison=
                comparison,

            task_result=
                task_result,

            timestamp_sec=
                timestamp,
        )


    violation_summary = (
        violation_metrics.get_summary(
            timestamp_sec=
                0.4
        )
    )


    # =====================================================
    # PRINT
    # =====================================================

    print()

    print(
        "========================================================"
    )

    print(
        "AdaptiveSkill - Persistent Trial Outcome Metrics"
    )

    print(
        "========================================================"
    )


    print()

    print(
        "SAFE ALTERNATIVE ROUTE"
    )

    print(
        "-" * 56
    )

    print(
        "Goal reached:",
        safe_summary.goal_reached_seen
    )

    print(
        "Violation occurred:",
        safe_summary.constraint_violation_occurred
    )

    print(
        "Violation episodes:",
        safe_summary.constraint_violation_episodes
    )

    print(
        "Alternative-valid preserved:",
        safe_summary.alternative_valid_route_preserved
    )

    print(
        "Outcome:",
        safe_summary.task_outcome
    )


    safe_pass = (

        safe_summary.goal_reached_seen

        and

        not safe_summary.constraint_violation_occurred

        and

        safe_summary.task_outcome
        ==
        OUTCOME_TASK_SUCCESS
    )


    print(
        "PASS:",
        safe_pass
    )


    print()

    print(
        "VIOLATION THEN RECOVERY + GOAL"
    )

    print(
        "-" * 56
    )

    print(
        "Goal reached:",
        violation_summary.goal_reached_seen
    )

    print(
        "Violation occurred:",
        violation_summary.constraint_violation_occurred
    )

    print(
        "Violation episodes:",
        violation_summary.constraint_violation_episodes
    )

    print(
        "Current task decision:",
        violation_summary.current_task_decision
    )

    print(
        "Returned to quiet:",
        violation_summary.returned_to_quiet
    )

    print(
        "Outcome:",
        violation_summary.task_outcome
    )


    violation_pass = (

        violation_summary.goal_reached_seen

        and

        violation_summary.constraint_violation_occurred

        and

        violation_summary.constraint_violation_episodes
        ==
        1

        and

        violation_summary.task_outcome
        ==
        OUTCOME_GOAL_WITH_VIOLATION
    )


    print(
        "PASS:",
        violation_pass
    )


    final_pass = (

        safe_pass

        and

        violation_pass
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
            "PERSISTENT TRIAL OUTCOME METRICS: PASSED"
        )

    else:

        print(
            "PERSISTENT TRIAL OUTCOME METRICS: FAILED"
        )


    print(
        "========================================================"
    )

    print()