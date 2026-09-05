import math
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
# SUMMARY
# =========================================================

@dataclass
class TrialInterventionSummary:
    """
    Episode-level summary for one demonstration trial.

    IMPORTANT:
    Counts refer to contiguous intervention EPISODES,
    not frames.

    Example:

        INTERVENE
        INTERVENE
        INTERVENE
        STAY_QUIET

    counts as ONE intervention episode.
    """

    trial_duration_sec: float

    reference_intervention_episodes: int
    task_intervention_episodes: int

    unnecessary_intervention_episodes_avoided: int
    task_risk_missed_by_reference_episodes: int
    task_relevant_intervention_episodes: int

    alternative_valid_episodes: int
    alternative_valid_route_preserved: bool

    time_reference_intervention_sec: float
    time_task_intervention_sec: float

    time_unnecessary_intervention_sec: float
    time_task_risk_missed_by_reference_sec: float

    max_reference_distance: float

    goal_reached_seen: bool

    returned_to_quiet: bool

    current_reference_decision: str
    current_task_decision: str


    def to_dict(self):
        return asdict(self)


# =========================================================
# ASSISTANCE METRICS
# =========================================================

class AssistanceMetrics:
    """
    Tracks intervention behaviour across an entire trial.

    The main purpose is to avoid a misleading
    frame-by-frame interpretation.

    It records:

        - intervention EPISODES
        - disagreement episodes
        - intervention time
        - alternative-valid episodes
        - whether the task-aware policy returned to quiet

    It does NOT claim:

        - improved learning
        - reduced workload
        - better user experience
        - validated intervention superiority

    Those require a future user study.
    """


    def __init__(self):
        self.reset()


    # =====================================================
    # RESET
    # =====================================================

    def reset(self):
        """
        Clear all trial-level state.
        """

        self.started = False

        self.start_timestamp = None
        self.last_timestamp = None


        # -------------------------------------------------
        # Episode counts
        # -------------------------------------------------

        self.reference_intervention_episodes = 0
        self.task_intervention_episodes = 0

        self.unnecessary_intervention_episodes_avoided = 0

        self.task_risk_missed_by_reference_episodes = 0

        self.task_relevant_intervention_episodes = 0

        self.alternative_valid_episodes = 0


        # -------------------------------------------------
        # Durations
        # -------------------------------------------------

        self.time_reference_intervention_sec = 0.0

        self.time_task_intervention_sec = 0.0

        self.time_unnecessary_intervention_sec = 0.0

        self.time_task_risk_missed_by_reference_sec = 0.0


        # -------------------------------------------------
        # Trial observations
        # -------------------------------------------------

        self.max_reference_distance = 0.0

        self.goal_reached_seen = False

        self.ever_task_intervened = False


        # -------------------------------------------------
        # Previous active states
        #
        # Used for edge detection:
        #
        # False -> True
        # means a NEW episode.
        # -------------------------------------------------

        self.previous_reference_intervention = False

        self.previous_task_intervention = False

        self.previous_unnecessary_intervention = False

        self.previous_task_risk_missed = False

        self.previous_task_relevant_intervention = False

        self.previous_alternative_valid = False


        # -------------------------------------------------
        # Current active states
        # -------------------------------------------------

        self.current_reference_intervention = False

        self.current_task_intervention = False

        self.current_unnecessary_intervention = False

        self.current_task_risk_missed = False

        self.current_task_relevant_intervention = False

        self.current_alternative_valid = False


        # -------------------------------------------------
        # Current decisions
        # -------------------------------------------------

        self.current_reference_decision = (
            DECISION_PAUSE
        )

        self.current_task_decision = (
            DECISION_PAUSE
        )


    # =====================================================
    # TIME
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


    def _accumulate_previous_duration(
        self,
        timestamp_sec
    ):
        """
        Attribute elapsed time to the state that was
        active during the PREVIOUS interval.
        """

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


    # =====================================================
    # UPDATE
    # =====================================================

    def update(
        self,
        comparison,
        task_result,
        timestamp_sec=None
    ):
        """
        Add one current frame / system decision.

        Episode counts increase only when a condition
        changes from inactive -> active.
        """

        timestamp_sec = self._get_timestamp(
            timestamp_sec
        )


        # =================================================
        # START TRIAL CLOCK
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
        # ACTIVE CONDITIONS
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


        # -------------------------------------------------
        # Reference wants correction,
        # task-aware system says stay quiet.
        # -------------------------------------------------

        unnecessary_intervention = (

            reference_decision
            ==
            DECISION_INTERVENE

            and

            task_decision
            ==
            DECISION_STAY_QUIET
        )


        # -------------------------------------------------
        # Reference stays quiet,
        # but task-aware policy sees real task risk.
        # -------------------------------------------------

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


        # -------------------------------------------------
        # Any task-relevant intervention.
        # -------------------------------------------------

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


        # -------------------------------------------------
        # Alternative-valid episode.
        #
        # Far enough from expert prior that the internal
        # reference baseline wants intervention,
        #
        # but the task remains VALID.
        # -------------------------------------------------

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
        # EDGE DETECTION -> EPISODE COUNTS
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


        # =================================================
        # OTHER TRIAL OBSERVATIONS
        # =================================================

        if math.isfinite(
            comparison.reference_distance
        ):

            self.max_reference_distance = max(
                self.max_reference_distance,
                float(
                    comparison.reference_distance
                )
            )


        if task_result.goal_reached:

            self.goal_reached_seen = True


        if task_intervention:

            self.ever_task_intervened = True


        # =================================================
        # SAVE CURRENT ACTIVE STATES
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


        # =================================================
        # PREVIOUS FLAGS FOR NEXT EDGE
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
        """
        Update durations even when no new state transition
        has occurred.

        Useful immediately before displaying or saving
        the trial summary.
        """

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


        # -------------------------------------------------
        # Returned to quiet:
        #
        # task-aware policy intervened at least once,
        # and is currently quiet again.
        # -------------------------------------------------

        returned_to_quiet = (

            self.ever_task_intervened

            and

            self.current_task_decision
            ==
            DECISION_STAY_QUIET
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

            time_reference_intervention_sec=
                self.time_reference_intervention_sec,

            time_task_intervention_sec=
                self.time_task_intervention_sec,

            time_unnecessary_intervention_sec=
                self.time_unnecessary_intervention_sec,

            time_task_risk_missed_by_reference_sec=
                self.time_task_risk_missed_by_reference_sec,

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
        )


# =========================================================
# SELF TEST
# =========================================================

if __name__ == "__main__":

    evaluator = (
        TaskConstraintEvaluator()
    )


    # =====================================================
    # UPPER EXPERT PRIOR
    # =====================================================

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


    metrics = (
        AssistanceMetrics()
    )


    # =====================================================
    # TEST SEQUENCE
    #
    # We intentionally repeat several frames inside
    # the same condition.
    #
    # Episode counts MUST NOT increase every frame.
    # =====================================================

    sequence = [

        # -------------------------------------------------
        # Normal / agreement
        # -------------------------------------------------

        (
            0.0,
            0.70,
            -0.75,
            0.0,
        ),


        # -------------------------------------------------
        # Alternative valid route episode 1
        #
        # Repeated three frames.
        #
        # Must count as ONE episode.
        # -------------------------------------------------

        (
            0.1,
            1.65,
            0.90,
            0.0,
        ),

        (
            0.2,
            1.65,
            0.90,
            0.0,
        ),

        (
            0.3,
            1.65,
            0.90,
            0.0,
        ),


        # -------------------------------------------------
        # Back to agreement
        # -------------------------------------------------

        (
            0.4,
            0.70,
            -0.75,
            0.0,
        ),


        # -------------------------------------------------
        # Alternative valid route episode 2
        # -------------------------------------------------

        (
            0.5,
            1.65,
            0.90,
            0.0,
        ),


        # -------------------------------------------------
        # Back to agreement
        # -------------------------------------------------

        (
            0.6,
            0.70,
            -0.75,
            0.0,
        ),


        # -------------------------------------------------
        # Orientation risk
        #
        # Reference stays quiet.
        # Task-aware intervenes.
        #
        # Repeated two frames.
        #
        # Must count as ONE missed-risk episode.
        # -------------------------------------------------

        (
            0.7,
            0.70,
            -0.75,
            12.0,
        ),

        (
            0.8,
            0.70,
            -0.75,
            12.0,
        ),


        # -------------------------------------------------
        # Recover orientation
        # -------------------------------------------------

        (
            0.9,
            0.70,
            -0.75,
            0.0,
        ),


        # -------------------------------------------------
        # Obstacle risk
        #
        # Both policies intervene.
        # -------------------------------------------------

        (
            1.0,
            1.65,
            -0.55,
            0.0,
        ),

        (
            1.1,
            1.65,
            -0.55,
            0.0,
        ),


        # -------------------------------------------------
        # Recover
        # -------------------------------------------------

        (
            1.2,
            0.70,
            -0.75,
            0.0,
        ),
    ]


    for (
        timestamp_sec,
        x,
        y,
        orientation_deg,
    ) in sequence:

        task_result = (
            evaluator.evaluate(
                x=x,
                y=y,
                orientation_relative_deg=
                    orientation_deg,
                tracking_missing_sec=
                    0.0,
            )
        )


        comparison = (
            comparator.compare(
                x=x,
                y=y,
                task_result=
                    task_result,
            )
        )


        metrics.update(
            comparison=
                comparison,

            task_result=
                task_result,

            timestamp_sec=
                timestamp_sec,
        )


    summary = (
        metrics.get_summary(
            timestamp_sec=
                1.3
        )
    )


    # =====================================================
    # PRINT
    # =====================================================

    print()

    print(
        "======================================================"
    )

    print(
        "AdaptiveSkill - Episode-Level Assistance Metrics"
    )

    print(
        "======================================================"
    )

    print()

    print(
        "Reference intervention episodes:",
        summary.reference_intervention_episodes
    )

    print(
        "Task-aware intervention episodes:",
        summary.task_intervention_episodes
    )

    print()

    print(
        "Unnecessary intervention episodes avoided:",
        summary.unnecessary_intervention_episodes_avoided
    )

    print(
        "Task-risk episodes missed by reference:",
        summary.task_risk_missed_by_reference_episodes
    )

    print(
        "Task-relevant intervention episodes:",
        summary.task_relevant_intervention_episodes
    )

    print()

    print(
        "Alternative-valid episodes:",
        summary.alternative_valid_episodes
    )

    print(
        "Alternative-valid route preserved:",
        summary.alternative_valid_route_preserved
    )

    print()

    print(
        "Reference intervention time:",
        round(
            summary.time_reference_intervention_sec,
            3
        )
    )

    print(
        "Task-aware intervention time:",
        round(
            summary.time_task_intervention_sec,
            3
        )
    )

    print()

    print(
        "Max reference distance:",
        round(
            summary.max_reference_distance,
            3
        )
    )

    print(
        "Returned to quiet:",
        summary.returned_to_quiet
    )


    # =====================================================
    # EXPECTED
    # =====================================================

    expected = {

        "reference_intervention_episodes":
            3,

        "task_intervention_episodes":
            2,

        "unnecessary_intervention_episodes_avoided":
            2,

        "task_risk_missed_by_reference_episodes":
            1,

        "task_relevant_intervention_episodes":
            2,

        "alternative_valid_episodes":
            2,

        "alternative_valid_route_preserved":
            True,

        "returned_to_quiet":
            True,
    }


    actual = {

        "reference_intervention_episodes":
            summary.reference_intervention_episodes,

        "task_intervention_episodes":
            summary.task_intervention_episodes,

        "unnecessary_intervention_episodes_avoided":
            summary.unnecessary_intervention_episodes_avoided,

        "task_risk_missed_by_reference_episodes":
            summary.task_risk_missed_by_reference_episodes,

        "task_relevant_intervention_episodes":
            summary.task_relevant_intervention_episodes,

        "alternative_valid_episodes":
            summary.alternative_valid_episodes,

        "alternative_valid_route_preserved":
            summary.alternative_valid_route_preserved,

        "returned_to_quiet":
            summary.returned_to_quiet,
    }


    passed = (
        actual
        ==
        expected
    )


    print()

    print(
        "======================================================"
    )

    print(
        "EXPECTED:"
    )

    for key, value in expected.items():

        print(
            f"  {key}: {value}"
        )


    print()

    print(
        "ACTUAL:"
    )

    for key, value in actual.items():

        print(
            f"  {key}: {value}"
        )


    print()

    print(
        "PASS:",
        passed
    )


    if passed:

        print(
            "EPISODE-LEVEL METRICS: PASSED"
        )

    else:

        print(
            "EPISODE-LEVEL METRICS: FAILED"
        )


    print(
        "======================================================"
    )

    print()