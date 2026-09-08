from dataclasses import dataclass
import json
from pathlib import Path


# =========================================================
# STATES / CHOICES
# =========================================================

STATE_IDLE = "IDLE"
STATE_HOLDING = "HOLDING"
STATE_WAITING = "WAITING_FOR_USER"
STATE_COOLDOWN = "COOLDOWN"

CHOICE_NONE = "NONE"
CHOICE_KEEP = "KEEP"
CHOICE_GUIDANCE = "GUIDANCE"

TASK_VALID = "VALID"


# =========================================================
# DECISION
# =========================================================

@dataclass
class IntentDecision:
    state: str
    prompt_visible: bool
    choice: str
    user_confirmed_alternative: bool
    guidance_requested: bool
    persistent_deviation_sec: float
    reason: str


# =========================================================
# MINIMAL INTENT MANAGER
#
# Purpose:
# - Do NOT infer user intention.
# - Detect only a narrow uncertainty condition:
#     task-valid + reliably tracked + persistently far
#     from expert prior.
# - Ask the user:
#     KEEP MY STRATEGY
#     SHOW GUIDANCE
#
# This is a user-confirmation / override mechanism, not an
# intent-recognition model.
# =========================================================

class IntentManager:

    def __init__(
        self,
        config_path=None,
        reference_deviation_threshold=0.30,
        reference_deviation_hold_sec=None,
        clarification_cooldown_sec=None,
    ):

        self.reference_deviation_threshold = float(
            reference_deviation_threshold
        )

        config_hold = 0.8
        config_cooldown = 3.0

        if config_path is not None:
            config_path = Path(
                config_path
            )

            with open(
                config_path,
                "r",
                encoding="utf-8",
            ) as file:
                config = json.load(
                    file
                )

            intent_config = config.get(
                "intent",
                {},
            )

            config_hold = float(
                intent_config.get(
                    "reference_deviation_hold_sec",
                    config_hold,
                )
            )

            config_cooldown = float(
                intent_config.get(
                    "clarification_cooldown_sec",
                    config_cooldown,
                )
            )

        self.reference_deviation_hold_sec = float(
            config_hold
            if reference_deviation_hold_sec is None
            else reference_deviation_hold_sec
        )

        self.clarification_cooldown_sec = float(
            config_cooldown
            if clarification_cooldown_sec is None
            else clarification_cooldown_sec
        )

        self.reset()


    # =====================================================
    # RESET
    # =====================================================

    def reset(self):

        self.state = STATE_IDLE

        self._deviation_start_time = None
        self._cooldown_until = None

        self._choice = CHOICE_NONE
        self._user_confirmed_alternative = False
        self._guidance_requested = False


    # =====================================================
    # INTERNAL
    # =====================================================

    def _decision(
        self,
        now,
        reason,
    ):

        persistent_sec = 0.0

        if self._deviation_start_time is not None:
            persistent_sec = max(
                0.0,
                float(now)
                -
                float(self._deviation_start_time),
            )

        return IntentDecision(
            state=self.state,
            prompt_visible=(
                self.state
                ==
                STATE_WAITING
            ),
            choice=self._choice,
            user_confirmed_alternative=(
                self._user_confirmed_alternative
            ),
            guidance_requested=(
                self._guidance_requested
            ),
            persistent_deviation_sec=round(
                persistent_sec,
                4,
            ),
            reason=reason,
        )


    def _clear_transient_choice_flags(
        self,
    ):

        self._choice = CHOICE_NONE
        self._guidance_requested = False


    # =====================================================
    # UPDATE
    # =====================================================

    def update(
        self,
        task_state,
        reference_distance,
        tracking_reliable,
        now,
    ):

        now = float(
            now
        )

        self._clear_transient_choice_flags()

        # -------------------------------------------------
        # Cooldown after either explicit user choice.
        # -------------------------------------------------

        if (
            self._cooldown_until
            is not None
            and
            now
            <
            self._cooldown_until
        ):

            self.state = STATE_COOLDOWN

            self._deviation_start_time = None

            return self._decision(
                now,
                "clarification_cooldown",
            )

        if (
            self._cooldown_until
            is not None
            and
            now
            >=
            self._cooldown_until
        ):

            self._cooldown_until = None

            self.state = STATE_IDLE

            # A KEEP confirmation is deliberately local/
            # temporary. After cooldown we can ask again if
            # a later distinct alternative remains ambiguous.
            self._user_confirmed_alternative = False


        # -------------------------------------------------
        # Tracking uncertainty must NOT be interpreted as
        # user intention or user error.
        # -------------------------------------------------

        if not bool(
            tracking_reliable
        ):

            self.state = STATE_IDLE
            self._deviation_start_time = None

            return self._decision(
                now,
                "tracking_unreliable",
            )


        # -------------------------------------------------
        # Intent clarification is ONLY for task-valid states.
        # Risk/violation should go through task assistance,
        # not an intent prompt.
        # -------------------------------------------------

        if str(
            task_state
        ) != TASK_VALID:

            self.state = STATE_IDLE
            self._deviation_start_time = None
            self._user_confirmed_alternative = False

            return self._decision(
                now,
                "task_not_valid",
            )


        # -------------------------------------------------
        # Close to expert prior -> no ambiguity.
        # -------------------------------------------------

        if (
            float(
                reference_distance
            )
            <=
            self.reference_deviation_threshold
        ):

            self.state = STATE_IDLE
            self._deviation_start_time = None
            self._user_confirmed_alternative = False

            return self._decision(
                now,
                "reference_similarity_acceptable",
            )


        # -------------------------------------------------
        # Far but task-valid: start/continue hold timer.
        # -------------------------------------------------

        if self._deviation_start_time is None:

            self._deviation_start_time = now
            self.state = STATE_HOLDING

            return self._decision(
                now,
                "reference_deviation_started",
            )


        persistent_sec = max(
            0.0,
            now
            -
            self._deviation_start_time,
        )


        if (
            persistent_sec
            <
            self.reference_deviation_hold_sec
        ):

            self.state = STATE_HOLDING

            return self._decision(
                now,
                "waiting_for_persistent_deviation",
            )


        # -------------------------------------------------
        # Persistent + task-valid + reliable:
        # system admits uncertainty and asks the user.
        # -------------------------------------------------

        self.state = STATE_WAITING

        return self._decision(
            now,
            "task_valid_but_different",
        )


    # =====================================================
    # USER CHOICES
    # =====================================================

    def choose_keep(
        self,
        now,
    ):

        now = float(
            now
        )

        if self.state != STATE_WAITING:

            return self._decision(
                now,
                "keep_ignored_no_active_prompt",
            )

        self._choice = CHOICE_KEEP
        self._user_confirmed_alternative = True
        self._guidance_requested = False

        self._cooldown_until = (
            now
            +
            self.clarification_cooldown_sec
        )

        self._deviation_start_time = None
        self.state = STATE_COOLDOWN

        return self._decision(
            now,
            "user_confirmed_alternative",
        )


    def choose_guidance(
        self,
        now,
    ):

        now = float(
            now
        )

        if self.state != STATE_WAITING:

            return self._decision(
                now,
                "guidance_ignored_no_active_prompt",
            )

        self._choice = CHOICE_GUIDANCE
        self._user_confirmed_alternative = False
        self._guidance_requested = True

        self._cooldown_until = (
            now
            +
            self.clarification_cooldown_sec
        )

        self._deviation_start_time = None
        self.state = STATE_COOLDOWN

        return self._decision(
            now,
            "user_requested_guidance",
        )
