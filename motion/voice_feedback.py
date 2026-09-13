import os
import queue
import subprocess
import threading
import time


# =========================================================
# VOICE FEEDBACK
# =========================================================

class VoiceFeedback:
    """
    Non-blocking Windows voice feedback.

    Real-time camera loop:
        voice.speak(message)
            |
            v
        queue
            |
            v
        background worker
            |
            v
        Windows SAPI speech

    Speech runs outside the camera thread, so the
    real-time vision loop is not blocked.

    Windows only.
    """


    def __init__(
        self,
        enabled=True,
        rate=0,
        volume=100
    ):

        self.enabled = bool(
            enabled
        )

        # Windows SAPI rate:
        # roughly -10 to +10
        self.rate = int(
            rate
        )

        # Windows SAPI volume:
        # 0 to 100
        self.volume = int(
            volume
        )


        self._queue = queue.Queue()

        self._running = True

        self._ready = False

        self._error = None


        self._thread = threading.Thread(

            target=
                self._worker,

            name=
                "AdaptiveSkillVoiceWorker",

            daemon=True
        )


        self._thread.start()


    # =====================================================
    # WORKER
    # =====================================================

    def _worker(self):

        # -------------------------------------------------
        # Verify Windows PowerShell + System.Speech
        # -------------------------------------------------

        try:

            check_command = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object "
                "System.Speech.Synthesis.SpeechSynthesizer; "
                "$s.Dispose();"
            )


            result = subprocess.run(

                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    check_command
                ],

                stdout=
                    subprocess.DEVNULL,

                stderr=
                    subprocess.PIPE,

                text=True,

                creationflags=
                    getattr(
                        subprocess,
                        "CREATE_NO_WINDOW",
                        0
                    )
            )


            if result.returncode != 0:

                raise RuntimeError(
                    result.stderr.strip()
                )


            self._ready = True


        except Exception as exc:

            self._error = str(
                exc
            )

            print(
                "[VoiceFeedback] Initialization error:",
                exc
            )


        # =================================================
        # MAIN QUEUE LOOP
        # =================================================

        while self._running:

            item = self._queue.get()


            try:

                # -----------------------------------------
                # Shutdown sentinel
                # -----------------------------------------

                if item is None:

                    break


                if not self.enabled:

                    continue


                message = str(
                    item
                ).strip()


                if not message:

                    continue


                if not self._ready:

                    continue


                self._speak_windows(
                    message
                )


            except Exception as exc:

                self._error = str(
                    exc
                )

                print(
                    "[VoiceFeedback] Speech error:",
                    exc
                )


            finally:

                self._queue.task_done()


        self._ready = False


    # =====================================================
    # WINDOWS SAPI
    # =====================================================

    def _speak_windows(
        self,
        message
    ):

        """
        Speak one sentence using Windows built-in
        System.Speech.

        This method blocks ONLY the background worker,
        never the camera thread.
        """

        environment = (
            os.environ.copy()
        )


        # Pass text through an environment variable
        # instead of inserting it directly into a
        # PowerShell command. This avoids quote problems.
        environment[
            "ADAPTIVESKILL_TTS_TEXT"
        ] = message


        environment[
            "ADAPTIVESKILL_TTS_RATE"
        ] = str(
            self.rate
        )


        environment[
            "ADAPTIVESKILL_TTS_VOLUME"
        ] = str(
            self.volume
        )


        command = r"""
Add-Type -AssemblyName System.Speech

$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer

$rate = [int][Environment]::GetEnvironmentVariable(
    'ADAPTIVESKILL_TTS_RATE'
)

$volume = [int][Environment]::GetEnvironmentVariable(
    'ADAPTIVESKILL_TTS_VOLUME'
)

$text = [Environment]::GetEnvironmentVariable(
    'ADAPTIVESKILL_TTS_TEXT'
)

$speaker.Rate = $rate
$speaker.Volume = $volume

$speaker.Speak($text)

$speaker.Dispose()
"""


        result = subprocess.run(

            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command
            ],

            env=
                environment,

            stdout=
                subprocess.DEVNULL,

            stderr=
                subprocess.PIPE,

            text=True,

            creationflags=
                getattr(
                    subprocess,
                    "CREATE_NO_WINDOW",
                    0
                )
        )


        if result.returncode != 0:

            raise RuntimeError(
                result.stderr.strip()
            )


    # =====================================================
    # SPEAK
    # =====================================================

    def speak(
        self,
        message
    ):

        """
        Queue one sentence.

        Returns immediately.
        """

        if not self.enabled:

            return False


        if not self._running:

            return False


        if message is None:

            return False


        message = str(
            message
        ).strip()


        if not message:

            return False


        self._queue.put(
            message
        )


        return True


    # =====================================================
    # CLEAR PENDING SPEECH
    # =====================================================

    def clear(
        self
    ):

        """
        Remove messages that have not yet started.
        """

        removed = 0


        while True:

            try:

                item = (
                    self._queue.get_nowait()
                )


                self._queue.task_done()


                if item is not None:

                    removed += 1


            except queue.Empty:

                break


        return removed


    # =====================================================
    # ENABLE / DISABLE
    # =====================================================

    def set_enabled(
        self,
        enabled
    ):

        self.enabled = bool(
            enabled
        )


        if not self.enabled:

            self.clear()


    # =====================================================
    # STATUS
    # =====================================================

    @property
    def ready(
        self
    ):

        return self._ready


    @property
    def error(
        self
    ):

        return self._error


    @property
    def pending_count(
        self
    ):

        return self._queue.qsize()


    # =====================================================
    # WAIT
    # =====================================================

    def wait_until_idle(
        self
    ):

        """
        Only use for standalone tests.

        Do NOT use this in the real-time camera loop.
        """

        self._queue.join()


    # =====================================================
    # SHUTDOWN
    # =====================================================

    def shutdown(
        self,
        wait=True
    ):

        if not self._running:

            return


        self._running = False


        self._queue.put(
            None
        )


        if wait:

            self._thread.join(
                timeout=5.0
            )


# =========================================================
# SELF TEST
# =========================================================

if __name__ == "__main__":

    print()

    print(
        "=============================================="
    )

    print(
        "AdaptiveSkill - Windows Voice Feedback"
    )

    print(
        "=============================================="
    )

    print()


    voice = VoiceFeedback(

        enabled=True,

        rate=0,

        volume=100
    )


    # Give worker enough time to initialize.
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


    print(
        "Voice ready:",
        voice.ready
    )


    print(
        "Voice error:",
        voice.error
    )


    print()


    if not voice.ready:

        print(
            "Voice system is not ready."
        )

        voice.shutdown()

        raise SystemExit(
            1
        )


    # =====================================================
    # NON-BLOCKING TEST
    # =====================================================

    print(
        "Testing non-blocking speech queue..."
    )


    start = (
        time.perf_counter()
    )


    voice.speak(
        "Move back toward the reference path."
    )


    voice.speak(
        "Rotate your hand more."
    )


    voice.speak(
        "You may need human assistance."
    )


    elapsed = (

        time.perf_counter()

        -

        start
    )


    print(
        "Three messages queued."
    )


    print(
        "Queue calls returned in:",
        round(
            elapsed,
            4
        ),
        "seconds"
    )


    print(
        "Main Python thread is still responsive."
    )


    print()

    print(
        "Waiting for speech test to finish..."
    )


    voice.wait_until_idle()


    print()

    print(
        "Speech queue finished."
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
        "=============================================="
    )

    print(
        "Expected:"
    )

    print(
        "1. Voice ready = True"
    )

    print(
        "2. Hear ALL THREE English sentences"
    )

    print(
        "3. Queue calls return immediately"
    )

    print(
        "4. Pending messages = 0"
    )

    print(
        "5. Voice error = None"
    )

    print(
        "=============================================="
    )

    print()