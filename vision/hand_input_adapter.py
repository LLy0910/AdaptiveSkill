from dataclasses import dataclass
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp


@dataclass
class HandInputState:
    camera_ready: bool
    hand_detected: bool
    index_x: float | None
    index_y: float | None
    tracking_missing_sec: float
    message: str


class HandInputAdapter:
    """
    Final AdaptiveSkill hand-input adapter.

    Tracking behavior intentionally stays close to the earlier working
    vision/hand_tracker.py:
      - MediaPipe Tasks HandLandmarker in VIDEO mode
      - one hand
      - 0.5 / 0.5 / 0.5 confidence thresholds
      - mirrored camera frame
      - raw landmark-8 fingertip x/y (no EMA smoothing, no jump filter)

    The important integration change is architectural: camera capture and
    MediaPipe inference run in a dedicated background thread. The main demo
    only reads the latest completed state, so heavier policy/UI rendering does
    not throttle the tracker loop as much as a synchronous integration.

    During tracking loss, the last coordinate is retained but marked as lost;
    the main demo decides to freeze/pause rather than invent new motion.
    """

    def __init__(
        self,
        project_root,
        camera_index=0,
    ):
        self.project_root = Path(project_root)
        self.model_path = (
            self.project_root
            / "models"
            / "hand_landmarker.task"
        )
        self.camera_index = int(camera_index)

        self.cap = None
        self.landmarker = None
        self.video_start_time = None

        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self.camera_ready = False
        self._latest_frame = None
        self._last_index_x = None
        self._last_index_y = None
        self._last_seen_time = None

        self._state = HandInputState(
            camera_ready=False,
            hand_detected=False,
            index_x=None,
            index_y=None,
            tracking_missing_sec=999.0,
            message="HAND CAMERA OFF",
        )

    @property
    def latest_frame(self):
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def start(self):
        if self.camera_ready:
            return True

        if not self.model_path.exists():
            with self._lock:
                self._state = HandInputState(
                    camera_ready=False,
                    hand_detected=False,
                    index_x=self._last_index_x,
                    index_y=self._last_index_y,
                    tracking_missing_sec=999.0,
                    message="HAND LANDMARKER MODEL NOT FOUND",
                )
            return False

        cap = cv2.VideoCapture(
            self.camera_index,
            cv2.CAP_DSHOW,
        )

        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            with self._lock:
                self._state = HandInputState(
                    camera_ready=False,
                    hand_detected=False,
                    index_x=self._last_index_x,
                    index_y=self._last_index_y,
                    tracking_missing_sec=999.0,
                    message="CAMERA CANNOT BE OPENED",
                )
            return False

        # Best-effort low-latency hint. Some DSHOW drivers ignore it.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(self.model_path)
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        landmarker = HandLandmarker.create_from_options(options)

        self.cap = cap
        self.landmarker = landmarker
        self.video_start_time = time.perf_counter()
        self._stop_event.clear()
        self.camera_ready = True

        self._thread = threading.Thread(
            target=self._capture_loop,
            name="AdaptiveSkillHandTracker",
            daemon=True,
        )
        self._thread.start()
        return True

    def _capture_loop(self):
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            now = time.perf_counter()

            if not ret:
                with self._lock:
                    self._state = HandInputState(
                        camera_ready=False,
                        hand_detected=False,
                        index_x=self._last_index_x,
                        index_y=self._last_index_y,
                        tracking_missing_sec=999.0,
                        message="CAMERA READ FAILED",
                    )
                time.sleep(0.01)
                continue

            # Same mirrored interaction as the earlier standalone tracker.
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            timestamp_ms = int(
                (now - self.video_start_time) * 1000
            )

            try:
                result = self.landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )
            except Exception:
                # Keep the thread alive; main UI will see tracking lost.
                result = None

            hand_detected = bool(
                result is not None
                and result.hand_landmarks
            )

            if hand_detected:
                landmarks = result.hand_landmarks[0]
                index_tip = landmarks[8]

                self._last_index_x = float(index_tip.x)
                self._last_index_y = float(index_tip.y)
                self._last_seen_time = now
                tracking_missing_sec = 0.0
                message = "HAND TRACKING | RAW INDEX"
            else:
                if self._last_seen_time is None:
                    tracking_missing_sec = 999.0
                else:
                    tracking_missing_sec = max(
                        0.0,
                        now - self._last_seen_time,
                    )
                message = "HAND TRACKING LOST | CUP FROZEN"

            state = HandInputState(
                camera_ready=True,
                hand_detected=hand_detected,
                index_x=self._last_index_x,
                index_y=self._last_index_y,
                tracking_missing_sec=float(tracking_missing_sec),
                message=message,
            )

            with self._lock:
                self._latest_frame = frame.copy()
                self._state = state

    def update(self):
        if not self.camera_ready:
            if not self.start():
                with self._lock:
                    return HandInputState(**self._state.__dict__)

        with self._lock:
            return HandInputState(**self._state.__dict__)

    def close(self):
        self._stop_event.set()

        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None

        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        with self._lock:
            self.camera_ready = False
            self._latest_frame = None
            self._state = HandInputState(
                camera_ready=False,
                hand_detected=False,
                index_x=self._last_index_x,
                index_y=self._last_index_y,
                tracking_missing_sec=999.0,
                message="HAND CAMERA OFF",
            )
