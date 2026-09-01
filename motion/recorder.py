import cv2
import mediapipe as mp
import time
import math
import csv
from pathlib import Path

import numpy as np


# =========================================================
# 1. 项目路径
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_PATH = DATA_DIR / "reference.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    print("ERROR: hand_landmarker.task not found.")
    print("Expected:", MODEL_PATH)
    raise SystemExit


# =========================================================
# 2. MediaPipe 设置
# =========================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================================================
# 3. 参数
# =========================================================

# 大约 1 秒的起始姿势校准
CALIBRATION_FRAMES = 30

# 只用于 Calibration 之前排除非常极端的距离
MIN_PALM_SIZE = 0.08
MAX_PALM_SIZE = 0.28

# Calibration 后，按 R 之前允许 ±15% 的尺度变化
PALM_SIZE_TOLERANCE = 0.15

# X/Y 轻度平滑
SMOOTHING_WINDOW = 5


# =========================================================
# 4. 工具函数
# =========================================================

def euclidean_distance(p1, p2):
    """
    计算两个 MediaPipe landmark
    在二维图像中的距离。
    """
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    return math.sqrt(dx * dx + dy * dy)


def circular_angle_difference(current_angle, reference_angle):
    """
    返回 -180° ~ +180° 范围内的角度差。

    例如：
    current = -170
    reference = +170

    真实差异是 +20°，
    而不是 -340°。
    """
    return (
        current_angle
        - reference_angle
        + 180
    ) % 360 - 180


def moving_average(values, window_size=5):
    """
    简单 moving average。
    """
    if len(values) == 0:
        return []

    result = []

    for i in range(len(values)):
        start = max(0, i - window_size + 1)
        window = values[start:i + 1]
        result.append(float(np.mean(window)))

    return result


# =========================================================
# 5. Calibration
# =========================================================

def calculate_calibration(samples):
    """
    从约 30 帧 starting pose 中得到：

    start_x
    start_y
    palm_size
    start_angle
    """

    if len(samples) == 0:
        return None

    start_x = float(
        np.median([
            sample["x"]
            for sample in samples
        ])
    )

    start_y = float(
        np.median([
            sample["y"]
            for sample in samples
        ])
    )

    palm_size = float(
        np.median([
            sample["palm_size"]
            for sample in samples
        ])
    )

    # Angle 先 unwrap，
    # 避免 179° / -179° 被误认为巨大跳变
    angles = np.array(
        [
            sample["angle"]
            for sample in samples
        ],
        dtype=float
    )

    unwrapped_angles = np.rad2deg(
        np.unwrap(
            np.deg2rad(angles)
        )
    )

    start_angle = float(
        np.median(unwrapped_angles)
    )

    return {
        "start_x": start_x,
        "start_y": start_y,
        "palm_size": palm_size,
        "start_angle": start_angle
    }


# =========================================================
# 6. Calibration 后的起始尺度检查
# =========================================================

def check_scale_against_calibration(
    current_palm_size,
    calibration
):
    """
    注意：
    这不是测真实摄像头距离。

    palm_size 只是一个 image-space scale proxy。

    这个函数只用于：
    Recording 开始之前检查起始状态。
    """

    if calibration is None:
        return "NOT_CALIBRATED"

    baseline = calibration["palm_size"]

    lower_bound = baseline * (
        1 - PALM_SIZE_TOLERANCE
    )

    upper_bound = baseline * (
        1 + PALM_SIZE_TOLERANCE
    )

    if current_palm_size < lower_bound:
        return "TOO_FAR"

    elif current_palm_size > upper_bound:
        return "TOO_CLOSE"

    else:
        return "READY"


# =========================================================
# 7. 保存 Reference
# =========================================================

def save_reference(records, calibration):

    if calibration is None:
        print("ERROR: No calibration available.")
        return

    if len(records) < 2:
        print("Recording too short. Nothing saved.")
        return

    raw_x = [
        record["x_raw"]
        for record in records
    ]

    raw_y = [
        record["y_raw"]
        for record in records
    ]

    smooth_x_raw = moving_average(
        raw_x,
        SMOOTHING_WINDOW
    )

    smooth_y_raw = moving_average(
        raw_y,
        SMOOTHING_WINDOW
    )

    initial_scale = calibration["palm_size"]

    if initial_scale < 1e-6:
        print("ERROR: Invalid calibration palm size.")
        return

    with open(
        REFERENCE_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "x_raw",
            "y_raw",
            "x_norm",
            "y_norm",
            "angle_raw",
            "angle_rel",
            "palm_size"
        ])

        for i, record in enumerate(records):

            # -------------------------------------------------
            # 起点 + 起始手掌尺度 normalization
            # -------------------------------------------------

            x_norm = (
                smooth_x_raw[i]
                - calibration["start_x"]
            ) / initial_scale

            y_norm = (
                smooth_y_raw[i]
                - calibration["start_y"]
            ) / initial_scale

            # -------------------------------------------------
            # 相对于 starting pose 的 angle
            # -------------------------------------------------

            angle_rel = circular_angle_difference(
                record["angle_raw"],
                calibration["start_angle"]
            )

            writer.writerow([
                record["timestamp"],

                smooth_x_raw[i],
                smooth_y_raw[i],

                x_norm,
                y_norm,

                record["angle_raw"],
                angle_rel,

                record["palm_size"]
            ])

    duration = (
        records[-1]["timestamp"]
        - records[0]["timestamp"]
    )

    print()
    print("======================================")
    print("REFERENCE SAVED")
    print("======================================")
    print("Frames:", len(records))
    print("Duration:", round(duration, 2), "sec")
    print(
        "Start X:",
        round(calibration["start_x"], 3)
    )
    print(
        "Start Y:",
        round(calibration["start_y"], 3)
    )
    print(
        "Initial Palm Size:",
        round(calibration["palm_size"], 3)
    )
    print(
        "Initial Angle:",
        round(calibration["start_angle"], 1),
        "deg"
    )
    print("File:", REFERENCE_PATH)
    print()


# =========================================================
# 8. 加载 Reference
# =========================================================

def load_reference():
    """
    同时兼容：

    新版：
    timestamp,x_raw,y_raw,x_norm,y_norm,
    angle_raw,angle_rel,palm_size

    旧版：
    timestamp,x,y,angle
    """

    if not REFERENCE_PATH.exists():
        print("No existing reference.csv")
        return []

    reference = []

    with open(
        REFERENCE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            return []

        # -------------------------------------------------
        # 新版
        # -------------------------------------------------

        if "x_raw" in reader.fieldnames:

            for row in reader:

                reference.append({
                    "timestamp":
                        float(row["timestamp"]),

                    "x_raw":
                        float(row["x_raw"]),

                    "y_raw":
                        float(row["y_raw"]),

                    "x_norm":
                        float(row["x_norm"]),

                    "y_norm":
                        float(row["y_norm"]),

                    "angle_raw":
                        float(row["angle_raw"]),

                    "angle_rel":
                        float(row["angle_rel"]),

                    "palm_size":
                        float(row["palm_size"])
                })

        # -------------------------------------------------
        # 旧版
        # -------------------------------------------------

        elif (
            "x" in reader.fieldnames
            and
            "y" in reader.fieldnames
        ):

            for row in reader:

                reference.append({
                    "timestamp":
                        float(row["timestamp"]),

                    "x_raw":
                        float(row["x"]),

                    "y_raw":
                        float(row["y"]),

                    "x_norm": 0.0,
                    "y_norm": 0.0,

                    "angle_raw":
                        float(row["angle"]),

                    "angle_rel": 0.0,

                    "palm_size": 0.0
                })

    print(
        f"Loaded reference: {len(reference)} frames"
    )

    return reference


# =========================================================
# 9. 画 Reference
# =========================================================

def draw_reference(frame, reference):
    """
    屏幕上继续显示 raw trajectory。

    normalized trajectory 后续 Day 3
    用来做真正的比较。
    """

    if len(reference) < 2:
        return

    height, width, _ = frame.shape

    for i in range(
        1,
        len(reference)
    ):

        previous = reference[i - 1]
        current = reference[i]

        x1 = int(
            previous["x_raw"]
            * width
        )

        y1 = int(
            previous["y_raw"]
            * height
        )

        x2 = int(
            current["x_raw"]
            * width
        )

        y2 = int(
            current["y_raw"]
            * height
        )

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )


# =========================================================
# 10. 打开摄像头
# =========================================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not cap.isOpened():
    print("ERROR: Camera cannot be opened.")
    raise SystemExit


# =========================================================
# 11. 状态变量
# =========================================================

recording = False
calibrating = False
calibration_ready = False

calibration_samples = []
calibration = None

records = []
record_start_time = None

reference = load_reference()


print()
print("==========================================")
print("AdaptiveSkill v1.2")
print("Calibration-aware Reference Recorder")
print()
print("C = Calibrate starting pose")
print("R = Start recording")
print("S = Stop + Save")
print("Q = Quit")
print("==========================================")
print()


# =========================================================
# 12. MediaPipe 主循环
# =========================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:

    video_start_time = time.perf_counter()

    while True:

        # =================================================
        # 摄像头
        # =================================================

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Cannot read camera frame.")
            break

        frame = cv2.flip(
            frame,
            1
        )

        height, width, _ = frame.shape


        # =================================================
        # OpenCV → MediaPipe
        # =================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (
                time.perf_counter()
                - video_start_time
            )
            * 1000
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # =================================================
        # 当前帧数据
        # =================================================

        hand_detected = False

        current_x = None
        current_y = None

        current_angle = None
        current_palm_size = None


        # =================================================
        # 13. 检测到手
        # =================================================

        if result.hand_landmarks:

            hand_detected = True

            landmarks = result.hand_landmarks[0]

            # ---------------------------------------------
            # 关键点
            # ---------------------------------------------

            wrist = landmarks[0]

            index_mcp = landmarks[5]

            index_tip = landmarks[8]

            middle_mcp = landmarks[9]

            pinky_mcp = landmarks[17]


            # ---------------------------------------------
            # Raw X/Y
            # ---------------------------------------------

            current_x = index_tip.x
            current_y = index_tip.y


            # ---------------------------------------------
            # 2D Hand Orientation
            # ---------------------------------------------

            dx = (
                middle_mcp.x
                - wrist.x
            )

            dy = (
                middle_mcp.y
                - wrist.y
            )

            current_angle = math.degrees(
                math.atan2(
                    dy,
                    dx
                )
            )


            # ---------------------------------------------
            # Palm Size
            #
            # landmark 5 ↔ landmark 17
            #
            # 只是 image-space scale proxy
            # ---------------------------------------------

            current_palm_size = (
                euclidean_distance(
                    index_mcp,
                    pinky_mcp
                )
            )


            # ---------------------------------------------
            # 画食指尖
            # ---------------------------------------------

            index_px = int(
                current_x
                * width
            )

            index_py = int(
                current_y
                * height
            )

            cv2.circle(
                frame,
                (index_px, index_py),
                10,
                (255, 255, 255),
                3
            )


            # =================================================
            # 14. Calibration
            # =================================================

            if calibrating:

                calibration_samples.append({
                    "x":
                        current_x,

                    "y":
                        current_y,

                    "angle":
                        current_angle,

                    "palm_size":
                        current_palm_size
                })

                progress = int(
                    len(calibration_samples)
                    /
                    CALIBRATION_FRAMES
                    *
                    100
                )

                progress = min(
                    progress,
                    100
                )

                cv2.putText(
                    frame,
                    f"CALIBRATING {progress}%",
                    (20, 165),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2
                )

                # -----------------------------------------
                # Calibration 完成
                # -----------------------------------------

                if (
                    len(calibration_samples)
                    >=
                    CALIBRATION_FRAMES
                ):

                    calibrating = False

                    calibration = (
                        calculate_calibration(
                            calibration_samples
                        )
                    )

                    calibration_ready = True

                    baseline = (
                        calibration["palm_size"]
                    )

                    lower_bound = (
                        baseline
                        *
                        (
                            1
                            -
                            PALM_SIZE_TOLERANCE
                        )
                    )

                    upper_bound = (
                        baseline
                        *
                        (
                            1
                            +
                            PALM_SIZE_TOLERANCE
                        )
                    )

                    print()
                    print(
                        "======================================"
                    )
                    print(
                        "CALIBRATION COMPLETE"
                    )
                    print(
                        "======================================"
                    )

                    print(
                        "Start X:",
                        round(
                            calibration["start_x"],
                            3
                        )
                    )

                    print(
                        "Start Y:",
                        round(
                            calibration["start_y"],
                            3
                        )
                    )

                    print(
                        "Baseline Palm Size:",
                        round(
                            baseline,
                            3
                        )
                    )

                    print(
                        "Start Angle:",
                        round(
                            calibration["start_angle"],
                            1
                        ),
                        "deg"
                    )

                    print(
                        "Allowed start scale:",
                        f"{lower_bound:.3f}",
                        "to",
                        f"{upper_bound:.3f}"
                    )

                    print(
                        "Return to approximately "
                        "the same starting pose, "
                        "then press R."
                    )

                    print()


            # =================================================
            # 15. Recording
            #
            # 注意：
            # Recording 开始以后，
            # palm size 只记录，
            # 不做实时 TOO FAR / TOO CLOSE 判断。
            # =================================================

            if recording:

                relative_time = (
                    time.perf_counter()
                    -
                    record_start_time
                )

                records.append({
                    "timestamp":
                        relative_time,

                    "x_raw":
                        current_x,

                    "y_raw":
                        current_y,

                    "angle_raw":
                        current_angle,

                    "palm_size":
                        current_palm_size
                })


            # =================================================
            # 16. 实时数据显示
            # =================================================

            cv2.putText(
                frame,
                f"X RAW: {current_x:.3f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Y RAW: {current_y:.3f}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"ANGLE: {current_angle:.1f} deg",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"PALM SIZE: {current_palm_size:.3f}",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2
            )


        # =================================================
        # 17. 没检测到手
        # =================================================

        else:

            cv2.putText(
                frame,
                "TRACKING LOST",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                3
            )


        # =================================================
        # 18. Scale / Calibration 提示
        #
        # ★ 核心修改 ★
        #
        # 只有 recording == False 时
        # 才允许显示这些提示。
        #
        # 一旦 Recording 开始，
        # 完全停止 TOO FAR / TOO CLOSE。
        # =================================================

        if (
            not recording
            and
            hand_detected
            and
            current_palm_size is not None
        ):

            # ---------------------------------------------
            # 尚未 Calibration
            # ---------------------------------------------

            if (
                not calibration_ready
                and
                not calibrating
            ):

                if (
                    current_palm_size
                    <
                    MIN_PALM_SIZE
                ):

                    distance_status = (
                        "MOVE CLOSER FOR CALIBRATION"
                    )

                elif (
                    current_palm_size
                    >
                    MAX_PALM_SIZE
                ):

                    distance_status = (
                        "MOVE BACK FOR CALIBRATION"
                    )

                else:

                    distance_status = (
                        "READY FOR CALIBRATION"
                    )


            # ---------------------------------------------
            # 正在 Calibration
            # ---------------------------------------------

            elif calibrating:

                distance_status = (
                    "HOLD STARTING POSE STILL"
                )


            # ---------------------------------------------
            # Calibration 已完成，
            # 但还没开始 Recording
            # ---------------------------------------------

            else:

                scale_status = (
                    check_scale_against_calibration(
                        current_palm_size,
                        calibration
                    )
                )

                if scale_status == "TOO_FAR":

                    distance_status = (
                        "MOVE SLIGHTLY CLOSER"
                    )

                elif scale_status == "TOO_CLOSE":

                    distance_status = (
                        "MOVE SLIGHTLY BACK"
                    )

                else:

                    distance_status = (
                        "START POSITION READY"
                    )


            cv2.putText(
                frame,
                distance_status,
                (20, height - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.67,
                (255, 255, 255),
                2
            )


        # =================================================
        # 19. Reference
        #
        # Recording 时隐藏旧轨迹
        # =================================================

        if (
            not recording
            and
            not calibrating
        ):

            draw_reference(
                frame,
                reference
            )


        # =================================================
        # 20. 底部状态
        # =================================================

        if recording:

            cv2.putText(
                frame,
                "RECORDING - Press S to Stop",
                (20, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                3
            )


        elif calibrating:

            cv2.putText(
                frame,
                "CALIBRATING...",
                (20, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )


        elif calibration_ready:

            cv2.putText(
                frame,
                "CALIBRATION READY - Press R",
                (20, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


        else:

            cv2.putText(
                frame,
                "Press C to Calibrate",
                (20, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 255, 255),
                2
            )


        # =================================================
        # 21. 显示
        # =================================================

        cv2.imshow(
            "AdaptiveSkill v1.2 - Teach Reference",
            frame
        )

        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        # =================================================
        # 22. C = Calibration
        # =================================================

        if key == ord("c"):

            if recording:

                print(
                    "Cannot calibrate while recording."
                )

            elif calibrating:

                print(
                    "Calibration already running."
                )

            elif not hand_detected:

                print(
                    "Cannot calibrate: "
                    "hand not detected."
                )

            elif (
                current_palm_size
                <
                MIN_PALM_SIZE
            ):

                print(
                    "Cannot calibrate: "
                    "move closer."
                )

            elif (
                current_palm_size
                >
                MAX_PALM_SIZE
            ):

                print(
                    "Cannot calibrate: "
                    "move back."
                )

            else:

                print()
                print(
                    "Calibration started."
                )
                print(
                    "Hold the STARTING POSE "
                    "still for about one second..."
                )

                calibrating = True
                calibration_ready = False

                calibration_samples = []
                calibration = None


        # =================================================
        # 23. R = Start Recording
        #
        # ★ 只在这一刻最后检查一次 scale ★
        # =================================================

        elif key == ord("r"):

            if recording:

                print(
                    "Recording already running."
                )

            elif calibrating:

                print(
                    "Wait until calibration finishes."
                )

            elif not calibration_ready:

                print(
                    "Please calibrate first with C."
                )

            elif not hand_detected:

                print(
                    "Cannot start: "
                    "hand not detected."
                )

            else:

                scale_status = (
                    check_scale_against_calibration(
                        current_palm_size,
                        calibration
                    )
                )

                if scale_status == "TOO_FAR":

                    print(
                        "Cannot start: "
                        "move slightly closer."
                    )

                elif scale_status == "TOO_CLOSE":

                    print(
                        "Cannot start: "
                        "move slightly back."
                    )

                else:

                    print()
                    print(
                        "Recording started."
                    )
                    print(
                        "Scale checking is now disabled "
                        "during the movement."
                    )

                    records = []

                    recording = True

                    record_start_time = (
                        time.perf_counter()
                    )


        # =================================================
        # 24. S = Stop + Save
        # =================================================

        elif key == ord("s"):

            if recording:

                recording = False

                print(
                    "Recording stopped."
                )

                save_reference(
                    records,
                    calibration
                )

                reference = (
                    load_reference()
                )

                # 下一次 Reference
                # 必须重新 Calibration
                calibration_ready = False
                calibration = None
                calibration_samples = []

            else:

                print(
                    "No recording is running."
                )


        # =================================================
        # 25. Q = Quit
        # =================================================

        elif key == ord("q"):

            break


# =========================================================
# 26. 收尾
# =========================================================

cap.release()
cv2.destroyAllWindows()

print()
print(
    "AdaptiveSkill v1.2 Closed."
)