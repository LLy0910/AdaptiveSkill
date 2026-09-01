import cv2
import mediapipe as mp
import time
import math
from pathlib import Path


# =========================================================
# 1. 项目路径
# =========================================================

# 当前文件：
# AdaptiveSkill/vision/hand_tracker.py
#
# parent        = vision
# parent.parent = AdaptiveSkill
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

print("Project root:", PROJECT_ROOT)
print("Model path:", MODEL_PATH)

if not MODEL_PATH.exists():
    print("ERROR: hand_landmarker.task not found.")
    print("Expected location:", MODEL_PATH)
    exit()


# =========================================================
# 2. MediaPipe Hand Landmarker 设置
# =========================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),

    # 摄像头视频流
    running_mode=VisionRunningMode.VIDEO,

    # Day 1 只追踪一只手
    num_hands=1,

    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================================================
# 3. MediaPipe 21 个手部关键点之间的连线
# =========================================================

HAND_CONNECTIONS = [
    # 拇指
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # 食指
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # 中指
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # 无名指
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # 小拇指
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # 手掌
    (0, 17)
]


# =========================================================
# 4. 打开 Windows 摄像头
# =========================================================

# 你的电脑之前已经验证 CAP_DSHOW 可以正常打开摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera cannot be opened.")
    exit()

print()
print("======================================")
print("AdaptiveSkill Hand Tracker Started")
print("Press Q to quit.")
print("======================================")
print()


# =========================================================
# 5. 创建 MediaPipe Hand Landmarker
# =========================================================

with HandLandmarker.create_from_options(options) as landmarker:

    # VIDEO 模式需要不断增加的时间戳
    start_time = time.perf_counter()

    while True:

        # -------------------------------------------------
        # 读取摄像头
        # -------------------------------------------------

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Cannot read camera frame.")
            break

        # 镜像，让你看起来像照镜子
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape


        # -------------------------------------------------
        # OpenCV BGR → RGB
        # -------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # -------------------------------------------------
        # NumPy 图像 → MediaPipe Image
        # -------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # -------------------------------------------------
        # 生成 VIDEO 模式需要的 timestamp
        # -------------------------------------------------

        timestamp_ms = int(
            (time.perf_counter() - start_time) * 1000
        )


        # -------------------------------------------------
        # MediaPipe 手部检测
        # -------------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # =================================================
        # 6. 检测到了手
        # =================================================

        if result.hand_landmarks:

            # 这里只拿第一只手
            landmarks = result.hand_landmarks[0]


            # -------------------------------------------------
            # 画手部骨架
            # -------------------------------------------------

            for start_idx, end_idx in HAND_CONNECTIONS:

                point1 = landmarks[start_idx]
                point2 = landmarks[end_idx]

                x1 = int(point1.x * width)
                y1 = int(point1.y * height)

                x2 = int(point2.x * width)
                y2 = int(point2.y * height)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )


            # -------------------------------------------------
            # 画 21 个 landmark
            # -------------------------------------------------

            for landmark in landmarks:

                cx = int(landmark.x * width)
                cy = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (cx, cy),
                    5,
                    (255, 255, 255),
                    -1
                )


            # =================================================
            # 7. 获取 Day 1 需要的三个点
            # =================================================

            # 0 = wrist
            wrist = landmarks[0]

            # 8 = index fingertip
            index_tip = landmarks[8]

            # 9 = middle finger MCP
            middle_mcp = landmarks[9]


            # -------------------------------------------------
            # 食指尖 X / Y
            # MediaPipe 是归一化坐标 0～1
            # -------------------------------------------------

            x = index_tip.x
            y = index_tip.y


            # =================================================
            # 8. 计算二维 hand orientation
            # =================================================

            # 想象一条线：
            #
            # wrist(0) --------> middle MCP(9)
            #
            # 我们看这条线现在朝哪个方向

            dx = middle_mcp.x - wrist.x
            dy = middle_mcp.y - wrist.y

            angle = math.degrees(
                math.atan2(dy, dx)
            )


            # =================================================
            # 9. 把食指尖特别圈出来
            # =================================================

            index_px = int(index_tip.x * width)
            index_py = int(index_tip.y * height)

            cv2.circle(
                frame,
                (index_px, index_py),
                12,
                (255, 255, 255),
                3
            )


            # =================================================
            # 10. 左上角显示数据
            # =================================================

            cv2.putText(
                frame,
                f"INDEX X: {x:.3f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"INDEX Y: {y:.3f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"ANGLE: {angle:.1f} deg",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "TRACKING",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )


        # =================================================
        # 11. 没检测到手
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
        # 12. 显示窗口
        # =================================================

        cv2.imshow(
            "AdaptiveSkill - Hand Tracking",
            frame
        )


        # 按 Q 退出
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


# =========================================================
# 13. 释放摄像头
# =========================================================

cap.release()
cv2.destroyAllWindows()

print("AdaptiveSkill Hand Tracker Closed.")