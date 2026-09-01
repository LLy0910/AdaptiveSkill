import csv
import cv2
import numpy as np
from pathlib import Path


# =========================================================
# 1. 项目路径
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

REFERENCE_FILES = [
    DATA_DIR / "reference_1.csv",
    DATA_DIR / "reference_2.csv",
    DATA_DIR / "reference_3.csv"
]


# =========================================================
# 2. 读取新版 Reference
# =========================================================

def load_reference(path):

    reference = []

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return reference

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        required_columns = [
            "timestamp",
            "x_raw",
            "y_raw",
            "x_norm",
            "y_norm",
            "angle_raw",
            "angle_rel",
            "palm_size"
        ]

        if reader.fieldnames is None:
            print(f"ERROR: Empty CSV: {path}")
            return reference

        missing = [
            column
            for column in required_columns
            if column not in reader.fieldnames
        ]

        if missing:
            print(
                f"ERROR: {path.name} is not v1.2 format."
            )
            print(
                "Missing columns:",
                missing
            )
            return reference

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

    return reference


# =========================================================
# 3. Angle unwrap
# =========================================================

def unwrap_angles(values):

    if len(values) == 0:
        return np.array([])

    values = np.array(
        values,
        dtype=float
    )

    return np.rad2deg(
        np.unwrap(
            np.deg2rad(values)
        )
    )


# =========================================================
# 4. 画轨迹
# =========================================================

def draw_trajectory(
    panel,
    reference,
    x_key,
    y_key,
    title,
    common_bounds=None
):

    panel[:] = 255

    height, width, _ = panel.shape

    cv2.putText(
        panel,
        title,
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )

    if len(reference) < 2:
        return

    xs = np.array([
        point[x_key]
        for point in reference
    ])

    ys = np.array([
        point[y_key]
        for point in reference
    ])

    if common_bounds is None:

        min_x = float(xs.min())
        max_x = float(xs.max())

        min_y = float(ys.min())
        max_y = float(ys.max())

    else:

        (
            min_x,
            max_x,
            min_y,
            max_y
        ) = common_bounds


    range_x = max(
        max_x - min_x,
        0.001
    )

    range_y = max(
        max_y - min_y,
        0.001
    )


    left = 35
    right = 25
    top = 50
    bottom = 25


    available_width = (
        width
        -
        left
        -
        right
    )

    available_height = (
        height
        -
        top
        -
        bottom
    )


    # X/Y 保持相同比例
    scale_x = (
        available_width
        /
        range_x
    )

    scale_y = (
        available_height
        /
        range_y
    )

    scale = min(
        scale_x,
        scale_y
    )


    drawing_width = (
        range_x
        *
        scale
    )

    drawing_height = (
        range_y
        *
        scale
    )


    offset_x = (
        left
        +
        (
            available_width
            -
            drawing_width
        )
        / 2
    )

    offset_y = (
        top
        +
        (
            available_height
            -
            drawing_height
        )
        / 2
    )


    def convert(point):

        px = int(
            offset_x
            +
            (
                point[x_key]
                -
                min_x
            )
            *
            scale
        )

        py = int(
            offset_y
            +
            (
                point[y_key]
                -
                min_y
            )
            *
            scale
        )

        return px, py


    # 轨迹
    for i in range(
        1,
        len(reference)
    ):

        p1 = convert(
            reference[i - 1]
        )

        p2 = convert(
            reference[i]
        )

        cv2.line(
            panel,
            p1,
            p2,
            (0, 0, 0),
            2
        )


    # START
    start_point = convert(
        reference[0]
    )

    cv2.circle(
        panel,
        start_point,
        5,
        (0, 0, 0),
        -1
    )

    cv2.putText(
        panel,
        "START",
        (
            start_point[0] + 6,
            max(
                start_point[1] - 5,
                15
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 0, 0),
        1
    )


    # END
    end_point = convert(
        reference[-1]
    )

    cv2.circle(
        panel,
        end_point,
        6,
        (0, 0, 0),
        2
    )

    cv2.putText(
        panel,
        "END",
        (
            end_point[0] + 6,
            max(
                end_point[1] - 5,
                15
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 0, 0),
        1
    )


# =========================================================
# 5. 画 Angle 曲线
# =========================================================

def draw_angle(
    panel,
    reference,
    key,
    title,
    common_min,
    common_max
):

    panel[:] = 255

    height, width, _ = panel.shape

    cv2.putText(
        panel,
        title,
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )


    if len(reference) < 2:
        return


    values = unwrap_angles(
        [
            point[key]
            for point in reference
        ]
    )


    left = 50
    right = 20
    top = 50
    bottom = 35


    graph_width = (
        width
        -
        left
        -
        right
    )

    graph_height = (
        height
        -
        top
        -
        bottom
    )


    value_range = max(
        common_max
        -
        common_min,
        1
    )


    # 坐标轴
    cv2.line(
        panel,
        (left, top),
        (left, height - bottom),
        (100, 100, 100),
        1
    )

    cv2.line(
        panel,
        (left, height - bottom),
        (width - right, height - bottom),
        (100, 100, 100),
        1
    )


    points = []


    for i, value in enumerate(
        values
    ):

        progress = (
            i
            /
            max(
                len(values) - 1,
                1
            )
        )


        px = int(
            left
            +
            progress
            *
            graph_width
        )


        normalized = (
            value
            -
            common_min
        ) / value_range


        py = int(
            (
                height
                -
                bottom
            )
            -
            normalized
            *
            graph_height
        )


        points.append(
            (px, py)
        )


    for i in range(
        1,
        len(points)
    ):

        cv2.line(
            panel,
            points[i - 1],
            points[i],
            (0, 0, 0),
            2
        )


    cv2.putText(
        panel,
        f"{common_max:.0f}",
        (3, top + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 0, 0),
        1
    )

    cv2.putText(
        panel,
        f"{common_min:.0f}",
        (3, height - bottom),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 0, 0),
        1
    )


    cv2.putText(
        panel,
        "0%",
        (
            left,
            height - 10
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 0, 0),
        1
    )

    cv2.putText(
        panel,
        "100%",
        (
            width - right - 35,
            height - 10
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 0, 0),
        1
    )


# =========================================================
# 6. 加载三条数据
# =========================================================

references = []


for path in REFERENCE_FILES:

    reference = load_reference(
        path
    )

    references.append(
        reference
    )

    if len(reference) > 0:

        duration = (
            reference[-1]["timestamp"]
            -
            reference[0]["timestamp"]
        )

        palm_sizes = [
            point["palm_size"]
            for point in reference
        ]

        print(
            f"{path.name}:"
        )

        print(
            f"  Frames      : {len(reference)}"
        )

        print(
            f"  Duration    : {duration:.2f} sec"
        )

        print(
            f"  Mean palm   : "
            f"{np.mean(palm_sizes):.3f}"
        )

        print()


# =========================================================
# 7. 计算三条共同的范围
# =========================================================

valid_refs = [
    ref
    for ref in references
    if len(ref) > 0
]


# ---------------------------------------------------------
# Raw trajectory 共同范围
# ---------------------------------------------------------

all_raw_x = [
    point["x_raw"]
    for ref in valid_refs
    for point in ref
]

all_raw_y = [
    point["y_raw"]
    for ref in valid_refs
    for point in ref
]


raw_bounds = (
    min(all_raw_x),
    max(all_raw_x),
    min(all_raw_y),
    max(all_raw_y)
)


# ---------------------------------------------------------
# Normalized trajectory 共同范围
# ---------------------------------------------------------

all_norm_x = [
    point["x_norm"]
    for ref in valid_refs
    for point in ref
]

all_norm_y = [
    point["y_norm"]
    for ref in valid_refs
    for point in ref
]


norm_bounds = (
    min(all_norm_x),
    max(all_norm_x),
    min(all_norm_y),
    max(all_norm_y)
)


# ---------------------------------------------------------
# Raw angle 共同范围
# ---------------------------------------------------------

all_angle_raw = []

for ref in valid_refs:

    values = unwrap_angles(
        [
            p["angle_raw"]
            for p in ref
        ]
    )

    all_angle_raw.extend(
        values.tolist()
    )


raw_angle_min = min(
    all_angle_raw
)

raw_angle_max = max(
    all_angle_raw
)


raw_angle_padding = max(
    5,
    (
        raw_angle_max
        -
        raw_angle_min
    )
    *
    0.08
)


raw_angle_min -= (
    raw_angle_padding
)

raw_angle_max += (
    raw_angle_padding
)


# ---------------------------------------------------------
# Relative angle 共同范围
# ---------------------------------------------------------

all_angle_rel = []

for ref in valid_refs:

    values = unwrap_angles(
        [
            p["angle_rel"]
            for p in ref
        ]
    )

    all_angle_rel.extend(
        values.tolist()
    )


rel_angle_min = min(
    all_angle_rel
)

rel_angle_max = max(
    all_angle_rel
)


rel_angle_padding = max(
    5,
    (
        rel_angle_max
        -
        rel_angle_min
    )
    *
    0.08
)


rel_angle_min -= (
    rel_angle_padding
)

rel_angle_max += (
    rel_angle_padding
)


# =========================================================
# 8. 整体窗口
# =========================================================

# 三行：
# Reference 1
# Reference 2
# Reference 3
#
# 四列：
# Raw XY
# Normalized XY
# Raw Angle
# Relative Angle

PANEL_WIDTH = 390
PANEL_HEIGHT = 240


TOTAL_WIDTH = (
    PANEL_WIDTH
    *
    4
)

TOTAL_HEIGHT = (
    PANEL_HEIGHT
    *
    3
)


canvas = np.ones(
    (
        TOTAL_HEIGHT,
        TOTAL_WIDTH,
        3
    ),
    dtype=np.uint8
) * 255


# =========================================================
# 9. 绘制
# =========================================================

for i, reference in enumerate(
    references
):

    y_start = (
        i
        *
        PANEL_HEIGHT
    )

    y_end = (
        y_start
        +
        PANEL_HEIGHT
    )


    # -----------------------------------------------------
    # Raw X/Y
    # -----------------------------------------------------

    panel_raw = canvas[
        y_start:y_end,
        0:PANEL_WIDTH
    ]

    draw_trajectory(
        panel_raw,
        reference,
        "x_raw",
        "y_raw",
        f"Reference {i + 1} - RAW XY",
        raw_bounds
    )


    # -----------------------------------------------------
    # Normalized X/Y
    # -----------------------------------------------------

    panel_norm = canvas[
        y_start:y_end,
        PANEL_WIDTH:
        PANEL_WIDTH * 2
    ]

    draw_trajectory(
        panel_norm,
        reference,
        "x_norm",
        "y_norm",
        "NORMALIZED XY",
        norm_bounds
    )


    # -----------------------------------------------------
    # Raw Angle
    # -----------------------------------------------------

    panel_angle_raw = canvas[
        y_start:y_end,
        PANEL_WIDTH * 2:
        PANEL_WIDTH * 3
    ]

    draw_angle(
        panel_angle_raw,
        reference,
        "angle_raw",
        "RAW ANGLE",
        raw_angle_min,
        raw_angle_max
    )


    # -----------------------------------------------------
    # Relative Angle
    # -----------------------------------------------------

    panel_angle_rel = canvas[
        y_start:y_end,
        PANEL_WIDTH * 3:
        PANEL_WIDTH * 4
    ]

    draw_angle(
        panel_angle_rel,
        reference,
        "angle_rel",
        "RELATIVE ANGLE",
        rel_angle_min,
        rel_angle_max
    )


# =========================================================
# 10. 分隔线
# =========================================================

for column in range(
    1,
    4
):

    x = (
        PANEL_WIDTH
        *
        column
    )

    cv2.line(
        canvas,
        (x, 0),
        (x, TOTAL_HEIGHT),
        (190, 190, 190),
        1
    )


for row in range(
    1,
    3
):

    y = (
        PANEL_HEIGHT
        *
        row
    )

    cv2.line(
        canvas,
        (0, y),
        (TOTAL_WIDTH, y),
        (190, 190, 190),
        1
    )


# =========================================================
# 11. 显示
# =========================================================

WINDOW_NAME = (
    "AdaptiveSkill v1.2 - Reference Comparison"
)


cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)


cv2.resizeWindow(
    WINDOW_NAME,
    TOTAL_WIDTH,
    TOTAL_HEIGHT
)


cv2.imshow(
    WINDOW_NAME,
    canvas
)


print(
    "========================================="
)

print(
    "LEFT 1 : Raw X/Y"
)

print(
    "LEFT 2 : Normalized X/Y"
)

print(
    "RIGHT 1: Raw Angle"
)

print(
    "RIGHT 2: Relative Angle"
)

print(
    "Press Q to close."
)

print(
    "========================================="
)


while True:

    key = (
        cv2.waitKey(0)
        &
        0xFF
    )

    if key == ord("q"):
        break


cv2.destroyAllWindows()