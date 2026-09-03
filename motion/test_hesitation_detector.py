from hesitation_detector import HesitationDetector


FPS = 30
DT = 1.0 / FPS


def run_test(
    name,
    samples
):
    detector = HesitationDetector()

    events = []

    states = []

    for timestamp, speed, progress in samples:

        result = detector.update(
            timestamp=timestamp,
            speed=speed,
            progress=progress,
            tracking=True
        )

        states.append(
            result["state"]
        )

        if result["event"]:

            events.append({
                "time": timestamp,
                "progress": progress,
                "duration":
                    result[
                        "candidate_duration"
                    ]
            })

    print()
    print(
        "========================================"
    )

    print(name)

    print(
        "========================================"
    )

    print(
        "Hesitation events:",
        len(events)
    )

    if events:

        for event in events:

            print(
                "Detected at:",
                round(
                    event["time"],
                    3
                ),
                "sec"
            )

            print(
                "Progress:",
                round(
                    event["progress"],
                    3
                )
            )

            print(
                "Candidate duration:",
                event["duration"],
                "sec"
            )

    print(
        "Final state:",
        states[-1]
    )


# ========================================================
# TEST 1
# NORMAL MOTION
#
# 速度始终高于 threshold。
# 应该没有 hesitation。
# ========================================================

normal_samples = []

t = 0.0

for i in range(90):

    progress = (
        0.20
        +
        0.60
        *
        i
        /
        89
    )

    speed = 1.30

    normal_samples.append(
        (
            t,
            speed,
            progress
        )
    )

    t += DT


# ========================================================
# TEST 2
# TRUE HESITATION
#
# 做到 50% 时完全停住约 1 秒。
# 应该检测到 hesitation。
# ========================================================

hesitation_samples = []

t = 0.0


# 正常移动到 50%
for i in range(45):

    progress = (
        0.20
        +
        0.30
        *
        i
        /
        44
    )

    hesitation_samples.append(
        (
            t,
            1.30,
            progress
        )
    )

    t += DT


# 在 50% 停住 1 秒
for i in range(30):

    hesitation_samples.append(
        (
            t,
            0.10,
            0.50
        )
    )

    t += DT


# 停顿后继续移动
for i in range(45):

    progress = (
        0.50
        +
        0.30
        *
        i
        /
        44
    )

    hesitation_samples.append(
        (
            t,
            1.30,
            progress
        )
    )

    t += DT


# ========================================================
# TEST 3
# SLOW BUT CONTINUOUS
#
# 速度低于 threshold，
# 但 progress 一直前进。
#
# 应该是 LOW_SPEED，
# 但不能触发 hesitation。
# ========================================================

slow_samples = []

t = 0.0

for i in range(180):

    progress = (
        0.20
        +
        0.60
        *
        i
        /
        179
    )

    speed = 0.50

    slow_samples.append(
        (
            t,
            speed,
            progress
        )
    )

    t += DT


# ========================================================
# TEST 4
# SHORT NATURAL PAUSE
#
# 只停 0.30 秒，
# 小于 0.525 秒 threshold。
#
# 不应该触发 hesitation。
# ========================================================

short_pause_samples = []

t = 0.0


for i in range(30):

    progress = (
        0.20
        +
        0.25
        *
        i
        /
        29
    )

    short_pause_samples.append(
        (
            t,
            1.30,
            progress
        )
    )

    t += DT


# 0.30 sec pause
for i in range(9):

    short_pause_samples.append(
        (
            t,
            0.10,
            0.45
        )
    )

    t += DT


for i in range(30):

    progress = (
        0.45
        +
        0.30
        *
        i
        /
        29
    )

    short_pause_samples.append(
        (
            t,
            1.30,
            progress
        )
    )

    t += DT


# ========================================================
# RUN
# ========================================================

print()
print(
    "AdaptiveSkill - Hesitation Logic Test"
)


run_test(
    "TEST 1 - NORMAL MOTION",
    normal_samples
)

run_test(
    "TEST 2 - TRUE HESITATION",
    hesitation_samples
)

run_test(
    "TEST 3 - SLOW CONTINUOUS",
    slow_samples
)

run_test(
    "TEST 4 - SHORT PAUSE",
    short_pause_samples
)


print()
print(
    "========================================"
)

print(
    "EXPECTED"
)

print(
    "========================================"
)

print(
    "Normal Motion     -> 0 events"
)

print(
    "True Hesitation   -> 1 event"
)

print(
    "Slow Continuous   -> 0 events"
)

print(
    "Short Pause       -> 0 events"
)

print()