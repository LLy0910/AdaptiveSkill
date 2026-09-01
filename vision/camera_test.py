import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera cannot be opened.")
    exit()

print("Camera opened successfully.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Cannot read frame.")
        break

    # 镜像画面，操作起来更自然
    frame = cv2.flip(frame, 1)

    cv2.putText(
        frame,
        "AdaptiveSkill - Camera Test",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("AdaptiveSkill", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()