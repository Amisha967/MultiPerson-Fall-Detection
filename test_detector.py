
import cv2
from person_detector import PersonDetector

VIDEO_PATH = "test_cctv.mp4"

print("Starting person detection...")

detector = PersonDetector()

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

print("Video opened successfully!")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    detections = detector.detect(frame)

    for i, person in enumerate(detections, start=1):

        x1, y1, x2, y2 = map(int, person["bbox"])

        cv2.rectangle(
            frame, (x1, y1), (x2, y2),
            (0, 255, 0), 2
        )

        cv2.putText(
            frame,
            f"Person {i}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        for x, y, confidence in person["keypoints"]:

            if confidence > 0.3:
                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    3,
                    (0, 0, 255),
                    -1
                )

    cv2.imshow("Multi-Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Detection completed!")
