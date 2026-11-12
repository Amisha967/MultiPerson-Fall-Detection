
import cv2
import time
from pathlib import Path

from person_detector import PersonDetector
from person_tracker import PersonTracker
from fall_classifier import FallClassifier
from visualizer import Visualizer


# ---------------- CONFIGURATION ----------------

VIDEO_PATH = "test_cctv.mp4"

MODEL_PATH = "yolov8n-pose.pt"

DETECTION_CONFIDENCE = 0.35

IMAGE_SIZE = 640

WINDOW_NAME = "Multi-Person Fall Detection"


# ---------------- MAIN FUNCTION ----------------

def main():

    video_path = Path(VIDEO_PATH).resolve()

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    print("\nOpening video:", video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        raise RuntimeError(
            "OpenCV could not open the video."
        )

    # Read video properties.
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    video_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if video_fps <= 0:
        video_fps = 25.0

    expected_duration = (
        total_frames / video_fps
        if total_frames > 0
        else 0
    )

    print("\nVIDEO INFORMATION")
    print("-------------------------")
    print("Resolution:", width, "x", height)
    print("FPS:", video_fps)
    print("Total frames:", total_frames)
    print("Duration:", round(expected_duration, 2), "seconds")
    print("-------------------------")

    # Initialize detector.
    detector = PersonDetector(
        model_path=MODEL_PATH,
        confidence=DETECTION_CONFIDENCE,
        image_size=IMAGE_SIZE
    )

    # Initialize tracker.
    tracker = PersonTracker(
        distance_threshold=150,
        keypoint_confidence=0.3,
        min_full_body_keypoints=5,
        history_cleanup_interval=5
    )

    # Initialize fall classifier.
    classifier = FallClassifier(
        horizontal_ar_threshold=0.6,
        angle_threshold=140,
        minimum_fall_duration=1.0,
        keypoint_confidence=0.3
    )

    # Initialize visualizer.
    visualizer = Visualizer()

    frame_count = 0

    fall_events = set()

    processing_start = time.time()

    stop_reason = "Unknown"

    print("\nVideo opened successfully!")
    print("Processing all frames...")
    print("Press Q or ESC to stop manually.\n")

    # ---------------- VIDEO LOOP ----------------

    while True:

        ret, frame = cap.read()

        if not ret:

            stop_reason = "End of video or decoder could not read another frame."

            print("\nFinished reading video.")
            break

        frame_count += 1

        # Use the video's timeline rather than computer processing time.
        video_time = frame_count / video_fps

        # 1. Detect people.
        detections = detector.detect(frame)

        # 2. Track people.
        tracked_people = tracker.update(
            detections,
            current_time=video_time
        )

        people_to_draw = []

        # 3. Classify each person independently.
        for person in tracked_people:

            fall_result = classifier.update(
                person_id=person["person_id"],
                keypoints=person["keypoints"],
                aspect_ratio=person["aspect_ratio"],
                is_full_body=person["is_full_body"],
                current_time=video_time
            )

            person["fall_result"] = fall_result

            if fall_result["fall_detected"]:

                fall_events.add(
                    person["person_id"]
                )

            people_to_draw.append(person)

        # 4. Visualize detections and fall status.
        output_frame = visualizer.draw(
            frame,
            people_to_draw
        )

        # 5. Display video information.
        cv2.putText(
            output_frame,
            f"Frame: {frame_count}/{total_frames}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output_frame,
            f"Video time: {video_time:.1f}s",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output_frame,
            f"People detected: {len(detections)}",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output_frame,
            f"Confirmed fall IDs: {len(fall_events)}",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )

        # 6. Display frame.
        cv2.imshow(
            WINDOW_NAME,
            output_frame
        )

        # 7. Print processing progress.
        if frame_count % 100 == 0:

            fall_count = sum(
                1
                for person in people_to_draw
                if person["fall_result"]["fall_detected"]
            )

            print(
                f"Frame {frame_count}/{total_frames} | "
                f"Video time: {video_time:.1f}s | "
                f"People: {len(detections)} | "
                f"Tracked: {len(tracked_people)} | "
                f"Falls currently visible: {fall_count}"
            )

        # 8. Playback delay.
        # No frames are skipped by this delay.
        delay = max(
            1,
            int(1000 / video_fps)
        )

        key = cv2.waitKey(delay) & 0xFF

        if key == ord("q") or key == 27:

            stop_reason = "Manually stopped by user."

            print("\nManual stop requested.")
            break

    # ---------------- RELEASE RESOURCES ----------------

    cap.release()

    cv2.destroyAllWindows()

    processing_end = time.time()

    processing_duration = (
        processing_end - processing_start
    )

    print("\n========== PROCESSING SUMMARY ==========")

    print("Expected frames:", total_frames)

    print("Successfully processed:", frame_count)

    print(
        "Video duration:",
        round(frame_count / video_fps, 2),
        "seconds"
    )

    print(
        "Processing time:",
        round(processing_duration, 2),
        "seconds"
    )

    print(
        "Confirmed fall IDs:",
        sorted(fall_events)
    )

    print("Stop reason:", stop_reason)

    if total_frames > 0:

        completion = (
            frame_count / total_frames
        ) * 100

        print(
            "Video processing completion:",
            round(completion, 2),
            "%"
        )

    print("========================================")

    print("\nFall detection program finished.")


if __name__ == "__main__":
    main()
