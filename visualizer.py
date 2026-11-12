
import cv2


SKELETON = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]


class Visualizer:

    def __init__(self, keypoint_confidence=0.3):

        self.keypoint_confidence = keypoint_confidence

    def draw(self, frame, people):

        for person in people:

            x1, y1, x2, y2 = person["box"]

            person_id = person["person_id"]

            fall = person["fall_result"]

            keypoints = person["keypoints"]

            status = fall["status"]

            if status == "Fall":
                color = (0, 0, 255)

            elif status == "Potential Fall":
                color = (0, 165, 255)

            else:
                color = (0, 255, 0)

            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Status label
            label = f"ID {person_id}: {status}"

            duration = fall.get("duration", 0.0)

            if duration is None:
                duration = 0.0

            if fall["potential_fall"]:

                label += f" ({duration:.1f}s)"

            label_y = max(y1 - 10, 20)

            cv2.putText(
                frame,
                label,
                (max(x1, 5), label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            # Draw skeleton
            for start, end in SKELETON:

                if (
                    start >= len(keypoints)
                    or end >= len(keypoints)
                ):
                    continue

                x_start, y_start, conf_start = keypoints[start][:3]

                x_end, y_end, conf_end = keypoints[end][:3]

                if (
                    conf_start > self.keypoint_confidence
                    and conf_end > self.keypoint_confidence
                    and x_start > 0
                    and y_start > 0
                    and x_end > 0
                    and y_end > 0
                ):

                    cv2.line(
                        frame,
                        (int(x_start), int(y_start)),
                        (int(x_end), int(y_end)),
                        color,
                        2
                    )

            # Draw keypoints
            for point in keypoints:

                x, y, confidence = point[:3]

                if (
                    confidence > self.keypoint_confidence
                    and x > 0
                    and y > 0
                ):

                    cv2.circle(
                        frame,
                        (int(x), int(y)),
                        3,
                        (255, 255, 0),
                        -1
                    )

            # Display hip angle
            if fall["angle"] is not None:

                cv2.putText(
                    frame,
                    f"Angle: {fall['angle']:.1f}",
                    (max(x1, 5), min(y2 + 20, frame.shape[0] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1
                )

        return frame
