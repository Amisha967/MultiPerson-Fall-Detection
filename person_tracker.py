
import time
import numpy as np


class PersonTracker:

    REQUIRED_KEYPOINTS = [0, 5, 6, 11, 12, 15, 16]

    def __init__(
        self,
        distance_threshold=150,
        keypoint_confidence=0.3,
        min_full_body_keypoints=5,
        history_cleanup_interval=5
    ):

        self.distance_threshold = distance_threshold
        self.keypoint_confidence = keypoint_confidence
        self.min_full_body_keypoints = min_full_body_keypoints
        self.history_cleanup_interval = history_cleanup_interval

        self.tracks = {}
        self.next_person_id = 1
        self.last_cleanup_time = 0

    def _get_pose_info(self, keypoints):

        keypoints = np.asarray(keypoints)
        visible_points = []

        for index in self.REQUIRED_KEYPOINTS:

            if index >= len(keypoints):
                continue

            x, y, confidence = keypoints[index][:3]

            if (
                x > 0
                and y > 0
                and confidence > self.keypoint_confidence
            ):
                visible_points.append([x, y])

        is_full_body = (
            len(visible_points) >= self.min_full_body_keypoints
        )

        if len(visible_points) < 2:
            return None

        points = np.array(visible_points)

        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)

        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            return None

        box = tuple(map(int, [x1, y1, x2, y2]))

        center = (
            int((x1 + x2) / 2),
            int((y1 + y2) / 2)
        )

        aspect_ratio = height / width

        return {
            "box": box,
            "center": center,
            "aspect_ratio": aspect_ratio,
            "is_full_body": is_full_body
        }

    def update(self, detections, current_time=None):

        if current_time is None:
            current_time = time.time()

        processed = []

        for detection in detections:

            pose_info = self._get_pose_info(
                detection["keypoints"]
            )

            if pose_info is None:
                continue

            processed.append({
                **detection,
                **pose_info
            })

        # Find possible matches between existing tracks and detections.
        possible_matches = []

        for detection_index, detection in enumerate(processed):

            cx, cy = detection["center"]

            for person_id, track in self.tracks.items():

                tx, ty = track["center"]

                distance = np.hypot(
                    cx - tx,
                    cy - ty
                )

                if distance < self.distance_threshold:

                    possible_matches.append(
                        (distance, detection_index, person_id)
                    )

        # Match nearest detections first.
        possible_matches.sort(key=lambda item: item[0])

        matched_detections = set()
        matched_persons = set()
        assignments = {}

        for distance, detection_index, person_id in possible_matches:

            if detection_index in matched_detections:
                continue

            if person_id in matched_persons:
                continue

            assignments[detection_index] = person_id

            matched_detections.add(detection_index)
            matched_persons.add(person_id)

        results = []

        for index, detection in enumerate(processed):

            person_id = assignments.get(index)

            if person_id is None:

                # Create a new ID only when sufficient body
                # keypoints are visible.
                if not detection["is_full_body"]:
                    continue

                person_id = self.next_person_id
                self.next_person_id += 1

                self.tracks[person_id] = {
                    "center": detection["center"],
                    "last_seen": current_time
                }

            else:

                self.tracks[person_id]["center"] = (
                    detection["center"]
                )

                self.tracks[person_id]["last_seen"] = current_time

            results.append({
                **detection,
                "person_id": person_id
            })

        # Remove expired tracks.
        if (
            current_time - self.last_cleanup_time
            > self.history_cleanup_interval
        ):

            expired_ids = [
                person_id
                for person_id, track in self.tracks.items()
                if (
                    current_time - track["last_seen"]
                    > self.history_cleanup_interval
                )
            ]

            for person_id in expired_ids:
                del self.tracks[person_id]

            self.last_cleanup_time = current_time

        return results
