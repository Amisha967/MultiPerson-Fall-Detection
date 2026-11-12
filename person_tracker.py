import time
import numpy as np


class PersonTracker:
    """Adapter that preserves the project's existing tracker interface.
    Person identities are supplied by YOLO's persistent ByteTrack tracker.
    """

    REQUIRED_KEYPOINTS = [0,  5, 6, 11, 12, 15, 16]

    def __init__(self, distance_threshold=150, keypoint_confidence=0.3,
                 min_full_body_keypoints=5, history_cleanup_interval=5):
        self.keypoint_confidence = keypoint_confidence
        self.min_full_body_keypoints = min_full_body_keypoints
        self.history_cleanup_interval = history_cleanup_interval
        self.last_seen = {}

    def _pose_info(self, keypoints):
        keypoints = np.asarray(keypoints)
        points = []
        for idx in self.REQUIRED_KEYPOINTS:
            if idx >= len(keypoints):
                continue
            x, y, conf = keypoints[idx][:3]
            if x > 0 and y > 0 and conf >= self.keypoint_confidence:
                points.append((x, y))

        is_full_body = len(points) >= self.min_full_body_keypoints
        if len(points) < 2:
            return None

        points = np.asarray(points, dtype=float)
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        width, height = x2 - x1, y2 - y1
        if width <= 1 or height <= 1:
            return None

        return {
            "box": tuple(map(int, (x1, y1, x2, y2))),
            "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
            "aspect_ratio": float(height / width),
            "is_full_body": bool(is_full_body)
        }

    def update(self, detections, current_time=None):
        if current_time is None:
            current_time = time.time()

        tracked_people = []
        for detection in detections:
            person_id = detection.get("track_id")
            # Do not create a guessed identity if ByteTrack has not assigned one.
            if person_id is None:
                continue

            pose_info = self._pose_info(detection["keypoints"])
            if pose_info is None:
                continue

            person_id = int(person_id)
            self.last_seen[person_id] = current_time
            tracked_people.append({
                **detection,
                **pose_info,
                "person_id": person_id
            })

        expired = [
            pid for pid, seen in self.last_seen.items()
            if current_time - seen > self.history_cleanup_interval
        ]
        for pid in expired:
            del self.last_seen[pid]

        return tracked_people
