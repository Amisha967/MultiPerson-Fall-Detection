
import time
import numpy as np


class FallClassifier:

    def __init__(
        self,
        horizontal_ar_threshold=0.6,
        angle_threshold=140,
        minimum_fall_duration=1.0,
        keypoint_confidence=0.3
    ):

        self.horizontal_ar_threshold = horizontal_ar_threshold
        self.angle_threshold = angle_threshold
        self.minimum_fall_duration = minimum_fall_duration
        self.keypoint_confidence = keypoint_confidence

        self.person_states = {}

    @staticmethod
    def calculate_angle(shoulder, hip, knee):

        a = np.array(shoulder, dtype=float)
        b = np.array(hip, dtype=float)
        c = np.array(knee, dtype=float)

        if np.array_equal(a, b) or np.array_equal(b, c):
            return 180.0

        vector_ba = a - b
        vector_bc = c - b

        radians = (
            np.arctan2(vector_bc[1], vector_bc[0])
            - np.arctan2(vector_ba[1], vector_ba[0])
        )

        angle = abs(np.degrees(radians))

        if angle > 180:
            angle = 360 - angle

        return float(angle)

    def _get_hip_angle(self, keypoints):

        required_indices = [5, 11, 13]

        for index in required_indices:

            if index >= len(keypoints):
                return None

            x, y, confidence = keypoints[index][:3]

            if (
                x <= 0
                or y <= 0
                or confidence <= self.keypoint_confidence
            ):
                return None

        shoulder = keypoints[5][:2]
        hip = keypoints[11][:2]
        knee = keypoints[13][:2]

        return self.calculate_angle(
            shoulder,
            hip,
            knee
        )

    def update(
        self,
        person_id,
        keypoints,
        aspect_ratio,
        is_full_body,
        current_time=None
    ):

        if current_time is None:
            current_time = time.time()

        if person_id not in self.person_states:

            self.person_states[person_id] = {
                "potential_fall": False,
                "fall_start_time": None,
                "fall_detected": False
            }

        state = self.person_states[person_id]

        angle = None

        # Update the fall state only for sufficiently visible bodies.
        if is_full_body:

            angle = self._get_hip_angle(keypoints)

            low_aspect_ratio = (
                aspect_ratio is not None
                and aspect_ratio < self.horizontal_ar_threshold
            )

            low_hip_angle = (
                angle is not None
                and angle < self.angle_threshold
            )

            potential_fall = (
                low_aspect_ratio or low_hip_angle
            )

            if potential_fall:

                if not state["potential_fall"]:

                    state["potential_fall"] = True
                    state["fall_start_time"] = current_time

            else:

                state["potential_fall"] = False
                state["fall_start_time"] = None
                state["fall_detected"] = False

        # Calculate duration using the video timeline.
        duration = 0.0

        if (
            state["potential_fall"]
            and state["fall_start_time"] is not None
        ):

            duration = max(
                0.0,
                current_time - state["fall_start_time"]
            )

            if duration >= self.minimum_fall_duration:
                state["fall_detected"] = True

        if state["fall_detected"]:
            status = "Fall"

        elif state["potential_fall"]:
            status = "Potential Fall"

        else:
            status = "Normal"

        return {
            "status": status,
            "fall_detected": state["fall_detected"],
            "potential_fall": state["potential_fall"],
            "angle": angle,
            "duration": float(duration)
        }
