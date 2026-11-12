import time
from collections import deque

import numpy as np


class FallClassifier:
    """
    Heuristic fall detector for one tracked person.

    A fall event is confirmed when all of the following hold:
      1. The hips drop quickly relative to the person's body height, at a
         plausible speed (very fast jumps are treated as tracking errors).
      2. The torso becomes close to horizontal and the bounding box is low.
      3. That posture persists for `minimum_fall_duration` seconds and starts
         within `drop_window` seconds of the hip drop.

    Torso angle and aspect ratio are smoothed with a rolling median so that
    single bad keypoint frames do not reset the posture timer.

    A confirmed fall stays latched until the person has been upright for
    `recovery_time` seconds. A cooldown stops a new fall from firing right
    after a reset.

    This is a prototype heuristic, not a medically validated detector.
    """

    def __init__(
        self,
        horizontal_ar_threshold=0.8,   # bbox height / width; below this = low box
        angle_threshold=55.0,          # torso angle from vertical (deg); above this = horizontal
        minimum_fall_duration=0.8,     # seconds the horizontal posture must persist
        keypoint_confidence=0.3,
        drop_ratio_threshold=0.30,     # hip drop as a fraction of body height
        max_drop_speed=5.0,            # body heights per second; faster = tracking jump
        drop_lookback=(0.1, 1.0),      # reference sample age window in seconds
        drop_window=2.0,               # posture must start within this many seconds of a drop
        posture_gap_tolerance=0.6,     # seconds a bad frame may interrupt the posture
        smooth_window=7,               # frames used for the rolling median
        upright_angle=30.0,            # torso angle below which the person counts as upright
        recovery_time=2.0,             # seconds upright before a fall latch clears
        cooldown=3.0,                  # seconds after clearing before a new fall can fire
        min_body_height=10.0,          # pixels
        verbose=False,                 # print per-frame condition flags when True
    ):
        self.horizontal_ar_threshold = horizontal_ar_threshold
        self.angle_threshold = angle_threshold
        self.minimum_fall_duration = minimum_fall_duration
        self.keypoint_confidence = keypoint_confidence
        self.drop_ratio_threshold = drop_ratio_threshold
        self.max_drop_speed = max_drop_speed
        self.drop_lookback = drop_lookback
        self.drop_window = drop_window
        self.posture_gap_tolerance = posture_gap_tolerance
        self.smooth_window = smooth_window
        self.upright_angle = upright_angle
        self.recovery_time = recovery_time
        self.cooldown = cooldown
        self.min_body_height = min_body_height
        self.verbose = verbose
        self.person_states = {}

    def _new_state(self):
        return {
            "history": deque(maxlen=60),                    # (time, hip_y, body_height)
            "angle_hist": deque(maxlen=self.smooth_window),
            "ar_hist": deque(maxlen=self.smooth_window),
            "last_drop_time": None,
            "horizontal_start": None,
            "last_posture_time": None,
            "fall_detected": False,
            "upright_start": None,
            "cleared_at": None,
        }

    def _point(self, keypoints, idx):
        if idx >= len(keypoints):
            return None
        x, y, conf = keypoints[idx][:3]
        if x <= 0 or y <= 0 or conf < self.keypoint_confidence:
            return None
        return np.array([x, y], dtype=float)

    def _measure(self, keypoints):
        shoulders = [p for p in (self._point(keypoints, i) for i in (5, 6)) if p is not None]
        hips = [p for p in (self._point(keypoints, i) for i in (11, 12)) if p is not None]

        if not hips:
            return None, None, None

        hip_center = np.mean(hips, axis=0)
        hip_y = float(hip_center[1])

        # Torso angle from vertical: 0 degrees = upright, 90 degrees = horizontal.
        torso_angle = None
        if shoulders:
            shoulder_center = np.mean(shoulders, axis=0)
            dx = abs(float(hip_center[0] - shoulder_center[0]))
            dy = abs(float(hip_center[1] - shoulder_center[1]))
            torso_angle = float(np.degrees(np.arctan2(dx, max(dy, 1e-6))))

        # Body height from stable central landmarks. Wrists are excluded because
        # they move independently and distort the estimate.
        vertical_points = [
            p for p in (self._point(keypoints, i) for i in (0, 5, 6, 11, 12, 13, 14, 15, 16))
            if p is not None
        ]
        body_height = None
        if len(vertical_points) >= 4:
            ys = [float(p[1]) for p in vertical_points]
            body_height = max(ys) - min(ys)

        return hip_y, body_height, torso_angle

    def _is_rapid_drop(self, state, now, hip_y, body_height):
        """Compare the current hip position with the oldest sample in the lookback window."""
        min_age, max_age = self.drop_lookback

        # Oldest first, so the reference is as far back as max_age allows.
        ref = None
        for t, y, h in state["history"]:
            age = now - t
            if age > max_age:
                continue
            if age >= min_age:
                ref = (t, y, h)
                break

        if ref is None:
            return False

        t0, y0, h0 = ref
        dt = now - t0
        reference_height = max(h0, body_height)
        drop_ratio = (hip_y - y0) / reference_height   # positive = moving down the image
        speed = drop_ratio / dt                         # body heights per second

        return drop_ratio >= self.drop_ratio_threshold and speed <= self.max_drop_speed

    def update(
        self,
        person_id,
        keypoints,
        aspect_ratio,
        is_full_body,
        current_time=None,
    ):
        if current_time is None:
            current_time = time.time()

        state = self.person_states.setdefault(person_id, self._new_state())

        duration = 0.0
        drop_now = False

        # 1. Measure this frame and detect a rapid, plausible hip drop.
        if is_full_body:
            hip_y, body_height, angle_raw = self._measure(keypoints)
            if angle_raw is not None:
                state["angle_hist"].append(angle_raw)
            if hip_y is not None and body_height is not None and body_height > self.min_body_height:
                drop_now = self._is_rapid_drop(state, current_time, hip_y, body_height)
                state["history"].append((current_time, hip_y, body_height))
                if drop_now:
                    state["last_drop_time"] = current_time

        if aspect_ratio is not None:
            state["ar_hist"].append(aspect_ratio)

        # Smoothed posture signals: a single bad frame cannot flip them.
        torso_angle = float(np.median(state["angle_hist"])) if state["angle_hist"] else None
        ar = float(np.median(state["ar_hist"])) if state["ar_hist"] else None

        drop_recent = (
            state["last_drop_time"] is not None
            and current_time - state["last_drop_time"] <= self.drop_window
        )

        # 2. Posture conditions. Torso angle is only meaningful on full-body frames.
        torso_horizontal = (
            is_full_body and torso_angle is not None and torso_angle >= self.angle_threshold
        )
        low_pose = ar is not None and ar < self.horizontal_ar_threshold
        fall_like_pose = torso_horizontal and low_pose

        # 3. Track how long the horizontal posture has lasted, tolerating brief gaps.
        if fall_like_pose:
            state["last_posture_time"] = current_time
            if state["horizontal_start"] is None and drop_recent:
                state["horizontal_start"] = current_time
        elif state["horizontal_start"] is not None:
            last = state["last_posture_time"]
            if last is None or current_time - last > self.posture_gap_tolerance:
                state["horizontal_start"] = None

        # 4. Confirm the fall once the posture has lasted long enough.
        if state["horizontal_start"] is not None:
            duration = current_time - state["horizontal_start"]
            if (
                duration >= self.minimum_fall_duration
                and not state["fall_detected"]
                and (state["cleared_at"] is None or current_time - state["cleared_at"] >= self.cooldown)
            ):
                state["fall_detected"] = True
                state["upright_start"] = None

        # 5. Clear a latched fall once the person has been upright long enough.
        if state["fall_detected"]:
            upright = is_full_body and torso_angle is not None and torso_angle <= self.upright_angle
            if upright:
                if state["upright_start"] is None:
                    state["upright_start"] = current_time
                if current_time - state["upright_start"] >= self.recovery_time:
                    state["fall_detected"] = False
                    state["horizontal_start"] = None
                    state["upright_start"] = None
                    state["cleared_at"] = current_time
            else:
                state["upright_start"] = None

        if state["fall_detected"]:
            status = "Fall"
        elif state["horizontal_start"] is not None:
            status = "Potential Fall"
        else:
            status = "Normal"

        if self.verbose:
            print(
                f"[DBG] id={person_id} t={current_time:.2f} torso={torso_angle} "
                f"AR={ar} low={low_pose} horiz={torso_horizontal} "
                f"drop_now={drop_now} drop_recent={drop_recent} "
                f"duration={duration:.2f} status={status}"
            )

        return {
            "status": status,
            "fall_detected": bool(state["fall_detected"]),
            "potential_fall": state["horizontal_start"] is not None,
            "angle": torso_angle,
            "duration": float(duration),
        }
