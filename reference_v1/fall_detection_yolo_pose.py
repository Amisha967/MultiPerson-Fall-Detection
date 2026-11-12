
#!/usr/bin/env python3
"""
Multi-Person Fall Detection - Customized v2

Features:
- Existing YOLOv8n-Pose weights
- BoT-SORT multi-person tracking
- EMA keypoint smoothing
- Biomechanical fall scoring
- Occlusion persistence
- Scene-cut detection
- Annotated video output
- Frame-level evaluation only with independent ground-truth windows
"""

import argparse
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

SKELETON_EDGES = [
    (5, 6), (5, 11), (6, 12), (11, 12),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16)
]


def select_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pose_model(preferred_model):
    candidates = [
        preferred_model,
        "yolov8n-pose.pt",
        "yolo11n-pose.pt",
        "yolo11s-pose.pt",
        "yolo26n-pose.pt"
    ]

    seen = set()

    for weight in candidates:
        if weight in seen:
            continue

        seen.add(weight)

        try:
            print(f"[INFO] Loading pose model: {weight}")
            return YOLO(weight), weight
        except Exception as error:
            print(f"[WARN] Could not load {weight}: {error}")

    raise RuntimeError("Could not load any YOLO pose model.")


def parse_fall_windows(window_str):
    if not window_str:
        return []

    windows = []

    for part in window_str.split(","):
        part = part.strip()

        if ":" in part:
            start, end = part.split(":", 1)
            windows.append((float(start), float(end)))

    return windows


def merge_frame_intervals(
    frames,
    fps,
    max_gap_sec=1.2,
    min_dur_sec=0.45
):
    if not frames:
        return []

    sorted_frames = sorted(set(frames))
    max_gap = int(fps * max_gap_sec)
    min_length = max(int(fps * min_dur_sec), 3)

    intervals = []
    start_frame = sorted_frames[0]
    previous_frame = sorted_frames[0]

    for frame in sorted_frames[1:]:
        if frame - previous_frame <= max_gap:
            previous_frame = frame
        else:
            if previous_frame - start_frame + 1 >= min_length:
                intervals.append((
                    round(start_frame / fps, 2),
                    round(previous_frame / fps, 2)
                ))

            start_frame = frame
            previous_frame = frame

    if previous_frame - start_frame + 1 >= min_length:
        intervals.append((
            round(start_frame / fps, 2),
            round(previous_frame / fps, 2)
        ))

    return intervals


def get_valid_point(
    keypoints,
    confidences,
    indices,
    confidence_threshold=0.22
):
    valid_points = [
        keypoints[index]
        for index in indices
        if index < len(confidences)
        and confidences[index] >= confidence_threshold
        and keypoints[index][0] > 0
        and keypoints[index][1] > 0
    ]

    if not valid_points:
        return None

    return np.mean(valid_points, axis=0)


class PersonFallTracker:

    def __init__(self, track_id, fps):
        self.track_id = track_id
        self.fps = max(fps, 15.0)

        self.max_height = 0.0
        self.baseline_height = 0.0

        self.standing_heights = deque(
            maxlen=int(self.fps * 2.5)
        )

        self.upper_y_history = deque(
            maxlen=int(self.fps)
        )

        self.ar_history = deque(maxlen=6)
        self.angle_history = deque(maxlen=6)

        self.smoothed_kpts = None
        self.last_bbox = None
        self.last_kpts_conf = None
        self.last_metrics = None

        self.missed_frames = 0
        self.fall_score = 0
        self.is_fallen = False

    def update(
        self,
        bbox,
        keypoints,
        confidences,
        frame_idx
    ):
        self.missed_frames = 0
        self.last_bbox = bbox.copy()
        self.last_kpts_conf = confidences.copy()

        # Exponential Moving Average smoothing
        if self.smoothed_kpts is None:
            self.smoothed_kpts = keypoints.copy()
        else:
            mask = (
                (confidences >= 0.20)[:, None]
                & (keypoints > 0)
            )

            self.smoothed_kpts = np.where(
                mask,
                0.65 * keypoints + 0.35 * self.smoothed_kpts,
                self.smoothed_kpts
            )

        points = self.smoothed_kpts

        x1, y1, x2, y2 = bbox

        box_width = max(float(x2 - x1), 1.0)
        box_height = max(float(y2 - y1), 1.0)

        raw_aspect_ratio = box_width / box_height

        self.ar_history.append(raw_aspect_ratio)

        aspect_ratio = float(np.mean(self.ar_history))

        upper_point = get_valid_point(
            points,
            confidences,
            [L_SHOULDER, R_SHOULDER, NOSE, L_EAR, R_EAR]
        )

        hip_point = get_valid_point(
            points,
            confidences,
            [L_HIP, R_HIP]
        )

        lower_point = get_valid_point(
            points,
            confidences,
            [L_ANKLE, R_ANKLE, L_KNEE, R_KNEE]
        )

        if upper_point is None:
            upper_point = np.array([
                (x1 + x2) / 2,
                y1 + 0.2 * box_height
            ])

        if hip_point is None:
            hip_point = np.array([
                (x1 + x2) / 2,
                y1 + 0.55 * box_height
            ])

        if lower_point is None:
            lower_point = np.array([
                (x1 + x2) / 2,
                y2
            ])

        dx = float(upper_point[0] - hip_point[0])
        dy = float(hip_point[1] - upper_point[1])

        raw_angle = math.degrees(
            math.atan2(abs(dx), max(dy, 1e-3))
        )

        self.angle_history.append(raw_angle)
        torso_angle = float(np.mean(self.angle_history))

        # Update standing-height baseline when upright
        if torso_angle < 28.0 and aspect_ratio < 0.78:

            self.standing_heights.append(box_height)

            self.max_height = max(
                self.max_height,
                box_height
            )

            self.baseline_height = (
                float(np.percentile(self.standing_heights, 75))
                if len(self.standing_heights) >= 4
                else self.max_height
            )

        elif self.baseline_height == 0.0:

            self.baseline_height = max(
                box_height,
                box_width * 1.35
            )

        reference_height = max(
            self.baseline_height,
            self.max_height,
            1.0
        )

        height_ratio = box_height / reference_height

        self.upper_y_history.append(
            (frame_idx, float(upper_point[1]))
        )

        vertical_drop = 0.0

        if len(self.upper_y_history) >= 3:
            _, oldest_y = self.upper_y_history[0]

            vertical_drop = (
                float(upper_point[1]) - oldest_y
            ) / reference_height

        # Biomechanical fall criteria
        posture_horizontal = (
            torso_angle >= 45.0
            or aspect_ratio >= 0.95
        )

        height_collapsed = (
            height_ratio <= 0.68
            or aspect_ratio >= 1.15
        )

        rapid_descent = vertical_drop >= 0.18

        confirmation_threshold = max(
            int(self.fps * 0.25),
            4
        )

        maximum_score = int(self.fps * 2.0)

        if (
            posture_horizontal and height_collapsed
        ) or (
            rapid_descent
            and (posture_horizontal or height_collapsed)
        ):

            step = (
                3
                if rapid_descent or aspect_ratio >= 1.20
                else 2
            )

            self.fall_score = min(
                self.fall_score + step,
                maximum_score
            )

        elif posture_horizontal:

            self.fall_score = min(
                self.fall_score + 1,
                maximum_score
            )

        elif (
            torso_angle < 30.0
            and height_ratio > 0.76
            and aspect_ratio < 0.80
        ):

            self.fall_score = max(
                self.fall_score - 3,
                0
            )

        else:

            self.fall_score = max(
                self.fall_score - 1,
                0
            )

        if self.fall_score >= confirmation_threshold:
            self.is_fallen = True

        elif self.fall_score <= 1:
            self.is_fallen = False

        fall_confidence = min(
            100.0,
            (
                self.fall_score
                / max(confirmation_threshold, 1)
            ) * 100.0
        )

        status = (
            "FALL DETECTED"
            if self.is_fallen
            else (
                "WARNING"
                if self.fall_score >= 2
                else "NORMAL"
            )
        )

        self.last_metrics = {
            "status": status,
            "torso_angle": torso_angle,
            "aspect_ratio": aspect_ratio,
            "height_ratio": height_ratio,
            "fall_confidence": fall_confidence,
            "is_raw_horizontal": bool(
                (
                    torso_angle >= 44.0
                    and height_ratio <= 0.70
                )
                or aspect_ratio >= 1.02
            )
        }

        return self.last_metrics

    def step_missing(self):

        self.missed_frames += 1

        max_hold = (
            int(self.fps * 1.5)
            if self.is_fallen
            else int(self.fps * 0.4)
        )

        if (
            self.last_metrics is not None
            and self.missed_frames <= max_hold
        ):
            return self.last_metrics

        return None


def draw_person_annotations(
    frame,
    bbox,
    keypoints,
    confidences,
    track_id,
    metrics,
    is_held=False
):

    x1, y1, x2, y2 = map(int, bbox)

    status = metrics["status"]

    color = (
        (0, 0, 255)
        if status == "FALL DETECTED"
        else (
            (0, 165, 255)
            if status == "WARNING"
            else (0, 210, 80)
        )
    )

    if (
        not is_held
        and keypoints is not None
        and confidences is not None
    ):

        for i, j in SKELETON_EDGES:

            if (
                i < len(confidences)
                and j < len(confidences)
                and confidences[i] >= 0.22
                and confidences[j] >= 0.22
            ):

                pt1 = tuple(map(int, keypoints[i]))
                pt2 = tuple(map(int, keypoints[j]))

                if (
                    pt1[0] > 0 and pt1[1] > 0
                    and pt2[0] > 0 and pt2[1] > 0
                ):
                    cv2.line(
                        frame,
                        pt1,
                        pt2,
                        (255, 220, 0),
                        2,
                        cv2.LINE_AA
                    )

        for idx in range(len(confidences)):

            if confidences[idx] >= 0.22:

                px, py = map(int, keypoints[idx])

                if px > 0 and py > 0:
                    cv2.circle(
                        frame,
                        (px, py),
                        4,
                        (0, 255, 255),
                        -1,
                        cv2.LINE_AA
                    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        3 if status == "FALL DETECTED" else 2
    )

    hold_tag = " [HOLD]" if is_held else ""

    label = (
        f"ID:{track_id} {status}{hold_tag} "
        f"({metrics['fall_confidence']:.0f}%) | "
        f"Ang:{metrics['torso_angle']:.0f}d "
        f"H:{metrics['height_ratio']:.2f} "
        f"AR:{metrics['aspect_ratio']:.2f}"
    )

    (text_width, text_height), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        2
    )

    banner_y = max(y1 - text_height - 10, 0)

    cv2.rectangle(
        frame,
        (x1, banner_y),
        (x1 + text_width + 8, banner_y + text_height + 8),
        color,
        -1
    )

    cv2.putText(
        frame,
        label,
        (x1 + 4, banner_y + text_height + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


def process_video(
    video_path,
    model,
    model_name,
    device,
    gt_windows,
    imgsz=640,
    show_window=False
):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    project_root = Path(__file__).resolve().parent.parent

    output_dir = (
        project_root
        / "evaluation"
        / "results"
        / "reference_v1"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir
        / f"{video_path.parent.name}_{video_path.stem}_fall_detected.mp4"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not create output video: {output_path}"
        )

    trackers = {}

    frame_idx = 0
    frames_with_raw_person = 0
    frames_with_active_track = 0

    box_conf_list = []
    keypoint_conf_list = []
    predicted_fall_frames = []

    previous_gray = None
    scene_cuts = 0

    print(
        f"[INFO] Processing {video_path.name} "
        f"({width}x{height} @ {fps:.1f} FPS) "
        f"using {model_name} on {device}"
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_idx += 1

        # Scene-cut detection
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        if previous_gray is not None:

            mean_difference = float(
                np.mean(cv2.absdiff(gray, previous_gray))
            )

            if mean_difference > 48.0:
                trackers.clear()
                scene_cuts += 1

        previous_gray = gray

        results = model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            device=device,
            imgsz=imgsz,
            conf=0.18,
            iou=0.45,
            verbose=False
        )

        any_fall = False
        seen_ids = set()

        if results and len(results) > 0:

            result = results[0]

            if (
                result.boxes is not None
                and len(result.boxes) > 0
                and result.keypoints is not None
                and result.keypoints.xy is not None
            ):

                frames_with_raw_person += 1

                boxes = result.boxes.xyxy.cpu().numpy()

                if result.boxes.conf is not None:
                    box_conf_list.extend(
                        result.boxes.conf.cpu().numpy().tolist()
                    )

                keypoints = result.keypoints.xy.cpu().numpy()

                if result.keypoints.conf is not None:
                    confidences = (
                        result.keypoints.conf.cpu().numpy()
                    )
                else:
                    confidences = np.ones(
                        (len(boxes), 17),
                        dtype=np.float32
                    )

                visible = confidences[confidences >= 0.20]

                if len(visible) > 0:
                    keypoint_conf_list.append(
                        float(np.mean(visible))
                    )

                if result.boxes.id is not None:
                    track_ids = (
                        result.boxes.id.int().cpu().tolist()
                    )
                else:
                    track_ids = list(
                        range(1, len(boxes) + 1)
                    )

                for idx, track_id in enumerate(track_ids):

                    seen_ids.add(track_id)

                    if track_id not in trackers:
                        trackers[track_id] = PersonFallTracker(
                            track_id,
                            fps
                        )

                    metrics = trackers[track_id].update(
                        boxes[idx],
                        keypoints[idx],
                        confidences[idx],
                        frame_idx
                    )

                    if metrics["status"] == "FALL DETECTED":
                        any_fall = True

                    draw_person_annotations(
                        frame,
                        boxes[idx],
                        trackers[track_id].smoothed_kpts,
                        confidences[idx],
                        track_id,
                        metrics
                    )

        # Handle temporarily missing people
        for track_id, tracker in list(trackers.items()):

            if track_id not in seen_ids:

                held_metrics = tracker.step_missing()

                if (
                    held_metrics is not None
                    and tracker.last_bbox is not None
                ):

                    if held_metrics["status"] == "FALL DETECTED":
                        any_fall = True

                    draw_person_annotations(
                        frame,
                        tracker.last_bbox,
                        tracker.smoothed_kpts,
                        tracker.last_kpts_conf,
                        track_id,
                        held_metrics,
                        is_held=True
                    )

        if seen_ids or any_fall:
            frames_with_active_track += 1

        if any_fall:

            predicted_fall_frames.append(frame_idx)

            cv2.rectangle(
                frame,
                (0, 0),
                (width, 46),
                (0, 0, 220),
                -1
            )

            cv2.putText(
                frame,
                "ALERT: FALL DETECTED",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                (255, 255, 255),
                3,
                cv2.LINE_AA
            )

        writer.write(frame)

        if show_window:

            cv2.imshow(
                "Multi-Person Fall Detection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()

    if show_window:
        cv2.destroyAllWindows()

    detected_episodes = merge_frame_intervals(
        predicted_fall_frames,
        fps,
        max_gap_sec=1.2,
        min_dur_sec=0.40
    )

    raw_coverage = (
        frames_with_raw_person / max(frame_idx, 1)
    ) * 100.0

    tracking_coverage = (
        frames_with_active_track / max(frame_idx, 1)
    ) * 100.0

    mean_box_conf = (
        float(np.mean(box_conf_list)) * 100.0
        if box_conf_list else 0.0
    )

    mean_keypoint_conf = (
        float(np.mean(keypoint_conf_list)) * 100.0
        if keypoint_conf_list else 0.0
    )

    print("\n" + "=" * 65)
    print(f"FALL DETECTION REPORT: {video_path.name}")
    print("=" * 65)

    print(f"Model                  : {model_name}")
    print(f"Device                 : {device}")
    print(f"Inference size         : {imgsz}")
    print(f"Frames processed       : {frame_idx}")
    print(f"Video duration         : {frame_idx / fps:.2f} seconds")
    print(f"Scene cuts             : {scene_cuts}")

    print(
        f"Raw detection coverage: {raw_coverage:.2f}%"
    )

    print(
        f"Tracking coverage     : {tracking_coverage:.2f}%"
    )

    print(
        f"Mean box confidence   : {mean_box_conf:.2f}%"
    )

    print(
        f"Mean keypoint confidence: {mean_keypoint_conf:.2f}%"
    )

    print(
        f"Predicted fall frames : {len(predicted_fall_frames)}"
    )

    print(
        f"Detected episodes     : {detected_episodes}"
    )

    # Evaluate only with independent manual ground truth
    if gt_windows:

        tp = fp = tn = fn = 0
        predicted_set = set(predicted_fall_frames)

        for frame_number in range(1, frame_idx + 1):

            time_seconds = frame_number / fps

            predicted = frame_number in predicted_set

            actual = any(
                start <= time_seconds <= end
                for start, end in gt_windows
            )

            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
            elif not predicted and not actual:
                tn += 1
            else:
                fn += 1

        total = max(tp + tn + fp + fn, 1)

        accuracy = 100.0 * (tp + tn) / total
        precision = 100.0 * tp / max(tp + fp, 1)
        recall = 100.0 * tp / max(tp + fn, 1)

        f1 = (
            2 * precision * recall
            / max(precision + recall, 1e-6)
        )

        print("-" * 65)
        print(f"Ground-truth windows : {gt_windows}")
        print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        print(f"Accuracy             : {accuracy:.2f}%")
        print(f"Precision            : {precision:.2f}%")
        print(f"Recall               : {recall:.2f}%")
        print(f"F1-score             : {f1:.2f}%")

    else:

        print("-" * 65)
        print("Frame metrics        : Not calculated")
        print(
            "Reason: independent ground-truth "
            "timestamps were not supplied."
        )

    print("=" * 65)

    print(f"[INFO] Annotated video saved: {output_path}")


def main():

    project_root = Path(__file__).resolve().parent.parent

    default_source = project_root / "evaluation"
    default_weights = project_root / "yolov8n-pose.pt"

    parser = argparse.ArgumentParser(
        description="Multi-Person Fall Detection"
    )

    parser.add_argument(
        "--source",
        type=str,
        default=str(default_source),
        help="Video file or folder; default is evaluation/"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(default_weights),
        help="YOLO pose model weights"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size"
    )

    parser.add_argument(
        "--fall-window",
        type=str,
        default="",
        help="Independent GT interval(s), e.g. 1.2:3.8"
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable display window"
    )

    args = parser.parse_args()

    source_path = Path(args.source).expanduser().resolve()

    extensions = {
        ".mp4", ".mov", ".avi",
        ".mkv", ".webm", ".m4v"
    }

    if source_path.is_dir():

        videos = sorted(
            path
            for path in source_path.rglob("*")
            if path.is_file()
            and path.suffix.lower() in extensions
            and not path.stem.endswith("_fall_detected")
        )

    elif source_path.is_file():

        videos = [source_path]

    else:

        raise FileNotFoundError(
            f"Source path does not exist: {source_path}"
        )

    if not videos:
        raise FileNotFoundError(
            f"No input videos found under: {source_path}"
        )

    gt_windows = parse_fall_windows(args.fall_window)

    if gt_windows and len(videos) != 1:
        raise ValueError(
            "--fall-window applies to one video only. "
            "Run each labelled video separately."
        )

    device = select_device()

    model_path = Path(args.model).expanduser()

    model_arg = (
        str(model_path.resolve())
        if model_path.exists()
        else args.model
    )

    model, loaded_name = load_pose_model(model_arg)

    print(f"[INFO] Found {len(videos)} video(s)")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Model: {loaded_name}")

    for video in videos:

        process_video(
            video,
            model,
            loaded_name,
            device,
            gt_windows,
            imgsz=args.imgsz,
            show_window=not args.no_show
        )


if __name__ == "__main__":
    main()
