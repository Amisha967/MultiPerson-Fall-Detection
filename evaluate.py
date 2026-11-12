from __future__ import annotations

import argparse
import contextlib
import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import cv2

from fall_classifier import FallClassifier
from person_detector import PersonDetector
from person_tracker import PersonTracker

# ---------------- CONFIGURATION ----------------

MODEL_PATH = "yolov8n-pose.pt"
DETECTION_CONFIDENCE = 0.35
IMAGE_SIZE = 640
DEFAULT_FPS = 30.0
LOG_FRAME_INTERVAL = 50

TRACKER_PARAMS: dict[str, Any] = {
    "distance_threshold": 150,
    "keypoint_confidence": 0.3,
    "min_full_body_keypoints": 5,
    "history_cleanup_interval": 5,
}

CLASSIFIER_PARAMS: dict[str, Any] = {
    "horizontal_ar_threshold": 1.0,
    "angle_threshold": 55,
    "minimum_fall_duration": 1.0,
    "keypoint_confidence": 0.3,
}

RESULTS_DIR = Path("evaluation/results")

VIDEOS: list[tuple[Path, str]] = [
    (Path("evaluation/fall_videos/fall-01.mp4"), "Fall"),
    (Path("evaluation/normal_videos/adl-01.mp4"), "Normal"),
]

VALID_LABELS = ("Fall", "Normal")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------- DATA STRUCTURES ----------------

@dataclass(frozen=True)
class VideoPredictionRow:
    """Stores evaluation results for a single video."""

    video: str
    actual: str | None
    predicted: str
    fall_ids: list[int]
    frames: int
    seconds: float

    @property
    def processing_fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0


@dataclass(frozen=True)
class EvaluationMetrics:
    """Stores video-level binary classification metrics."""

    tp: int
    tn: int
    fp: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    labelled_videos: int


# ---------------- VIDEO CAPTURE HELPER ----------------

@contextlib.contextmanager
def open_video_capture(video_path: Path) -> Iterator[cv2.VideoCapture]:
    """Context manager ensuring cv2.VideoCapture is always released."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        yield cap
    finally:
        cap.release()


# ---------------- EVALUATION ----------------

def evaluate_video(
    video_path: Path,
    detector: PersonDetector,
    tracker_params: dict[str, Any] = TRACKER_PARAMS,
    classifier_params: dict[str, Any] = CLASSIFIER_PARAMS,
    log_interval: int = LOG_FRAME_INTERVAL,
) -> tuple[str, list[int], int, float]:
    """Run person detection, tracking, and fall classification on a single video."""
    tracker = PersonTracker(**tracker_params)
    classifier = FallClassifier(**classifier_params)

    fall_ids: set[int] = set()
    frame_idx = 0

    with open_video_capture(video_path) as cap:
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = raw_fps if raw_fps and raw_fps > 0 else DEFAULT_FPS
        total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)

        start_time = time.perf_counter()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            video_time = frame_idx / fps

            detections = detector.detect(frame)
            tracked_people = tracker.update(
                detections,
                current_time=video_time,
            )

            for person in tracked_people:
                person_id = person["person_id"]
                result = classifier.update(
                    person_id=person_id,
                    keypoints=person["keypoints"],
                    aspect_ratio=person["aspect_ratio"],
                    is_full_body=person["is_full_body"],
                    current_time=video_time,
                )
                if result.get("fall_detected"):
                    fall_ids.add(person_id)

            if log_interval > 0 and frame_idx % log_interval == 0:
                total_str = str(total_frames) if total_frames > 0 else "?"
                logger.info("  Progress: %d/%s frames", frame_idx, total_str)

        elapsed = time.perf_counter() - start_time

    predicted = "Fall" if fall_ids else "Normal"
    return predicted, sorted(fall_ids), frame_idx, elapsed


# ---------------- METRICS ----------------

def compute_metrics(rows: Sequence[VideoPredictionRow]) -> EvaluationMetrics:
    """Compute video-level confusion matrix and classification metrics."""
    tp = tn = fp = fn = 0

    for row in rows:
        if row.actual not in VALID_LABELS:
            continue
        if row.actual == "Fall" and row.predicted == "Fall":
            tp += 1
        elif row.actual == "Normal" and row.predicted == "Normal":
            tn += 1
        elif row.actual == "Normal" and row.predicted == "Fall":
            fp += 1
        elif row.actual == "Fall" and row.predicted == "Normal":
            fn += 1

    n = tp + tn + fp + fn
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return EvaluationMetrics(
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
        labelled_videos=n,
    )


# ---------------- OUTPUTS ----------------

def write_outputs(
    rows: Sequence[VideoPredictionRow],
    metrics: EvaluationMetrics,
    results_dir: Path = RESULTS_DIR,
) -> None:
    """Write CSV predictions, confusion matrix, and text summary report."""
    results_dir.mkdir(parents=True, exist_ok=True)

    predictions_csv = results_dir / "video_predictions.csv"
    with predictions_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video",
            "actual",
            "predicted",
            "fall_ids",
            "frames",
            "processing_seconds",
            "processing_fps",
        ])
        for r in rows:
            writer.writerow([
                r.video,
                r.actual or "Unlabelled",
                r.predicted,
                " ".join(map(str, r.fall_ids)),
                r.frames,
                f"{r.seconds:.2f}",
                f"{r.processing_fps:.2f}",
            ])

    if metrics.labelled_videos > 0:
        confusion_csv = results_dir / "confusion_matrix.csv"
        with confusion_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["", "Predicted Normal", "Predicted Fall"])
            writer.writerow(["Actual Normal", metrics.tn, metrics.fp])
            writer.writerow(["Actual Fall", metrics.fn, metrics.tp])

        report = (
            "PRELIMINARY FALL-DETECTION EVALUATION\n"
            "=====================================\n"
            f"Labelled videos: {metrics.labelled_videos}\n\n"
            f"Accuracy   : {metrics.accuracy:.4f}\n"
            f"Precision  : {metrics.precision:.4f}\n"
            f"Recall     : {metrics.recall:.4f}\n"
            f"Specificity: {metrics.specificity:.4f}\n"
            f"F1-score   : {metrics.f1:.4f}\n\n"
            "Confusion Matrix\n"
            f"{'':18s}{'Predicted Normal':>18s}{'Predicted Fall':>16s}\n"
            f"{'Actual Normal':18s}{metrics.tn:>18d}{metrics.fp:>16d}\n"
            f"{'Actual Fall':18s}{metrics.fn:>18d}{metrics.tp:>16d}\n\n"
            "Metrics are video-level, not frame-level.\n"
        )
    else:
        report = (
            "CCTV VIDEO TEST\n"
            "===============\n\n"
            "No independent ground-truth label was supplied.\n\n"
            "Prediction is recorded, but accuracy, precision,\n"
            "recall, specificity, F1-score, and confusion matrix\n"
            "are not calculated.\n"
        )

    report_path = results_dir / "evaluation_report.txt"
    report_path.write_text(report, encoding="utf-8")

    print("\n" + report)
    logger.info("Saved results to: %s", results_dir.resolve())


# ---------------- CLI & MAIN ----------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="YOLO26s Multi-Person Fall Detection Evaluation"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Path to a single CCTV/test video file",
    )
    parser.add_argument(
        "--label",
        choices=VALID_LABELS,
        default=None,
        help="Verified ground-truth video label ('Fall' or 'Normal')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help=f"Path to YOLO pose weights (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DETECTION_CONFIDENCE,
        help=f"Person detection confidence threshold (default: {DETECTION_CONFIDENCE})",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=IMAGE_SIZE,
        help=f"Inference image size (default: {IMAGE_SIZE})",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Directory to store output reports (default: {RESULTS_DIR})",
    )
    return parser.parse_args()


def resolve_video_list(
    source: Path | None,
    label: str | None,
) -> list[tuple[Path, str | None]]:
    """Validate and resolve the list of videos to evaluate."""
    if source is not None:
        resolved = source.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Video not found: {resolved}")
        return [(resolved, label)]

    videos: list[tuple[Path, str | None]] = [
        (path.expanduser().resolve(), video_label)
        for path, video_label in VIDEOS
    ]
    missing = [path for path, _ in videos if not path.is_file()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Video file(s) not found: {missing_str}")

    return videos


def main() -> None:
    args = parse_args()
    videos = resolve_video_list(args.source, args.label)

    logger.info("Loading YOLO26s-Pose model (%s)...", args.model)
    detector = PersonDetector(
        model_path=args.model,
        confidence=args.conf,
        image_size=args.imgsz,
    )
    logger.info("YOLO26s-Pose model loaded successfully.")

    rows: list[VideoPredictionRow] = []

    for video_path, label in videos:
        logger.info("Evaluating: %s (Actual: %s)", video_path.name, label or "Unlabelled CCTV")

        predicted, fall_ids, frames, seconds = evaluate_video(
            video_path=video_path,
            detector=detector,
        )

        row = VideoPredictionRow(
            video=video_path.name,
            actual=label,
            predicted=predicted,
            fall_ids=fall_ids,
            frames=frames,
            seconds=seconds,
        )
        rows.append(row)

        logger.info(
            "Result: Predicted=%s | Confirmed fall IDs=%s | Time=%.2fs (%.1f FPS)",
            row.predicted,
            row.fall_ids,
            row.seconds,
            row.processing_fps,
        )

    metrics = compute_metrics(rows)
    write_outputs(rows, metrics, results_dir=args.results_dir)


if __name__ == "__main__":
    main()