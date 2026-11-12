from ultralytics import YOLO


class PersonDetector:
    """YOLOv8 pose detection with persistent ByteTrack IDs."""

    def __init__(self, model_path="yolov8n-pose.pt", confidence=0.35, image_size=640):
        print("Loading YOLO Pose model...")
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.image_size = image_size
        print("YOLO Pose model loaded successfully!")

    def detect(self, frame):
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence,
            imgsz=self.image_size,
            classes=[0],
            verbose=False
        )
        result = results[0]
        detections = []

        if result.boxes is None or result.keypoints is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        poses = result.keypoints.data.cpu().numpy()
        track_ids = None
        if result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)

        for i, (box, score, pose) in enumerate(zip(boxes, scores, poses)):
            detections.append({
                "bbox": box.astype(float),
                "confidence": float(score),
                "keypoints": pose.astype(float),
                "track_id": int(track_ids[i]) if track_ids is not None else None
            })
        return detections
