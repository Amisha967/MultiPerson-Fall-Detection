
from ultralytics import YOLO


class PersonDetector:

    def __init__(
        self,
        model_path="yolov8n-pose.pt",
        confidence=0.35,
        image_size=640
    ):
        print("Loading YOLO Pose model...")

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.image_size = image_size

        print("YOLO Pose model loaded successfully!")

    def detect(self, frame):

        results = self.model.predict(
            source=frame,
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
        keypoints = result.keypoints.data.cpu().numpy()

        for box, score, pose in zip(boxes, scores, keypoints):

            detections.append({
                "bbox": box.astype(float),
                "confidence": float(score),
                "keypoints": pose.astype(float)
            })

        return detections
