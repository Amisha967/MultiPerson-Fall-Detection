# MultiPerson-Fall-Detection
# Multi-Person Fall Detection Using YOLOv8 Pose

A computer vision-based fall detection system designed to detect and monitor multiple people in video streams using YOLOv8 Pose estimation, keypoint analysis, and person tracking.

The system identifies human body poses, tracks individuals across consecutive frames, and classifies their activity as **Normal**, **Potential Fall**, or **Fall**.

## Overview

Falls are a significant safety concern in environments such as hospitals, elderly care facilities, workplaces, and public spaces.

This project explores an automated computer vision approach to fall detection using human pose estimation. Unlike a basic single-person detection system, this implementation maintains separate tracking IDs and fall states for multiple detected individuals.

## Key Features

- **Multi-Person Detection:** Detects multiple people in a video using YOLOv8 Pose.
- **Human Pose Estimation:** Extracts 17 body keypoints for each detected person.
- **Person Tracking:** Assigns individual IDs and matches people across consecutive frames.
- **Pose-Based Fall Classification:** Uses body aspect ratio and shoulder–hip–knee angle.
- **Temporal Fall Confirmation:** Confirms a potential fall when the posture persists for at least one second.
- **Visual Monitoring:** Displays bounding boxes, pose skeletons, individual IDs, fall status, and video progress.
- **Frame-by-Frame Processing:** Processes the input video sequentially without intentionally skipping frames.

## System Architecture

The project is organized into independent modules:

| Module | Responsibility |
|---|---|
| `main.py` | Coordinates the complete detection pipeline |
| `person_detector.py` | Detects people and extracts pose keypoints |
| `person_tracker.py` | Assigns and maintains individual person IDs |
| `fall_classifier.py` | Analyzes posture and determines fall status |
| `visualizer.py` | Displays bounding boxes, skeletons, and alerts |

### Processing Pipeline

1. **Video Input:** Reads frames from a video file.
2. **Person Detection:** YOLOv8 Pose identifies people and their body keypoints.
3. **Person Tracking:** A centroid-distance matching approach maintains individual IDs.
4. **Pose Analysis:** Extracts relevant body landmarks and calculates posture measurements.
5. **Fall Classification:** Evaluates horizontal body orientation and hip angle.
6. **Temporal Verification:** Checks whether a potential fall posture persists for the configured duration.
7. **Visualization:** Displays the detection results for each tracked individual.

## Fall Detection Methodology

The system uses two primary posture indicators.

### 1. Body Aspect Ratio

The height-to-width ratio of the visible body keypoint bounding box is calculated:

`Aspect Ratio = Body Height / Body Width`

A lower ratio indicates a more horizontal body posture.

The current horizontal posture threshold is:

`Aspect Ratio < 0.6`

### 2. Shoulder–Hip–Knee Angle

The system calculates the angle formed by the left shoulder, left hip, and left knee.

The current implementation considers an angle below `140°` as a potential fall indicator.

### 3. Temporal Confirmation

A potential fall is confirmed when the detected posture persists for at least one second.

This temporal condition helps reduce momentary posture misclassifications.

### Fall Status Visualization

| Status | Display color | Meaning |
|---|---|---|
| Normal | Green | No potential fall detected |
| Potential Fall | Orange | Suspicious posture detected |
| Fall | Red | Potential fall persisted for the confirmation duration |

## Technology Stack

- **Programming Language:** Python
- **Object Detection & Pose Estimation:** YOLOv8 Pose
- **Computer Vision:** OpenCV
- **Numerical Computation:** NumPy
- **Tracking:** Centroid-distance matching
- **Visualization:** OpenCV

## Project Structure

```text
MultiPerson-Fall-Detection/
│
├── main.py
├── person_detector.py
├── person_tracker.py
├── fall_classifier.py
├── visualizer.py
│
├── requirements.txt
├── README.md
├── .gitignore
│
└── test_videos/
    └── Add your own test video here
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Amisha967/MultiPerson-Fall-Detection.git

cd MultiPerson-Fall-Detection
```

### 2. Create a Virtual Environment

```bash
python3 -m venv fallenv
```

Activate it on macOS/Linux:

```bash
source fallenv/bin/activate
```

On Windows:

```bash
fallenv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the YOLOv8 Pose Model

The default model is:

`yolov8n-pose.pt`

Ultralytics can automatically download the model when the detector is initialized, provided an internet connection is available.

### 5. Add a Test Video

Place your video in the project root directory and configure its path in `main.py`:

```python
VIDEO_PATH = "test_cctv.mp4"
```

### 6. Run the Project

```bash
python main.py
```

The detection window displays the processed video and individual fall classifications.

Press `Q` or `ESC` to exit.

## Configuration

The following parameters can be adjusted in `main.py`:

| Parameter | Current value | Purpose |
|---|---:|---|
| Detection confidence | 0.35 | Minimum person detection confidence |
| Image size | 640 | YOLO inference image size |
| Horizontal aspect ratio | 0.6 | Horizontal posture threshold |
| Hip angle | 140° | Potential fall angle threshold |
| Fall duration | 1.0 second | Fall confirmation duration |
| Tracking distance | 150 pixels | Maximum centroid matching distance |

These are initial heuristic settings and should be calibrated against different camera angles, environments, and datasets.

## Limitations and Future Improvements

The current implementation is a prototype. Some areas for improvement include:

- More robust multi-object tracking when people overlap or cross paths.
- Improved fall classification to distinguish sitting, bending, lying down, and actual falls.
- Better handling of partially visible individuals.
- Evaluation using labelled fall-detection datasets.
- Quantitative evaluation using precision, recall, F1-score, and false-alarm rate.
- Real-time alert integration and event logging.
- Performance optimization for longer CCTV streams.

## Acknowledgements

This project was independently implemented using Python, OpenCV, and Ultralytics YOLOv8 Pose.

The implementation was informed by the approach described in [Real-Time-Fall-Detection](https://github.com/haashi-r/Real-Time-Fall-Detection), particularly its use of pose-based posture analysis, person tracking, and temporal fall confirmation.

## Disclaimer

This project is developed for educational and experimental purposes. It is not a certified medical, emergency-response, or safety-monitoring system. Its predictions should not be relied upon as the sole basis for real-world emergency decisions.

## Author

**Amisha Singh**

GitHub: [@Amisha967](https://github.com/Amisha967)