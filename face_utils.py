"""The same crop is used during data collection and cloud inference."""
from pathlib import Path
import sys

import cv2
import numpy as np

BASE = Path(__file__).parent
DETECTOR_PATH = BASE / "face_detection_yunet_2023mar.onnx"


def make_detector():
    if not DETECTOR_PATH.exists():
        raise FileNotFoundError("Run: python download_models.py")
    # Official API: https://docs.opencv.org/4.x/df/d20/classcv_1_1FaceDetectorYN.html
    return cv2.FaceDetectorYN.create(str(DETECTOR_PATH), "", (640, 480), 0.65, 0.3, 5000)


def square_box(face, width, height):
    x, y, w, h = map(float, face[:4])
    side = min(int(round(max(w, h) * 1.3)), width, height)
    if side < 2:
        return None
    x1 = max(0, min(int(round(x + w / 2 - side / 2)), width - side))
    y1 = max(0, min(int(round(y + h / 2 - side / 2)), height - side))
    return [x1, y1, x1 + side, y1 + side]


def find_faces(detector, frame):
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)
    if faces is None:
        return []
    result = []
    for face in faces:
        box = square_box(face, width, height)
        if box:
            x1, y1, x2, y2 = box
            result.append((box, frame[y1:y2, x1:x2]))
    return result


def tm_input(crop):
    image = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # TM's float-image input convention:
    # https://github.com/googlecreativelab/teachablemachine-community/blob/master/snippets/converter/image/api.py
    return (image.astype(np.float32) / 127.5 - 1.0)[None, ...]


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Close Camera/Teams/Zoom, check permissions, or try --camera 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def fit_frame(frame):
    height, width = frame.shape[:2]
    ratio = min(640 / width, 480 / height, 1.0)
    if ratio < 1:
        frame = cv2.resize(frame, (int(width * ratio), int(height * ratio)))
    return frame
