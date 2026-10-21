"""Cloud: full image -> YuNet faces -> TM TFLite classifications -> JSON."""
from contextlib import asynccontextmanager
from io import BytesIO
from threading import Lock
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from tflite_runtime.interpreter import Interpreter
from face_utils import BASE, make_detector, find_faces, tm_input

LOCK = Lock()
MAX_BYTES = 2 * 1024 * 1024
THRESHOLD = 0.70


def read_labels(path):
    result = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            line = parts[1]
        result.append(line)
    if len(result) != 3 or set(result) != {"with_mask", "without_mask", "incorrect_mask"}:
        raise ValueError("labels.txt must contain the 3 trained labels. Keep the original order.")
    return result


@asynccontextmanager
async def lifespan(app):
    cv2.setNumThreads(1)
    app.state.labels = read_labels(BASE / "labels.txt")
    app.state.detector = make_detector()
    # Linux-only lightweight runtime. https://www.tensorflow.org/lite/guide/python
    interpreter = Interpreter(model_path=str(BASE / "model.tflite"), num_threads=2)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    if tuple(inp["shape"]) != (1, 224, 224, 3) or inp["dtype"] != np.float32:
        raise ValueError("Export TensorFlow Lite > Floating point, not Quantized or Edge TPU.")
    if tuple(out["shape"]) != (1, 3):
        raise ValueError("The trained model must have exactly 3 output classes.")
    app.state.interpreter, app.state.inp, app.state.out = interpreter, inp, out
    interpreter.set_tensor(inp["index"], np.zeros((1, 224, 224, 3), dtype=np.float32))
    interpreter.invoke()  # Warm the classifier before accepting requests.
    yield


app = FastAPI(title="IoT Lab 3 - Mask Detection", lifespan=lifespan)


@app.get("/")
def root():
    return {"message": "Mask Cloud API is running", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "labels": app.state.labels}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    started = perf_counter()
    raw = file.file.read(MAX_BYTES + 1)
    if not raw or len(raw) > MAX_BYTES:
        raise HTTPException(413, "Image must be nonempty and no larger than 2 MiB.")
    try:
        with Image.open(BytesIO(raw)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise HTTPException(415, "Only JPEG and PNG are supported.")
            width, height = image.size
            if width < 2 or height < 2 or width > 1280 or height > 1280:
                raise HTTPException(422, "Resize the image to at most 1280x1280.")
            frame = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, "Cannot decode this image.") from exc
    results, inference_ms = [], 0.0
    # ponytail: one locked detector/interpreter for a lab; use isolated workers for concurrent production traffic.
    with LOCK:
        for box, crop in find_faces(app.state.detector, frame):
            tensor = tm_input(crop)
            app.state.interpreter.set_tensor(app.state.inp["index"], tensor)
            begin = perf_counter()
            app.state.interpreter.invoke()
            scores = app.state.interpreter.get_tensor(app.state.out["index"])[0]
            inference_ms += (perf_counter() - begin) * 1000
            index = int(np.argmax(scores))
            label, confidence = app.state.labels[index], float(scores[index])
            results.append({"box": box, "label": label if confidence >= THRESHOLD else "uncertain",
                            "top_label": label, "confidence": confidence,
                            "probabilities": {name: float(score) for name, score in zip(app.state.labels, scores)}})
    return {"faces": results, "face_count": len(results), "width": width, "height": height,
            "inference_ms": round(inference_ms, 3),
            "server_processing_ms": round((perf_counter() - started) * 1000, 3)}
