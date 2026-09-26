import os
import base64
import numpy as np
import cv2
import requests

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
MODEL_PATH = os.path.join(WEIGHTS_DIR, "lv_model.onnx")

IMAGE_SIZE = 112
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_session = None


def _model_available() -> bool:
    return os.path.exists(MODEL_PATH)


def load_model():
    global _session
    if _session is None and _model_available():
        import onnxruntime as ort

        _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return _session


def _load_image_from_url(file_url: str) -> np.ndarray:
    response = requests.get(file_url, timeout=30)
    response.raise_for_status()
    arr = np.frombuffer(response.content, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _preprocess(image: np.ndarray) -> np.ndarray:
    if image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    normalized = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return normalized.transpose(2, 0, 1)[None, ...].astype(np.float32)


def predict(file_url: str) -> dict:
    load_model()

    if _session is None:
        return {
            "outline_mask_base64": None,
            "lv_area_pixels": 0,
            "summary": "LV outline model weights not found — training not yet completed.",
            "confidence": 0.0,
        }

    image = _load_image_from_url(file_url)
    tensor = _preprocess(image)
    logits = _session.run(["logits"], {"image": tensor})[0]
    probs = 1 / (1 + np.exp(-logits))
    mask = (probs[0, 0] > 0.5).astype(np.uint8)

    overlay = image.copy()
    if overlay.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        overlay = cv2.resize(overlay, (IMAGE_SIZE, IMAGE_SIZE))
    colored = overlay.copy()
    colored[mask > 0] = [0, 0, 255]
    overlay = cv2.addWeighted(overlay, 0.6, colored, 0.4, 0)
    _, buffer = cv2.imencode(".png", overlay)

    area = int(mask.sum())

    return {
        "outline_mask_base64": base64.b64encode(buffer.tobytes()).decode("utf-8"),
        "lv_area_pixels": area,
        "summary": "Left ventricle outline generated for the analyzed frame."
                   if area > 0 else "No clear left ventricle boundary detected in this frame.",
        "confidence": 0.9 if area > 0 else 0.3,
    }
