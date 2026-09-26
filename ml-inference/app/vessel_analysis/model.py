import os
import base64
import numpy as np
import cv2
import requests

from .severity import estimate_stenosis_percent, severity_band

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
VESSEL_MODEL_PATH = os.path.join(WEIGHTS_DIR, "vessel_model.onnx")
STENOSIS_MODEL_PATH = os.path.join(WEIGHTS_DIR, "stenosis_model.onnx")

IMAGE_SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_vessel_session = None
_stenosis_session = None


def _models_available() -> bool:
    return os.path.exists(VESSEL_MODEL_PATH) and os.path.exists(STENOSIS_MODEL_PATH)


def load_model():
    global _vessel_session, _stenosis_session
    if _vessel_session is None and _models_available():
        import onnxruntime as ort

        _vessel_session = ort.InferenceSession(VESSEL_MODEL_PATH, providers=["CPUExecutionProvider"])
        _stenosis_session = ort.InferenceSession(STENOSIS_MODEL_PATH, providers=["CPUExecutionProvider"])
    return _vessel_session


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


def _run_model(session, image_tensor) -> np.ndarray:
    logits = session.run(["logits"], {"image": image_tensor})[0]
    probs = 1 / (1 + np.exp(-logits))
    return (probs[0, 0] > 0.5).astype(np.uint8)


def _find_lesions(stenosis_mask: np.ndarray):
    contours, _ = cv2.findContours(stenosis_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < 5:
            continue
        boxes.append(cv2.boundingRect(c))
    return boxes


def _make_overlay(image: np.ndarray, vessel_mask: np.ndarray, lesions: list) -> bytes:
    overlay = image.copy()
    if overlay.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        overlay = cv2.resize(overlay, (IMAGE_SIZE, IMAGE_SIZE))

    vessel_color = overlay.copy()
    vessel_color[vessel_mask > 0] = [0, 200, 0]
    overlay = cv2.addWeighted(overlay, 0.75, vessel_color, 0.25, 0)

    for (x, y, w, h) in lesions:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)

    _, buffer = cv2.imencode(".png", overlay)
    return buffer.tobytes()


def predict(file_url: str) -> dict:
    load_model()

    if _vessel_session is None:
        return {
            "overlay_image_base64": None,
            "lesions": [],
            "summary": "Vessel analysis model weights not found — training not yet completed.",
            "confidence": 0.0,
        }

    image = _load_image_from_url(file_url)
    tensor = _preprocess(image)
    vessel_mask = _run_model(_vessel_session, tensor)
    stenosis_mask = _run_model(_stenosis_session, tensor)

    lesion_boxes = _find_lesions(stenosis_mask)
    lesions = []
    for bbox in lesion_boxes:
        pct = estimate_stenosis_percent(vessel_mask, bbox)
        lesions.append({
            "bbox": list(bbox),
            "stenosis_percent": round(pct, 1),
            "severity_band": severity_band(pct),
        })

    overlay_bytes = _make_overlay(image, vessel_mask, lesion_boxes)
    overlay_base64 = base64.b64encode(overlay_bytes).decode("utf-8")

    if lesions:
        worst = max(lesions, key=lambda l: l["stenosis_percent"])
        summary = f"{len(lesions)} lesion(s) detected. Most severe: ~{worst['stenosis_percent']}% narrowing ({worst['severity_band']})."
        confidence = 0.75
    else:
        summary = "No significant stenosis detected in the analyzed frame."
        confidence = 0.6

    return {
        "overlay_image_base64": overlay_base64,
        "vessel_mask_present": bool(vessel_mask.sum() > 0),
        "lesions": lesions,
        "summary": summary,
        "confidence": confidence,
    }
