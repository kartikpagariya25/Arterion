import os
import base64
import numpy as np
import cv2
import requests
from PIL import Image, ImageDraw, ImageFont

from .severity import estimate_stenosis_percent, severity_band, display_confidence

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
VESSEL_MODEL_PATH = os.path.join(WEIGHTS_DIR, "vessel_model.onnx")
STENOSIS_MODEL_PATH = os.path.join(WEIGHTS_DIR, "stenosis_model.onnx")

IMAGE_SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SEVERITY_COLORS_BGR = {
    "mild": (0, 200, 0),
    "moderate": (0, 200, 255),
    "severe": (0, 0, 255),
}

_vessel_session = None
_stenosis_session = None
_font_cache = None

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

def _run_model_probs(session, image_tensor) -> np.ndarray:
    logits = session.run(["logits"], {"image": image_tensor})[0]
    return 1 / (1 + np.exp(-logits[0, 0]))

def _run_model_probs_tta(session, image: np.ndarray) -> np.ndarray:
    """Average probability maps from the original image and its horizontal flip."""
    tensor = _preprocess(image)
    probs = _run_model_probs(session, tensor)

    flipped = cv2.flip(image, 1)
    tensor_flipped = _preprocess(flipped)
    probs_flipped = _run_model_probs(session, tensor_flipped)
    probs_flipped = np.fliplr(probs_flipped)

    return (probs + probs_flipped) / 2.0

def _run_model(session, image_tensor) -> np.ndarray:
    probs = _run_model_probs(session, image_tensor)
    return (probs > 0.5).astype(np.uint8)

def _find_lesions(stenosis_mask: np.ndarray):
    cleaned = cv2.morphologyEx(stenosis_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < 8:
            continue
        boxes.append(cv2.boundingRect(c))
    return boxes

def _get_font(size: int):
    global _font_cache
    if _font_cache is None:
        _font_cache = {}
    if size not in _font_cache:
        try:
            _font_cache[size] = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
            )
        except Exception:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]

def _draw_legend(draw: ImageDraw.ImageDraw, font, origin=(12, 12)):
    x, y = origin
    entries = [("Mild", (0, 200, 0)), ("Moderate", (255, 200, 0)), ("Severe", (255, 0, 0))]
    draw.rectangle([x - 6, y - 6, x + 118, y + len(entries) * 20 + 2], fill=(0, 0, 0, 140))
    for label, color in entries:
        draw.ellipse([x, y + 4, x + 10, y + 14], fill=color)
        draw.text((x + 16, y), label, font=font, fill=(255, 255, 255))
        y += 20

def _encode_png(image_bgr: np.ndarray) -> bytes:
    _, buffer = cv2.imencode(".png", image_bgr)
    return buffer.tobytes()

def _make_overlay(image: np.ndarray, vessel_mask: np.ndarray, lesions: list) -> bytes:
    """Draws a clean vessel outline plus a colored box per lesion with a severity/percent/confidence label.
    No probability heatmap is applied — only the vessel tree and the detected lesion boxes are marked,
    so the output stays readable instead of tinting unrelated regions of the image.
    """
    base = image.copy()
    if base.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        base = cv2.resize(base, (IMAGE_SIZE, IMAGE_SIZE))

    vessel_contours, _ = cv2.findContours(vessel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(base, vessel_contours, -1, (0, 200, 0), 1, cv2.LINE_AA)

    rgb = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).convert("RGBA")
    draw_layer = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(draw_layer)
    font = _get_font(14)

    for (bbox, pct, band, confidence) in lesions:
        x, y, w, h = bbox
        color_bgr = SEVERITY_COLORS_BGR.get(band, (0, 0, 255))
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0], 255)

        draw.rectangle([x, y, x + w, y + h], outline=color_rgb, width=2)

        label = f"{band.upper()} · {pct}% narrowing · {confidence}% confidence"
        bbox_text = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
        label_x, label_y = x, max(0, y - th - 10)
        draw.rectangle([label_x, label_y, label_x + tw + 8, label_y + th + 8], fill=(color_bgr[2], color_bgr[1], color_bgr[0], 210))
        draw.text((label_x + 4, label_y + 3), label, font=font, fill=(255, 255, 255, 255))

    if lesions:
        _draw_legend(draw, font)

    composited = Image.alpha_composite(pil_img, draw_layer).convert("RGB")
    result_bgr = cv2.cvtColor(np.array(composited), cv2.COLOR_RGB2BGR)
    return _encode_png(result_bgr)

def predict(file_url: str) -> dict:
    load_model()

    if _vessel_session is None:
        return {
            "original_image_base64": None,
            "overlay_image_base64": None,
            "lesions": [],
            "summary": "Vessel analysis model weights not found — training not yet completed.",
        }

    image = _load_image_from_url(file_url)
    if image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    original_base64 = base64.b64encode(_encode_png(image)).decode("utf-8")

    vessel_probs = _run_model_probs_tta(_vessel_session, image)
    vessel_mask = (vessel_probs > 0.5).astype(np.uint8)

    stenosis_probs = _run_model_probs_tta(_stenosis_session, image)
    stenosis_mask = (stenosis_probs > 0.5).astype(np.uint8)

    lesion_boxes = _find_lesions(stenosis_mask)
    lesions = []
    overlay_lesions = []
    for bbox in lesion_boxes:
        pct = estimate_stenosis_percent(vessel_mask, bbox)
        band = severity_band(pct)
        confidence = display_confidence(band)
        lesions.append({
            "bbox": list(bbox),
            "stenosis_percent": round(pct, 1),
            "severity_band": band,
            "confidence": confidence,
        })
        overlay_lesions.append((bbox, round(pct, 1), band, confidence))

    overlay_bytes = _make_overlay(image, vessel_mask, overlay_lesions)
    overlay_base64 = base64.b64encode(overlay_bytes).decode("utf-8")

    if lesions:
        worst = max(lesions, key=lambda l: l["stenosis_percent"])
        summary = f"{len(lesions)} lesion(s) detected. Most severe: ~{worst['stenosis_percent']}% narrowing ({worst['severity_band']}, {worst['confidence']}% confidence)."
    else:
        summary = "No significant stenosis detected in the analyzed frame."

    return {
        "original_image_base64": original_base64,
        "overlay_image_base64": overlay_base64,
        "vessel_mask_present": bool(vessel_mask.sum() > 0),
        "lesions": lesions,
        "summary": summary,
    }