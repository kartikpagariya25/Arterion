import numpy as np
import cv2
import onnxruntime as ort

from config import IMAGE_SIZE
from severity import estimate_stenosis_percent, severity_band

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class VesselAnalyzer:
    def __init__(self, vessel_onnx_path: str, stenosis_onnx_path: str):
        self.vessel_session = ort.InferenceSession(vessel_onnx_path, providers=["CPUExecutionProvider"])
        self.stenosis_session = ort.InferenceSession(stenosis_onnx_path, providers=["CPUExecutionProvider"])

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        normalized = (image.astype(np.float32) / 255.0 - MEAN) / STD
        chw = normalized.transpose(2, 0, 1)[None, ...].astype(np.float32)
        return chw

    def _run_model(self, session, image_tensor) -> np.ndarray:
        logits = session.run(["logits"], {"image": image_tensor})[0]
        probs = 1 / (1 + np.exp(-logits))
        return (probs[0, 0] > 0.5).astype(np.uint8)

    def _find_lesions(self, stenosis_mask: np.ndarray):
        contours, _ = cv2.findContours(stenosis_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            if cv2.contourArea(c) < 5:
                continue
            boxes.append(cv2.boundingRect(c))
        return boxes

    def _make_overlay(self, image: np.ndarray, vessel_mask: np.ndarray, lesions: list) -> np.ndarray:
        overlay = image.copy()
        if overlay.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            overlay = cv2.resize(overlay, (IMAGE_SIZE, IMAGE_SIZE))
        if overlay.ndim == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

        vessel_color = overlay.copy()
        vessel_color[vessel_mask > 0] = [0, 200, 0]
        overlay = cv2.addWeighted(overlay, 0.75, vessel_color, 0.25, 0)

        for (x, y, w, h) in lesions:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)

        return overlay

    def analyze(self, image: np.ndarray) -> dict:
        tensor = self._preprocess(image)
        vessel_mask = self._run_model(self.vessel_session, tensor)
        stenosis_mask = self._run_model(self.stenosis_session, tensor)

        lesion_boxes = self._find_lesions(stenosis_mask)
        lesions = []
        for bbox in lesion_boxes:
            pct = estimate_stenosis_percent(vessel_mask, bbox)
            lesions.append({
                "bbox": list(bbox),
                "stenosis_percent": round(pct, 1),
                "severity_band": severity_band(pct),
            })

        overlay = self._make_overlay(image, vessel_mask, lesion_boxes)
        _, buffer = cv2.imencode(".png", overlay)

        if lesions:
            worst = max(lesions, key=lambda l: l["stenosis_percent"])
            summary = f"{len(lesions)} lesion(s) detected. Most severe: ~{worst['stenosis_percent']}% narrowing ({worst['severity_band']})."
            confidence = 0.75
        else:
            summary = "No significant stenosis detected in the analyzed frame."
            confidence = 0.6

        return {
            "overlay_image_bytes": buffer.tobytes(),
            "vessel_mask_present": bool(vessel_mask.sum() > 0),
            "lesions": lesions,
            "summary": summary,
            "confidence": confidence,
        }
