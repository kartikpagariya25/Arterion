import numpy as np
import cv2
import onnxruntime as ort

from config import IMAGE_SIZE

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class LVOutlineAnalyzer:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        return normalized.transpose(2, 0, 1)[None, ...].astype(np.float32)

    def analyze(self, image: np.ndarray) -> dict:
        tensor = self._preprocess(image)
        logits = self.session.run(["logits"], {"image": tensor})[0]
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
            "overlay_image_bytes": buffer.tobytes(),
            "lv_area_pixels": area,
            "summary": "Left ventricle outline generated for the analyzed frame."
                       if area > 0 else "No clear left ventricle boundary detected in this frame.",
            "confidence": 0.9 if area > 0 else 0.3,
        }
