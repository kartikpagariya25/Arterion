import tempfile
import numpy as np
import cv2
import onnxruntime as ort

from config import CLIP_LENGTH, FRAME_SIZE

KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
KINETICS_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


def classify_ef(ef: float) -> str:
    if ef >= 55:
        return "normal"
    if ef >= 45:
        return "mildly_reduced"
    if ef >= 30:
        return "moderately_reduced"
    return "severely_reduced"


class EFAnalyzer:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def _read_video(self, video_bytes: bytes) -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as f:
            f.write(video_bytes)
            path = f.name

        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.shape[0] != FRAME_SIZE or frame.shape[1] != FRAME_SIZE:
                frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
            frames.append(frame)
        cap.release()
        return np.stack(frames)

    def _sample_clip(self, video: np.ndarray) -> np.ndarray:
        n = video.shape[0]
        if n <= CLIP_LENGTH:
            indices = list(range(n)) + [n - 1] * (CLIP_LENGTH - n)
        else:
            start = (n - CLIP_LENGTH) // 2
            indices = list(range(start, start + CLIP_LENGTH))
        return video[indices]

    def analyze(self, video_bytes: bytes) -> dict:
        video = self._read_video(video_bytes)
        clip = self._sample_clip(video)

        clip = clip.astype(np.float32) / 255.0
        clip = (clip - KINETICS_MEAN) / KINETICS_STD
        clip = clip.transpose(3, 0, 1, 2)[None, ...].astype(np.float32)

        ef_normalized = self.session.run(["ef"], {"clip": clip})[0]
        ef = float(ef_normalized[0]) * 100

        classification = classify_ef(ef)
        summary = f"Estimated EF {ef:.1f}% — {classification.replace('_', ' ')}."

        return {
            "ejection_fraction": round(ef, 1),
            "classification": classification,
            "summary": summary,
            "confidence": 0.8,
        }
