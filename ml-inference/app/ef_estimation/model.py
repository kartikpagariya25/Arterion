import os
import tempfile
import numpy as np
import cv2
import requests

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
MODEL_PATH = os.path.join(WEIGHTS_DIR, "ef_model.onnx")

CLIP_LENGTH = 32
FRAME_SIZE = 112
KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
KINETICS_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

_session = None


def _model_available() -> bool:
    return os.path.exists(MODEL_PATH)


def load_model():
    global _session
    if _session is None and _model_available():
        import onnxruntime as ort

        _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return _session


def classify_ef(ef: float) -> str:
    if ef >= 55:
        return "normal"
    if ef >= 45:
        return "mildly_reduced"
    if ef >= 30:
        return "moderately_reduced"
    return "severely_reduced"


def _download_video(file_url: str) -> bytes:
    response = requests.get(file_url, timeout=60)
    response.raise_for_status()
    return response.content


def _read_video(video_bytes: bytes) -> np.ndarray:
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


def _sample_clip(video: np.ndarray) -> np.ndarray:
    n = video.shape[0]
    if n <= CLIP_LENGTH:
        indices = list(range(n)) + [n - 1] * (CLIP_LENGTH - n)
    else:
        start = (n - CLIP_LENGTH) // 2
        indices = list(range(start, start + CLIP_LENGTH))
    return video[indices]


def predict(file_url: str) -> dict:
    load_model()

    if _session is None:
        return {
            "ejection_fraction": 0.0,
            "classification": "unknown",
            "summary": "EF estimation model weights not found — training not yet completed.",
            "confidence": 0.0,
        }

    video_bytes = _download_video(file_url)
    video = _read_video(video_bytes)
    clip = _sample_clip(video)

    clip = clip.astype(np.float32) / 255.0
    clip = (clip - KINETICS_MEAN) / KINETICS_STD
    clip = clip.transpose(3, 0, 1, 2)[None, ...].astype(np.float32)

    ef_normalized = _session.run(["ef"], {"clip": clip})[0]
    ef = float(ef_normalized[0]) * 100

    classification = classify_ef(ef)
    summary = f"Estimated EF {ef:.1f}% — {classification.replace('_', ' ')}."

    return {
        "ejection_fraction": round(ef, 1),
        "classification": classification,
        "summary": summary,
        "confidence": 0.8,
    }
