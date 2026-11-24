"""Reference inference: list of (path, view) videos -> dominance. Used as the template for the backend."""
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

from config import ARTIFACTS_DIR

SIZE, CACHE_SIZE, NFRAMES = 160, 192, 8
MEAN, STD = 0.449, 0.226
VIEW_ID = {"RCA": 0, "LCA": 1}


def load_frames(path: str) -> np.ndarray:
    """.npz (key pixel_array) or any video file -> (8, 192, 192) uint8 grayscale."""
    if path.lower().endswith(".npz"):
        arr = np.load(path)["pixel_array"]
        if arr.ndim == 4:
            arr = arr[..., 0]
    else:
        cap, frames = cv2.VideoCapture(path), []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        cap.release()
        if not frames:
            raise ValueError("no frames could be read from the file")
        arr = np.stack(frames)
    idx = np.linspace(0, arr.shape[0] - 1, NFRAMES).round().astype(int)
    return np.stack([cv2.resize(arr[i], (CACHE_SIZE, CACHE_SIZE), interpolation=cv2.INTER_AREA) for i in idx])


def predict(session, videos: list) -> dict:
    clips, views = [], []
    for path, view in videos:
        arr = load_frames(path)
        sel = np.linspace(0, arr.shape[0] - 1, 4).round().astype(int)
        clip = np.stack([cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_AREA) for f in arr[sel]])
        clip = (clip.astype(np.float32) / 255.0 - MEAN) / STD
        clips.append(np.repeat(clip[:, None], 3, axis=1))
        views.append(VIEW_ID[view.upper()])
    logits = session.run(["logits"], {"frames": np.stack(clips)[None].astype(np.float32),
                                       "views": np.array([views], dtype=np.int64)})[0][0]
    e = np.exp(logits - logits.max())
    p_left = float((e / e.sum())[1])
    return {
        "dominance_type": "left" if p_left > 0.5 else "right",
        "confidence": round(max(p_left, 1 - p_left), 4),
        "p_left": round(p_left, 4),
        "views_used": sorted({v.upper() for _, v in videos}),
    }


if __name__ == "__main__":
    sess = ort.InferenceSession(os.path.join(ARTIFACTS_DIR, "dominance_model.onnx"), providers=["CPUExecutionProvider"])
    print(predict(sess, [(sys.argv[1], sys.argv[2])]))
