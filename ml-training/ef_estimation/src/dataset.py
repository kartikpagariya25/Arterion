import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset

from config import ECHONET_EXTRACTED_DIR, CLIP_LENGTH, FRAME_SIZE

KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
KINETICS_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


def _find_root():
    for entry in os.listdir(ECHONET_EXTRACTED_DIR):
        candidate = os.path.join(ECHONET_EXTRACTED_DIR, entry)
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "FileList.csv")):
            return candidate
    raise FileNotFoundError("Could not locate FileList.csv under the extracted EchoNet directory")


ROOT = _find_root()
VIDEOS_DIR = os.path.join(ROOT, "Videos")
FILE_LIST = pd.read_csv(os.path.join(ROOT, "FileList.csv"))


def _read_video(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def window_starts(n: int, clip_length: int, period: int, k: int) -> list:
    """Start indices of up to k evenly spread windows; a window spans (clip_length-1)*period+1 frames."""
    span = (clip_length - 1) * period + 1
    if n <= span:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, n - span, k)})


def _clip_at(video: np.ndarray, start: int, clip_length: int, period: int) -> np.ndarray:
    n = video.shape[0]
    return video[[min(start + i * period, n - 1) for i in range(clip_length)]]  # pad short videos with the last frame


def _sample_clip(video: np.ndarray, clip_length: int, train: bool, period: int = 1) -> np.ndarray:
    n = video.shape[0]
    span = (clip_length - 1) * period + 1
    if n <= span:
        start = 0
    else:
        start = np.random.randint(0, n - span + 1) if train else (n - span) // 2
    return _clip_at(video, start, clip_length, period)


def to_tensor(clip: np.ndarray) -> np.ndarray:
    clip = (clip.astype(np.float32) / 255.0 - KINETICS_MEAN) / KINETICS_STD
    return clip.transpose(3, 0, 1, 2)


class EFDataset(Dataset):
    def __init__(self, split: str, period: int = 1):
        self.period = period
        self.split = split.upper()
        self.df = FILE_LIST[FILE_LIST["Split"] == self.split].reset_index(drop=True)
        self.train = self.split == "TRAIN"

    def __len__(self):
        return len(self.df)

    def load_video(self, idx):
        row = self.df.iloc[idx]
        video = _read_video(os.path.join(VIDEOS_DIR, f"{row['FileName']}.avi"))
        if video.shape[1] != FRAME_SIZE or video.shape[2] != FRAME_SIZE:
            video = np.stack([cv2.resize(f, (FRAME_SIZE, FRAME_SIZE)) for f in video])
        return video, float(row["EF"]) / 100.0

    def __getitem__(self, idx):
        video, ef = self.load_video(idx)
        clip = to_tensor(_sample_clip(video, CLIP_LENGTH, self.train, self.period))
        row = self.df.iloc[idx]

        return torch.from_numpy(clip).float(), torch.tensor(ef, dtype=torch.float32)
