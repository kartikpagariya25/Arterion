"""Coronary dominance dataset: study index, stratified study-level splits, cached frames."""
import hashlib
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from config import EXTRACTED_DIR, PROCESSED_DIR

CACHE_DIR = os.path.join(PROCESSED_DIR, "cache")
INDEX_PATH = os.path.join(PROCESSED_DIR, "index.json")
SPLITS_PATH = os.path.join(PROCESSED_DIR, "splits.json")
VIEWS = {"RCA": 0, "LCA": 1}
LABELS = {"Right_Dominance": 0, "Left_Dominance": 1}  # 0 = right, 1 = left
CACHED_FRAMES = 8
CACHE_SIZE = 192
MEAN, STD = 0.449, 0.226  # ImageNet grayscale-ish stats


def cache_name(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest() + ".npy"


def scan_index(root: str = EXTRACTED_DIR) -> list:
    """Walk <root>/**/{Left,Right}_Dominance/<study>/{RCA,LCA}/*.npz into study records."""
    studies = {}
    for dirpath, _, files in os.walk(root):
        parts = dirpath.replace("\\", "/").split("/")
        if len(parts) < 3 or parts[-1] not in VIEWS or parts[-3] not in LABELS:
            continue
        view, study, label_dir = parts[-1], parts[-2], parts[-3]
        rec = studies.setdefault(
            study, {"study": study, "label": LABELS[label_dir], "videos": []}
        )
        for f in sorted(files):
            if f.endswith(".npz"):
                rec["videos"].append({"view": view, "path": os.path.join(dirpath, f)})
    return sorted(studies.values(), key=lambda r: r["study"])


def build_splits(index: list, seed: int = 42, val_frac: float = 0.15, test_frac: float = 0.15) -> dict:
    """Stratified study-level split (no study appears in more than one split)."""
    rng = random.Random(seed)
    out = {"train": [], "val": [], "test": []}
    for label in sorted({r["label"] for r in index}):
        ids = sorted(r["study"] for r in index if r["label"] == label)
        rng.shuffle(ids)
        n_test = round(len(ids) * test_frac)
        n_val = round(len(ids) * val_frac)
        out["test"] += ids[:n_test]
        out["val"] += ids[n_test:n_test + n_val]
        out["train"] += ids[n_test + n_val:]
    return out


class DiskCache:
    """Mapping from original npz path to its cached (CACHED_FRAMES, S, S) uint8 array."""

    def __getitem__(self, path):
        return np.load(os.path.join(CACHE_DIR, cache_name(path)))


def _pick(n_available: int, n_wanted: int, train: bool) -> np.ndarray:
    if train:
        idx = np.sort(np.random.choice(n_available, n_wanted, replace=n_available < n_wanted))
    else:
        idx = np.linspace(0, n_available - 1, n_wanted).round().astype(int)
    return idx


def study_to_tensors(study, cache, max_videos=4, frames_per_video=4, size=160, train=False):
    """Return frames float32 (V, F, 3, size, size) and views int64 (V,)."""
    videos = study["videos"]
    order = _pick(len(videos), min(max_videos, len(videos)), train) if len(videos) > 1 else np.array([0])
    if not train and len(videos) > max_videos:
        order = np.linspace(0, len(videos) - 1, max_videos).round().astype(int)
    clips, views = [], []
    for i in np.unique(order) if not train else order:
        v = videos[int(i)]
        arr = cache[v["path"]]
        sel = _pick(arr.shape[0], frames_per_video, train)
        clip = arr[sel].astype(np.float32) / 255.0
        if clip.shape[-1] != size:
            import cv2
            clip = np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in clip])
        if train:
            clip = np.clip(clip * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.08, 0.08), 0, 1)
            shift = np.random.randint(-8, 9, size=2)
            clip = np.roll(clip, tuple(shift), axis=(1, 2))
        clip = (clip - MEAN) / STD
        clips.append(np.repeat(clip[:, None], 3, axis=1))
        views.append(VIEWS[v["view"]])
    return np.stack(clips).astype(np.float32), np.array(views, dtype=np.int64)


class StudyDataset(Dataset):
    def __init__(self, split: str, train: bool, size: int = 160, max_videos: int = 4, frames_per_video: int = 4):
        with open(INDEX_PATH) as f:
            index = {r["study"]: r for r in json.load(f)}
        with open(SPLITS_PATH) as f:
            ids = json.load(f)[split]
        self.studies = [index[i] for i in ids]
        self.train, self.size = train, size
        self.max_videos, self.frames_per_video = max_videos, frames_per_video
        self.cache = DiskCache()

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        s = self.studies[i]
        frames, views = study_to_tensors(
            s, self.cache, self.max_videos, self.frames_per_video, self.size, self.train
        )
        return torch.from_numpy(frames), torch.from_numpy(views), s["label"]


def collate(batch):
    """Pad variable video counts to the batch max; mask marks real videos."""
    vmax = max(b[0].shape[0] for b in batch)
    f = batch[0][0].shape[1:]
    frames = torch.zeros(len(batch), vmax, *f)
    views = torch.zeros(len(batch), vmax, dtype=torch.long)
    mask = torch.zeros(len(batch), vmax, dtype=torch.bool)
    for i, (fr, vw, _) in enumerate(batch):
        n = fr.shape[0]
        frames[i, :n], views[i, :n], mask[i, :n] = fr, vw, True
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return frames, views, mask, labels
