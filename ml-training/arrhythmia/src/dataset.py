import numpy as np
import torch
from torch.utils.data import Dataset


class BeatDataset(Dataset):
    def __init__(self, npz_path, rr_mean=None, rr_std=None, augment=False):
        data = np.load(npz_path)
        self.beats = data["beats"]
        self.rr = data["rr"]
        self.labels = data["labels"]
        self.augment = augment

        if rr_mean is not None and rr_std is not None:
            self.rr = (self.rr - rr_mean) / (rr_std + 1e-8)

    def __len__(self):
        return len(self.labels)

    def _apply_augmentations(self, beat: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            beat = beat * np.random.uniform(0.9, 1.1)

        if np.random.rand() < 0.5:
            noise_std = np.random.uniform(0.01, 0.05)
            beat = beat + np.random.normal(0, noise_std, size=beat.shape)

        if np.random.rand() < 0.5:
            shift = np.random.randint(-5, 6)
            beat = np.roll(beat, shift)
            if shift > 0:
                beat[:shift] = beat[shift]
            elif shift < 0:
                beat[shift:] = beat[shift - 1]

        if np.random.rand() < 0.5:
            freq = np.random.uniform(0.1, 0.5)
            amp = np.random.uniform(0, 0.1)
            t = np.arange(len(beat)) / 360.0
            beat = beat + amp * np.sin(2 * np.pi * freq * t)

        if np.random.rand() < 0.3:
            kernel = np.ones(3) / 3
            beat = np.convolve(beat, kernel, mode="same")

        return beat.astype(np.float32)

    def __getitem__(self, idx):
        beat = self.beats[idx].copy()
        rr = self.rr[idx]
        label = self.labels[idx]

        if self.augment:
            beat = self._apply_augmentations(beat)

        beat = torch.tensor(beat, dtype=torch.float32).unsqueeze(0)
        rr = torch.tensor(rr, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return beat, rr, label
