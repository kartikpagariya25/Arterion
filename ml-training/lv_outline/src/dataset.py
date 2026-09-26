import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

from config import PROCESSED_DIR, IMAGE_SIZE


def build_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.ElasticTransform(alpha=20, sigma=5, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def build_eval_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


class LVSegmentationDataset(Dataset):
    def __init__(self, split: str):
        self.images_dir = os.path.join(PROCESSED_DIR, split, "images")
        self.masks_dir = os.path.join(PROCESSED_DIR, split, "masks")
        self.filenames = sorted(os.listdir(self.images_dir))
        self.transform = build_train_transform() if split == "train" else build_eval_transform()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        image = cv2.imread(os.path.join(self.images_dir, fname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[0] != IMAGE_SIZE or image.shape[1] != IMAGE_SIZE:
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        mask = cv2.imread(os.path.join(self.masks_dir, fname), cv2.IMREAD_GRAYSCALE)
        if mask.shape[0] != IMAGE_SIZE or mask.shape[1] != IMAGE_SIZE:
            mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)

        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask
