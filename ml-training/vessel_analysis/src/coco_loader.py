import os
import numpy as np
import cv2
from pycocotools.coco import COCO

from config import (
    SEGMENTATION_SPLITS, STENOSIS_SPLITS, PROCESSED_DIR,
    VESSEL_CATEGORY_IDS, STENOSIS_CATEGORY_ID, IMAGE_SIZE,
)


def build_mask(coco: COCO, image_id: int, category_ids: list, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    ann_ids = coco.getAnnIds(imgIds=image_id, catIds=category_ids)
    for ann in coco.loadAnns(ann_ids):
        ann_mask = coco.annToMask(ann)
        mask = np.maximum(mask, ann_mask)
    return mask


def process_split(task: str, split: str) -> int:
    splits = SEGMENTATION_SPLITS if task == "segmentation" else STENOSIS_SPLITS
    category_ids = VESSEL_CATEGORY_IDS if task == "segmentation" else [STENOSIS_CATEGORY_ID]

    images_dir = splits[split]["images"]
    ann_path = splits[split]["annotations"]

    coco = COCO(ann_path)
    out_dir = os.path.join(PROCESSED_DIR, task, split)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)

    count = 0
    positive_count = 0
    for image_id in coco.getImgIds():
        info = coco.loadImgs(image_id)[0]
        img_path = os.path.join(images_dir, info["file_name"])
        image = cv2.imread(img_path)
        if image is None:
            print(f"warning: could not read {img_path}, skipping")
            continue

        mask = build_mask(coco, image_id, category_ids, info["height"], info["width"])
        if mask.sum() > 0:
            positive_count += 1

        if image.shape[0] != IMAGE_SIZE or image.shape[1] != IMAGE_SIZE:
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

        stem = os.path.splitext(info["file_name"])[0]
        cv2.imwrite(os.path.join(out_dir, "images", f"{stem}.png"), image)
        cv2.imwrite(os.path.join(out_dir, "masks", f"{stem}.png"), mask * 255)
        count += 1

    print(f"{task}/{split}: processed {count} images, {positive_count} with a non-empty mask")
    return count


def save_overlay_spotcheck(task: str, split: str, n: int = 5):
    out_dir = os.path.join(PROCESSED_DIR, task, split)
    images_dir = os.path.join(out_dir, "images")
    masks_dir = os.path.join(out_dir, "masks")
    spotcheck_dir = os.path.join(PROCESSED_DIR, "_spotcheck", task, split)
    os.makedirs(spotcheck_dir, exist_ok=True)

    filenames = sorted(os.listdir(images_dir))[:n]
    for fname in filenames:
        image = cv2.imread(os.path.join(images_dir, fname))
        mask = cv2.imread(os.path.join(masks_dir, fname), cv2.IMREAD_GRAYSCALE)

        overlay = image.copy()
        overlay[mask > 0] = [0, 0, 255]
        blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
        cv2.imwrite(os.path.join(spotcheck_dir, fname), blended)

    print(f"saved {len(filenames)} spot-check overlays to {spotcheck_dir}")


if __name__ == "__main__":
    for task in ("segmentation", "stenosis"):
        for split in ("train", "val", "test"):
            process_split(task, split)
        save_overlay_spotcheck(task, "train")
