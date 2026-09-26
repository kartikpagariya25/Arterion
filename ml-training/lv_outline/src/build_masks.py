import os
import numpy as np
import pandas as pd
import cv2
from skimage.draw import polygon

from config import ECHONET_EXTRACTED_DIR, PROCESSED_DIR


def _find_root():
    for entry in os.listdir(ECHONET_EXTRACTED_DIR):
        candidate = os.path.join(ECHONET_EXTRACTED_DIR, entry)
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "FileList.csv")):
            return candidate
    raise FileNotFoundError("Could not locate FileList.csv under the extracted EchoNet directory")


ROOT = _find_root()
VIDEOS_DIR = os.path.join(ROOT, "Videos")


def build_lv_mask(tracing_group: pd.DataFrame, height: int, width: int) -> np.ndarray:
    """tracing_group: all VolumeTracings.csv rows for one (FileName, Frame) pair.
    Row 0 is the long-axis line; the remaining rows are the perpendicular
    diameter segments used in the Simpson's-rule tracing. Concatenating one
    side forward and the other side reversed gives a closed LV boundary."""
    rows = tracing_group.iloc[1:]
    x = np.concatenate([rows["X1"].values, rows["X2"].values[::-1]])
    y = np.concatenate([rows["Y1"].values, rows["Y2"].values[::-1]])

    rr, cc = polygon(
        np.clip(np.rint(y), 0, height - 1).astype(int),
        np.clip(np.rint(x), 0, width - 1).astype(int),
        shape=(height, width),
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[rr, cc] = 1
    return mask


def extract_frame(video_path: str, frame_index: int):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def build_dataset(limit: int = None):
    file_list = pd.read_csv(os.path.join(ROOT, "FileList.csv"))
    tracings = pd.read_csv(os.path.join(ROOT, "VolumeTracings.csv"))

    split_map = dict(zip(file_list["FileName"], file_list["Split"]))
    grouped = list(tracings.groupby(["FileName", "Frame"]))
    if limit:
        grouped = grouped[:limit]

    counts = {"TRAIN": 0, "VAL": 0, "TEST": 0}
    skipped = 0

    for (filename_avi, frame_idx), group in grouped:
        stem = filename_avi.replace(".avi", "")
        split = split_map.get(stem)
        if split is None:
            skipped += 1
            continue

        video_path = os.path.join(VIDEOS_DIR, filename_avi)
        frame = extract_frame(video_path, int(frame_idx))
        if frame is None:
            skipped += 1
            continue

        height, width = frame.shape[:2]
        mask = build_lv_mask(group, height, width)

        out_dir = os.path.join(PROCESSED_DIR, split.lower())
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)

        out_name = f"{stem}_{int(frame_idx)}.png"
        cv2.imwrite(os.path.join(out_dir, "images", out_name), frame)
        cv2.imwrite(os.path.join(out_dir, "masks", out_name), mask * 255)
        counts[split] += 1

    print(f"processed: {counts}, skipped: {skipped}")


def save_spotcheck(split: str = "train", n: int = 5):
    out_dir = os.path.join(PROCESSED_DIR, split)
    images_dir = os.path.join(out_dir, "images")
    masks_dir = os.path.join(out_dir, "masks")
    spotcheck_dir = os.path.join(PROCESSED_DIR, "_spotcheck", split)
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
    import sys

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    build_dataset(limit=limit)
    save_spotcheck("train")
