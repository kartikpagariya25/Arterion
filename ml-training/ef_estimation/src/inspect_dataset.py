import os
import glob
import zipfile
import pandas as pd
import cv2

from config import ECHONET_RAW_DIR, ECHONET_EXTRACTED_DIR


def extract():
    if os.path.exists(ECHONET_EXTRACTED_DIR) and os.listdir(ECHONET_EXTRACTED_DIR):
        print("already extracted, skipping")
        return

    zip_candidates = glob.glob(os.path.join(ECHONET_RAW_DIR, "*.zip"))
    if not zip_candidates:
        raise FileNotFoundError(
            f"No .zip found in {ECHONET_RAW_DIR} — place the downloaded EchoNet-Dynamic archive there first."
        )

    zip_path = zip_candidates[0]
    print(f"extracting {zip_path} (7GB can take several minutes)...")
    os.makedirs(ECHONET_EXTRACTED_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(ECHONET_EXTRACTED_DIR)
    print("extraction complete")


def find_file(name: str) -> str:
    matches = glob.glob(os.path.join(ECHONET_EXTRACTED_DIR, "**", name), recursive=True)
    if not matches:
        raise FileNotFoundError(f"{name} not found under {ECHONET_EXTRACTED_DIR}")
    return matches[0]


def inspect_csv(name: str):
    path = find_file(name)
    print(f"\n--- {path} ---")
    df = pd.read_csv(path)
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print(df.head(3).to_string())
    if "Split" in df.columns:
        print("\nSplit value counts:")
        print(df["Split"].value_counts())
    return df


def inspect_sample_video(file_list_df: pd.DataFrame):
    videos_dirs = glob.glob(os.path.join(ECHONET_EXTRACTED_DIR, "**", "Videos"), recursive=True)
    if not videos_dirs:
        print("\nno 'Videos' directory found — check extracted structure manually")
        return

    videos_dir = videos_dirs[0]
    print(f"\n--- sample video check ({videos_dir}) ---")
    sample_files = os.listdir(videos_dir)[:3]
    for fname in sample_files:
        path = os.path.join(videos_dir, fname)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        print(f"{fname}: {width}x{height}, {fps:.1f} fps, {frame_count} frames")


if __name__ == "__main__":
    extract()
    file_list = inspect_csv("FileList.csv")
    inspect_csv("VolumeTracings.csv")
    inspect_sample_video(file_list)
